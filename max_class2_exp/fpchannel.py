"""Channel P — spectrum -> fingerprint prediction (README §3, the centerpiece).

Every pool candidate carries a 2048-bit Morgan r2 fingerprint (the packed
`fps.npy` universe, `common.fingerprint_matrix`). Channel P predicts that
fingerprint *from the query spectra* and scores each candidate's bits against
the prediction (features.py: `fp_ll`, `fp_tanimoto`, `fp_available`):

  fp_ll        FingerID-style log-likelihood, prior-normalised per bit:
               sum_b [c_b log(p_b / prior_b) + (1-c_b) log((1-p_b)/(1-prior_b))]
               — "everything predicted 1" cannot win, because a bit that is
               always on in the training molecules carries no information;
  fp_tanimoto  Tanimoto of the thresholded prediction vs the candidate bits;
  fp_available whether the channel produced a prediction for this query.

Two implementations, auto-selected (no flags — presence of artifacts decides):

  v1 (default, always available)   a multi-label *linear* model trained on
      the fly inside every stage2 run: tabular spectrum features (500
      intensity-weighted linear m/z bins, adduct one-hot, scaled precursor
      m/z) -> 2048 sigmoid outputs, BCE, a couple of CPU minutes. It trains
      ONLY on rows visible under the mask in force — under the split that
      already excludes every eval molecule's rows (R1), and in clean mode the
      sampled pseudo-truths' rows are hidden too before the model trains
      (R2). Leak-free by construction, asserted in `build_fp_model`.
  v2 (used when both artifacts exist)  the frozen de novo transformer
      encoder (`train_denovo.py`) + fingerprint head (`train_fp_head.py`).
      Both trainers exclude every molecule of every dataset-creation split
      *by default* and record the excluded keys in the checkpoint `meta`;
      `build_fp_model` refuses an artifact whose guard does not cover the
      split being evaluated (retrain it, or move it away to fall back to v1).

Determined per run, never cached to disk: a stale cached model is exactly
the kind of silent leak this channel is guarded against.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "dataset-creation"))

import common  # noqa: E402
from dataloader import clean_peaks, read_rows  # noqa: E402
from spectra_utils import CleanParams, clean_spectrum  # noqa: E402

# channel-P v2 artifacts (auto-detected; both must exist and pass the guard)
ENCODER_PATHS = (HERE / "out" / "denovo_v1.pt", HERE / "out" / "denovo_v1.pt.last.pt")
HEAD_PATHS = (HERE / "out" / "denovo_fp_head.pt",)

N_BITS = 2048
N_BINS = 500                       # linear m/z bins over [0, MZ_MAX]
MZ_MAX = 2000.0
FP_MAX_ROWS = 200_000              # training spectra cap (seeded subsample)
FP_EPOCHS = 2
FP_BATCH = 8192
FP_LR = 0.02
FP_ROW_COLUMNS = ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct"]

# the ten competition adducts; same tables as nnspec.py /
# src/enveda_casmi26/fragmentation.py (dimer spectra predict like the monomer)
ADDUCTS = ("[M+H]+", "[M+Na]+", "[M+K]+", "[M+NH4]+", "[M+Cl]-", "[M+CH2O2-H]-",
           "[M-H]-", "[M+H-H2O]+", "[M+H-2H2O]+", "[M-H-H2O]-")
ADDUCT_ALIASES = {
    "[2M+H]+": "[M+H]+", "[2M+Na]+": "[M+Na]+", "[2M+K]+": "[M+K]+",
    "[2M+NH4]+": "[M+NH4]+", "[2M-H]-": "[M-H]-",
    "[M-H2O+H]+": "[M+H-H2O]+", "[M-2H2O+H]+": "[M+H-2H2O]+",
    "[M-H2O-H]-": "[M-H-H2O]-", "[M-H2O]+": "[M+H-H2O]+",
}
_ADDUCT_IDX = {a: i for i, a in enumerate(ADDUCTS)}

# fork-shared state for the parallel spectrum featuriser
_FP_PARAMS: CleanParams | None = None


def ce_of(val) -> float:
    """Single collision energy in eV (mean over merged multi-CE rows; NaN if absent)."""
    arr = np.atleast_1d(np.asarray(val, dtype=float))
    ok = arr == arr
    return float(np.mean(arr[ok])) if ok.any() else float("nan")


def ce_bucket(ce) -> float:
    """CE rounded to 5 eV — the F2 prediction cache key (test CEs are the
    20/40/60 eV grid; rounding gives the cache cross-run reuse)."""
    ce = float(ce) if ce is not None else float("nan")
    if not np.isfinite(ce):
        return 40.0
    return float(np.clip(round(ce / 5.0) * 5.0, 10.0, 100.0))


def _adduct_index(adduct) -> int:
    return _ADDUCT_IDX.get(ADDUCT_ALIASES.get(adduct, adduct), len(ADDUCTS))


def _tabular(mz, p, prec, adduct) -> np.ndarray:
    """One spectrum -> (N_BINS + n_adducts + 1) float32 feature row.
    `p` must sum to 1 (clean_spectrum output); bins are intensity-weighted."""
    x = np.zeros(N_BINS + len(ADDUCTS) + 1, dtype=np.float32)
    if len(mz):
        idx = np.clip((np.asarray(mz, dtype=float) / MZ_MAX * N_BINS).astype(np.int64),
                      0, N_BINS - 1)
        np.add.at(x, idx, np.asarray(p, dtype=float))
    x[N_BINS + _adduct_index(adduct)] = 1.0
    x[-1] = float(prec) / 1000.0
    return x


def _featurize_row(args):
    """Fork worker: raw parquet row -> tabular feature row (None if unusable).
    Mirrors the query path exactly (clean_peaks, then clean_spectrum with the
    run's CleanParams) so v1 trains on the same spectra it will score."""
    mz, it, prec, adduct = args
    m, i = clean_peaks(mz, it, float(prec))
    if m is None:
        return None
    cm, cp = clean_spectrum(m, i, float(prec), _FP_PARAMS)
    if len(cm) == 0:
        return None
    return _tabular(cm, cp, float(prec), adduct)


def bit_priors(fps: np.ndarray, mol_idx, chunk: int = 50_000) -> np.ndarray:
    """Per-bit on-rate over a set of molecules (the LLR priors)."""
    mol_idx = np.unique(np.asarray(mol_idx, dtype=np.int64))
    total = np.zeros(N_BITS)
    for g in range(0, len(mol_idx), chunk):
        sub = np.asarray(fps[mol_idx[g: g + chunk]]).view(np.uint8)
        total += np.unpackbits(sub, axis=1).sum(axis=0)
    return np.clip(total / max(1, len(mol_idx)), 1e-3, 1.0 - 1e-3)


def unpack_bits(fps: np.ndarray, rows) -> np.ndarray:
    """Packed uint64 fingerprint rows -> (n, 2048) bool bit matrix."""
    return np.unpackbits(np.asarray(fps[rows]).view(np.uint8), axis=1).astype(bool)


# ---------------------------------------------------------------------- v1

class FpLinear:
    """Multi-label linear fingerprint model (numpy weights; cheap to fork)."""
    kind = "v1-linear"

    def __init__(self, W: np.ndarray, b: np.ndarray, priors: np.ndarray, n_train: int):
        self.W, self.b, self.priors = W, b, priors
        self.n_train = n_train

    def predict_proba_batch(self, spectra_lists) -> list:
        """Mean predicted probability vector per query (2048,) or None.
        One matmul for every spectrum of every query at once."""
        flat, seg = [], []
        for spectra in spectra_lists:
            k = 0
            for s in spectra:
                if len(s[0]):
                    flat.append(_tabular(s[0], s[1], float(s[3]), s[4]))
                    k += 1
            seg.append(k)
        if not flat:
            return [None] * len(spectra_lists)
        P = 1.0 / (1.0 + np.exp(-(np.stack(flat) @ self.W.T + self.b)))
        out, pos = [], 0
        for k in seg:
            out.append(P[pos: pos + k].mean(axis=0) if k else None)
            pos += k
        return out


def train_fp_linear(rows: np.ndarray, row_key: np.ndarray, fps: np.ndarray,
                    train_path: Path, params: CleanParams, seed: int,
                    workers: int, log=common.log):
    global _FP_PARAMS
    rng = np.random.default_rng(seed + 1_000_003)      # never perturbs the sampler's rng
    if len(rows) > FP_MAX_ROWS:
        rows = np.sort(rng.choice(rows, FP_MAX_ROWS, replace=False))
    log(f"[fp] v1: featurising {len(rows):,} visible spectra "
        f"(cap {FP_MAX_ROWS:,}, seed {seed})")
    df = read_rows(rows, FP_ROW_COLUMNS, train_path).to_pandas()
    items = list(zip(df["ms2_mzs"], df["ms2_normalized_intensities"],
                     df["precursor_mz"], df["adduct"]))
    del df
    _FP_PARAMS = params
    from multiprocessing import Pool
    with Pool(min(workers, max(1, len(items)))) as p:
        feats = common.pool_map(p, _featurize_row, items, "[fp] cleaning spectra", 64)
    keep = [i for i, f in enumerate(feats) if f is not None]
    X = np.stack([feats[i] for i in keep]).astype(np.float32)
    mols = row_key[rows[keep]]
    del feats, items
    log(f"[fp] v1: {len(X):,} usable spectra of {len(np.unique(mols)):,} molecules")

    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    y_packed = np.asarray(fps[mols]).view(np.uint8)     # (n, 256) packed bits
    lin = torch.nn.Linear(X.shape[1], N_BITS)
    opt = torch.optim.AdamW(lin.parameters(), lr=FP_LR)
    n = len(X)
    for epoch in range(FP_EPOCHS):
        perm = np.random.default_rng(seed + epoch).permutation(n)
        losses = []
        for g in range(0, n, FP_BATCH):
            idx = perm[g: g + FP_BATCH]
            xb = torch.from_numpy(X[idx])
            yb = torch.from_numpy(np.unpackbits(y_packed[idx], axis=1)).float()
            loss = F.binary_cross_entropy_with_logits(lin(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        log(f"[fp] v1: epoch {epoch + 1}/{FP_EPOCHS} bce {np.mean(losses):.4f}")
    priors = bit_priors(fps, mols)
    return FpLinear(lin.weight.detach().numpy().astype(np.float32),
                    lin.bias.detach().numpy().astype(np.float32), priors, n)


# ---------------------------------------------------------------------- v2

class FpEncoder:
    """Frozen de novo encoder + fingerprint head (mean probabilities)."""
    kind = "v2-encoder"

    def __init__(self, model, head, priors: np.ndarray, device, info: dict):
        self.model, self.head, self.device = model, head, device
        self.priors, self.info = priors, info

    def predict_proba_batch(self, spectra_lists) -> list:
        from enveda_casmi26.fp_model import encode_batch, prep_spectrum
        import torch
        flat_mz, flat_int, flat_prec, seg = [], [], [], []
        for spectra in spectra_lists:
            k = 0
            for s in spectra:
                mz, ints = prep_spectrum(s[0], s[1])
                if len(mz) == 0:
                    continue
                flat_mz.append(mz)
                flat_int.append(ints)
                flat_prec.append(float(s[3]))
                k += 1
            seg.append(k)
        if not flat_mz:
            return [None] * len(spectra_lists)
        pooled = encode_batch(self.model, flat_mz, flat_int, flat_prec, self.device, batch=32)
        t = torch.from_numpy(pooled).to(next(self.head.parameters()).device)
        probs = torch.sigmoid(self.head(t)).detach().cpu().numpy()
        out, pos = [], 0
        for k in seg:
            out.append(probs[pos: pos + k].mean(axis=0) if k else None)
            pos += k
        return out


def _load_guarded(path: Path, eval_keys: set, what: str) -> dict:
    """Load a channel-P v2 checkpoint and enforce its leak guard.

    Raises RuntimeError when the artifact cannot PROVE it excluded every
    eval molecule of the split being evaluated — leak-free is the default,
    so a checkpoint that predates the guard (no `meta`) is refused too."""
    import torch
    from enveda_casmi26.fp_model import allow_checkpoint_globals
    allow_checkpoint_globals()      # older checkpoints embed Path objects / numpy priors
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    meta = ckpt.get("meta") if isinstance(ckpt, dict) else None
    excluded = (meta or {}).get("excluded_keys")
    if not isinstance(excluded, (list, tuple)):
        raise RuntimeError(
            f"[fp] {path.name} ({what}) predates the channel-P leak guard — it "
            f"cannot prove that no eval molecule was trained on. Retrain it with "
            f"max_class2_exp/train_denovo.py / train_fp_head.py (exclusion is the "
            f"default there), or move the file away to fall back to v1.")
    missing = sorted(k for k in eval_keys if k not in set(excluded))
    if missing:
        raise RuntimeError(
            f"[fp] {path.name} ({what}) was trained on {len(missing)} molecule(s) "
            f"of the split being evaluated (e.g. {missing[:3]}). Leak-free is the "
            f"default: retrain the artifact (its trainer excludes every "
            f"dataset-creation/out split by default), or move it away to use v1.")
    return ckpt


def build_fp_model(mask, row_key, train_path, fps, params, seed, workers,
                   eval_row_ids, eval_keys, query_rows=None, log=common.log):
    """The run's channel-P model: v2 artifacts when they exist (guarded),
    else a v1 linear model trained on the visible rows only.

    mask: the visibility mask in force (clean mode: split mask minus the
    pseudo-truths' rows). query_rows: the pseudo-query rows of this run
    (R2 assert, clean mode only). Returns (model | None, info dict)."""
    t0 = time.time()
    eval_keys = {str(k) for k in eval_keys}

    # ---- v2: frozen artifacts, guarded
    enc_path = next((p for p in ENCODER_PATHS if p.exists()), None)
    head_path = next((p for p in HEAD_PATHS if p.exists()), None)
    if enc_path is not None and head_path is not None:
        import torch
        from enveda_casmi26.fp_model import DeNovoModel, build_fp_head
        enc_ckpt = _load_guarded(enc_path, eval_keys, "encoder")
        head_ckpt = _load_guarded(head_path, eval_keys, "fp head")
        cfg = enc_ckpt["config"]
        model = DeNovoModel(cfg["embed_dim"], cfg["n_layers"], cfg["n_heads"],
                            cfg["ff_dim"], vocab_size=cfg["vocab_size"])
        model.load_state_dict(enc_ckpt["state_dict"])
        model.eval()
        head = build_fp_head(head_ckpt["config"])
        head.load_state_dict(head_ckpt["state_dict"])
        head.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        head.to(device)
        priors = np.asarray(head_ckpt.get("priors"), dtype=float) \
            if head_ckpt.get("priors") is not None else None
        if priors is None:      # head without stored priors: bit rates of the
            priors = bit_priors(  # currently visible molecules as a fallback
                fps, np.unique(row_key[np.asarray(mask, dtype=bool)]))
        info = {"kind": "v2-encoder", "encoder": enc_path.name, "head": head_path.name,
                "device": str(device), "priors": "stored" if head_ckpt.get("priors")
                is not None else "visible-molecule fallback",
                "seconds": round(time.time() - t0, 1)}
        log(f"[fp] channel P: v2 encoder ({enc_path.name} + {head_path.name}, "
            f"{device}) — leak guard passed for all {len(eval_keys)} eval keys")
        return FpEncoder(model, head, priors, device, info), info

    if enc_path is not None:
        log("[fp] encoder found but no fingerprint head (train it with "
            "max_class2_exp/train_fp_head.py) -> training a v1 linear model")

    # ---- v1: trained here, leak-free by construction
    rows = np.flatnonzero(np.asarray(mask, dtype=bool))
    assert not np.isin(np.asarray(eval_row_ids, dtype=np.int64), rows).any(), \
        "R1 (channel P): an eval row entered the fingerprint model's training rows"
    if query_rows is not None and len(query_rows):
        assert not np.isin(np.asarray(query_rows, dtype=np.int64), rows).any(), \
            "R2 (channel P): a pseudo-query row entered the fingerprint model's training rows"
    if not len(rows):
        log("[fp] no visible rows for the v1 model -> channel off")
        return None, {"kind": "off", "reason": "no visible rows",
                      "seconds": round(time.time() - t0, 1)}
    model = train_fp_linear(rows, row_key, fps, train_path, params, seed, workers, log)
    info = {"kind": "v1-linear", "n_train_spectra": model.n_train,
            "cap": FP_MAX_ROWS, "epochs": FP_EPOCHS,
            "seconds": round(time.time() - t0, 1)}
    log(f"[fp] channel P: v1 linear model on {model.n_train:,} spectra "
        f"({info['seconds']}s) — trained on split-visible rows only")
    return model, info
