"""Channel F2 — neural in-silico spectrum prediction (ICEBERG 2.1, pretrained).

Complements the rule-based channel F (`insilico.py`): instead of matching
observed peaks against an unweighted theoretical m/z list, a neural model
trained on ~1M real MS/MS spectra predicts a full *peak list with
intensities* for each candidate structure, conditioned on adduct, collision
energy (eV) and instrument. The predicted spectrum is compared to the
observed query spectrum; match statistics become re-ranker features.

Model: ICEBERG 2.1 (Wang et al., bioRxiv 2025.05.28.656653; code MIT,
github.com/coleygroup/ms-pred), zero-shot with the public *MassSpecGym*
checkpoint (msg_all). No training of this channel happens here — the weights
are frozen; this module only runs inference.

Provenance & leakage notes (see README §F2):
  * The checkpoint was trained ONLY on public experimental MS/MS data
    (MassSpecGym 1.5: GNPS/MassBank/MoNA-type repositories) — no NIST, no
    Enveda data. Class-2 answers are never-measured molecules, so their
    spectra exist in no corpus: no answer leakage is possible.
  * Measured *decoys*, however, may appear in that public data (soft
    memorization). The re-ranker must be trained under the honest protocol:
    the pseudo-truth's rows must be treated as unmeasured by this channel
    too (see R1/R2 in the README — extend them to F2 before trusting local
    numbers).

Environment: runs inside the vendored ms-pred environment, NOT the repo
venv (different rdkit/torch pins). From the repo root:

    extern/ms-pred/.venv/bin/python max_class2_exp/nnspec.py --help

Weights live in `out/iceberg/` (git-ignored); auto-discovered below.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "extern" / "ms-pred" / "src"))

WEIGHTS_DIR = HERE / "out" / "iceberg"
DEFAULT_INSTRUMENT = "QTOF"  # timsTOF is a QTOF; ICEBERG's instrument vocab: Orbitrap/QTOF/IT-FT/Unknown

# Competition adduct -> (precursor m/z offset from the neutral monoisotopic
# mass, charge). Same table as src/enveda_casmi26/fragmentation.py ADDUCTS.
ADDUCT_MZ_OFFS = {
    "[M+H]+": (1.00727647, 1),
    "[M+Na]+": (22.989220, 1),
    "[M+K]+": (38.963158, 1),
    "[M+NH4]+": (18.033823, 1),
    "[M+Cl]-": (34.969402, 1),
    "[M+CH2O2-H]-": (44.998201, 1),   # formate, a.k.a. [M+CHO2]-
    "[M-H]-": (-1.00727647, 1),
    "[M+H-H2O]+": (-17.003288, 1),
    "[M+H-2H2O]+": (-35.013853, 1),
    "[M-H-H2O]-": (-19.017841, 1),
}

# Competition adduct -> name in ICEBERG's adduct vocabulary
# (ms_pred.common.chem_utils.ion2onehot_pos). All ten are covered.
ICEBERG_ADDUCT = {
    "[M+H]+": "[M+H]+",
    "[M+Na]+": "[M+Na]+",
    "[M+K]+": "[M+K]+",
    "[M+NH4]+": "[M+NH4]+",
    "[M+Cl]-": "[M+Cl]-",
    "[M+CH2O2-H]-": "[M+CHO2]-",
    "[M-H]-": "[M-H]-",
    "[M+H-H2O]+": "[M+H-H2O]+",
    "[M+H-2H2O]+": "[M+H-2H2O]+",
    "[M-H-H2O]-": "[M-H-H2O]-",
}

# Raw train adduct names -> the competition adduct they predict like
# (same aliases as src/enveda_casmi26/fragmentation.py). Dimer spectra are
# predicted as the monomer adduct: fragments below the monomer mass behave
# alike; dimer-region peaks simply stay unmatched.
ADDUCT_ALIASES = {
    "[2M+H]+": "[M+H]+",
    "[2M+Na]+": "[M+Na]+",
    "[2M+K]+": "[M+K]+",
    "[2M+NH4]+": "[M+NH4]+",
    "[2M-H]-": "[M-H]-",
    "[M-H2O+H]+": "[M+H-H2O]+",
    "[M-2H2O+H]+": "[M+H-2H2O]+",
    "[M-H2O-H]-": "[M-H-H2O]-",
    "[M-H2O]+": "[M+H-H2O]+",
}

_PRED_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray] | None] = {}
_MAX_CACHE = 200_000

_MODEL = None  # lazy singleton


def _find_ckpt(kind: str) -> Path:
    """Locate the ICEBERG gen/inten checkpoint under out/iceberg (msg_all)."""
    if kind == "gen":
        pats = ("**/gen/best.ckpt", "**/gen/*.ckpt")
    else:
        pats = ("**/inten*/best.ckpt", "**/inten*/*.ckpt")
    for pat in pats:
        hits = sorted(WEIGHTS_DIR.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No {kind} checkpoint under {WEIGHTS_DIR} — see max_class2_exp/README.md (F2 setup)")


def _get_model():
    """Lazy-load the joint ICEBERG gen+inten model (CPU). Heavy: ~10 s."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import pathlib
    import torch
    import pytorch_lightning as pl
    from ms_pred.iceberg import gen_model, inten_model, joint_model
    # torch>=2.6 loads checkpoints with weights_only=True by default; the
    # ICEBERG checkpoints pickle pathlib objects in their hparams.
    torch.serialization.add_safe_globals(
        [pathlib.PosixPath, pathlib.PurePosixPath, pathlib.Path])
    def _load(cls, path):
        try:
            return cls.load_from_checkpoint(path)
        except Exception:
            # trusted source: MIT-licensed public research weights
            return cls.load_from_checkpoint(path, weights_only=False)
    gen = _load(gen_model.FragGNN, str(_find_ckpt("gen")))
    inten = _load(inten_model.IntenGNN, str(_find_ckpt("inten")))
    _MODEL = joint_model.JointModel(gen_model_obj=gen, inten_model_obj=inten)
    _MODEL.eval()
    _MODEL.freeze()
    return _MODEL


def _canon_and_mass(smis: list[str]) -> list[tuple[str, float] | None]:
    """Canonicalize SMILES (single fragment only) and get the exact mass."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    out = []
    for smi in smis:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None or "." in smi:
                out.append(None)
            else:
                out.append((Chem.MolToSmiles(mol), rdMolDescriptors.CalcExactMolWt(mol)))
        except Exception:
            out.append(None)
    return out


def predict_spectra(entries, threshold: float = 0.0, top_k: int = 100,
                     max_nodes: int = 100) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Predict MS2 spectra for a batch of candidates.

    entries: list of (smiles, adduct, ce_ev); adduct from ADDUCT_MZ_OFFS.
    Returns one (mz, intensity) tuple per entry — intensities normalized to
    sum 1, peaks sorted by m/z — or None where the SMILES is unparseable /
    the model declined the input.
    """
    model = _get_model()
    canon, ces, adducts, precs = [], [], [], []
    keep = []
    parsed = _canon_and_mass([e[0] for e in entries])
    for i, (smi, adduct, ce) in enumerate(entries):
        adduct = ADDUCT_ALIASES.get(adduct, adduct)
        if adduct not in ADDUCT_MZ_OFFS or parsed[i] is None:
            continue
        off, _z = ADDUCT_MZ_OFFS[adduct]
        canon.append(parsed[i][0])
        adducts.append(ICEBERG_ADDUCT[adduct])
        ces.append(float(ce))
        precs.append(float(parsed[i][1] + off))
        keep.append(i)
    res: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(entries)
    if not keep:
        return res
    out = model.predict_mol(
        canon, precursor_mz=precs, collision_eng=ces, adduct=adducts,
        instrument=[DEFAULT_INSTRUMENT] * len(canon), threshold=threshold,
        device="cpu", max_nodes=max_nodes, adduct_shift=False,
        canonical_root_smi=True, binned_out=False)
    for j, i in enumerate(keep):
        spec = out["spec"][j].detach().cpu().numpy()
        if spec.ndim != 2 or spec.shape[0] == 0:
            continue
        order = np.argsort(spec[:, 1])[::-1][:top_k]     # keep top-k by intensity
        peaks = spec[order]
        s = peaks[:, 1].sum()
        if not np.isfinite(s) or s <= 0:
            continue
        peaks[:, 1] = peaks[:, 1] / s
        # merge near-identical masses (H-shift branches often give the same
        # m/z twice); 0.001 Da is far below any real formula separation here
        uniq, inv = np.unique(np.round(peaks[:, 0], 3), return_inverse=True)
        it = np.zeros(len(uniq), dtype=float)
        np.add.at(it, inv, peaks[:, 1])
        res[i] = (uniq[np.argsort(uniq)], it[np.argsort(uniq)])
    return res


def predict_spectrum(smiles: str, adduct: str, ce_ev: float,
                     threshold: float = 0.0, top_k: int = 100
                     ) -> tuple[np.ndarray, np.ndarray] | None:
    """Cached single prediction, mirroring insilico.fragment_mzs's contract."""
    key = (smiles, adduct, round(float(ce_ev), 1), threshold, top_k)
    if key in _PRED_CACHE:
        return _PRED_CACHE[key]
    (r,) = predict_spectra([(smiles, adduct, ce_ev)], threshold=threshold, top_k=top_k)
    if len(_PRED_CACHE) >= _MAX_CACHE:
        _PRED_CACHE.clear()
    _PRED_CACHE[key] = r
    return r


def matched_cosine(q_mz: np.ndarray, q_p: np.ndarray,
                   p_mz: np.ndarray, p_p: np.ndarray, ppm: float) -> float:
    """Cosine between an observed and a predicted spectrum with ppm matching.

    Each predicted peak is matched to its closest observed peak within
    tolerance; unmatched peaks on either side reduce the score (they are in
    the norms but contribute nothing to the dot product).
    """
    q_mz, q_p = np.asarray(q_mz, float), np.asarray(q_p, float)
    p_mz, p_p = np.asarray(p_mz, float), np.asarray(p_p, float)
    if len(q_mz) == 0 or len(p_mz) == 0:
        return 0.0
    pos = np.searchsorted(p_mz, q_mz)
    left = np.clip(pos - 1, 0, len(p_mz) - 1)
    right = np.clip(pos, 0, len(p_mz) - 1)
    d_lo, d_hi = np.abs(q_mz - p_mz[left]), np.abs(q_mz - p_mz[right])
    take_left = d_lo <= d_hi
    j = np.where(take_left, left, right)          # nearest predicted peak
    d = np.where(take_left, d_lo, d_hi)
    tol = np.maximum(q_mz * ppm * 1e-6, 1e-9)     # tolerance scales with m/z
    hit = d <= tol
    dot = float((q_p[hit] * p_p[j[hit]]).sum())
    return dot / (float(np.sqrt((q_p ** 2).sum() * (p_p ** 2).sum())) + 1e-12)


# ---------------------------------------------------------------- CLI / demo

def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--adduct", default="[M+H]+")
    ap.add_argument("--ce", type=float, default=40.0, help="collision energy in eV")
    ap.add_argument("--ppm", type=float, default=20.0)
    ap.add_argument("--compare-train", action="store_true",
                    help="score the prediction against this molecule's measured "
                         "spectra in data/train.parquet (same adduct)")
    args = ap.parse_args()

    t0 = time.time()
    r = predict_spectrum(args.smiles, args.adduct, args.ce)
    dt = time.time() - t0
    if r is None:
        print(f"prediction unavailable for {args.smiles!r} ({args.adduct})")
        return
    mz, it = r
    print(f"predicted {len(mz)} peaks for {args.smiles} {args.adduct} @ {args.ce:.0f} eV "
          f"({dt:.1f}s incl. model load)")
    top = np.argsort(it)[::-1][:10]
    for i in sorted(top, key=lambda i: mz[i]):
        print(f"  {mz[i]:10.4f}  {it[i]:6.3f}")

    if args.compare_train:
        import pandas as pd
        from rdkit import Chem
        mol = Chem.MolFromSmiles(args.smiles)
        canon = Chem.MolToSmiles(mol) if mol else args.smiles
        df = pd.read_parquet(PROJECT_ROOT / "data" / "train.parquet",
                             columns=["normalized_smiles", "adduct",
                                      "ms2_mzs", "ms2_normalized_intensities"])
        sub = df[(df["adduct"] == args.adduct) & (df["normalized_smiles"] == canon)]
        if len(sub) == 0:
            print("no measured spectra of this molecule/adduct in train.parquet")
            return
        sims = [matched_cosine(r_[0], r_[1], mz, it, args.ppm)
                for r_ in zip(sub["ms2_mzs"], sub["ms2_normalized_intensities"])]
        print(f"vs {len(sims)} measured spectra: cosine mean {np.mean(sims):.3f} "
              f"median {np.median(sims):.3f} max {np.max(sims):.3f} (ppm {args.ppm})")


if __name__ == "__main__":
    _cli()
