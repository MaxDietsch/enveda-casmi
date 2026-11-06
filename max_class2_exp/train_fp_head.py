"""Train the channel-P v2 fingerprint head on the frozen de novo encoder.

Spectrum -> encoder (frozen, `out/denovo_v1.pt` from train_denovo.py) ->
masked-mean pooling -> MLP head -> 2048-bit Morgan r2 fingerprint (BCE) —
the same packed-bit fingerprint universe as the candidate pool, so
stage2 can score every candidate's actual bits against the prediction.

Leak-free by default, exactly like train_denovo.py: every molecule of every
dataset-creation/out/*.parquet split (plus data/test_cut.parquet when it
exists) is excluded from training, and the excluded keys are written into
the checkpoint's `meta`. stage2 (`fpchannel.build_fp_model`) refuses an
encoder/head pair whose guard does not cover the split being evaluated —
so local numbers can never be inflated by a molecule the model memorised.
`--include-eval-molecules` opts out for the final Kaggle model only (the
guard will then refuse the artifact locally, which is the point).

Run from the repo root (CPU or a visible GPU):

    uv run python max_class2_exp/train_fp_head.py
    uv run python max_class2_exp/train_fp_head.py --encoder out/denovo_v1.pt.last.pt

Embeddings are cached (`out/cache/fp_embed_cache.npz`, keyed by data file +
excluded-key hash + encoder name), so iterating on head hyperparameters is
cheap after the first pass.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from enveda_casmi26 import tracking  # noqa: E402
from enveda_casmi26.fp_model import DeNovoModel, allow_checkpoint_globals, encode_batch  # noqa: E402
from train_denovo import eval_split_keys, load_training_data, process_spectrum, resolve_data_path  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", type=Path, default=HERE / "out" / "denovo_v1.pt",
                    help="frozen encoder checkpoint (train_denovo.py output)")
    ap.add_argument("--data", type=Path, default=None,
                    help="training parquet (default: data/train_clean.parquet when "
                         "present, else data/train.parquet)")
    ap.add_argument("--out", type=Path, default=HERE / "out" / "denovo_fp_head.pt",
                    help="head checkpoint path")
    ap.add_argument("--cache", type=Path, default=HERE / "out" / "cache" / "fp_embed_cache.npz",
                    help="encoder-embedding cache (deleted/rebuilt automatically "
                         "when data or encoder changes)")
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--spectra", type=int, default=0, help="training spectra cap (0 = all)")
    ap.add_argument("--max-rows", type=int, default=0, help="read only the first N rows (smoke tests)")
    ap.add_argument("--max-peaks", type=int, default=128,
                    help="peaks per spectrum given to the encoder (as in train_denovo.py)")
    ap.add_argument("--val-structures", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-eval-molecules", action="store_true",
                    help="DISABLE the leak guard for the final Kaggle model: train on "
                         "every molecule. stage2 will refuse this artifact for local "
                         "evaluation, which is exactly what it is for")
    tracking.add_arguments(ap)
    return ap.parse_args()


def morgan_bits(keys, smiles_map, n_bits):
    """{key: packed uint64 bits} — same packing as common.fingerprint_matrix."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    out = {}
    for k, s in smiles_map.items():
        mol = Chem.MolFromSmiles(s) if s else None
        if mol is None:
            continue
        out[k] = np.packbits(gen.GetFingerprintAsNumPy(mol).astype(np.uint8)).view(np.uint64)
    return out


def embed_all(model, device, df, max_peaks, batch=256):
    """Masked-mean pooled encoder embeddings for every row of `df`."""
    mz_list, int_list = [], []
    for mzs, ints in zip(df["ms2_mzs"].values, df["ms2_normalized_intensities"].values):
        m, i = process_spectrum(mzs, ints, max_peaks)
        mz_list.append(m)
        int_list.append(i)
    prec = df["precursor_mz"].values.astype(np.float32)
    t0 = time.time()
    x = encode_batch(model, mz_list, int_list, prec, device, batch=batch)
    print(f"embedded {len(x):,} spectra in {time.time() - t0:.0f}s "
          f"(dim {x.shape[1]})", flush=True)
    return x


def run(args, tracker):
    torch.manual_seed(args.seed)
    t0 = time.time()

    if not args.encoder.exists():
        sys.exit(f"encoder checkpoint not found: {args.encoder} "
                 f"(train it with max_class2_exp/train_denovo.py first)")
    data_path = resolve_data_path(args.data)
    if not data_path.exists():
        sys.exit(f"training parquet not found: {data_path}")
    excluded = set() if args.include_eval_molecules else eval_split_keys()
    if args.include_eval_molecules:
        print("WARNING: --include-eval-molecules set — this head will see eval-split "
              "molecules; stage2's leak guard will refuse it locally. Final "
              "submission model only.")
    (train_df, _, _), (val_df, _, _), excluded_keys = load_training_data(
        data_path, args.spectra, args.max_peaks, args.seed,
        max_rows=args.max_rows, val_structures=args.val_structures,
        exclude=excluded)
    excluded_sha = hashlib.sha256("|".join(excluded_keys).encode()).hexdigest()[:16]
    # cache key must distinguish re-trainings of the SAME-named encoder
    # (a resumed run overwrites denovo_v1.pt; embeddings would go stale)
    enc_stat = args.encoder.stat()
    enc_stamp = f"{args.encoder.name}:{enc_stat.st_mtime_ns}:{enc_stat.st_size}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allow_checkpoint_globals()      # older checkpoints embed Path objects / numpy priors
    ckpt = torch.load(args.encoder, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    enc_meta = ckpt.get("meta") or {}
    if not args.include_eval_molecules:
        enc_excluded = set(enc_meta.get("excluded_keys") or [])
        gap = sorted(excluded - enc_excluded)
        if not enc_meta or gap:
            sys.exit(f"the encoder {args.encoder.name} does not cover {len(gap)} key(s) "
                     f"of the current eval splits (e.g. {gap[:3]}): it predates the leak "
                     f"guard or the splits changed. Retrain the encoder first "
                     f"(max_class2_exp/train_denovo.py) so encoder and head agree.")
    model = DeNovoModel(cfg["embed_dim"], cfg["n_layers"], cfg["n_heads"],
                        cfg["ff_dim"], vocab_size=cfg["vocab_size"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    print(f"encoder loaded (val_exact_hit={ckpt.get('val_exact_hit')}, {device})")

    # ---- embeddings (cached; keyed by data + exclusion + encoder)
    frames = [("train", train_df), ("val", val_df)]
    x = None
    if args.cache.exists():
        z = np.load(args.cache, allow_pickle=True)
        if (str(z["meta_data"]) == data_path.name
                and str(z["meta_sha"]) == excluded_sha
                and str(z["meta_enc"]) == enc_stamp
                and int(z["x"].shape[0]) == len(train_df) + len(val_df)):
            x = z["x"]
            print(f"embedding cache hit: {args.cache}")
        else:
            print("embedding cache stale (data/exclusion/encoder/size changed) -> re-embedding")
    if x is None:
        x = embed_all(model, device, pd.concat([f for _, f in frames], ignore_index=True),
                      args.max_peaks)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.cache, x=x,
                 meta_data=data_path.name, meta_sha=excluded_sha,
                 meta_enc=enc_stamp)
    n_train = len(train_df)
    x_train, x_val = x[:n_train], x[n_train:]

    # ---- labels: one fingerprint per structure, expanded to rows
    smiles_map = dict(zip(train_df["inchikey14"], train_df["normalized_smiles"]))
    smiles_map.update(dict(zip(val_df["inchikey14"], val_df["normalized_smiles"])))
    all_keys = sorted(set(train_df["inchikey14"]) | set(val_df["inchikey14"]))
    bits = morgan_bits(all_keys, smiles_map, args.n_bits)
    def _labels(keys):
        packed = np.stack([bits[k] if k in bits else np.zeros(args.n_bits // 64, np.uint64)
                           for k in keys])
        return torch.from_numpy(np.unpackbits(packed.view(np.uint8), axis=1)).float()
    y_train = _labels(train_df["inchikey14"].values)
    y_val = _labels(val_df["inchikey14"].values)
    print(f"train: {len(x_train):,} spectra / {train_df['inchikey14'].nunique():,} structures; "
          f"val: {len(x_val):,}")

    # ---- head
    head = nn.Sequential(nn.Linear(cfg["embed_dim"], args.hidden), nn.ReLU(),
                         nn.Dropout(0.2), nn.Linear(args.hidden, args.n_bits)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    xt = torch.from_numpy(x_train).to(device)
    yt = y_train.to(device)
    xv = torch.from_numpy(x_val).to(device) if len(x_val) else None
    yv = y_val.to(device) if len(x_val) else None
    best = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        head.train()
        perm = np.random.default_rng(args.seed + epoch).permutation(len(xt))
        losses = []
        for g in range(0, len(perm), args.batch_size):
            idx = perm[g: g + args.batch_size]
            logits = head(xt[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        head.eval()
        msg = f"epoch {epoch + 1}/{args.epochs} train bce {np.mean(losses):.4f}"
        if xv is not None:
            with torch.no_grad():
                vl = float(nn.functional.binary_cross_entropy_with_logits(head(xv), yv))
            msg += f" val bce {vl:.4f}"
            tracker.log_metrics({"val/bce": vl}, step=epoch + 1)
            if vl < best:
                best = vl
                uniq = train_df["inchikey14"].drop_duplicates().values
                priors = np.clip(_labels(uniq).numpy().mean(axis=0), 1e-3, 1.0 - 1e-3)
                torch.save({"state_dict": head.state_dict(),
                            "config": {"in_dim": cfg["embed_dim"], "hidden": args.hidden,
                                       "n_bits": args.n_bits},
                            "priors": torch.from_numpy(priors), "val_bce": vl,
                            "meta": {"guard": "eval-molecules-v1",
                                     "excluded_keys": excluded_keys,
                                     "data": data_path.name,
                                     "encoder": args.encoder.name,
                                     "include_eval_molecules": bool(args.include_eval_molecules)}},
                           args.out)
                msg += " (saved)"
        tracker.log_metrics({"train/bce": float(np.mean(losses))}, step=epoch + 1)
        print(msg, flush=True)
    if best == float("inf"):       # no validation slice: save the final head
        uniq = train_df["inchikey14"].drop_duplicates().values
        priors = np.clip(_labels(uniq).numpy().mean(axis=0), 1e-3, 1.0 - 1e-3)
        torch.save({"state_dict": head.state_dict(),
                    "config": {"in_dim": cfg["embed_dim"], "hidden": args.hidden,
                               "n_bits": args.n_bits},
                    "priors": torch.from_numpy(priors),
                    "meta": {"guard": "eval-molecules-v1",
                             "excluded_keys": excluded_keys,
                             "data": data_path.name,
                             "encoder": args.encoder.name,
                             "include_eval_molecules": bool(args.include_eval_molecules)}},
                   args.out)
        print(f"saved (no validation slice) -> {args.out}")
    print(f"done in {time.time() - t0:.0f}s -> {args.out} (val bce {best:.4f})")
    tracker.log_params({"encoder": str(args.encoder), "n_bits": args.n_bits,
                        "hidden": args.hidden, "val_bce": round(best, 4)})
    tracker.log_artifact(str(args.out))


def main():
    args = parse_args()
    with tracking.session(args, run_name=f"class2-fp-head-{args.out.stem}",
                          tags={"script": "max_class2_exp/train_fp_head.py",
                                "model": "fp_head", "channel": "P-v2"}) as tracker:
        run(args, tracker)


if __name__ == "__main__":
    main()
