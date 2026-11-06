"""Train the de novo spectrum -> SMILES transformer (channel P, model v2).

This is the class-2 FP channel's optional deep model (README §3, "Model v2"):
the transformer encoder defined repo-wide in ``src/enveda_casmi26/fp_model.py``
(``DeNovoModel``), trained here on visible train spectra only. Everything the
script needs lives in this folder or in the repo-wide ``src`` package — nothing
is imported from ``scripts/`` or ``max_class1_exp/``.

- GPU: CUDA is used automatically when available (``--device`` to override);
  TF32 matmuls are enabled on Ampere+ GPUs, training runs under bf16 autocast.
- MLflow: parameters, losses, generation metrics, and the checkpoint + tokenizer
  are logged through ``src/enveda_casmi26/tracking.py`` (local ``mlflow.db`` by
  default, ``--no-mlflow`` to disable, see the tracking module for remote setup).
- Smoke test: ``--spectra 3 --epochs 1 --max-rows 400 --val-structures 10`` with
  tiny model dims trains on a handful of spectra in seconds (see README §7).

Leak-free by default: every molecule of every evaluation split
(``dataset-creation/out/*.parquet`` — the difficulty splits and their
variants — plus ``data/test_cut.parquet`` when it exists) is excluded from
training, and the excluded keys are recorded in the checkpoint ``meta``.
stage2's channel-P guard refuses an artifact that cannot prove it covered
the split being evaluated, so local numbers can never be inflated by a
memorised molecule. ``--include-eval-molecules`` opts out for the final
Kaggle model only (the guard will refuse that artifact locally — by
design). Default training data is ``data/train_clean.parquet`` when present
(else ``data/train.parquet``); ``ingest_lib == 'enveda-180'`` rows are
excluded like in the organizers' tutorial, and a structure-level validation
split is carved for generation metrics.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, models, trainers, processors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from enveda_casmi26 import tracking  # noqa: E402
from enveda_casmi26.fp_model import DeNovoModel, allow_checkpoint_globals  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TRAIN_CUT = ROOT / "data" / "train_cut.parquet"
TRAIN = ROOT / "data" / "train.parquet"
TRAIN_CLEAN = ROOT / "data" / "train_clean.parquet"


# ---------------------------------------------------------------- data
def process_spectrum(mzs, intensities, max_len):
    mzs = np.asarray(mzs, dtype=np.float32)
    ints = np.asarray(intensities, dtype=np.float32)
    if len(mzs) == 0:
        return mzs, ints
    order = np.argsort(ints)[::-1]
    mzs = mzs[order][: max_len - 1]
    ints = ints[order][: max_len - 1]
    if ints.max() > 0:
        ints = ints / ints.max()
    return mzs, ints


def read_spectra(path, cols, max_rows=0):
    """Read the training parquet; with max_rows > 0 only the leading rows are read."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not max_rows:
        return pd.read_parquet(path, columns=cols)
    batches, n = [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000, columns=cols):
        batches.append(batch)
        n += batch.num_rows
        if n >= max_rows:
            break
    if not batches:
        return pd.DataFrame(columns=cols)
    return pa.concat_batches(batches).slice(0, min(max_rows, n)).to_pandas()


def resolve_data_path(path=None):
    """Default training data: the cleaned train parquet when present, else
    the raw one (explicit --data always wins; train_cut.parquet is used only
    when passed explicitly)."""
    if path is not None:
        return Path(path)
    return TRAIN_CLEAN if TRAIN_CLEAN.exists() else TRAIN


def eval_split_keys() -> set:
    """Every molecule key held by any local evaluation set: the ``key``
    column of every parquet in dataset-creation/out/ (the difficulty splits
    and their variants — files without a key column are skipped), plus
    data/test_cut.parquet when it exists. Channel-P v2 training excludes
    all of them by default, and the trainers record the set in the
    checkpoint meta (the stage2 guard). NB: the molecules cache lives in
    dataset-creation/out/cache/, one level deeper — a non-recursive glob
    keeps it out."""
    import pandas as pd

    keys = set()
    for f in sorted((ROOT / "dataset-creation" / "out").glob("*.parquet")):
        try:
            keys.update(pd.read_parquet(f, columns=["key"])["key"].dropna().astype(str))
        except Exception:
            continue                      # not a split (no key column) — skip
    tc = ROOT / "data" / "test_cut.parquet"
    if tc.exists():
        try:
            df = pd.read_parquet(tc)
            col = "key" if "key" in df.columns else \
                "inchikey14" if "inchikey14" in df.columns else None
            if col:
                keys.update(df[col].dropna().astype(str))
        except Exception:
            pass
    return keys


def load_training_data(path, n_spectra, max_len, seed=0, max_rows=0, val_structures=500,
                       exclude=None):
    """Training + validation spectra. `exclude` is a set of InChIKey14 keys
    whose every row is dropped before anything else (the leak guard). Returns
    (train, val, excluded_keys) — excluded_keys also contains the carved
    validation structures, i.e. every key NOT trained on."""
    cols = ["inchikey14", "normalized_smiles", "precursor_mz", "ms2_mzs",
            "ms2_normalized_intensities", "ingest_lib"]
    df = read_spectra(path, cols, max_rows)
    df = df[df.ingest_lib != "enveda-180"]
    df = df.dropna(subset=["normalized_smiles"])
    print(f"after enveda-180 filter: {len(df)} spectra")
    if exclude:
        hit = df.inchikey14.isin(exclude)
        print(f"leak guard: excluded {int(hit.sum()):,} spectra of "
              f"{df.loc[hit, 'inchikey14'].nunique():,} eval-split molecules "
              f"({len(exclude):,} keys)")
        df = df[~hit]

    rng = np.random.RandomState(seed)
    structs = df.inchikey14.unique()
    n_val = min(val_structures, max(1, len(structs) // 2))
    val_structs = set(structs[rng.permutation(len(structs))[:n_val]])
    is_val = df.inchikey14.isin(val_structs)
    val_df = df[is_val].drop_duplicates(subset=["inchikey14"], keep="first")
    train_df = df[~is_val]

    if n_spectra and len(train_df) > n_spectra:
        train_df = train_df.sample(n=n_spectra, random_state=seed)
    print(f"train: {len(train_df)} spectra / {train_df.inchikey14.nunique()} structures")
    print(f"val:   {len(val_df)} spectra (one per structure)")
    if len(train_df) == 0:
        raise SystemExit("no training spectra left after the enveda-180 and leak-guard filters")

    def prep(d):
        mz_list, int_list = [], []
        for mzs, ints in zip(d.ms2_mzs.values, d.ms2_normalized_intensities.values):
            m, i = process_spectrum(mzs, ints, max_len)
            mz_list.append(m)
            int_list.append(i)
        return mz_list, int_list

    train_mz, train_int = prep(train_df)
    val_mz, val_int = prep(val_df)
    excluded_keys = sorted(set(str(k) for k in (exclude or [])) | {str(k) for k in val_structs})
    return (train_df, train_mz, train_int), (val_df, val_mz, val_int), excluded_keys


def build_tokenizer(smiles_list, vocab_size):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    alphabet = sorted(set("".join(smiles_list)))
    tok = Tokenizer(models.BPE())
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<pad>", 0), ("<s>", 1), ("</s>", 2)],
    )
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=alphabet,
        special_tokens=["<pad>", "<s>", "</s>"],
        show_progress=False,
    )
    tok.train_from_iterator(iter(smiles_list), trainer)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return tok


# ---------------------------------------------------------------- batching
def make_batch(mz_list, int_list, precursors, token_list, pad_id):
    peaks = max(len(m) for m in mz_list) + 1
    b = len(mz_list)
    mzs = np.zeros((b, peaks), dtype=np.float32)
    ints = np.zeros((b, peaks), dtype=np.float32)
    mask = np.zeros((b, peaks), dtype=np.float32)
    for i, (m, inten) in enumerate(zip(mz_list, int_list)):
        k = len(m) + 1
        mzs[i, 0] = precursors[i]
        mzs[i, 1:k] = m
        ints[i, 0] = 2.0
        ints[i, 1:k] = inten
        mask[i, :k] = 1.0
    tlen = max(len(t) for t in token_list)
    toks = np.full((b, tlen), pad_id, dtype=np.int64)
    for i, t in enumerate(token_list):
        toks[i, : len(t)] = t
    return (torch.from_numpy(mzs), torch.from_numpy(ints),
            torch.from_numpy(mask), torch.from_numpy(toks))


def decode_ids(ids, tokenizer):
    return tokenizer.decode(ids).replace(" ", "")


def canonical(smiles):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None, None
    return Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol).split("-")[0]


# ---------------------------------------------------------------- train
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None,
                    help=f"training parquet (default: {TRAIN_CLEAN} when present, "
                         f"else {TRAIN}; train_cut.parquet only when passed explicitly)")
    ap.add_argument("--include-eval-molecules", action="store_true",
                    help="DISABLE the leak guard for the final Kaggle model: train on "
                         "every molecule. stage2's channel-P guard will refuse this "
                         "checkpoint for local evaluation, which is exactly the point")
    ap.add_argument("--spectra", type=int, default=500_000,
                    help="training spectra to sample (0 = all)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--ff-dim", type=int, default=2048)
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--max-peaks", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--val-structures", type=int, default=500,
                    help="structures held out for validation")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="read only the first N rows of the parquet (0 = all; smoke tests)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="training device (auto = CUDA when available, else CPU)")
    ap.add_argument("--out", type=Path, default=HERE / "out" / "denovo_v1.pt",
                    help="best-checkpoint path (default max_class2_exp/out/denovo_v1.pt)")
    ap.add_argument("--tokenizer", type=Path, default=HERE / "out" / "denovo_tokenizer.json",
                    help="trained tokenizer path")
    ap.add_argument("--resume", type=Path, default=None,
                    help="checkpoint to resume from (.last.pt)")
    ap.add_argument("--seed", type=int, default=0)
    tracking.add_arguments(ap)
    return ap.parse_args()


def run(args, tracker):
    torch.manual_seed(args.seed)

    t0 = time.time()
    data_path = resolve_data_path(args.data)
    if not data_path.exists():
        raise SystemExit(f"training parquet not found: {data_path}")
    excluded = set() if args.include_eval_molecules else eval_split_keys()
    if args.include_eval_molecules:
        print("WARNING: --include-eval-molecules set — the checkpoint will see "
              "eval-split molecules. Final Kaggle model only; stage2's guard "
              "will refuse it for local evaluation.")
    (train_df, train_mz, train_int), (val_df, val_mz, val_int), excluded_keys = \
        load_training_data(data_path, args.spectra, args.max_peaks, args.seed,
                           max_rows=args.max_rows, val_structures=args.val_structures,
                           exclude=excluded)
    meta = {"guard": "eval-molecules-v1", "excluded_keys": excluded_keys,
            "data": data_path.name,
            "include_eval_molecules": bool(args.include_eval_molecules),
            "enveda180_rows_excluded": True}

    tok = build_tokenizer(train_df.normalized_smiles.unique().tolist(), args.vocab_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(args.tokenizer))

    train_tokens = [np.asarray(tok.encode(s).ids, dtype=np.int64)
                    for s in train_df.normalized_smiles.values]
    val_tokens = [np.asarray(tok.encode(s).ids, dtype=np.int64)
                  for s in val_df.normalized_smiles.values]
    train_prec = train_df.precursor_mz.values.astype(np.float32)
    val_prec = val_df.precursor_mz.values.astype(np.float32)
    val_smiles = val_df.normalized_smiles.values
    val_keys = val_df.inchikey14.values
    print(f"data ready at {time.time() - t0:.0f}s")
    tracker.log_params({
        "data_train_spectra": len(train_mz),
        "data_train_structures": int(train_df.inchikey14.nunique()),
        "data_val_spectra": len(val_mz),
        "data_val_structures": int(val_df.inchikey14.nunique()),
    })

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = DeNovoModel(args.embed_dim, args.n_layers, args.n_heads, args.ff_dim,
                        vocab_size=args.vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M params on {device}")
    tracker.log_params({"model_n_params": n_params, "train_device": str(device),
                        "train_device_name": (torch.cuda.get_device_name(0)
                                              if device.type == "cuda" else "cpu")})
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    base_step = 0
    if args.resume and args.resume.exists():
        allow_checkpoint_globals()    # older .last.pt configs embed Path objects
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        base_step = ckpt.get("step", 0)
        print(f"resumed from {args.resume} at step {base_step} "
              f"(val_exact_hit {ckpt.get('val_exact_hit', '?')})")
        tracker.log_params({"resumed_from": str(args.resume), "resume_step": base_step})
    scaler_dtype = torch.bfloat16

    steps_per_epoch = max(1, len(train_mz) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    print(f"{steps_per_epoch} optim steps/epoch")
    best_metric = -1.0
    step = base_step
    last_path = args.out.with_suffix(".pt.last.pt")
    config = tracking.public_args(args)
    for epoch in range(args.epochs):
        order = np.random.RandomState(args.seed + epoch + 1000 * base_step).permutation(len(train_mz))
        for g in range(0, len(order), args.batch_size):
            batch_idx = order[g: g + args.batch_size]
            mz_b, int_b, mask_b, tok_b = make_batch(
                [train_mz[i] for i in batch_idx], [train_int[i] for i in batch_idx],
                train_prec[batch_idx], [train_tokens[i] for i in batch_idx], 0)
            with torch.autocast(device_type=device.type, dtype=scaler_dtype):
                memory = model.encode(mz_b.to(device), int_b.to(device), mask_b.to(device))
                loss = model.loss(tok_b.to(device), memory, mask_b.to(device)) / args.grad_accum
            loss.backward()
            if (g // args.batch_size + 1) % args.grad_accum == 0 or g + args.batch_size >= len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 500 == 0:
                    torch.save({"state_dict": model.state_dict(),
                                 "optimizer": opt.state_dict(),
                                 "config": config, "step": step,
                                 "meta": meta}, last_path)
                if step % 100 == 0:
                    tracker.log_metrics(
                        {"train/loss": loss.item() * args.grad_accum}, step=step)
                    print(f"ep {epoch} step {step}/{total_steps} "
                          f"loss {loss.item() * args.grad_accum:.4f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                if step % 500 == 0:
                    model.eval()
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=scaler_dtype):
                        vm, vi, vk, vt = make_batch(val_mz[:64], val_int[:64], val_prec[:64],
                                                    val_tokens[:64], 0)
                        memory = model.encode(vm.to(device), vi.to(device), vk.to(device))
                        vloss = model.loss(vt.to(device), memory, vk.to(device)).item()
                    tracker.log_metrics({"val/loss": vloss}, step=step)
                    print(f"  val loss {vloss:.4f}", flush=True)
                    model.train()
                if step % 1500 == 0:
                    model.eval()
                    hits, valid, tani = gen_metrics(
                        model, tok, device, val_mz, val_int, val_prec, val_smiles, val_keys)
                    tracker.log_metrics({"val/exact_hit": hits, "val/validity": valid,
                                         "val/tanimoto": tani}, step=step)
                    print(f"  GEN: exact {hits:.3f} valid {valid:.3f} tanimoto {tani:.3f}", flush=True)
                    if hits > best_metric:
                        best_metric = hits
                        torch.save({
                            "state_dict": model.state_dict(),
                            "config": config,
                            "val_exact_hit": hits,
                            "meta": meta,
                        }, args.out)
                        print(f"  saved -> {args.out}", flush=True)
                    model.train()
    model.eval()
    hits, valid, tani = gen_metrics(
        model, tok, device, val_mz, val_int, val_prec, val_smiles, val_keys)
    print(f"FINAL GEN: exact {hits:.3f} valid {valid:.3f} tanimoto {tani:.3f}")
    if hits > best_metric:
        torch.save({"state_dict": model.state_dict(), "config": config,
                    "val_exact_hit": hits, "meta": meta}, args.out)
        print(f"saved -> {args.out}")
        best_metric = hits
    tracker.log_metrics({"final/exact_hit": hits, "final/validity": valid,
                         "final/tanimoto": tani, "final/best_exact_hit": best_metric})
    tracker.log_artifact(str(args.out))
    tracker.log_artifact(str(args.tokenizer))


def main():
    args = parse_args()
    with tracking.session(args, run_name=f"class2-denovo-{args.out.stem}",
                          tags={"script": "max_class2_exp/train_denovo.py",
                                "model": "denovo", "channel": "P-v2"}) as tracker:
        run(args, tracker)


def gen_metrics(model, tok, device, val_mz, val_int, val_prec, val_smiles, val_keys, n=64):
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    exact, valid_cnt, attempt_cnt, tani_sum = 0, 0, 0, 0.0
    for i in range(min(n, len(val_mz))):
        mz_b, int_b, mask_b, _ = make_batch([val_mz[i]], [val_int[i]], [val_prec[i]], [[1]], 0)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            memory = model.encode(mz_b.to(device), int_b.to(device), mask_b.to(device))
            gen, scores = model.generate(memory, mask_b.to(device), n_samples=8)
        truth_smiles = val_smiles[i]
        truth_key = val_keys[i]
        mol_t = __import__("rdkit").Chem.MolFromSmiles(truth_smiles)
        fp_t = fp_gen.GetSparseCountFingerprint(mol_t) if mol_t else None
        best_tani = 0.0
        seen = set()
        for j in range(gen.shape[1]):
            ids = gen[0, j].tolist()
            if 2 in ids:
                ids = ids[: ids.index(2)]
            attempt_cnt += 1
            smiles = decode_ids([t for t in ids if t not in (0, 1, 2)], tok)
            c, key = canonical(smiles)
            if c is None or key in seen:
                continue
            seen.add(key)
            valid_cnt += 1
            if key == truth_key:
                exact += 1
                break
            if fp_t is not None:
                fp_c = fp_gen.GetSparseCountFingerprint(__import__("rdkit").Chem.MolFromSmiles(c))
                t = DataStructs.TanimotoSimilarity(fp_t, fp_c)
                best_tani = max(best_tani, t)
        tani_sum += best_tani
    n = min(n, len(val_mz))
    return exact / n, valid_cnt / max(1, attempt_cnt), tani_sum / n


if __name__ == "__main__":
    main()
