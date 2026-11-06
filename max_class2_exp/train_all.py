"""One command for every GPU-trained artifact the stage2 fusion consumes.

stage2.py itself only trains light models inside a run (the channel-P v1
linear fallback and the fusion classifier — both CPU, minutes). The heavy
train-once part is the channel-P v2 pair, and this script runs both of its
trainers in order and verifies their leak guards before declaring success:

    1. train_denovo.py  — the spectrum -> SMILES transformer encoder (GPU)
    2. train_fp_head.py — the 2048-bit fingerprint head on the frozen encoder

Run from the repo root on the GPU box:

    python max_class2_exp/train_all.py             # full training (defaults)
    python max_class2_exp/train_all.py --smoke     # ~1-minute end-to-end check
    python max_class2_exp/train_all.py --include-eval-molecules   # final
                                                   # Kaggle model ONLY

It needs next to it (README 7.1 has the exact scp list):

    data/train_clean.parquet         the training spectra (else data/train.parquet)
    dataset-creation/out/*.parquet   the eval splits — the leak guard excludes
                                     their molecules; without them this script
                                     refuses to train (the checkpoints could
                                     not prove anything and stage2 would
                                     reject them)
    src/                             the repo-wide model code (fp_model, tracking)

Artifacts land in out/: denovo_v1.pt, denovo_tokenizer.json,
denovo_fp_head.pt (plus denovo_v1.pt.last.pt, a mid-training resume point).
scp them back over the local max_class2_exp/out/ and the next stage2 run
auto-detects channel-P v2 (guarded — fpchannel._load_guarded).

Never --resume from a denovo_v1.pt.last.pt that predates the leak guard
(no `meta` in it): those weights saw eval molecules, and resuming would
carry that into the "fresh" checkpoint. Retrain from scratch.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))              # train_denovo import below
sys.path.insert(0, str(ROOT / "src"))      # enveda_casmi26


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="end-to-end sanity check: tiny model, ~2k rows, about a "
                         "minute; writes out/denovo_smoke.pt & friends and touches "
                         "no real artifact")
    ap.add_argument("--epochs", type=int, default=None,
                    help="encoder epochs (default: train_denovo.py's, 2)")
    ap.add_argument("--spectra", type=int, default=None,
                    help="encoder training spectra cap (default: train_denovo.py's, 500k)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for both trainers (default: each script's own)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None,
                    help="encoder device (default: CUDA when available, else CPU)")
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume the encoder from a .last.pt of a GUARDED run "
                         "(never from a pre-guard checkpoint — see module docstring)")
    ap.add_argument("--include-eval-molecules", action="store_true",
                    help="DISABLE the leak guard for the final Kaggle model: train "
                         "on every molecule. stage2 will refuse these artifacts for "
                         "local evaluation, which is exactly what they are for")
    ap.add_argument("--no-mlflow", action="store_true", help="disable MLflow tracking")
    return ap.parse_args()


def step(name, cmd):
    print(f"\n[train_all] === {name} ===\n[train_all] $ "
          + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    if subprocess.run([str(c) for c in cmd]).returncode != 0:
        sys.exit(f"[train_all] {name} failed (see above) — stopping; later steps did not run")
    print(f"[train_all] {name} done in {time.time() - t0:.0f}s", flush=True)


def preflight(args):
    """Fail fast, before any GPU time is spent: imports, data, splits, device."""
    import torch
    from train_denovo import eval_split_keys, resolve_data_path

    if not (ROOT / "src" / "enveda_casmi26" / "fp_model.py").exists():
        sys.exit(f"[train_all] missing {ROOT / 'src' / 'enveda_casmi26'} — scp the "
                 "repo's src/ next to max_class2_exp (README 7.1)")
    data_path = resolve_data_path(None)
    if not data_path.exists():
        sys.exit(f"[train_all] training parquet not found: {data_path} — scp "
                 "data/train_clean.parquet (README 7.1)")
    keys = eval_split_keys()
    if not keys and not args.include_eval_molecules:
        sys.exit("[train_all] no dataset-creation/out/*.parquet split found — the "
                 "leak guard would exclude nothing, the checkpoints could not "
                 "prove exclusion, and stage2 would refuse them. scp the split "
                 "parquets next to the repo (README 7.1) and re-run.")
    if args.include_eval_molecules:
        print("[train_all] WARNING: --include-eval-molecules — the artifacts will "
              "see eval-split molecules. Final Kaggle model only; stage2's guard "
              "will refuse them locally (by design).")
    else:
        print(f"[train_all] leak guard armed: {len(keys):,} eval-split molecules "
              f"excluded and recorded in every checkpoint")
    print(f"[train_all] training data: {data_path.name}")
    if torch.cuda.is_available():
        print(f"[train_all] GPU: {torch.cuda.get_device_name(0)}")
    elif not (args.device == "cpu" or args.smoke):
        sys.exit("[train_all] no CUDA device visible — refusing the full run (it "
                 "would take days on CPU). Install a CUDA build of torch, or pass "
                 "--device cpu to override.")
    return data_path


def verify(paths, include_eval_molecules):
    """Re-load every artifact and check its guard covers the current splits."""
    import torch
    from train_denovo import eval_split_keys
    from enveda_casmi26.fp_model import allow_path_globals

    eval_keys = set() if include_eval_molecules else eval_split_keys()
    allow_path_globals()              # older checkpoint configs embed Path objects
    print(f"\n[train_all] === verifying the leak guards ===")
    for path, what in paths:
        if not path.exists():
            sys.exit(f"[train_all] {what} artifact missing: {path} — the trainer "
                     "did not save it (see its output above)")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        meta = (ckpt.get("meta") or {}) if isinstance(ckpt, dict) else {}
        if include_eval_molecules:
            if not meta:
                sys.exit(f"[train_all] {path.name} ({what}) has no meta at all")
            print(f"[train_all] {path.name} ({what}): meta present "
                  "(Kaggle model — guard deliberately disabled)")
            continue
        if meta.get("guard") != "eval-molecules-v1":
            sys.exit(f"[train_all] {path.name} ({what}) has no valid guard tag — "
                     "do not scp it back; retrain")
        missing = sorted(eval_keys - set(meta.get("excluded_keys") or []))
        if missing:
            sys.exit(f"[train_all] {path.name} ({what}) does not cover "
                     f"{len(missing)} eval key(s) (e.g. {missing[:3]}) — the splits "
                     "changed after training; retrain")
        print(f"[train_all] {path.name} ({what}): guard ok — covers all "
              f"{len(eval_keys):,} eval keys")


def main():
    args = parse_args()
    preflight(args)

    if args.smoke:
        enc_out = HERE / "out" / "denovo_smoke.pt"
        tok_out = HERE / "out" / "denovo_tokenizer_smoke.json"
        head_out = HERE / "out" / "denovo_fp_head_smoke.pt"
        enc_cmd = [sys.executable, HERE / "train_denovo.py",
                   "--spectra", "200", "--epochs", "1", "--max-rows", "2000",
                   "--val-structures", "10",
                   "--embed-dim", "64", "--n-layers", "1", "--n-heads", "2",
                   "--ff-dim", "128", "--out", enc_out, "--tokenizer", tok_out]
        head_cmd = [sys.executable, HERE / "train_fp_head.py",
                    "--encoder", enc_out, "--out", head_out,
                    "--cache", HERE / "out" / "cache" / "fp_embed_cache_smoke.npz",
                    "--epochs", "1", "--max-rows", "2000", "--val-structures", "10",
                    "--hidden", "128"]
    else:
        enc_out = HERE / "out" / "denovo_v1.pt"
        tok_out = HERE / "out" / "denovo_tokenizer.json"
        head_out = HERE / "out" / "denovo_fp_head.pt"
        enc_cmd = [sys.executable, HERE / "train_denovo.py"]
        head_cmd = [sys.executable, HERE / "train_fp_head.py", "--encoder", enc_out]

    for cmd in (enc_cmd, head_cmd):
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        if args.include_eval_molecules:
            cmd.append("--include-eval-molecules")
        if args.no_mlflow:
            cmd.append("--no-mlflow")
    if args.epochs is not None:
        enc_cmd += ["--epochs", str(args.epochs)]
    if args.spectra is not None:
        enc_cmd += ["--spectra", str(args.spectra)]
    if args.device:
        enc_cmd += ["--device", args.device]
    if args.resume:
        enc_cmd += ["--resume", args.resume]

    step("encoder (train_denovo.py)", enc_cmd)
    step("fingerprint head (train_fp_head.py)", head_cmd)
    if not tok_out.exists():
        sys.exit(f"[train_all] tokenizer missing: {tok_out} — the encoder run "
                 "did not save it (see its output above)")
    verify([(enc_out, "encoder"), (head_out, "fp head")],
           args.include_eval_molecules)

    print(f"""
[train_all] all artifacts verified. scp these back over the local
            max_class2_exp/out/ (the tokenizer too — stage2 needs it for v2):
            {enc_out.name}
            {enc_out.name}.last.pt   (optional, resume point)
            {tok_out.name}
            {head_out.name}

            then, locally from the repo root:
            uv run python max_class2_exp/stage2.py --limit 30 --blind-control
            uv run python max_class2_exp/stage2.py --blind-control --model-seeds 3""")


if __name__ == "__main__":
    main()
