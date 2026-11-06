"""The analogue channel, part 1: a candidate's nearest visible molecules.

Class-2 candidates have no spectra of their own (that is the definition of
the class), but most train molecules have close structural relatives that
*were* measured - `enveda-180` consists of combinatorial families, and the
difficulty-2 split reports a median `max_visible_tc` of 0.58. Fragmentation
spectra are strongly conserved within a family, so the spectra of a
candidate's neighbours are a usable proxy for the candidate's own spectrum.

This module finds, for a set of candidate structures, the top-K molecules
with visible rows by Morgan r2 Tanimoto, over the packed uint64
fingerprints of the whole train universe (the sparse-word popcount trick of
`common._tc_row`, which skips the query's empty 64-bit words). A visible
candidate's own molecule is allowed and will be its top neighbour at
Tanimoto 1.0 - that is the "this structure *was* measured" evidence, and it
is exactly the asymmetry the re-ranker has to learn (the truth, whose rows
are all hidden, never has it).
"""

import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "dataset-creation"))

import common  # noqa: E402

# state shared with forked workers (the same pattern as common.Similarity)
_NB_FPS = None
_NB_COLS = None
_NB_CNT = None


def _tc_row(q: int) -> np.ndarray:
    """Tanimoto of structure q to every structure (popcount over packed words)."""
    inter = np.zeros(len(_NB_CNT), dtype=np.int32)
    qrow = _NB_FPS[q]
    for w in np.nonzero(qrow)[0]:
        inter += np.bitwise_count(_NB_COLS[w] & qrow[w])
    union = _NB_CNT + _NB_CNT[q] - inter
    return np.divide(inter, union, out=np.zeros(len(union), dtype=np.float32), where=union > 0)


def _topk_one(args):
    q, allowed, k, min_tc = args
    tc = _tc_row(q)
    tc[~allowed] = -1.0
    if 0 < k < len(tc):
        idx = np.argpartition(-tc, k - 1)[:k]
    else:
        idx = np.arange(len(tc))
    idx = idx[tc[idx] >= min_tc]
    order = np.argsort(-tc[idx], kind="stable")
    idx = idx[order]
    return q, idx.astype(np.int64), tc[idx]


def topk_neighbours(fps: np.ndarray, queries, allowed: np.ndarray, k: int, min_tc: float,
                    workers: int | None = None, label: str = "[neighbours]") -> dict:
    """{structure index: (neighbour indices sorted by Tanimoto desc, Tanimotos)}.

    fps: packed uint64 (n_structures, 32) Morgan fingerprints (rows for
    external structures appended is fine - they index the same universe as
    `queries` and `allowed`). allowed: bool mask of the structures that may
    serve as neighbours (the molecules with >= 1 visible row under the mask
    in force). k <= 0 turns the channel off (empty lists).
    """
    global _NB_FPS, _NB_COLS, _NB_CNT
    queries = sorted({int(q) for q in queries})
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if not queries or k <= 0:
        return out
    _NB_FPS = np.ascontiguousarray(fps)
    _NB_COLS = np.ascontiguousarray(_NB_FPS.T)
    _NB_CNT = np.bitwise_count(_NB_FPS).sum(axis=1, dtype=np.int32)
    allowed = np.asarray(allowed, dtype=bool)
    assert len(allowed) == len(_NB_FPS), "allowed mask does not match the fingerprint universe"
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with get_context("fork").Pool(min(workers, len(queries))) as pool:
        prog = common.Progress(f"{label} top-{k} neighbours", len(queries))
        for q, idx, tc in pool.imap(_topk_one, [(q, allowed, k, min_tc) for q in queries], chunksize=8):
            out[q] = (idx, tc)
            prog.update()
        prog.close()
    n_hit = sum(1 for idx, _ in out.values() if len(idx))
    common.log(f"{label} {len(out):,} structures, {n_hit:,} with at least one neighbour "
               f"(Tc >= {min_tc:g}); median neighbours "
               f"{np.median([len(i) for i, _ in out.values()]) if out else 0:.0f}")
    return out


def brute_force_topk(fps: np.ndarray, q: int, allowed, k: int, min_tc: float):
    """Reference implementation for tests (unpacks full bit vectors)."""
    bits = np.unpackbits(fps.view(np.uint8), axis=1).astype(bool)
    inter = (bits[q] & bits).sum(axis=1)
    union = (bits[q] | bits).sum(axis=1)
    tc = np.divide(inter, union, out=np.zeros(len(union), dtype=np.float32), where=union > 0)
    tc[~np.asarray(allowed, dtype=bool)] = -1.0
    order = np.argsort(-tc, kind="stable")
    order = order[tc[order] >= min_tc][:k]
    return order.astype(np.int64), tc[order]
