"""Correctness tests for the class-2 experiment (no competition data needed).

Run from the repo root:
    uv run python max_class2_exp/test_stage2.py

Covers: pool construction and window maths, the packed-bit top-K neighbour
search against a brute-force reference, in-silico fragment matching, the
MetFrag-lite channel (weights, cache, polarity), the shared featurizer
(analogue channels, metfrag block, per-candidate aggregation, determinism,
source/formula annotations), the model helpers (within-shortlist z-scoring,
seed ensembles, feature ablation), the annotations module (formula cache,
visible source stats), the ranking/eval helpers (failure buckets, same-formula
panel ranks, rank-1 protection, paired bootstrap) and spectra_utils.
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "dataset-creation"))

import common  # noqa: E402
from insilico import frag_match, fragment_mzs  # noqa: E402
from metfrag import metfrag_match, metfrag_mzs  # noqa: E402
from features import (BUCKETS, FEATURES, FEATURE_COLUMNS, FIDX, ModelEnsemble,  # noqa: E402
                      failure_bucket, featurize, fit_models, model_matrix,
                      paired_bootstrap, predict, protect_top1_order, rank_in_order,
                      rank_within, set_dropped_features, truth_rank, zscore_within_sets)
from neighbours import brute_force_topk, topk_neighbours  # noqa: E402
from pool import Pool, build_pool, external_rows  # noqa: E402
from speclib import Library, build_library, query_mass  # noqa: E402
from spectra_utils import centroid, clean_spectrum, entropy_similarity  # noqa: E402
import annotations  # noqa: E402

CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
ETHANOL = "CCO"
CAFFEINE_FORMULA = "C8H10N4O2"
checks = 0


def ok(name: str, cond: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if not cond:
        raise AssertionError(f"[FAIL] {name} {detail}")
    print(f"[ok] {name}")


def t(name: str, fn) -> None:
    global checks
    checks += 1
    t0 = time.time()
    try:
        n = fn()
        print(f"[ok] {name} ({time.time() - t0:.2f}s)")
        if n:
            checks = checks + int(n) - 1
    except AssertionError as e:
        raise AssertionError(f"[FAIL] {name}: {e}") from e


# ----------------------------------------------------------------------- pool

def test_pool() -> int:
    molecules = pd.DataFrame({
        "key": ["K0000", "K0001", "K0002", "K0003"],
        "smiles": ["CCO", "CCC", "CCCC", "CCCCC"],
        "exact_mass": [46.0419, 44.0262, 58.0783, 72.0939],
        "n_rows": [2, 2, 2, 2], "n_eligible": [2, 2, 2, 2],
        "has_tims": [True] * 4, "has_other": [False] * 4,
        "ion_mode": ["positive"] * 4})
    row_key = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    mask = np.array([True, True,                   # mol 0: both rows visible
                     True, False,                  # mol 1: one row visible
                     True, False,                  # mol 2: one row visible (but excluded)
                     False, False])                # mol 3: nothing visible
    # mol 3 has no visible row -> only reachable as an addition
    additions = pd.DataFrame({"key": ["K0003", "K0000"], "smiles": ["CCCCC", "IGNORED"]})
    pool, _ = build_pool(molecules, row_key, mask, additions, excluded_keys=["K0002"])
    ok("pool: hidden molecule only via additions", bool(pool.is_addition[np.flatnonzero(pool.keys == "K0003")[0]]))
    ok("pool: duplicate addition skipped", (pool.keys == "K0000").sum() == 1)
    ok("pool: excluded key dropped", "K0002" not in pool.keys)
    ok("pool: visible row counts", pool.visible_rows[np.flatnonzero(pool.keys == "K0001")[0]] == 1)
    idx, err = pool.window(46.0419, 10.0)
    ok("pool: window finds the member", (pool.keys[idx] == "K0000").any())
    ok("pool: signed ppm error", abs(err[np.flatnonzero(pool.keys[idx] == "K0000")[0]]) < 1e-9)
    idx, _ = pool.window(500.0, 10.0)
    ok("pool: empty window", len(idx) == 0)
    idx, _ = pool.window(np.nan, 10.0)
    ok("pool: nan query mass", len(idx) == 0)
    ok("pool: member_of_key", pool.member_of_key("K0003") >= 0 and pool.member_of_key("ZZZ") == -1)
    ok("pool: masses sorted", (np.diff(pool.mass) >= 0).all())
    # external pool
    ext = pd.DataFrame({"key": ["E1", "K0001"], "smiles": ["CCO", "CCC"]})
    pool2, fps_ext = build_pool(molecules, row_key, mask, None, (), ext)
    e = np.flatnonzero(pool2.keys == "E1")
    ok("pool: external member added once", len(e) == 1 and (pool2.keys == "K0001").sum() == 1)
    ok("pool: external fingerprint rows", fps_ext.shape == (2, 32) and fps_ext.dtype == np.uint64
       and fps_ext[0].any())
    ok("pool: external mass computed", abs(pool2.mass[e[0]] - 46.0419) < 0.01)
    ok("pool: external struct past train universe", int(pool2.struct[e[0]]) == 4)
    return 1


# ----------------------------------------------------------------- neighbours

def test_neighbours() -> int:
    rng = np.random.default_rng(0)
    n, words = 40, 3
    fps = np.zeros((n, words), dtype=np.uint64)
    for i in range(n):
        bits = rng.integers(0, 2, size=64 * words, dtype=np.uint8)
        fps[i] = np.packbits(bits).view(np.uint64)
    allowed = rng.random(n) < 0.7
    q = 7
    allowed[q] = True
    out = topk_neighbours(fps, [q], allowed, k=5, min_tc=0.15, workers=2)
    idx, tc = out[q]
    bidx, btc = brute_force_topk(fps, q, allowed, k=5, min_tc=0.15)
    ok("neighbours: packed popcount = brute force", len(idx) == len(bidx)
       and (idx == bidx).all() and np.allclose(tc, btc, atol=1e-6),
       f"{idx} vs {bidx}")
    ok("neighbours: self first at Tc 1.0", idx[0] == q and abs(tc[0] - 1.0) < 1e-6)
    ok("neighbours: sorted desc, min_tc kept", (np.diff(tc) <= 0).all() and (tc >= 0.15).all())
    out = topk_neighbours(fps, [q], np.zeros(n, dtype=bool), k=5, min_tc=0.0, workers=2)
    ok("neighbours: nothing allowed -> empty", len(out[q][0]) == 0)
    out = topk_neighbours(fps, [q], np.ones(n, dtype=bool), k=0, min_tc=0.0, workers=2)
    ok("neighbours: k=0 turns the channel off", out == {})
    return 1


# -------------------------------------------------------------------- insilico

def test_frag_match() -> int:
    theo = np.array([100.0, 200.0, 300.0])
    q_mz = np.array([100.0, 150.0, 200.0])
    q_p = np.array([0.5, 0.25, 0.25])
    n, ex, ppm = frag_match(q_mz, q_p, theo, ppm=10.0)
    ok("frag: counts and explained intensity", (n, ex) == (2, 0.75) and ppm < 1.0)
    # ppm-scaled tolerance: 100.0 at 10 ppm accepts 100.0009, rejects 100.002
    n, _, _ = frag_match(np.array([100.0009]), np.array([1.0]), theo, ppm=10.0)
    ok("frag: tolerance edge (in)", n == 1)
    n, _, _ = frag_match(np.array([100.002]), np.array([1.0]), theo, ppm=10.0)
    ok("frag: tolerance edge (out)", n == 0)
    n, ex, _ = frag_match(q_mz, q_p, np.empty(0), ppm=10.0)
    ok("frag: empty theoretical set", (n, ex) == (0, 0.0))
    n, _, _ = frag_match(np.empty(0), np.empty(0), theo, ppm=10.0)
    ok("frag: empty query", n == 0)
    n, ex, _ = frag_match(q_mz, q_p, None, ppm=10.0)
    ok("frag: unavailable structure", (n, ex) == (0, 0.0))
    return 1


def test_fragment_mzs() -> int:
    theo = fragment_mzs(CAFFEINE, "[M+H]+")
    ok("fragmzs: caffeine fragments", theo is not None and len(theo) > 20 and (np.diff(theo) >= 0).all())
    ok("fragmzs: cache identity", fragment_mzs(CAFFEINE, "[M+H]+") is theo)
    prec = 194.08038 + 1.00727647
    ok("fragmzs: precursor present", abs(theo - prec).min() < 1e-3,
       f"closest {theo[np.argmin(np.abs(theo - prec))]:.4f}")
    ok("fragmzs: m/z bounds", theo[0] >= 25.0 and theo[-1] <= prec + 2 * 1.008 + 0.11)
    ok("fragmzs: adduct changes the fragments", not np.array_equal(theo, fragment_mzs(CAFFEINE, "[M+Na]+")))
    ok("fragmzs: bad SMILES -> None", fragment_mzs("not a smiles!!", "[M+H]+") is None)
    return 1


# --------------------------------------------------------------------- metfrag

def test_metfrag() -> int:
    pos = metfrag_mzs(CAFFEINE, True)
    ok("metfrag: set built", pos is not None and len(pos[0]) > 20 and (np.diff(pos[0]) >= 0).all())
    ok("metfrag: cache identity", metfrag_mzs(CAFFEINE, True) is pos)
    ok("metfrag: weights are 1.0/0.6 only", set(np.unique(pos[1])) <= {1.0, 0.6})
    prec = 194.08038 + 1.00727647
    w_prec = pos[1][np.argmin(np.abs(pos[0] - prec))]
    ok("metfrag: precursor present at weight 1.0", abs(pos[0] - prec).min() < 1e-4 and w_prec == 1.0)
    ok("metfrag: 1-bond fragments exist besides the precursor",
       (pos[1] == 1.0).sum() >= 2 and (pos[1] == 0.6).sum() >= 1)
    neg = metfrag_mzs(CAFFEINE, False)
    pos_mz = pos[0]
    shifted = np.abs(neg[0][:, None] - (pos_mz - 2 * 1.007276466621)[None, :]).min(axis=1)
    i = int(np.argmin(np.abs(neg[0] - (194.08038 - 1.00727647))))
    ok("metfrag: negative polarity = same fragments, proton sign flipped",
       abs(neg[0] - (194.08038 - 1.00727647)).min() < 1e-4       # the [M-H]- ion exists
       and neg[1][i] == 1.0                                       # ... as the intact ion
       and (shifted < 1e-3).all())                                # and only differs by 2 protons
    ok("metfrag: bad SMILES -> None", metfrag_mzs("not a smiles!!", True) is None)
    ok("metfrag: single-atom molecule -> None", metfrag_mzs("C", True) is None)
    # matching: nearest theoretical peak wins; both-in-tolerance takes max weight
    theo_mz, w = pos
    hit, wv, d = metfrag_match(np.array([prec]), theo_mz, w, 1e-3)
    ok("metfrag: match the precursor", hit[0] and wv[0] == 1.0 and d[0] < 1e-4)
    hit, wv, _ = metfrag_match(np.array([99999.0]), theo_mz, w, 0.01)
    ok("metfrag: no match", not hit[0])
    hit, wv, d = metfrag_match(np.array([100.0]), np.array([100.0, 100.09]), np.array([0.6, 1.0]), 0.1)
    ok("metfrag: both within tol -> higher weight", hit[0] and wv[0] == 1.0)
    return 1


# ------------------------------------------------------------------ featurize

def _tiny_library() -> Library:
    """Mol 0 owns two spectra (junk first, then an exact copy of the query);
    mol 1 owns one. Multi-spectrum neighbours are the regression case for the
    slot alignment of the entropy_best reduceat."""
    rows = [{"row_id": 0, "mol": 0, "adduct": "[M+H]+", "ionization_mode": "positive",
              "instrument_type": common.TIMS, "precursor_mz": 150.0, "collision_energy_ev": [20.0],
              "ingest_lib": "x"},
             {"row_id": 1, "mol": 0, "adduct": "[M+H]+", "ionization_mode": "positive",
              "instrument_type": common.TIMS, "precursor_mz": 150.0, "collision_energy_ev": [20.0],
              "ingest_lib": "x"},
             {"row_id": 2, "mol": 1, "adduct": "[M+H]+", "ionization_mode": "positive",
              "instrument_type": common.TIMS, "precursor_mz": 150.0, "collision_energy_ev": [20.0],
              "ingest_lib": "x"}]
    table = pd.DataFrame(rows)
    mz = np.array([77.0, 100.0, 195.08765, 100.01])     # spec0=[77], spec1=[100,195.09], spec2=[100.01]
    p = np.array([1.0, 0.5, 0.5, 1.0])
    lens = np.array([1, 2, 1])
    return Library(table, mz, p, lens)


def _tiny_pool(smiles) -> Pool:
    pool = Pool(keys=np.array([f"k{i}" for i in range(len(smiles))]),
                smiles=np.array(smiles, dtype=object),
                mass=np.array([194.08038] * len(smiles)),
                struct=np.arange(len(smiles), dtype=np.int64),
                visible_rows=np.array([2, 1] + [0] * (len(smiles) - 2), dtype=np.int64),
                is_addition=np.array([False, False] + [True] * (len(smiles) - 2), dtype=bool),
                is_external=np.zeros(len(smiles), dtype=bool))
    pool.formulas = np.array([CAFFEINE_FORMULA if s == CAFFEINE else "C2H6O"
                              for s in smiles], dtype=object)
    pool.tims_rows = np.array([2, 1] + [0] * (len(smiles) - 2), dtype=np.float32)
    pool.public_rows = np.array([0, 5] + [0] * (len(smiles) - 2), dtype=np.float32)
    return pool


def test_featurize() -> int:
    lib = _tiny_library()
    pool = _tiny_pool([CAFFEINE, CAFFEINE, CAFFEINE])
    nbrs = {0: (np.array([0]), np.array([1.0], dtype=np.float32)),
            1: (np.array([1]), np.array([0.6], dtype=np.float32)),
            2: (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))}
    q_mz = np.array([100.0, 195.08765])          # one analogue-matched peak + the [M+H]+ precursor
    q_p = np.array([0.5, 0.5])
    spectra = [(q_mz, q_p, True, 195.08765, "[M+H]+")]
    # candidate masses sit exactly 1/2/3 ppm above the spectrum's implied
    # neutral mass (194.08038), so max-over-spectra == the passed median error
    m_spec = 195.08765 - 1.00727647
    pool.mass = m_spec * (1 + np.array([1.0, 2.0, 3.0]) * 1e-6)
    X = featurize(spectra, np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]), pool, nbrs, lib, 0.02, 10.0)
    ok("featurize: shape", X.shape == (3, len(FEATURE_COLUMNS)))
    ok("featurize: one positive-identical analogue wins",
       X[0, FIDX["entropy_best"]] == 1.0
       and X[0, FIDX["entropy_best"]] > X[1, FIDX["entropy_best"]] > X[2, FIDX["entropy_best"]] == 0)
    ok("featurize: no neighbours -> zero analogue features",
       (X[2, [FIDX["entropy_best"], FIDX["entropy_composite"], FIDX["cosine_best"],
              FIDX["nl_entropy_best"], FIDX["n_analogue_molecules"], FIDX["n_analogue_spectra"],
              FIDX["analogue_best_tc"]]] == 0).all())
    ok("featurize: neighbour bookkeeping",
       (X[:, FIDX["n_analogue_molecules"]] == [1, 1, 0]).all()
       and (X[:, FIDX["n_analogue_spectra"]] == [2, 1, 0]).all()
       and np.allclose(X[:, FIDX["analogue_best_tc"]], [1.0, 0.6, 0.0]))
    ok("featurize: mass error and visibility passthrough",
       np.allclose(X[:, FIDX["mass_error_ppm"]], [1, 2, 3])
       and np.allclose(X[:, FIDX["cand_visible_rows"]], [np.log(3), np.log(2), 0.0]))
    ok("featurize: single spectrum -> worst mass error = median error",
       np.allclose(X[:, FIDX["mass_error_ppm_max"]], [1, 2, 3]))
    ok("featurize: source flags passthrough",
       (X[:, FIDX["cand_is_addition"]] == [0, 0, 1]).all()
       and (X[:, FIDX["cand_is_external"]] == 0).all()
       and np.allclose(X[:, FIDX["cand_tims_rows"]], [np.log(3), np.log(2), 0.0])
       and np.allclose(X[:, FIDX["cand_public_rows"]], [0.0, np.log(6), 0.0]))
    ok("featurize: same-formula competition",
       (X[:, FIDX["cand_n_same_formula"]] == 2).all())
    ok("featurize: in-silico explains the precursor peak",
       (X[:, FIDX["frag_matched_peaks"]] >= 1).all()
       and (X[:, FIDX["frag_explained_intensity"]] >= 0.5).all()
       and (X[:, FIDX["frag_available"]] == 1).all()
       and (X[:, FIDX["n_theo_frags"]] > 3).all())
    ok("featurize: metfrag block alive (precursor matched at weight 1.0)",
       (X[:, FIDX["mtf_available"]] == 1).all()
       and (X[:, FIDX["mtf_matched_peaks"]] >= 1).all()
       and np.allclose(X[:, FIDX["mtf_explained_linear"]], 0.5)
       and (X[:, FIDX["mtf_mean_ppm"]] < 1.0).all())
    ok("featurize: rarity down-weights a peak every candidate explains",
       np.allclose(X[:, FIDX["mtf_rarity_explained"]], 0.5 / (1 + 3)))
    ok("featurize: gap to best is 0 for the best candidate",
       np.allclose(X[:, FIDX["mtf_gap_best"]], 0.0))
    ok("featurize: composite channel alive", X[0, FIDX["entropy_composite"]] > 0)
    X2 = featurize(spectra, np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]), pool, nbrs, lib, 0.02, 10.0)
    ok("featurize: deterministic", np.array_equal(X, X2))
    X0 = featurize(spectra, np.empty(0, dtype=np.int64), np.empty(0), pool, nbrs, lib, 0.02, 10.0)
    ok("featurize: empty shortlist", X0.shape == (0, len(FEATURE_COLUMNS)))
    return 1


def test_featurize_mixed() -> int:
    """Different molecules in one shortlist: rarity counts across candidates,
    the metfrag gap-to-best punishes the structure that explains less, and
    same-formula competition follows the formulas."""
    lib = _tiny_library()
    pool = _tiny_pool([CAFFEINE, ETHANOL, CAFFEINE])
    nbrs = {}
    q_mz = np.array([100.0, 195.08765])
    q_p = np.array([0.5, 0.5])
    spectra = [(q_mz, q_p, True, 250.0, "[M+H]+")]
    X = featurize(spectra, np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]), pool, nbrs, lib, 0.02, 10.0,
                  mtf_tol=0.01)
    ok("featurize mixed: only caffeine explains the precursor",
       np.allclose(X[:, FIDX["mtf_explained_linear"]], [0.5, 0.0, 0.5]))
    ok("featurize mixed: gap to best is 0 for winners, negative for the loser",
       np.allclose(X[:, FIDX["mtf_gap_best"]], [0.0, -0.5, 0.0]))
    ok("featurize mixed: rarity counts candidates explaining the peak",
       np.allclose(X[:, FIDX["mtf_rarity_explained"]], [0.5 / 3, 0.0, 0.5 / 3]))
    ok("featurize mixed: same-formula competition by candidate",
       (X[:, FIDX["cand_n_same_formula"]] == [1, 0, 1]).all())
    return 1


# --------------------------------------------------------------- model helpers

def test_models() -> int:
    X = np.zeros((6, len(FEATURE_COLUMNS)))
    X[:, FIDX["entropy_best"]] = [1.0, 0.0, -1.0, 1.0, 0.0, -1.0]   # the informative feature
    X[:, FIDX["cand_visible_rows"]] = 4.0                            # constant -> z-scores to 0
    X[:, FIDX["mass_error_ppm"]] = [7.0, 7.0, 7.0, 7.0, 7.0, 7.0]    # constant -> 0
    X[:, FIDX["frag_available"]] = 0.0                               # raw, stays 0
    X[:, FIDX["cand_is_addition"]] = [0, 0, 0, 0, 0, 0]
    X[:, FIDX["n_candidates"]] = [3, 3, 3, 3, 3, 3]                  # context only, dropped
    # default: n_candidates AND the source block are out of the model matrix
    # (the block marks the truth by simulator construction - see features.py)
    Z = model_matrix(X, [3, 3])
    ok("models: matrix drops n_candidates + source block by default",
       Z.shape == (6, len(FEATURES) - 4))
    j = FEATURES.index("entropy_best")
    ok("models: z-scored within shortlist",
       np.allclose(Z[:, j], [1.2247, 0.0, -1.2247] * 2, atol=1e-3))
    ok("models: constant columns -> zeros",
       (Z[:, FEATURES.index("mass_error_ppm")] == 0).all())
    ok("models: raw feature not z-scored", (Z[:, FEATURES.index("frag_available")] == 0).all())
    y = np.array([1, 0, 0, 1, 0, 0])
    models = fit_models(Z, y, 0, ("logistic", "hgb"))
    ok("models: both fitted", set(models) == {"logistic", "hgb"})
    for kind, m in models.items():
        s = predict(m, X, [3, 3])
        ok(f"models: {kind} ranks the positive first in both shortlists",
           s[0] == s[:3].max() and s[3] == s[3:].max() and ((s >= 0) & (s <= 1)).all())
    # seed ensembles (forum P0.6: pin seeds and average)
    ens = fit_models(Z, y, 11, ("hgb",), n_seeds=3)
    ok("models: hgb ensemble has 3 members", isinstance(ens["hgb"], ModelEnsemble)
       and len(ens["hgb"]) == 3)
    s_one = predict(ModelEnsemble([ens["hgb"].first()]), X, [3, 3])
    s_ens = predict(ens["hgb"], X, [3, 3])
    ok("models: ensemble prediction is the member average",
       np.allclose(s_ens, s_one))          # identical members would differ with real seeds; shape check here
    # --keep-source-features: nothing dropped
    set_dropped_features([], keep_source=True)
    Zk = model_matrix(X, [3, 3])
    ok("models: keep_source restores all features", Zk.shape == (6, len(FEATURES)))
    ok("models: kept source block z-scored / raw as before",
       (Zk[:, FEATURES.index("cand_visible_rows")] == 0).all()      # constant 4.0 -> z 0
       and (Zk[:, FEATURES.index("cand_is_addition")] == 0).all())  # raw flag stays 0
    # --drop-features on top of keep_source
    set_dropped_features(["mass_error_ppm"], keep_source=True)
    Za = model_matrix(X, [3, 3])
    ok("models: ablated feature dropped", Za.shape == (6, len(FEATURES) - 1))
    models_a = fit_models(Za, y, 0, ("logistic",))
    ok("models: fit/predict on the ablated matrix", predict(models_a["logistic"], X, [3, 3]).shape == (6,))
    # without keep_source the source block comes back on top of explicit drops
    set_dropped_features(["mass_error_ppm"])
    Za = model_matrix(X, [3, 3])
    ok("models: source block re-imposed without keep_source",
       Za.shape == (6, len(FEATURES) - 5))
    # replace= sets exactly the given list (saved-model reload semantics)
    set_dropped_features(["mass_error_ppm"], replace=True)
    ok("models: replace sets the exact drop set", model_matrix(X, [3, 3]).shape == (6, len(FEATURES) - 1))
    try:
        set_dropped_features(["no_such_feature"], keep_source=True)
        ok("models: unknown ablation rejected", False)
    except ValueError:
        ok("models: unknown ablation rejected", True)
    set_dropped_features([])          # restore the default (source block dropped)
    Zc = zscore_within_sets(np.ones((2, 2)), [2])
    ok("models: zscore of a constant set", (Zc == 0).all())
    return 1


# ------------------------------------------------------------ ranking helpers

def test_rank_helpers() -> int:
    ok("buckets: outside window", failure_bucket(0, False) == "outside_window")
    ok("buckets: rank 1", failure_bucket(1, True) == "rank_1")
    ok("buckets: rank 2-25", failure_bucket(25, True) == "rank_2_25")
    ok("buckets: beyond 25", failure_bucket(26, True) == "in_window_rank_gt25")
    ok("buckets: the four buckets", set(BUCKETS) == {"rank_1", "rank_2_25",
                                                     "in_window_rank_gt25", "outside_window"})
    # panel rank (P0.2): restrict scores to the panel positions
    scores = np.array([0.1, 0.5, 0.3, 0.9])
    panel = np.array([1, 3])
    ok("rank_within: best panel member", rank_within(scores, panel, 3) == 1)
    ok("rank_within: weaker panel member", rank_within(scores, panel, 1) == 2)
    ok("rank_within: truth not competing", rank_within(scores, panel, 0) == 0)
    ok("rank_within: ties break by the tiebreak asc (like |ppm|)",
       rank_within(np.array([0.5, 0.5, 0.1]), np.array([0, 1]), 1,
                   np.array([1.0, 9.0])) == 2
       and rank_within(np.array([0.5, 0.5, 0.1]), np.array([0, 1]), 1,
                       np.array([9.0, 1.0])) == 1)
    # rank within an explicit order + rank-1 protection (P1.5)
    order = np.array([2, 0, 1])
    ok("rank_in_order: basic", rank_in_order(order, 1) == 3)
    ok("rank_in_order: restricted to a panel", rank_in_order(order, 1, np.array([0, 1])) == 2)
    ok("rank_in_order: absent -> 0", rank_in_order(order, -1) == 0)
    prot = protect_top1_order(np.array([1, 0, 2]), 2)
    ok("protect: channel top-1 first, fused order after", prot.tolist() == [2, 1, 0])
    ok("protect: no-op when the channel already won",
       protect_top1_order(np.array([2, 0, 1]), 2).tolist() == [2, 0, 1])
    # paired bootstrap (P0.6)
    a = np.array([1.0, 0.5, 0.0, 0.0])
    b = np.array([0.5, 1.0, 0.0, 0.0])
    d = paired_bootstrap(a, a, 50, 0)
    ok("bootstrap: identical rr -> delta 0 with a CI around 0",
       d["mean_delta"] == 0.0 and d["ci95"][0] <= 0.0 <= d["ci95"][1])
    d = paired_bootstrap(a, b, 200, 0)
    ok("bootstrap: symmetric swap -> mean ~0", abs(d["mean_delta"]) < 0.01)
    w = np.array([1.0, 1.0, 0.5, 0.5])      # every molecule favors a: the CI must exclude 0
    d = paired_bootstrap(w, np.zeros(4), 50, 0)
    ok("bootstrap: a clear win is a clear win", d["mean_delta"] > 0.2 and d["ci95"][0] > 0)
    ok("bootstrap: empty / mismatched input -> None", paired_bootstrap([], [], 50, 0) is None
       and paired_bootstrap(a, np.zeros(3), 50, 0) is None)
    # truth_rank: rank + rr@25 + scores, with rank_order's tie-breaks
    q = {"cands": np.array([10, 11, 12]), "ppm_err": np.array([1.0, -2.0, 3.0])}
    ok("truth_rank: winner", truth_rank(np.array([0.1, 0.5, 0.3]), q, 11) == (1, 1.0, 0.5, 0.5))
    ok("truth_rank: runner-up", truth_rank(np.array([0.1, 0.5, 0.3]), q, 12) == (2, 0.5, 0.3, 0.5))
    r = truth_rank(np.array([0.1, 0.5, 0.3]), q, 99)
    ok("truth_rank: absent truth", r[0] == 0 and r[1] == 0.0 and np.isnan(r[2]) and r[3] == 0.5)
    ok("truth_rank: negative truth", truth_rank(np.array([0.1, 0.5, 0.3]), q, -1)[:2] == (0, 0.0))
    r = truth_rank(np.empty(0), {"cands": np.empty(0, dtype=np.int64), "ppm_err": np.empty(0)}, 1)
    ok("truth_rank: empty shortlist", r[0] == 0 and r[1] == 0.0 and np.isnan(r[2]) and np.isnan(r[3]))
    ties = np.array([1.0, 1.0, 1.0])
    ok("truth_rank: tie broken by |ppm|",
       truth_rank(ties, q, 10)[0] == 1 and truth_rank(ties, q, 11)[0] == 2
       and truth_rank(ties, q, 12)[0] == 3)
    big = {"cands": np.arange(30), "ppm_err": np.zeros(30)}
    r = truth_rank(np.arange(30, dtype=float), big, 4)     # 25 members score above 4
    ok("truth_rank: rr cut at TOP_K", r[0] == 26 and r[1] == 0.0 and r[2] == 4.0 and r[3] == 29.0)
    r = truth_rank(np.arange(30, dtype=float), big, 5)     # rank 25 is still inside
    ok("truth_rank: rank 25 still earns rr", r[0] == 25 and r[1] == 0.04)
    return 1


# ----------------------------------------------------------------- annotations

def test_annotations() -> int:
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td) / "formulas.parquet"
        f = annotations.formulas_for(["K1", "K2"], [ETHANOL, CAFFEINE], cache)
        ok("annotations: formulas computed", f == {"K1": "C2H6O", "K2": CAFFEINE_FORMULA})
        ok("annotations: cache file written", cache.exists())
        f2 = annotations.formulas_for(["K1", "K2"], [ETHANOL, CAFFEINE], cache)
        ok("annotations: cache round trip", f2 == f)
        f3 = annotations.formulas_for(["K3"], ["CC(=O)O"], cache)
        ok("annotations: new keys appended", f3["K3"] == "C2H4O2" and cache.exists())
        pool = _tiny_pool([CAFFEINE, ETHANOL, CAFFEINE])
        f_pool = annotations.formulas_for(pool.keys, pool.smiles, cache)
        annotations.annotate_pool_formulas(pool, f_pool)
        ok("annotations: pool formulas aligned",
           (pool.formulas == np.array([CAFFEINE_FORMULA, "C2H6O", CAFFEINE_FORMULA],
                                      dtype=object)).all())
    # visible source stats: hidden rows must never count (P0.4)
    row_key = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    mask = np.array([True, True, True, False, True])
    lib_codes = np.array([0, 1, 0, 2, 0], dtype=np.int16)          # enveda-180, massbank, np-examples
    tims = np.array([True, False, True, True, True])
    lib_names = ["enveda-180", "massbank", "enveda-np-examples"]
    st = annotations.visible_source_stats(row_key, mask, lib_codes, tims, lib_names, 3)
    ok("annotations: visible public rows", st["public_rows"].tolist() == [1, 0, 0])
    ok("annotations: visible tims rows", st["tims_rows"].tolist() == [1, 1, 1])
    st_all = annotations.visible_source_stats(row_key, np.ones(5, dtype=bool), lib_codes, tims,
                                              lib_names, 3)
    ok("annotations: all-rows mask sees hidden rows too",
       st_all["public_rows"].tolist() == [1, 0, 0] and st_all["tims_rows"].tolist() == [1, 2, 1])
    pool = _tiny_pool([CAFFEINE, ETHANOL, CAFFEINE])
    pool.struct = np.array([0, 1, 99], dtype=np.int64)             # the third member is external
    annotations.annotate_pool_sources(pool, st)
    ok("annotations: pool sources aligned, external zeroed",
       np.allclose(pool.public_rows, [1, 0, 0]) and np.allclose(pool.tims_rows, [1, 1, 0]))
    flags = annotations.truth_public_flags(["k0", "k1", "k2"], st_all,
                                           {"k0": 0, "k1": 1, "k2": 2})
    ok("annotations: truth public flags", flags.tolist() == [True, False, False])
    # load_row_sources: parquet -> npz cache -> round trip (a write-only cache
    # path shipped a format the read path could not parse; this pins it)
    with tempfile.TemporaryDirectory() as td:
        pq_path = Path(td) / "mini.parquet"
        pd.DataFrame({"ingest_lib": ["massbank", "enveda-180", "massbank"],
                      "instrument_type": [common.TIMS, common.TIMS, "qtof"]}).to_parquet(pq_path)
        old_cache = annotations.CACHE_DIR
        try:
            annotations.CACHE_DIR = Path(td)
            codes, tims_r, names = annotations.load_row_sources(pq_path, 3)
            ok("annotations: row sources computed",
               names == ["enveda-180", "massbank"] and codes.tolist() == [1, 0, 1]
               and tims_r.tolist() == [True, True, False])
            codes2, tims2, names2 = annotations.load_row_sources(pq_path, 3)
            ok("annotations: row sources cache round trip",
               names2 == names and (codes2 == codes).all() and (tims2 == tims_r).all()
               and isinstance(names2, list))
        finally:
            annotations.CACHE_DIR = old_cache
    return 1


# --------------------------------------------------------- spectra / speclib

def test_spectra_utils() -> int:
    mz, p = centroid(np.array([100.0, 100.005, 200.0]), np.array([1.0, 0.5, 1.0]), 0.02)
    ok("spectra: centroid merges close peaks", len(mz) == 2 and abs(p[0] - 1.5) < 1e-9)
    cm, cp = clean_spectrum([100.0, 100.005, 200.0, 500.0], [1.0, 0.5, 1.0, 0.2], 300.0)
    ok("spectra: cleaning drops the precursor region and normalises",
       len(cm) == 2 and abs(cp.sum() - 1.0) < 1e-9)
    ok("spectra: entropy similarity of identical spectra is 1",
       abs(entropy_similarity(cm, cp, cm, cp, 0.02) - 1.0) < 1e-6)
    return 1


def test_speclib() -> int:
    # the analogue channel is off by default: an empty library must be usable
    from spectra_utils import CleanParams
    lib = build_library(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                        Path("/nonexistent.parquet"), CleanParams(), {})
    ok("speclib: empty rows -> empty library", len(lib) == 0 and lib.spectra_of(0) == (0, 0))
    ok("speclib: query_mass median", abs(query_mass([195.0877, 217.0696],
                                                    ["[M+H]+", "[M+Na]+"]) - 194.08038) < 1e-3)
    ok("speclib: nan when nothing usable", np.isnan(query_mass([100.0], ["[M+X]+"])))
    lib = _tiny_library()
    ok("speclib: tiny library spectra_of", lib.spectra_of(0) == (0, 2)
       and lib.spectra_of(1) == (2, 3) and lib.spectra_of(9) == (0, 0))
    return 1


def main() -> None:
    t("pool", test_pool)
    t("neighbours", test_neighbours)
    t("frag_match", test_frag_match)
    t("fragment_mzs", test_fragment_mzs)
    t("metfrag", test_metfrag)
    t("featurize", test_featurize)
    t("featurize_mixed", test_featurize_mixed)
    t("models", test_models)
    t("rank_helpers", test_rank_helpers)
    t("annotations", test_annotations)
    t("spectra_utils", test_spectra_utils)
    t("speclib", test_speclib)
    print(f"\nall {checks} checks passed")


if __name__ == "__main__":
    main()
