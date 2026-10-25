"""Correctness tests for the stage-2 building blocks (no data files needed, ~10 s).

    uv run python max_class1_exp/test_stage2.py

The end-to-end self-tests with real data are separate commands (see README):
    uv run python max_class1_exp/stage2.py --leak-train --limit 100
    uv run python max_class1_exp/stage2.py --n-train 300 --limit 100
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from library import CandidateIndex, Library  # noqa: E402
import common  # noqa: E402  (dataset-creation)
from spectra import (CleanParams, clean_spectrum, cosine_similarity, entropy_similarity,  # noqa: E402
                     match_stats, nl_cosine_similarity, nl_entropy_similarity, neutral_losses)
import stage2  # noqa: E402

RNG = np.random.default_rng(0)
FAILED = []
TOL = 0.02


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAILED.append(name)


def random_spectrum(n, lo=50, hi=500):
    mz = np.sort(RNG.choice(np.arange(lo, hi, 0.5), size=n, replace=False))  # >= 0.5 Da apart
    p = RNG.random(n) + 0.01
    return mz, p / p.sum()


# ------------------------------------------------------------- neutral loss

def test_neutral_loss():
    ok_cos, ok_ent = True, True
    for _ in range(200):
        a_mz, a_p = random_spectrum(int(RNG.integers(3, 30)))
        b_mz, b_p = random_spectrum(int(RNG.integers(3, 30)))
        prec_a, prec_b = a_mz.max() + RNG.uniform(5, 50), b_mz.max() + RNG.uniform(5, 50)
        ok_cos &= abs(nl_cosine_similarity(a_mz, a_p, prec_a, b_mz, b_p, prec_b, TOL)
                      - cosine_similarity(a_mz, a_p, b_mz + (prec_a - prec_b), b_p, TOL)) < 1e-9
        ok_ent &= abs(nl_entropy_similarity(a_mz, a_p, prec_a, b_mz, b_p, prec_b, TOL)
                      - entropy_similarity(a_mz, a_p, b_mz + (prec_a - prec_b), b_p, TOL)) < 1e-9
    check("neutral loss: nl similarity == similarity on library peaks shifted by prec_q - prec_l "
          "(200 random pairs, cosine + entropy)", ok_cos and ok_ent)
    nl, order = neutral_losses(np.array([100.0, 200.0, 300.0]), 400.0)
    check("neutral losses: nl = precursor - mz, sorted ascending; order reverses the intensities",
          np.allclose(nl, [100.0, 200.0, 300.0]) and np.allclose(order, [2, 1, 0]))
    # tolerance edge: nl_q = 100.0; nl_l = 100.019 matches, 100.021 does not
    q_mz, q_p = np.array([300.0]), np.array([1.0])
    check("neutral loss: tolerance edge (0.019 Da matches, 0.021 Da does not)",
          nl_cosine_similarity(q_mz, q_p, 400.0, np.array([399.981]), np.array([1.0]), 500.0, TOL) > 0.99
          and nl_cosine_similarity(q_mz, q_p, 400.0, np.array([399.979]), np.array([1.0]), 500.0, TOL) == 0)


# ------------------------------------------------------------ match statistics

def test_match_stats():
    q_mz, q_p = np.array([100.0, 200.0, 300.0]), np.array([0.5, 0.3, 0.2])
    n, s = match_stats(q_mz, q_p, np.array([100.005, 199.99, 400.0]), np.array([0.5, 0.3, 0.2]), TOL)
    check("match stats: matched query peaks and their intensity share (2 peaks, 0.8)",
          n == 2 and abs(s - 0.8) < 1e-12)
    n, s = match_stats(q_mz, q_p, np.array([100.0, 100.015]), np.array([0.5, 0.5]), TOL)
    check("match stats: two library peaks near one query peak count once", n == 1 and abs(s - 0.5) < 1e-12)
    n, s = match_stats(q_mz, q_p, np.array([150.0, 250.0]), np.array([0.5, 0.5]), TOL)
    check("match stats: no overlap -> (0, 0)", n == 0 and s == 0.0)
    n, s = match_stats(np.empty(0), np.empty(0), q_mz, q_p, TOL)
    check("match stats: empty query -> (0, 0)", n == 0 and s == 0.0)


# ----------------------------------------------- featurize: pairwise + invariants

def tiny_pair():
    """One query spectrum vs a one-spectrum library (hand-checkable)."""
    lprec = 260.0
    cm, cp = clean_spectrum(np.array([100.0, 150.0, 200.0]), np.array([0.5, 0.3, 0.2]), lprec)
    tab = pd.DataFrame({"row_id": [0], "mol": [7], "adduct": ["[M+H]+"],
                        "ionization_mode": ["positive"], "instrument_type": ["timsTOF"],
                        "precursor_mz": [lprec], "collision_energy_ev": [[20.0]],
                        "ingest_lib": ["libA"]})
    lib = Library(tab, cm, cp, np.array([len(cm)], dtype=np.int64))
    qprec = 260.5
    qm, qp = clean_spectrum(np.array([100.01, 150.015, 200.0, 220.0]),
                           np.array([0.4, 0.3, 0.2, 0.1]), qprec)
    q = {"spectra": [(qm, qp, True, qprec, "[M+H]+", True, 25.0)],
         "cands": np.array([7]), "ppm_err": np.array([1.0])}
    return lib, q


def test_featurize_vs_pairwise():
    lib, q = tiny_pair()
    X = stage2.featurize(q["spectra"], q["cands"], np.abs(q["ppm_err"]), lib, TOL)[0]
    qm, qp, _, qprec = q["spectra"][0][:4]
    lmz, lp = lib.get(0)
    lprec = float(lib.table["precursor_mz"].iat[0])
    F = stage2.FIDX
    ok = abs(X[F["entropy_best"]] - entropy_similarity(qm, qp, lmz, lp, TOL)) < 1e-9 \
        and abs(X[F["cosine_best"]] - cosine_similarity(qm, qp, lmz, lp, TOL)) < 1e-9 \
        and abs(X[F["nl_entropy_best"]] - nl_entropy_similarity(qm, qp, qprec, lmz, lp, lprec, TOL)) < 1e-9 \
        and abs(X[F["nl_cosine_best"]] - nl_cosine_similarity(qm, qp, qprec, lmz, lp, lprec, TOL)) < 1e-9
    check("featurize: best/entropy/cosine/nl columns equal the pairwise similarities", ok)
    n, s = match_stats(qm, qp, lmz, lp, TOL)
    check("featurize: matched_peaks / explained_intensity equal match_stats (3 peaks, 0.9)",
          X[F["matched_peaks"]] == n == 3 and abs(X[F["explained_intensity"]] - 0.9) < 1e-12)
    check("featurize: context and flag columns",
          X[F["n_candidates"]] == 1 and X[F["n_library_spectra"]] == 1 and X[F["mass_error_ppm"]] == 1.0
          and X[F["has_same_adduct"]] == 1.0 and X[F["has_same_polarity"]] == 1.0
          and X[F["same_instrument_family"]] == 1.0 and X[F["min_ce_gap"]] == 5.0
          and X[F["n_source_libraries"]] == 1.0)
    check("featurize: composite of a single spectrum equals the spectrum itself",
          abs(X[F["entropy_composite"]] - X[F["entropy_best"]]) < 1e-9)
    X2 = stage2.featurize(q["spectra"], q["cands"], np.abs(q["ppm_err"]), lib, TOL)[0]
    check("featurize: deterministic (identical on a second run)", np.array_equal(X, X2))


# --------------------------------------------------------- within-set z-scoring

def test_zscore():
    X = np.array([[1.0, 5.0], [3.0, 5.0], [10.0, 2.0], [20.0, 2.0], [30.0, 2.0]])
    Z = stage2.zscore_within_sets(X, [2, 3])
    ok = np.allclose(Z[:2, 0], [-1.0, 1.0]) and np.allclose(Z[:2, 1], 0) \
        and abs(Z[2:, 0].mean()) < 1e-9 and abs(Z[2:, 0].std() - 1) < 1e-9 and np.allclose(Z[2:, 1], 0)
    check("z-scoring: mean 0 / std 1 within each shortlist; constant columns become 0", ok)
    Xm = RNG.random((6, len(stage2.FEATURE_COLUMNS)))
    Xm[:, stage2.FIDX["n_candidates"]] = 99.0                       # context, must be dropped
    Xm[:3, stage2.FIDX["min_ce_gap"]] = [np.nan, 1.0, 3.0]
    Xm[3:, stage2.FIDX["min_ce_gap"]] = np.nan
    Zm = stage2.model_matrix(Xm, [3, 3])
    col = stage2.FEATURES.index("entropy_best")
    flag = stage2.FEATURES.index("has_same_polarity")
    check("model matrix: n_candidates dropped, min_ce_gap imputed, no NaN, raw flags untouched",
          Zm.shape == (6, len(stage2.FEATURES)) and np.isfinite(Zm).all()
          and abs(Zm[:3, col].mean()) < 1e-9 and abs(Zm[:3, col].std() - 1) < 1e-9
          and np.allclose(Zm[:, flag], Xm[:, stage2.FIDX["has_same_polarity"]]))


# --------------------------------------------------------------- smoke test

N_QUERY, DECOY0 = 20, 100
MASSES = {m: 200.0 + 10.0 * m for m in range(N_QUERY)}
for d in range(N_QUERY):        # two decoys per query molecule: +3 ppm and -4 ppm
    MASSES[DECOY0 + d] = (200.0 + 10.0 * d) * (1 + 3e-6)
    MASSES[DECOY0 + N_QUERY + d] = (200.0 + 10.0 * d) * (1 - 4e-6)
CE_OF_ROW = {m * 4 + j: ((10.0,) if j < 2 else (40.0,)) for m in range(N_QUERY) for j in range(4)}
ROW_KEY = np.array([m for m in range(N_QUERY) for _ in range(4)]
                   + [DECOY0 + d for d in range(2 * N_QUERY) for _ in range(2)], dtype=np.int32)


def row_spectrum(r: int):
    m = int(ROW_KEY[r])
    prec = MASSES[m] + 1.007276
    if m >= DECOY0:      # decoys: peaks shifted 1.5 Da -> no match with any query
        offs, inten = np.array([118.5, 101.5, 84.5, 67.5, 50.5]), np.array([0.8, 0.6, 0.5, 0.4, 0.3])
    else:
        offs, inten = np.array([120.0, 103.0, 86.0, 69.0, 52.0]), np.array([0.9, 0.7, 0.5, 0.4, 0.3])
    return prec - offs + 0.003 * np.sin(r), inten * (1.0 + 0.05 * np.cos(r)), prec


def fake_read_rows(row_ids, columns, train_path):
    rows = []
    for r in np.asarray(row_ids, dtype=np.int64):
        mz, inten, prec = row_spectrum(int(r))
        rows.append({"ms2_mzs": mz.tolist(), "ms2_normalized_intensities": inten.tolist(),
                     "precursor_mz": prec, "adduct": "[M+H]+", "ionization_mode": "positive",
                     "instrument_type": "timsTOF", "collision_energy_ev": list(CE_OF_ROW.get(int(r), (25.0,)))})
    return pd.DataFrame(rows, index=np.asarray(row_ids, dtype=np.int64))[list(columns)]


def build_fake_library(mask, params):
    rows = np.flatnonzero(mask)
    mzs, ps, lens, tab = [], [], [], []
    for r in rows:
        mz, inten, prec = row_spectrum(int(r))
        cm, cp = clean_spectrum(mz, inten, prec, params)
        mzs.append(cm)
        ps.append(cp)
        lens.append(len(cm))
        ce = CE_OF_ROW.get(int(r), (25.0,))[0]
        tab.append({"row_id": int(r), "mol": int(ROW_KEY[r]), "adduct": "[M+H]+",
                    "ionization_mode": "positive", "instrument_type": "timsTOF",
                    "precursor_mz": prec, "collision_energy_ev": [float(ce)],
                    "ingest_lib": "libA" if int(r) % 2 else "libB"})
    return Library(pd.DataFrame(tab), np.concatenate(mzs), np.concatenate(ps),
                   np.asarray(lens, dtype=np.int64))


def test_smoke():
    params = CleanParams()
    n_rows = len(ROW_KEY)
    meta = pd.DataFrame({"mol": ROW_KEY, "eligible": np.ones(n_rows, dtype=bool),
                         "is_tims": np.ones(n_rows, dtype=bool),
                         "adduct": ["[M+H]+"] * n_rows,
                         "ce": [CE_OF_ROW.get(r, (25.0,)) for r in range(n_rows)],
                         "precursor_mz": [MASSES[int(ROW_KEY[r])] + 1.007276 for r in range(n_rows)],
                         "instrument_type": ["timsTOF"] * n_rows,
                         "ionization_mode": ["positive"] * n_rows})
    mol_ids = list(range(N_QUERY)) + list(range(20, DECOY0)) \
        + list(range(DECOY0, DECOY0 + 2 * N_QUERY))
    molecules = pd.DataFrame({"key": [f"K{m}" for m in mol_ids], "smiles": ["C"] * len(mol_ids),
                              "exact_mass": [MASSES.get(m, np.nan) for m in mol_ids]})
    order, start = common.mol_rows(meta)

    # -- sampler (R1, R2, R5): hold out one CE setting per molecule
    rng = np.random.default_rng(7)
    examples, shortfall = stage2.sample_training_examples(
        N_QUERY, "ce=1.0", False, rng, meta, order, start, set(), {})
    check("smoke: sampler returns 20 ce-change molecules from the 20-molecule pool",
          len(examples) == N_QUERY and all(e["change"] == "ce" for e in examples)
          and all(0 <= e["mol"] < N_QUERY for e in examples))
    mask = stage2.training_mask(SimpleNamespace(train_mask=np.ones(n_rows, dtype=bool)), examples)
    check("smoke: R2 - query rows hidden, visible rows kept",
          all(not mask[e["hidden_rows"]].any() and mask[e["visible_rows"]].all()
              and not mask[e["query_rows"]].any() for e in examples))

    cidx = CandidateIndex(molecules, ROW_KEY, mask)
    queries, dropped = stage2.build_training_queries(examples, fake_read_rows, "fake", meta,
                                                    cidx, 10.0, params)
    check("smoke: 20 shortlists kept, nothing dropped", len(queries) == N_QUERY
          and not any(dropped.values()))
    check("smoke: every shortlist holds exactly the truth and its 2 decoys",
          all(len(q["cands"]) == 3 and (q["cands"] == q["mol"]).sum() == 1 for q in queries))
    lib = build_fake_library(mask, params)
    q_rows = np.concatenate([e["query_rows"] for e in examples])
    check("smoke: R2 - no pseudo-query row in the library",
          not np.isin(q_rows, lib.table["row_id"].to_numpy()).any())

    Xs = [stage2.featurize(q["spectra"], q["cands"], np.abs(q["ppm_err"]), lib, params.tol)
          for q in queries]
    Xs2 = [stage2.featurize(q["spectra"], q["cands"], np.abs(q["ppm_err"]), lib, params.tol)
           for q in queries]
    check("smoke: featurize deterministic across a full training set",
          all(np.array_equal(a, b) for a, b in zip(Xs, Xs2)))
    eb = stage2.FIDX["entropy_best"]
    check("smoke: entropy_best separates truth from decoys in every shortlist",
          all(X[q["cands"] == q["mol"], eb].min() > X[q["cands"] != q["mol"], eb].max()
              for q, X in zip(queries, Xs)))

    # -- fit + predict (the --leak-train principle: a shortlist the model trained
    #    on must have its truth ranked first)
    sizes = [len(q["cands"]) for q in queries]
    X_all, y_all = np.concatenate(Xs), np.concatenate([q["cands"] == q["mol"] for q in queries])
    check("labels: exactly one positive per shortlist",
          all(y.sum() == 1 for y in [y_all[s:e] for s, e in zip(np.r_[0, np.cumsum(sizes)[:-1]],
                                                                np.cumsum(sizes))]))
    models = stage2.fit_models(stage2.model_matrix(X_all, sizes), y_all, 11, ("logistic", "hgb"))
    hits = {}
    for name, model in models.items():
        hits[name] = sum(int(np.argmax(stage2.predict(model, X, [len(q["cands"])]))
                             == np.flatnonzero(q["cands"] == q["mol"])[0])
                         for q, X in zip(queries, Xs))
        check(f"smoke: stage2_{name} ranks the trained truth first for >= 15/20 shortlists "
              f"({hits[name]}/20)", hits[name] >= 15)


def main():
    test_neutral_loss()
    test_match_stats()
    test_featurize_vs_pairwise()
    test_zscore()
    test_smoke()
    print(f"\n{'ALL TESTS PASSED' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}'}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
