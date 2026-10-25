"""Correctness tests for the stage-1 building blocks (no data needed, ~2 s).

    uv run python max_class1_exp/test_stage1.py

The end-to-end self-test with real data is a separate command (see README):
    uv run python max_class1_exp/stage1.py --leak-eval --limit 200
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from library import query_mass  # noqa: E402
from spectra import (LN4, CleanParams, centroid, clean_spectrum, cosine_one_vs_many,  # noqa: E402
                     cosine_similarity, entropy, entropy_one_vs_many, entropy_similarity,
                     entropy_weight, merge_spectra, unit_norm)

RNG = np.random.default_rng(0)
FAILED = []


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAILED.append(name)


def random_spectrum(n, lo=50, hi=500):
    mz = np.sort(RNG.choice(np.arange(lo, hi, 0.5), size=n, replace=False))  # >= 0.5 Da apart
    p = RNG.random(n) + 0.01
    return mz, p / p.sum()


def brute_entropy_similarity(mz_a, p_a, mz_b, p_b, tol):
    """1 - (2 S(AB) - S(A) - S(B)) / ln 4 on an explicitly merged spectrum
    (peaks here are >= 0.5 Da apart, so matching is unambiguous)."""
    ab = {}
    for m, a in zip(mz_a, p_a):
        ab[round(m, 6)] = ab.get(round(m, 6), 0) + a / 2
    for m, b in zip(mz_b, p_b):
        hit = [k for k in ab if abs(k - m) <= tol and k in set(np.round(mz_a, 6))]
        k = hit[0] if hit else round(m + 1e-7, 7)
        ab[k] = ab.get(k, 0) + b / 2
    return 1 - (2 * entropy(np.array(list(ab.values()))) - entropy(p_a) - entropy(p_b)) / LN4


def main():
    tol = 0.02

    # ---- entropy similarity
    mz, p = random_spectrum(20)
    check("entropy: identical spectra -> 1", abs(entropy_similarity(mz, p, mz, p, tol) - 1) < 1e-9)
    mz2, p2 = random_spectrum(15, 600, 900)
    check("entropy: no shared peaks -> 0", entropy_similarity(mz, p, mz2, p2, tol) == 0)
    ok_brute, ok_sym, ok_range = True, True, True
    for _ in range(200):
        a_mz, a_p = random_spectrum(int(RNG.integers(3, 30)))
        b_mz, b_p = random_spectrum(int(RNG.integers(3, 30)))
        b_mz = b_mz + RNG.uniform(-0.015, 0.015, len(b_mz)) * (RNG.random(len(b_mz)) < 0.5)
        b_mz = np.sort(b_mz)
        s1 = entropy_similarity(a_mz, a_p, b_mz, b_p, tol, weighted=False)
        s2 = entropy_similarity(b_mz, b_p, a_mz, a_p, tol, weighted=False)
        ref = brute_entropy_similarity(a_mz, a_p, b_mz, b_p, tol)
        ok_brute &= abs(s1 - ref) < 1e-9
        ok_sym &= abs(s1 - s2) < 1e-9
        ok_range &= -1e-12 <= s1 <= 1 + 1e-12
    check("entropy: equals brute-force 1 - (2S(AB) - S(A) - S(B))/ln4 (200 random pairs)", ok_brute)
    check("entropy: symmetric", ok_sym)
    check("entropy: within [0, 1]", ok_range)

    # ---- one query vs many == pairwise
    lib = [random_spectrum(int(RNG.integers(1, 25))) for _ in range(30)]
    lib.append(lib[3])                                       # a duplicate of another library spectrum
    q_mz, q_p = lib[7][0].copy(), lib[7][1].copy()           # query equals library spectrum 7
    l_mz = np.concatenate([m for m, _ in lib])
    l_spec = np.repeat(np.arange(len(lib)), [len(m) for m, _ in lib])
    l_pw = np.concatenate([entropy_weight(x) for _, x in lib])
    l_pu = np.concatenate([unit_norm(x) for _, x in lib])
    many_e = entropy_one_vs_many(q_mz, entropy_weight(q_p), l_mz, l_pw, l_spec, len(lib), tol)
    many_c = cosine_one_vs_many(q_mz, unit_norm(q_p), l_mz, l_pu, l_spec, len(lib), tol)
    pair_e = np.array([entropy_similarity(q_mz, q_p, m, x, tol) for m, x in lib])
    pair_c = np.array([cosine_similarity(q_mz, q_p, m, x, tol) for m, x in lib])
    check("entropy: one-vs-many equals pairwise", np.allclose(many_e, pair_e))
    check("cosine: one-vs-many equals pairwise", np.allclose(many_c, pair_c))
    check("one-vs-many: the identical library spectrum scores 1 and ranks first",
          abs(many_e[7] - 1) < 1e-9 and int(np.argmax(many_e)) == 7)

    # ---- cosine against a direct dot product
    a_mz, a_p = random_spectrum(10)
    b_p = RNG.random(10)
    b_p /= b_p.sum()
    direct = float(unit_norm(a_p) @ unit_norm(b_p))
    check("cosine: equals the dot product of aligned unit vectors",
          abs(cosine_similarity(a_mz, a_p, a_mz, b_p, tol) - direct) < 1e-12)

    # ---- one-to-one matching: two library peaks near one query peak count once
    q_mz, q_p = np.array([100.0]), np.array([1.0])
    l_mz, l_p = np.array([99.99, 100.012]), np.array([0.5, 0.5])
    s = entropy_one_vs_many(q_mz, q_p, l_mz, l_p, np.zeros(2, dtype=np.int64), 1, tol)[0]
    check("matching: one query peak matches at most one peak per library spectrum", s <= 1 + 1e-12)
    c = cosine_one_vs_many(q_mz, q_p, l_mz, unit_norm(l_p), np.zeros(2, dtype=np.int64), 1, tol)[0]
    check("matching: cosine also counts the query peak once", abs(c - unit_norm(l_p)[0]) < 1e-12)

    # ---- tolerance edge
    check("matching: peaks 0.019 Da apart match, 0.021 Da apart do not",
          entropy_similarity(np.array([100.0]), np.array([1.0]), np.array([100.019]), np.array([1.0]), tol) > 0.99
          and entropy_similarity(np.array([100.0]), np.array([1.0]), np.array([100.021]), np.array([1.0]), tol) == 0)

    # ---- entropy weighting
    flat = np.ones(40) / 40                                   # S = ln 40 = 3.69 >= 3 -> unchanged
    check("weighting: S >= 3 leaves intensities unchanged", np.allclose(entropy_weight(flat), flat))
    peaky = np.array([0.7, 0.2, 0.1])
    s = entropy(peaky)
    expect = peaky ** (0.25 + 0.25 * s)
    check("weighting: S < 3 raises to 0.25 + 0.25 S and renormalises",
          np.allclose(entropy_weight(peaky), expect / expect.sum()))

    # ---- cleaning
    mz = np.array([50.0, 80.0, 80.01, 120.0, 199.0, 200.5, 201.0])
    it = np.array([1.0, 50.0, 30.0, 100.0, 0.5, 40.0, 90.0])
    cm, cp = clean_spectrum(mz, it, precursor_mz=201.0, p=CleanParams(tol=0.02, noise=0.01))
    check("cleaning: peaks above precursor - 1.6 removed", cm.max() < 201.0 - 1.6 + 1e-9)
    check("cleaning: peaks below 1 % of the base peak removed", not np.isclose(cm, 199.0).any())
    check("cleaning: peaks within tol merged (80.0 + 80.01 -> one peak, weighted m/z)",
          np.isclose(cm, (80.0 * 50 + 80.01 * 30) / 80).any() and not np.isclose(cm, 80.01).any())
    check("cleaning: intensities sum to 1", abs(cp.sum() - 1) < 1e-12)
    m2, i2 = centroid(np.array([10.0, 10.015, 10.03, 10.045]), np.array([1.0, 5.0, 1.0, 4.0]), 0.02)
    check("centroid: greedy from the most intense peak (chain of 4 -> 2 peaks)",
          len(m2) == 2 and np.isclose(i2.sum(), 11.0))
    empty = clean_spectrum(np.array([300.0]), np.array([1.0]), precursor_mz=250.0)
    check("cleaning: a spectrum with only precursor-region peaks becomes empty", len(empty[0]) == 0)

    # ---- composite spectrum
    s1 = (np.array([100.0, 150.0]), np.array([0.5, 0.5]))
    s2 = (np.array([100.005, 200.0]), np.array([0.5, 0.5]))
    cm, cp = merge_spectra([s1, s2], tol)
    check("composite: shared peak merged, weights averaged, sums to 1",
          len(cm) == 3 and np.isclose(cp.sum(), 1) and np.isclose(cp.max(), 0.5))

    # ---- neutral mass of the query (caffeine C8H10N4O2, 194.08038)
    m = query_mass([195.08765, 217.06960], ["[M+H]+", "[M+Na]+"])
    check(f"query mass: caffeine from [M+H]+ and [M+Na]+ within 2 ppm ({m:.5f})",
          abs(m - 194.08038) / 194.08038 * 1e6 < 2)

    print(f"\n{'ALL TESTS PASSED' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}'}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
