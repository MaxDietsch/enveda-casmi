# Potential improvements

Known gaps between the simulated eval sets and the hidden test, with ideas to
close them. None of them is implemented yet. Numbers refer to the default
1000-molecule outputs (seed 2026).

## 1. Too few spectra per molecule

**Problem.** The test has a median of 3 spectra per molecule (~3.75 on
average, 1500 spectra for 400 molecules). Our sets reach a median of 2 and
about 2.5 on average (overall: 2,544 spectra for 1000 molecules instead of
~3,750). The draw `k ~ Geometric(0.25)` is capped by the number of candidate
rows, and most molecules have few eligible spectra (median 4). Difficulty 1
is hit hardest, because its eval spectra must come from one adduct (and, for
`ce`, from part of its collision energies).

**Why it matters.** Predictions are made per molecule from all of its
spectra. With fewer spectra, methods that combine evidence across spectra
look worse locally than on the test.

**Ideas.**
- Draw `k` first, then sample the molecule among those with at least `k`
  candidate rows (mass, ion and diversity weights still apply inside that set).
  The spectra-per-molecule distribution then matches by construction. Side
  effect: a slight preference for well-measured molecules.
- Difficulty 1 `instrument`: allow eval spectra from several adducts, with
  the visible rows being the non-timsTOF rows in those adducts.

## 2. Chemistry: mostly synthetic, drug-like molecules

**Problem.** Eval spectra must be timsTOF, and train's timsTOF data is almost
entirely `enveda-180` (synthetic screening compounds) plus the 250 natural
products of `enveda-np-examples`. The test is natural-product-like. Only 46
eligible molecules weigh more than 500 Da, so the 500–800 and 800–1160 Da
target shares (0.15 and 0.05) come out at about 0.03 and 0.01.

**Ideas.**
- Evaluation: report scores per source library (`ingest_lib` of the eval
  rows). This is the evaluation's job, not the dataset's.
- A natural-product variant (`--eval-instruments tims|qtof`) that also
  accepts Q-TOF spectra with CE in eV from `riken`, `gnps`, `mona`,
  `massbank` and `msdial`. timsTOF is a Q-TOF, so fragmentation is close.
  First count how many such spectra pass the eligibility rules.
- Weight molecule sampling by a natural-product-likeness score (or COCONUT
  membership as a proxy).

## 3. Analogues remain in train

**Problem.** Even with neighbourhood removal for 10% of the level 2/3
molecules, the median Tanimoto of a level 2/3 molecule to its nearest
visible train molecule is 0.59, and only ~15% are below 0.5. `enveda-180`
consists of combinatorial families, so near analogues are everywhere. Real
class 2/3 natural products probably have fewer.

**Ideas.**
- Use `max_visible_tc` to report scores per similarity band (for example
  < 0.4, 0.4–0.6, > 0.6). The column is already in every output.
- Allow a higher `--neighbour-removed-frac` (the current [0.05, 0.20] limit
  comes from the original plan) or a lower `--neighbour-tc`.

## 4. Fingerprint collisions

**Problem.** A few molecules (3 of 1000 in difficulty 2) have Tanimoto 1.0
to a different molecule: Morgan radius 2 cannot tell some homologues and
positional isomers apart.

**Idea.** Use count-based Morgan fingerprints or radius 3 for the similarity
queries.

## 5. Rare adducts

**Problem.** `[M+H]+` and `[M-H]-` dominate; `[M-H2O-H]-`, `[M-H2O+H]+`,
`[M-2H2O+H]+` and `[M+K]+` occur 0–4 times. The test's adduct distribution
is unknown, so this is not necessarily wrong.

**Idea.** An optional minimum number of spectra per adduct.

## 6. Noise in the score

**Problem.** Even with 1000 molecules the MRR has a standard error of about
±0.013.

**Idea.** Build several seeds (for example 2026–2030) and compare methods
paired on the same splits.
