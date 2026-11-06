# Kaggle forum findings for the Class-2 experiment

**Competition:** `enveda-CASMI26-molecule-id-mass-spectra` (id 143318)
**Forum:** competition discussion tab, forum id `11293906`
**Read on:** 2026-11-05
**Coverage:** all **150** topics listed; **149** full threads downloaded (OP + all
nested comments). The only failed fetch was `746430` (FRIGID licensing); the
host's FRIGID answer is in `745072` (answer C-5).
**Discussion URL pattern:**
`https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/<topic_id>`

Scores quoted below are self-reported at the time of each post (Sep–Nov 2026).
The public leaderboard moved from ~0.34 in mid-September to ~0.46–0.48 by
early November, so early "top-1" posts are historical, not current.

---

## 0. TL;DR — what to change in this experiment

The forum's single most important message for us: **class 2 is a ranking
problem, and the ranking errors are same-formula isomers.** Our README already
says "all difficulty is in ranking"; the forum supplies the numbers and the
failure taxonomy. The changes below are ordered by how much they can change
our conclusions, not by implementation effort.

| # | Change | Why (forum evidence) | Expected effect |
|---|---|---|---|
| **P0.1** | Stop treating "class 2 ≈ 60 %" as fact. Add class-share scenarios 16/45/39 and 28/20/52 next to our 20/60/20. | `741597` estimates **16 / 45 / 39**; `748524` LB-fits **28 / 20 / 52**. The hosts do not disclose it. | Prevents us from over-investing in class 2 if its true share is ~20 %. The method is share-independent; only the expected overall gain scales. |
| **P0.2** | Add a **same-formula isomer panel** metric (truth + same-formula candidates, and truth + top-N same-formula decoys) to `stage2.py` and `f2_eval.py`. | `742055`: removing same-formula blockers closes **95.1 %** of the class-2 gap; ~66 of ~83 in-window candidates share the truth's formula. `743254`: 98 % of wrong winners have the same formula. | The sensitive instrument for every class-2 change. Overall MRR dilutes isomer wins; this does not. |
| **P0.3** | Add a per-molecule **failure ledger**: truth outside window / in window rank >25 / rank 2–25 / rank 1, plus conditional MRR. | `742055` buckets: 1.6 % recall, 40 % rank 2–25, 2.4 % >25; 17:1 ranking-vs-recall headroom. | Tells us whether a change affects recall, tail ranking, or top-1 — they need different fixes. |
| **P0.4** | Stratify every **pretrained-model** result (F2/ICEBERG, any FPNet/DreaMS) by "structure has public spectra / is in MassSpecGym". Report the no-public-spectra slice separately. | `748524`: ICEBERG re-ranking +0.02 on class 2 overall, **~0 on structures not in MassSpecGym**. `749217` documents the same pretraining-corpus/provenance problem for otherwise-allowed models. | Stops us from shipping an F2 gain that is memorisation. |
| **P0.5** | Add a **spectrum-blind control** with a non-vacuous candidate-coverage check, and report candidate coverage for every run. | Pathik (`752735`): a blind control that reads 0.000 can be arithmetically forced when coverage is 0. | Catches shortcut detectors that cannot fail. |
| **P0.6** | Run **5 dataset seeds** (2026–2030) × at least 3 model seeds, paired bootstrap by molecule. Report SE. | `743254`: seed noise ±0.007 overall, class-2 MRR ~0.02 (`748524`); `753511`: best-of-7 public probes adds +0.019 of pure noise. `POTENTIAL_IMPROVEMENTS.md` §6 already flags ±0.013 SE. | Prevents chasing noise; makes "channel X helps" claims defensible. |
| **P1.1** | Add **simple MetFrag-lite fragmentation features** next to the Fragmentor: 1-bond and 2-bond cuts (2-bond ×0.6), H shifts −2..+3, 0.005–0.01 Da, **linear intensity**, explained intensity, **rarity-weighted** explained intensity, **gap to the best candidate**. | `742055`/`749251`: in-silico fragmentation was the single most important feature (0.502 → 0.627 on the no-public-spectra split); `742055` NeckBeard: +0.065 MRR on an isomer panel from four simple changes. | The strongest cheap class-2 lever we are not currently measuring in its simple form. |
| **P1.2** | Change defaults: `--ppm 10 → 8.5`; sweep `--tol 0.02 → 0.01 / 0.005`; sweep linear vs sqrt intensity. | `743254`: 99 % of timsTOF precursors within 5 ppm; 8.5 ppm is the community sweet spot; wider windows add decoys. NeckBeard: linear > sqrt, 0.005 Da > 0.01 Da (small effects, same direction). | Small but free; also makes the pool comparable to the public class-2 baselines. |
| **P1.3** | Add **multi-adduct mass-agreement**, per-adduct mass error, and candidate **source/visibility flags** to the ranker. | `741423`: multi-adduct neutral masses agree to 1.3 ppm median; 97/400 molecules have both polarities. `752356`: source flags were named the real class-3 ranker lever; test them for class 2 as context, not as a truth shortcut. | A confidence signal the current mass feature throws away; helps calibrate the asymmetry. |
| **P1.4** | Build channel P as an **FPNet-style** model, not just a linear FingerID-lite: ~6–7k fingerprint bits, BCE, hard negatives from the 10 ppm pool window, merge augmentation, bit-prior init, candidate score `f_c·z`. | `741745`: 6-layer transformer, 6,930 bits, `f_c·z`, 31 features, HistGBM 4 seeds → 0.339. `745952`: the exact training recipe; second FPNet adds only ~0.002. | The top solution's centerpiece; our v2 plan already points this way. |
| **P1.5** | Add a **safe rank-1 protection** mode: keep the best channel's top-1, insert the fused top-1 at rank 2; compare with full re-rank. | `749412`: +0.002 on two bases; cannot lose a rank-1 hit. `745952`: locking engine picks raised CV 0.63→0.89 but **LB 0.413→0.409** — caution against aggressive re-ranking. | Cheap insurance, measurable on our splits. |
| **P2.1** | Keep the external pool to **COCONUT only**. Do not add PubChem; if tested, formula-filter and tail-only, and report separately. | `743254`: PubChem top-50 expansion **0.335 → 0.205**; `741659`: class-2 MRR 0.52 → 0.35; `742088`: full PubChem +0.002 LB, tail +0.001. | Avoids the most repeatable negative result in the forum. |
| **P2.2** | Do **not** use COCONUT/NP-membership or popularity as ranker features. Use them for stratification only. | `743254`: membership flags looked great on public holdouts and hurt on LB; `748068`: popularity prior +0.50 local vs +0.075 LB, 67 % vs ~0 % on the no-public-spectra split. | Avoids a seductive shortcut. |
| **P2.3** | Add submission hygiene: metric-key dedupe, no filler/invalid guesses, ≤25 valid guesses, no nulls/duplicate ids; do not hardcode 400; pin the Docker image. | `742088`: filler guesses scored the whole submission **0.000**; `749412`: a failing assert silently scores the previous `submission.csv`; `749908`: unpinned copy 0.441 → 0.427. | Protects real submissions. |

---

## 1. What "Class 2" is, and how big it is

Host-confirmed (`742274`):

- **Class 1** — reference spectrum exists in public libraries.
- **Class 2** — structure is in **PubChem or COCONUT**, but has **no reference
  spectrum**; reachable by database retrieval, not by library matching.
- **Class 3** — absent from **both** PubChem and COCONUT; requires generation.
  (Host: "Class 3 is absent from both PubChem and COCONUT.")

Some threads call these **Regime A / B / C** (`755962`, `743974`).

### Class shares are not known — two conflicting estimates

| Source | Class 1 | Class 2 | Class 3 | How derived |
|---|---:|---:|---:|---|
| `741597` (haideptry, citing prvsiyan's notebook) | ~16 % | **~45 %** | ~39 % | LB-probing: library-only submission = 0.151, class-1 MRR ~0.93 → `f1 ≈ 16.2 %`; then analog-propagation calibration. |
| `748524` (Bull, 2026-10-12) | ~28 % | **~20 %** | ~52 % | Fitted from LB scores of several submissions under stated per-class MRR assumptions. |
| `dataset-creation/README` / our `overall` default | 20 % | **60 %** | 20 % | Our simulator default (`--share`). |

**Implication for us:** our README §0 says "Class 2 is assumed to be ~60 % of
the hidden test." That is the most optimistic of the three. The method work is
unaffected — class 2 is still the largest *addressable* class under every
estimate — but every "expected overall LB gain" claim must be shown at 20 %,
45 %, and 60 %. The forum has no ground truth; the hosts explicitly do not
disclose shares (`748882` answer C-3: "participants should expect members of
all three classes in both sets").

---

## 2. The central result: ranking, not retrieval

From `742055` ("Six variants, one plateau at 0.33–0.34"), the best-documented
class-2 diagnostic on the forum. Setup: 250 held-out natural products, **every
spectrum of the true structure removed from every library**, split by
InChIKey14, library search asserted at exactly 0.000.

| Bucket | Queries | MRR headroom |
|---|---:|---:|
| Truth **absent** from candidate set (recall) | 4 (1.6 %) | +0.016 |
| Truth present, ranked **2–25** | 100 (40 %) | **+0.274** |
| Truth present, ranked **>25** | 6 (2.4 %) | +0.024 |

**Ranking beats recall ~17:1.** Further:

- Oracle: shipped **0.7315** → same-formula blockers removed **0.9869** →
  truth rank 1 everywhere **1.0000**. Perfect same-formula separation is
  **95.1 % of the entire gap**; everything else including recall is ~0.013.
- **~83 candidates** in a window, **~66 share the truth's formula**. In rank-2
  cases, 40/42 blockers had the truth's exact formula; **69 % have a different
  Murcko scaffold**.
- Median Tanimoto(truth, winner) = **0.48**; only **20 % above 0.7**.
- The truth's mass-shifted analog score is already **~90 % of the best
  candidate's** (median); only **2 %** of ranking failures are "analog-poor".
  Stronger analog propagation is *not* what separates the truth from the
  isomer that beat it.

`743254` adds the same picture from a different pipeline:

- When the truth is in the candidate list but not ranked first, the wrong
  winner has the **same molecular formula 98 %** of the time.
- **Mass/formula scoring is not the lever; separating constitutional isomers
  is.** A small "same-formula isomer panel" (truth + best-scoring isomers) was
  a much more sensitive development metric than overall MRR.
- **8 %** of molecules had a tautomer copy of the truth under a different
  PubChem key — dedupe with the metric's key, not raw InChIKey.

**Consequence for our experiment:** our README's premise is correct, and our
`max_visible_tc` bands are the right analogue of the forum's "analog-poor"
analysis. But our headline MRR@25 will be dominated by easy candidates. The
same-formula panel (P0.2) is the metric we are missing.

---

## 3. What works

### 3.1 Analog propagation is a strong baseline, but saturated

The public baseline progression (`741597`, `749251`):

| Step | Public LB |
|---|---:|
| Spectral library search only | 0.158 |
| + COCONUT candidates ranked by analog propagation | 0.223 |
| + newer COCONUT release | 0.243 |
| + gated library term | 0.250 |
| + in-silico fragmentation channel | 0.283 |
| Public 4-channel pipeline (prvsiyan) | 0.335 |

Our optional analogue channel (`--enable-analogues`) is a version of this.
Keep it **off by default**; when enabled, use it as a measured baseline, not a
headline. `742055` says the signal is already ~90 % saturated, and
`741659` notes that adding candidates/analogs can hurt before it helps under
MRR@25.

### 3.2 In-silico fragmentation is the strongest cheap class-2 lever

Two independent measurements:

**Aryan Awasthi (`749251`, `748524`)** on his class-2-like split (structures
with **no public spectra**, 490 `enveda-180` molecules):

- in-silico fragmentation (1–2 bond cuts, ±2 H, 10 ppm, explained intensity,
  expressed as the **gap to the best candidate in the same molecule**):
  **0.502 → 0.627** — his single most important ranker feature;
- adding analog features: **0.627 → 0.660**;
- direction transferred to the LB every time; magnitude was **+0.03–0.04**
  instead of +0.14.

**NeckBeard (`742055` reply)** on a 241-molecule `enveda-np-examples` isomer
panel, all same-formula candidates, paired t-tests. Baseline: all cleavages
equal, sqrt intensity, 0.01 Da, H shifts −2..+2.

| Change | dMRR | t |
|---|---:|---:|
| Down-weight 2-bond cleavages (×0.6 vs 1-bond) | **+0.030** | +2.6 |
| Linear intensity instead of sqrt | **+0.017** | +2.0 |
| Tolerance 0.005 Da instead of 0.01 | +0.005 | +1.0 |
| H shifts −2..+3 instead of −2..+2 | +0.010 | +1.2 |
| **All four combined** | **+0.065** | +4.6 |

What did **not** help: hand-written bond-cleavage propensity weights (−0.002),
allowing 3-bond cleavages (**−0.031**, t=−2.1), penalising fragments reachable
many ways (−0.015, t=−2.3), subtracting a chance-match baseline (~0).

Read the pattern: **parsimony helps, hand-written cleavage chemistry does
not.** The useful signal is "how few cleavages does this candidate need", not
"which bond should break."

`745468` adds the chemical diagnosis: formula is predicted correctly in >95 %
of validation examples; the failures are isomer discrimination. The proposed
next step — measure the **formulas of the fragments themselves** and judge
whether the candidate formula can produce them — is a P2 extension.

**Our `insilico.py` currently uses the repo Fragmentor** (single cuts, ring
fissions, neutral-loss cascades, even-electron rule, functional-group gating;
~700 fragments/molecule). That is richer than the forum's MetFrag-lite, but
the forum's result is that the **simple** version is what produced the gain.
We should add the simple feature block as a separate channel and compare, not
replace the Fragmentor.

### 3.3 Spectrum→fingerprint (FPNet) is the other centerpiece

`741745` (0.339, rank 1 at the time) describes the public top pipeline:

- 4 channels: exact spectral matching, mass-shifted analog propagation,
  chemical/MetFrag-lite heuristics, **spectrum→fingerprint transformer**.
- **FPNet:** small 6-layer Transformer predicting **6,930 fingerprint bits**;
  candidate score is the Bernoulli log-likelihood `f_c · z` (the second term is
  constant across candidates), so tens of thousands of candidates score by one
  matrix multiply.
- Final ranker: **31 features**, HistGradientBoosting with **4 seeds × 2
  calibrated prior settings**, then confidence gating.
- What did **not** work: blind PubChem isomer expansion; overly wide mass
  windows. COCONUT 2.0 / ChEBI / LIPID MAPS were the useful pools; **±8.5 ppm**
  was the best precision/recall trade-off.

`745952` gives the training recipe used by a later fusion pipeline:

- **BCE + hard negatives from the 10 ppm pool window + merge augmentation +
  bit-prior initialisation**.
- A second FPNet trained with the same recipe lands at 0.548 vs 0.579 window
  MRR and adds **~0.002** in the ensemble. "More of the same model is not
  where the next points are."

Cautions from `743254` / `749251` / `748068`:

- Two new FP models reached hard-negative top-1 **0.694 / 0.640** on a random
  2 % structure split, then scored **lower** on the LB (0.330 vs 0.335).
  Random structure splits leak near-identical molecules.
- Fine-tuning FPNet on timsTOF: offline top-1 0.459 → 0.491, **LB 0.337 →
  0.328**. "Fine-tune on timsTOF" is in practice "fine-tune on `enveda-180`
  chemistry" (1,154,969 timsTOF rows, 1,153,785 are `enveda-180`; only 1,184
  lie outside).
- `748068`: `fpnet_full1` was trained on all training data, so the public
  pipeline as built **cannot be validated** on held-out training data.

**For us:** channel P v1 as a linear FingerID-lite is a reasonable baseline,
but the forum says the transformer head is where the value is. Implement the
FPNet details above, and validate only on the no-public-spectra slice.

### 3.4 The ranker: list-relative features, seeds, source flags

- `741745`: 31 features, HistGBM, 4 seeds, 2 priors, confidence gating.
- `748524`: LambdaRank; in-silico fragmentation 0.502 → 0.627, analog features
  → 0.660 on the class-2-like split.
- `752356` (0.464 base): engine-1's ranker uses **list-relative features**
  (ranks, z-scores, gap to best, softmax, formula share). This is why adding
  generated candidates re-scores the originals and can demote correct top-1s;
  their `genfix` freezes the original run and inserts only new structures on
  the old score scale minus a penalty.
- `752356`: retraining the engine-1 ranker **with source flags** was named the
  "real class-3 lever"; source flags are likely useful for class 2 too
  (candidate visibility, library source, instrument match).
- `742088`: structural proximity (Tanimoto to the library twin) and the
  learned ranker produced **different answer sets with the same total MRR**
  (0.324 vs 0.324, tails differed in ~90 % of rows) — blend the signals rather
  than choosing one.

Our `features.py` already z-scores continuous features within each shortlist
(the "contrast within set" trick). That is aligned. Add source flags and the
explicit gap-to-best features.

### 3.5 Forward models / simulators: coverage is the lever

`752356` audited 481 public versions:

- The big repeatable steps (**+0.008 each**) came from **giving
  ICEBERG/GLACIER simulator evidence on more molecules**, not from reweighting.
- PubChem promotion threshold tuning bought **+0.006 once**; ~20 other tweaks
  were flat.
- Gold was ~0.471 at the time; one molecule at rank 1 is worth ~+0.0076 on the
  public board.

`748524`: ICEBERG re-ranking gave **+0.02 on class 2 overall**, but **~0 on
structures not in MassSpecGym** — it was memorising, not generalising.

`745952`: a second FPNet adds only ~0.002; `752356`: the big repeatable steps
came from simulator coverage, not from stacking another copy of the same model.

Other allowed forward models, per host answers:

- **CFM-ID 4** (`743774`) — allowed.
- **ICEBERG PubChem atlas** (`745134`) — 91.4 M predicted spectra; the host
  confirmed the atlas is usable as model output, with retrieval/processing
  code; the membership list is also usable. (The generating weights were
  NIST'23-trained, but the host allowed the **atlas outputs** explicitly on
  2026-10-28. A 2026-11-01 follow-up asks whether the later Oct-30 guidance
  changes this and is unanswered in the thread — re-check before relying on
  the atlas for a prize-eligible submission.)
- **MIST** checkpoints (`751373`) — allowed (MassSpecGym-trained; MIT).
- **DreaMS** (`749217`) — allowed; but on truth-vs-blocker pairs it was below
  chance (20/42), so not a ranker. Use as an embedding/coverage probe only.
- **MS2DeepScore / FIORA-OS** (`751373`) — allowed if trained only on public
  libraries with public training code.
- **SIRIUS / CSI:FingerID** — **not** allowed (training data includes NIST).

**For us:** our `nnspec.py` (ICEBERG 2.1 zero-shot) is the right first step.
Add (a) **coverage** reporting: fraction of candidates with a finite
prediction, (b) a **GLACIER fallback** where ICEBERG has no evidence, (c) a
separate score on the no-public-spectra slice, (d) optionally CFM-ID 4 as a
second forward model with a different error structure. Do not stack a second
FPNet and expect more than ~0.002.

### 3.6 Multi-spectrum / multi-adduct evidence

`741423` and `741641`:

- 93 % of test peaks are below 1 % of the base peak; 77 % below 0.1 %. After a
  1 % floor the median spectrum has **18 peaks** (52 after 0.1 %).
- The precursor peak is missing in **44 %** of test spectra; treat
  `precursor_mz` as metadata, not a peak. About 3 % of peaks sit slightly
  above the precursor (the +2 Da isotope allowance).
- Different adducts of one molecule agree on neutral mass to a **median of
  1.3 ppm** (90th pct 3.1 ppm) — a very reliable anchor.
- 97 of 400 visible molecules have both positive and negative spectra; the
  metric scores per molecule, so combine.
- Collision energy behaves physically: intensity-weighted mean fragment /
  precursor = 0.64 at 20 eV, 0.44 at 40 eV, 0.36 at 60 eV; only 24 % of
  molecules have the full 20/40/60 + merged ladder.
- Use `collision_energy_ev`, not `collision_energy_orig` (negative in negative
  mode).

Our split already uses timsTOF, ten adducts, median 3 spectra/molecule, and a
median implied neutral mass. Add **mass-agreement spread** as a feature and
report per-adduct errors; don't assume a fixed CE ladder.

### 3.7 Safe rank-1 protection / insertion

`749412` (cheapest validated trick on the forum):

- On the E006 public pipeline, when engine 2's top-1 differs from the final
  rank 1, engine 2 is right in ~3 molecules and E006 in ~13. Swapping engine 2
  to rank 1 scored **0.403**; inserting engine 2's top-1 at **rank 2**,
  deduplicated by the metric key, scored **+0.002** on two bases
  (0.440 → 0.442; 0.441 → 0.443).
- The same insert for engine 1 at rank 3 was neutral.

`745952` warns the other way: on `enveda-np-examples`, locking the engine's
confident picks lifted validation 0.63 → 0.89, but the LB went **0.413 →
0.409**. The np-examples split is not the test's class 2 (well-documented
training structures).

**For us:** implement `--protect-top1` and compare:
(a) full re-rank, (b) keep the best single channel's top-1 and insert the
fused top-1 at rank 2, (c) confidence-gated re-rank. Measure on the
no-public-spectra slice, not overall.

### 3.8 Test-time processing across test spectra is allowed

`752297` + `751373`: host-confirmed that **fully automated, unsupervised use of
relationships across test spectra at inference** is allowed — clustering test
spectra, or letting a confidently identified test molecule inform the ranking
of a spectrally similar one.

This is not in our class-2 plan. A cheap experiment: build a spectral-similarity
graph over the eval queries, and for a query with a high-confidence neighbour,
add/boost the neighbour's top candidate. This is a genuinely different source
of evidence and is legal. Keep it as an optional channel; the private test has
only ~400 molecules, so clusters may be small.

---

## 4. What does NOT work (where most of the value is)

These are the forum's most repeatable negative results. Several are directly
relevant to choices we might otherwise make.

### 4.1 PubChem expansion

- `743254`: expanding the pool with PubChem top-50 isomers dropped the LB
  **0.335 → 0.205** even though raw coverage should improve.
- `741659` (citing prvsiyan): PubChem isomers reduced class-2 MRR
  **~0.52 → 0.35**.
- `742088`: full PubChem as a candidate source moved the LB **+0.002**;
  restricted to tail slots 16–25, **+0.001** — both inside resubmission noise.
  Their scorer reached MRR 0.27 alone at ~6,000 candidates per query.
- `748524`: inserting PubChem candidates at fixed ranks gave +0.01–0.02
  locally, but LB **0.295 → 0.288**. "There appear to be almost none" of the
  test truths that exist only in PubChem.
- `741745`: blind PubChem isomer expansion hurt the difficult classes;
  COCONUT / ChEBI / LIPID MAPS were much better.

### 4.2 Popularity / membership priors

- `748068` / Aryan: a PubChem **popularity prior** lifted class-2/3 CV MRR
  **0.11 → 0.61 (+0.50)**, but the LB only **0.146 → 0.221 (+0.075)**.
  The cause: by popularity alone the answer is #1 for **67 % of
  `enveda-np-examples`** but **~0 % of `enveda-180` structures with no public
  spectra**. Class-2 molecules have no public spectra by definition, so any CV
  built from library molecules overrates priors like this.
- `743254`: membership flags in curated databases looked great on public
  holdouts and hurt on the LB; holdout molecules are over-represented in
  curated DBs.

### 4.3 "Better offline" fingerprint models

- `749251`: two new FP models, hard-negative top-1 0.694 / 0.640 on a random
  2 % structure split → LB **lower** (0.330 vs 0.335).
- Fine-tuning FPNet on timsTOF: offline 0.459 → 0.491, **LB 0.337 → 0.328**.
- `748068`: the public `fpnet_full1` was trained on all data, so its pipeline
  cannot be validated on held-out training data.

### 4.4 Seven attempted same-formula separators (all ~0 marginal)

From `742055` (alex chilton, 249 held-out natural products, class-2 condition):

| Feature | Marginal result |
|---|---|
| MetFrag-lite fragment channel | +0.00367; leave-one-out says the whole channel +0.0025 ("free to delete") |
| Connectivity scoring | +0.0000; 0 of 10 splits positive |
| Formula/sub-formula gate | +0.0000; fit half chose weight 0.0 in all 10 splits |
| Collision-energy depth | 80.3 % separation on isomers → **4 molecules of 249 move** |
| S-CNLF (base neutral loss) | truth reaches its own base loss 45.4 % vs decoys 18.6 %, but fails its own test 54.6 % of the time |
| ICEBERG as a learned feature | +0.0015 in the >100-candidate stratum |
| Ring-cut infeasibility | −0.00178; 3 molecules of 247 |

Pattern: a real separation on a selected cohort, then nothing once it competes
with the ranker's own probability. Their coverage column explains it: these
features discriminate on a third to a half of pairs, and the ranker is already
right on most of those.

### 4.5 DreaMS as a ranker

`742055`: DreaMS separates same-molecule from same-formula spectrum pairs at
AUC 0.88 in general, but on the actual blocked pairs (measured spectra on both
sides) the truth was closer in only **20 of 42** — below chance. 51.9 % on
different-scaffold pairs (n=27), 40.0 % on near-twins. A structure encoder
trained against a frozen DreaMS teacher is capped by that number.

### 4.6 Generation for class 3 (context, not our problem)

- `753400`: rule-based biotransformation of the 15 nearest analogs, 65 rules,
  up to 2 steps. Truth generated 46 % (unlimited analogs), 32 % (≤0.70
  Tanimoto), 16.5 % (≤0.55). Even when generated, it ranked poorly: class-3
  MRR 0.080 / 0.056 / 0.030 at penalty 20. Net expected gain **+0.004–0.010**
  overall — below LB noise.
- `755089`: Monte-Carlo graph rewiring + fingerprint target; +0.006–0.011
  local, **0 on the LB**. Generated isomers are not unstable (GFN2-xTB), so
  stability filtering is not the explanation.
- `741659`: MRR geometry — moving a correct candidate from rank 1 to 2 costs
  0.5; recovering a class-3 molecule at rank 5 gains only 0.2. One generated
  false positive that demotes a confident rank-1 DB hit can wipe out several
  successful de novo discoveries. Safe augmentation / confidence gating is the
  real problem.

### 4.7 Wide windows, 3-bond cuts, cleavage propensities, etc.

- Wide mass windows add decoys; ±8.5 ppm was best (`741745`); 8.5 → 10 ppm
  changed almost nothing because the window is full of exact-mass isomers
  (`743254`).
- 3-bond cleavages: **−0.031** (`742055`).
- Hand-written cleavage propensities: −0.002, and they diluted the parsimony
  effect.
- Penalising fragments reachable many ways: −0.015.
- S-CNLF and collision-energy depth: ~0 marginal (`742055`).
- `755961` (fusion audit, 250-molecule masked proxy): combining engines
  without giving the second engine scores for newly generated candidates
  demoted correct top-1s — 96 → 78 after PubChem incorporation → 22 at the
  earlier fusion output; 74 of 96 top-1s lost. When the second engine *did*
  score generated structures, MRR recovered 0.2596 → 0.4041 locally (+0.144),
  but this is an exposed proxy, not an LB result. The lesson is mechanical:
  **an expert that has not scored a candidate is not negative evidence.**

---

## 5. Validation protocol — the section that matters most for us

Our simulator is already unusually good. The forum's warnings are mostly
about the *remaining* gaps.

### 5.1 What we already do right (per `dataset-creation/README`)

- Class-2 level = all train rows of the eval molecule hidden; the structure
  enters only through `split.library_additions()`. This matches the forum's
  strict class-2 condition (`742055`, `743254`).
- Eval spectra are **timsTOF**, ten adducts, median 3 spectra/molecule, mass
  bins and ion share matched to the test description.
- Candidate pool deduped by the **scorer's key** (`pool.py`).
- `max_visible_tc` and `neighbours_removed` bands are exactly the diagnostics
  the forum recommends.
- `candidate_recall` and top-1/10/25 are already reported; pseudo-class-2
  training hides all rows of the pseudo-truth (R2), preserving the asymmetry
  that the truth has no spectra and decoys do.
- Within-shortlist z-scoring is the "contrast within set" idea.

### 5.2 The `enveda-np-examples` trap

`742055` / `741597`:

- All 250 `enveda-np-examples` structures also appear in `riken`, `gnps`,
  `mona`, `massbank`, `pluskal_ms2` and others: **55,599 spectra across 8
  other libraries**.
- Holding out only by `ingest_lib` leaves the structure in the library; a
  spectral search finds it, giving CV 0.33–0.37 while the LB is 0.15–0.16.
- Correct check: remove all spectra of the held-out structure from **every**
  library, split by InChIKey14, and assert the library search returns **zero
  exact InChIKey14 self-matches** (best cosine 0.000).

Our `difficulty2` already hides all rows, so this specific leak is handled.
But any separate `enveda-np-examples` benchmark we use must pass the same
zero-self-match assertion.

### 5.3 The S1 / S2 distinction

Aryan (`748524`, `749251`):

- **S1:** 250 `enveda-np-examples` — all have public spectra elsewhere.
- **S2:** 490 `enveda-180` structures with **no public spectra** — class-2-like.
- S1 and S2 disagree sharply on priors: popularity puts the answer at #1 for
  67 % of S1 but ~0 % of S2. A popularity prior added +0.50 MRR on S1 but only
  +0.075 on the LB. **Accept a change only if it helps S2.**

Our `neighbours_removed` / `max_visible_tc` bands approximate this, but we
should add an explicit `has_public_spectra` flag and report S1/S2-like blocks.

### 5.4 Pretraining overlap (MassSpecGym, ICEBERG, DreaMS, FPNet banks)

- `748524`: evaluate MassSpecGym-trained models only on structures **not in
  MassSpecGym**. Re-ranking with ICEBERG gave +0.02 on class 2 overall, but
  ~0 on that subset.
- `749217`: DreaMS's base was trained on GeMS-A10; NIST20 was used only for
  evaluation. GLACIER/ICEBERG official MassSpecGym checkpoints are allowed.
- `748068`: a model trained on all training data cannot be validated on
  held-out training data. This is why the public `fpnet_full1` pipeline is
  unfalsifiable.

Practical implementation for us: add a molecule-level
`in_public_spectra_corpus` flag (at minimum `ingest_lib` not in
`enveda-180` / `enveda-np-examples`; ideally the actual MassSpecGym structure
list). Report F2/ICEBERG/FPNet metrics on both sides. If we cannot get the
MassSpecGym list, be conservative and label the flag "approximate public
overlap".

### 5.5 Spectra count and draws

- `742055`: the public test copy averages **3.03** spectra/molecule; their CV
  sets averaged **4.34 / 4.99**. Resample CV molecules to the test's count
  distribution, rebuilding fingerprint scores and cross-spectrum aggregates
  from the same sample. A single draw is worth ±0.011; average **5+** draws.
- Our `POTENTIAL_IMPROVEMENTS.md` §1 already flags that our sets reach median 2
  and ~2.5 average instead of ~3.75. The forum's fix (draw `k` first, then
  sample a molecule with at least `k` candidate rows) is worth implementing.

### 5.6 Noise floors and the public LB

- `743254`: one test molecule is worth ~0.0025 MRR; retraining the same GBDT
  ranker with a different seed moved holdout MRR by **±0.007**; don't chase
  differences below ~0.005. Class-2 seed noise specifically was **~0.02**
  (`748524`).
- `748068`: public LB is ~130 molecules; one molecule moving from outside the
  top 25 to rank 1 changes the score by ~0.008.
- `753511`: with ~125 public molecules, one public comparison cannot separate
  +0.005 from zero. A real +0.005 is shown as a loss 31 % of the time. Best of
  7 zero-gain variants adds ~+0.019 of pure noise on public, none of which
  carries to private. A paired offline comparison on 600 held-out molecules
  has SE ~0.0065, about half the public board's.
- `742088`: HistGBR with `random_state=None` gave 0.292 vs 0.298 on identical
  submissions; pin seeds and average.

### 5.7 Honest recall and blind controls (Pathik, `752735`)

- Build a validation split that can actually be **zero**: leave the true
  structure in the pool only if an external database genuinely has it. Under a
  naive library-search baseline, class 2 scores **0.0000 by construction**.
- Check that a shortcut detector detects shortcuts: his first spectrum-blind
  control read 0.0000, but candidate coverage was 0, so it was vacuous. Fixed,
  the blind floor was 0.0741 on a 36-candidate pool and 0.0348 on a 3,908
  candidate pool, against 0.3448 sighted.
- Per-molecule failure ledger (seed-0, n=500): in no database **21.6 %**, in
  window but ranked outside top 25 **14.8 %**, found but ranked 2–25 **39.4 %**,
  ranked first **23.8 %**. MRR 0.3448 overall but **0.4420 conditional on the
  answer being in the window**. "The biggest recoverable bucket is not what
  you miss, it is what you retrieve and rank badly."
- Representation was not the bottleneck: only **11.8 %** of in-window targets
  had even one bit-identical Morgan decoy; median nearest-decoy Tanimoto
  0.698. Larger fingerprints were not the answer.
- A channel can look great standalone and vanish in fusion: a fingerprint
  ranker 0.1475 / 0.1847 standalone, fused +0.0017 (t=0.36).
- Local gains did not predict the LB three times out of three: +0.0093 local →
  +0.019 LB; +0.1129 local → +0.003 LB; +0.0111 local → **−0.011** LB.

### 5.8 Same-formula panel (the metric we should add)

`743254` + `742055`: construct the panel as the truth plus its best-scoring
same-formula isomers (all same-formula candidates, or truth + top-10 by the
current score). Report top-1 and MRR@25 **within the panel**. This is far more
sensitive than overall MRR and directly measures the thing class 2 is about.
Also report the panel size (median ~46–168 same-formula candidates in their
pools; our pool may differ).

### 5.9 Class-share calibration

- `748524` fit 28/20/52 by matching local per-class MRRs to several LB scores;
  their first weighted local estimate was 0.334 vs LB 0.322.
- `741597` derived 16/45/39 from a library-only LB probe (0.151) and class-1
  MRR ~0.93.
- The hosts will not disclose shares (`748882` C-3).
- `755039` speculates that class 3 may be concentrated in the private split
  (69 % of the score); this is explicitly a guess, but it argues against
  overfitting the public board.

Use these as scenarios, not facts.

---

## 6. Data / chemistry facts to encode in features and tests

### 6.1 Instrument and adducts

`744011`:

| | Train | Visible test |
|---|---|---|
| Instrument | 46 % timsTOF, 30 % Orbitrap, rest QTOF / ion trap / QqQ | **100 % timsTOF** |
| Adducts | 52 % [M+H]+, 18 % [M−H]−, **13 % dimers** ([2M+Na]+, [2M+H]+, [2M−H]−) | 79 % [M+H]+, 16 % [M−H]−, **no dimers** |
| Precursor m/z | median 350, min 2, max ~2.4 M | 245–460, median 329 |
| CE | mixed `60`, `60.0`, `[60.0]` | clean 20/40/60 and stepped 20,40,60 |

Recommendations from that thread: weight/filter by instrument; drop dimer
adducts from the library; drop impossible precursors (50–2,000 m/z); parse CE
into numbers; build CV with the test's ~80/20 positive/negative ratio and
[M+H]+ / [M−H]− dominance; split by InChIKey14, never by spectrum.

`742087` (adduct labels):

- Library-wide, **6.9 %** of NP-library rows (174,977 / 2,532,125) have a
  labelled-adduct neutral mass >10 ppm from the true structure.
- On a 40,000-row sample, an alternative adduct hypothesis lands within 10 ppm
  for 40.1 % — but **12,043 of the 16,037 rescues are `[M+CH2O2-H]-`
  (formate)**, a GNPS/MassBank curation artefact, not an acquisition effect.
- On Enveda-curated `enveda-np-examples`: **0 of 250** miss the truth under
  their labelled adduct. On a GNPS-heavy held-out set: 8 of 350 miss, 7
  rescuable.
- **Do not spend candidate-generation effort on adduct re-hypothesis for the
  test.** Do clean/down-weight formate mislabels if training on library data.

### 6.2 Peaks

`741423` / `741641`:

- Median 230 peaks in test, but 93 % below 1 % of base peak (77 % below 0.1 %).
  After a 1 % floor, median 18 peaks; 0.1 % floor, 52. The intensity floor
  changes the input more than any model choice.
- Precursor peak present in only **56 %** of test spectra.
- Neutral losses that matter: H2O 18.0106, NH3 17.0265, CO2 43.9898,
  HCOOH 46.0055, hexose 162.0528. A molecule and the same molecule +OH have
  different fragments but many identical losses — include a neutral-loss
  channel.
- 56 % precursor presence means don't build features that assume it.
- 3 % of peaks sit slightly above the precursor (the +2 Da isotope allowance);
  the repo cleaner already removes peaks above precursor + 2 Da.

### 6.3 Collision energy

- Use `collision_energy_ev` (positive), not `collision_energy_orig`
  (negative in negative mode).
- 20/40/60 eV and merged spectra are complementary, not redundant; only 24 %
  of molecules have the full positive-mode ladder. Pooling logic must not
  assume a fixed ladder.
- Collision-energy depth as an isomer separator was weak in `742055`
  (80.3 % separation → 4 molecules of 249 move). Low priority.

### 6.4 Metric mechanics

- First-match MRR@25; guesses canonicalised (stereo removed, tautomer-
  canonical InChIKey first block); the loop stops at the first match.
- Metric updated to `RemoveStereochemistry` on 2026-10-07 (`746878`).
- Empty entries are discarded before ranks; `A;;B` reads as two guesses, B at
  rank 2; a trailing `;` is not a guess (`742274`).
- Appending valid distinct guesses after an unchanged rank-1 cannot lower a
  row's score; inserting before the answer can (`748882` C-1).
- **Filler/invalid guesses can poison a row**: `742088` measured 0.000 twice;
  `742274` host says empty entries are discarded, but unparseable/filler
  SMILES are still a risk. Use only valid, deduped candidates.
- 8 % of molecules had a tautomer copy of the truth under a different PubChem
  key (`743254`) — dedupe by the metric key.
- Electron-mass correction: applied **only to the hidden test**, not to
  `train.parquet` or the Enveda-180 Zenodo release; to match, subtract
  **0.000548 Da** from positive-mode m/z (`748882` C-2).
- Re-download `train.parquet`: water-loss adducts were added on 2026-09-15
  (`741471`).

### 6.5 Database coverage

`748552`:

- The four approved sources (COCONUT, PubChemLite, ChEBI, LIPID MAPS) are
  essentially PubChem re-indexed. The union adds **2 molecules out of 2,000**
  beyond PubChem. PubChem alone gives ~90.45 % coverage; the union ~90.55 %
  (scorer-key ruler: both 89.48 %).
- Running with one source only: COCONUT 0.1065, PubChemLite 0.1040, LIPID MAPS
  0.0140, ChEBI 0.0000 — all below their own pool (0.1480).
- **~11 % of the answers are not in PubChem** — that fraction is class 3 by
  definition; no retrieval pool fixes it.
- LOTUS adds 0.0435 % (+940 candidates) and NP Atlas 0.0007 % (+65) on their
  train+visible-test ruler. Marginal, but the only measured positive direction
  outside PubChem.

`742055` reaches the same conclusion from the class-2 side: NPAtlas adds 1,113
structures (0.16 % growth), 45 inside the test's 279–374 Da band, and **zero
holdout truths**; LOTUS is 97.5 % already contained, **zero truths added**.

`755965` (S3 mirrors, all open licences): Enveda-180 is Enamine screening
compounds; a plant isolate almost never has a library match there, only
chemical neighbours. 2,077 Enveda-180 compounds also have a positive-mode
[M+H]+ spectrum in GNPS — a usable cross-instrument check.

---

## 7. Compliance / allowed models (for anything we train or reuse)

Host answers, condensed (`748270` is the community table; verify against the
linked threads):

| Source / model | Allowed? | Note |
|---|---|---|
| `train.parquet` and models trained on it | **Yes** | "effectively open source"; redistributed artifacts fine (`745841`, `745072`) |
| MassSpecGym and models trained on it | **Yes** | `742991` |
| GLACIER / ICEBERG (official MassSpecGym checkpoints) | **Yes** | `744338`, `749217` |
| DreaMS (ssl backbone) | **Yes** | NIST used only for evaluation, not gradients (`749217`) |
| CFM-ID 4 | **Yes** | `743774` |
| MIST checkpoints (MassSpecGym, MIT) | **Yes** | `751373` |
| FRIGID official checkpoints | **Yes** | with authors' training code/data (`745072` C-5) |
| DiffMS (MIT; decoder pretrained on HMDB) | **Yes** | HMDB is allowed (`745072` C-4/6) |
| ChEBI / LIPID MAPS / NPAtlas / HMDB | **Yes** | open, non-commercial provisions permitted |
| PubChem / PubChemLite / COCONUT / LOTUS | **Yes** | retrieval and subsets fine (`741857`) |
| Newer GNPS / MassBank / MoNA releases | **Yes** | CC BY-SA records acceptable (`745228`) |
| GNPS-NIST14-MATCHES | **Yes** | explicit exception (`745832`) |
| Enveda-180 Zenodo release | **Yes** | `745148` |
| ICEBERG PubChem atlas (91.4 M predicted spectra) | **Yes** | model outputs; include retrieval/processing code (`745134`) |
| **NIST and anything NIST-derived** | **No** | including NIST-trained weights and newly built NIST-derived datasets (`744470`, `745832`) |
| **SIRIUS / CSI:FingerID** | **No** | training data includes NIST (`751373`) |

Other rules relevant to us:

- **Third-party weights** (including other competitors') require the original
  **training code + permitted training data**. If unavailable, retrain an
  equivalent model yourself (`745072`, `749217`, `748882`).
- The compliance check can cover **all** submissions; prior non-reproducible
  submissions are not eligible to win but do not disqualify future compliant
  ones (`745841`, `748882`).
- **Test-time, fully automated, label-free processing across test spectra is
  allowed** (`752297`; `751373` follow-up).
- `train.parquet` is CC BY-NC as distributed, but the host treats it as open
  source for this competition; the non-commercial provision was intended to
  protect the test data.

Our `max_class2_exp` currently uses:
- ICEBERG 2.1 MassSpecGym `msg_all` — allowed;
- repo Fragmentor — our own code;
- train.parquet — allowed;
- COCONUT optional pool — allowed.

No compliance blocker in the current plan.

---

## 8. Concrete changes to our experiment

This section maps the forum evidence onto our files. Priorities are about
**decision validity**, not code effort.

### P0 — changes that can invalidate our conclusions

**P0.1 Class-share sensitivity.**
Files: `max_class2_exp/README.md` §0, `dataset-creation/make_overall.py`
(`--share`), reporting in `stage2.py`.
Change:
- Replace "Class 2 is assumed to be ~60 %" with a scenario table:
  `16/45/39`, `28/20/52`, and our `20/60/20`.
- Report any expected overall gain as `share_class2 × ΔMRR_class2`, with the
  share as a parameter. Do not present a single overall number as "the" result.
- Keep the 60 % `overall` split as a stress scenario, but do not let
  architecture choices depend on it.

**P0.2 Same-formula isomer panel.**
Files: `max_class2_exp/stage2.py` (`summary()` and the per-query result frame),
`max_class2_exp/f2_eval.py`, `max_class2_exp/pool.py`.
Change:
- Add a cached `formula` array to `Pool` (RDKit `CalcMolFormula`; ~273k
  structures is a few minutes; cache to `out/`).
- For each query, define:
  - `panel_all`: all in-window candidates with the truth's formula (including
    the truth);
  - `panel_top`: the truth plus the top-10 same-formula decoys by the current
    baseline score.
- Report `top1_panel`, `mrr@25_panel`, `median_panel_size` for every method
  (mass_error, entropy_best, frag, fp, stage2) and every band.
- Use this as the primary development metric for class-2 changes; keep overall
  MRR@25 as the secondary, test-like metric.

**P0.3 Failure ledger.**
File: `max_class2_exp/stage2.py` result frame.
Change:
- Add `bucket ∈ {outside_window, in_window_rank_gt25, rank_2_25, rank_1}` per
  query, plus `rr` and `candidate_recall`.
- Report bucket counts and mean RR per bucket, overall and by `tc_band`.
- Add `mrr_conditional_on_recall` (MRR over queries where the truth is in the
  window). `742055`/Pathik show this is where the real differences appear.

**P0.4 Public-spectra / MassSpecGym overlap stratification.**
Files: `max_class2_exp/f2_eval.py`, `stage2.py`, any FPNet evaluation;
`dataset-creation` split annotation.
Change:
- Add a molecule-level `has_public_spectra` flag: any train row for the
  molecule in a public library (`riken`, `gnps`, `mona`, `massbank`,
  `pluskal_ms2`, `msdial`, `drug_plus`, `masaryk`, etc.) — i.e. anything
  outside `enveda-180` / `enveda-np-examples`.
- If available, add `in_massspecgym` from the actual MassSpecGym structure
  list; otherwise label the flag "approximate public overlap".
- Report every F2 / ICEBERG / FPNet / DreaMS result split by this flag, and
  make the **no-public-spectra** slice the headline for pretrained channels.
- This is the forum's +0.02 → ~0 ICEBERG lesson (`748524`).

**P0.5 Blind control + coverage assertion.**
Files: `max_class2_exp/stage2.py`, `f2_eval.py`.
Change:
- Add `--blind-control`: replace each query's peaks with a spectrum from
  another molecule in the same mass bin (or shuffle peak lists across
  molecules), keep the candidate window and all features, score.
- Before trusting the control, assert `candidate_recall > 0` and report
  `median_candidates`; a 0.000 blind score with 0 coverage is vacuous.
- Report the sighted-vs-blind gap and the coverage for every channel.

**P0.6 Seed protocol and error bars.**
Files: `dataset-creation/make_difficulty2.py` (already `--seed`),
`stage2.py` (`--train-seed`), reporting.
Change:
- Build 5 datasets: seeds 2026–2030. Run each method on all 5.
- Train each learned component with ≥3 model seeds.
- Report paired-by-molecule bootstrap 95 % CIs; treat <0.02 class-2
  differences as unresolved (forum: seed noise ~0.02; `743254` ±0.007;
  `753511` SE ~0.0065 on 600 molecules).
- Do not use the public LB for model selection at this stage; the public board
  has ~125–130 molecules and 0.008/molecule granularity.

### P1 — modelling changes with the strongest forum support

**P1.1 Simple MetFrag-lite feature block.**
Files: new `max_class2_exp/metfrag.py` (or extend `insilico.py`), `features.py`
`FEATURES`.
Change:
- For each (candidate, adduct, query spectrum):
  - enumerate 1-bond and 2-bond cleavages; **down-weight 2-bond ×0.6**;
  - H shifts −2..+3; tolerance 0.005–0.01 Da (sweep);
  - **linear intensity** (also keep sqrt for comparison);
  - explained intensity share, matched peak count, mean ppm;
  - **rarity-weighted** explained intensity: weight each matched peak by
    1/(1 + number of candidates in the shortlist that explain it), or an IDF
    over the shortlist;
  - **gap to the best candidate** in the same molecule (explained intensity
    minus the shortlist max), and the candidate's rank on this feature.
- Do **not** add 3-bond cuts, cleavage-propensity weights, or
  reachability penalties; they were negative.
- Evaluate against the Fragmentor as a separate channel (`frag_simple_*`) and
  in fusion. The forum's +0.065 was on an isomer panel; expect a smaller
  overall effect and measure the panel metric.

**P1.2 Parameter sweeps.**
Files: `stage2.py` defaults; `features.py` tolerances.
Change:
- `--ppm`: 10 → **8.5** (sweep 5 / 8.5 / 10).
- `--tol` (peak matching): 0.02 → **0.01** (sweep 0.005 / 0.01 / 0.02).
- `--frag-ppm`: sweep 5 / 10 / 20.
- Intensity weighting in the fragmentation channel: linear vs sqrt.
- Reason: `743254` 99 % within 5 ppm; `741745` ±8.5 ppm; NeckBeard's table.
  The mass window width itself barely matters *when the pool is already
  formula-filtered* — but wider windows add decoys and change the shortlist
  size, so keep the sweep.

**P1.3 New ranker features.**
Files: `max_class2_exp/features.py`.
Add:
- `mass_agreement_ppm`: spread (median absolute deviation) of the per-spectrum
  implied neutral masses of the query. Small spread = reliable mass; large
  spread = adduct/mass problem. (`741423`: 1.3 ppm median agreement.)
- `mass_error_ppm_per_adduct`: signed error for each adduct family (or the
  min/max across spectra).
- `formula_share`: fraction of the shortlist sharing the modal formula;
  `n_same_formula`; `cand_formula_mass_error`.
- `cand_visible_rows` already exists; add `cand_is_addition`,
  `cand_is_external`, `cand_n_public_libs`, `cand_n_spectra`,
  `cand_n_instruments`, `cand_instrument_match` (timsTOF), `cand_adduct_match`.
  Use these as **context**, and ablate: a simulator-only shortcut is a risk.
- `n_query_spectra`, `n_query_adducts`, `n_query_polarities`, `has_precursor_peak`
  (the last is a data-quality/context feature).
- Keep the within-shortlist z-scoring; features that are constant per
  shortlist are fine as context but should not drive the model.

**P1.4 Channel P as FPNet.**
Files: `max_class2_exp/fpmodel.py` (new), `train_denovo.py`,
`src/enveda_casmi26/fp_model.py`.
Change:
- Target fingerprint: **~6–7k bits** (e.g. Morgan 2048 + MACCS 166 + RDKit
  2048 + pharmacophore, or the repo's existing 6,930-bit space), matching
  `741745`.
- 4–6 layer transformer encoder (or the repo's de novo encoder) with a
  multi-label head; **BCE** loss.
- **Hard negatives from the 10 ppm pool window**; **merge augmentation**
  (merge multiple spectra of the same molecule); **bit-prior initialisation**.
- Candidate score: `Σ_b [c_b log p_b + (1−c_b) log(1−p_b)]`, normalized by bit
  priors; average predicted probabilities across the query's spectra, then
  score.
- Validate on the **no-public-spectra** slice, not a random structure split.
- Do not stack a second FPNet expecting more than ~0.002 (`745952`).

**P1.5 Safe rank-1 protection.**
Files: `stage2.py` fusion/ranking.
Change:
- Add `--protect-top1`:
  - keep the best single channel's top-1 (`entropy_best` or the best baseline
    on the split);
  - insert the fused/stage2 top-1 at rank 2, dedupe by metric key;
  - compare (a) full re-rank, (b) protect-top1, (c) confidence-gated re-rank.
- Evaluate on the same-formula panel and the no-public-spectra slice.
- This is `749412`'s +0.002 trick, made explicit; it also guards against the
  `745952` "locking picks raised CV but hurt LB" failure.

**P1.6 F2/ICEBERG coverage and fallback.**
Files: `nnspec.py`, `f2_eval.py`.
Change:
- Report `scored_fraction` per query/candidate and the distribution of
  predicted peak counts. A candidate with `None` prediction is not negative
  evidence; mask it and report coverage.
- Add GLACIER as a fallback where ICEBERG has no prediction (both allowed,
  MassSpecGym).
- Optional: CFM-ID 4 as a second forward model; only keep it if the
  no-public-spectra slice improves beyond noise.
- Do not evaluate F2 on the full train-derived set and call it a class-2
  result; split by `has_public_spectra` / MassSpecGym.

**P1.7 Analogue channel: keep off, measure marginal only.**
Files: `features.py`, `stage2.py --enable-analogues`.
Change:
- Keep default off.
- Report the analogue-only baseline and its marginal contribution to the
  fused model on the same-formula panel.
- Expect saturation (`742055`: truth analog score ~90 % of best, 2 % of
  failures analog-poor). If it adds nothing, drop it and reduce runtime.

**P1.8 Test-time spectral networking (optional, allowed).**
Files: new `max_class2_exp/networking.py` or an optional stage in `stage2.py`.
Change:
- Build a spectral-similarity graph over eval queries (entropy/cosine on the
  cleaned peaks; same polarity/adduct family).
- For a query with a high-confidence neighbour, add/boost the neighbour's
  top-1 candidate (with a penalty) or use the neighbour's predicted
  fingerprint as an extra channel.
- Gate on confidence/margin; evaluate with and without.
- Legal per `752297` / `751373`. The private test has ~400 molecules; clusters
  may be small, so measure coverage.

### P2 — data, pool, hygiene

**P2.1 Pool composition.**
- Keep `train ∪ COCONUT` (plus optional COCONUT external pool). COCONUT
  covers ~99.6 % of in-domain plant natural products in `enveda-np-examples`
  (`741597`).
- **Do not add PubChem** as a general candidate source. If we test it:
  formula-filtered, only tail slots 16–25, and report separately; expect ~0
  (`742088`, `743254`, `741659`).
- Do not add NP-membership / COCONUT-membership / popularity as features; use
  only for stratification (`743254`, `748068`).

**P2.2 Library cleaning.**
- If the FP model or any channel trains on public-library rows, clean/down-weight
  `[M+CH2O2-H]-` formate mislabels (`742087`: 6.9 % of NP-library rows; 12,043
  of 16,037 rescues are formate).
- Drop dimer adducts from the library/pool (`744011`: 13 % of train, none in
  test).
- Drop impossible precursors (outside ~50–2,000 m/z) (`744011`).
- Use `collision_energy_ev`, normalize instrument spellings, and do not assume
  a fixed CE ladder (`741423`, `744011`).

**P2.3 Submission hygiene.**
- Dedupe the 25 guesses by the **metric key** (tautomer-canonical InChIKey14,
  stereo removed), not raw InChIKey (`743254`, `746878`).
- No filler/unparseable guesses — they can score the row 0.000 (`742088`).
- Validate: no nulls, no duplicate `molecule_id`, ≤25 guesses, valid SMILES,
  non-empty; trailing `;` is fine, `A;;B` counts B at rank 2 (`742274`).
- Do not hardcode the visible test size (400); the hidden rerun has a different
  number of molecules. A failing assert can silently score a previous
  `submission.csv` (`749412`).
- Pin the Docker image in `kernels/*/kernel-metadata.json`; an unpinned copy of
  a 0.441 notebook scored 0.427 (`749908`).
- Never calibrate anything on the visible `test.parquet`; it is a placeholder
  drawn from train (molecule IDs and structures differ in the hidden rerun)
  (`745715`, `741423`).

**P2.4 Optional external assets.**
- `753724`: an 18 M+ structure candidate DB (COCONUT 460k + PubChem
  bioactives, deduped on InChIKey14) with a streaming Polars baseline. Useful
  as a one-time convenience; remember the PubChem caveat.
- `741874`: 139,744 harmonized MassBank spectra + Morgan/MACCS fingerprints
  (20,204 structures) — a ready reference library.
- `755965`: S3 mirrors of Enveda-180, COCONUT 2.0 (Oct 2026), LOTUS,
  MassBank open records, GNPS community libraries.
- `745134`: ICEBERG PubChem atlas (91.4 M predicted spectra), allowed.
- `755965`: Enveda-180 is Enamine screening compounds; 2,077 have a
  positive-mode [M+H]+ spectrum in GNPS — a cross-instrument check.

---

## 9. What is already aligned — do not change

These are worth stating explicitly so the forum evidence does not cause
unnecessary churn:

- `difficulty2` hides **all** train rows of the eval molecule and adds the
  structure via `split.library_additions()`. This is the forum's strict
  class-2 condition (`742055`, `743254`).
- Eval spectra are **timsTOF**, ten adducts, median 3 spectra/molecule, mass
  bins and ion share matched to the test description (`dataset-creation`).
- The pool is deduped by the scorer's key and sorted by exact mass; the mass
  window is the only retrieval step — exactly "all difficulty is in ranking".
- `max_visible_tc` and `neighbours_removed` bands are the right analogue of
  the forum's analog-poor analysis.
- `candidate_recall`, top-1/10/25, and per-band reporting already exist.
- Pseudo-class-2 training hides all rows of the pseudo-truth (R2), preserving
  the asymmetry that the truth has no spectra and decoys do.
- Within-shortlist z-scoring is the "contrast within set" idea the forum
  independently converged on (`752356`).
- The analogue channel is off by default; keep it that way.

The forum's main message is **not** "change the simulator". It is "the
simulator is right; the metric and the features need to target same-formula
ranking, and pretrained-model results need a no-public-spectra slice."

---

## 10. Repo observations (not forum findings, but relevant to the change list)

- `max_class2_exp/README.md` §0 states "Class 2 is assumed to be ~60 % of the
  hidden test." This is the highest of the three estimates; see P0.1.
- The README's status table says `features.py` / `stage2.py` are built, but
  `features.py` still imports `spectra` from `max_class1_exp` (line ~35),
  while the README says the folder is self-contained. This is an internal
  consistency issue, not a forum issue; finish the `spectra_utils.py` move
  before the next clean run.
- The README's "Results" section is empty ("to be filled after the first clean
  run"), so the P0 metrics should land before the first full run, not after.
- `stage2.py` defaults: `--ppm 10`, `--tol 0.02`, `--frag-ppm 10`. The forum
  suggests 8.5 / 0.01 / 5–10 as the first sweep.
- `f2_eval.py` evaluates all train-derived molecules; add the
  `has_public_spectra` / MassSpecGym split before interpreting ICEBERG.
- `kernels/casmi26-stage2-sub/kernel-metadata.json` has no `docker_image`;
  add a pinned image (P2.3).
- `make_submission.py` fills up to 25 slots; add the validation checks in
  P2.3 before the next submission.

---

## 11. Thread index (the ones worth reading)

All URLs are
`https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/<id>`.

### Class-2 core

| ID | Title | Why read it |
|---|---|---|
| `742055` | Six variants, one plateau at 0.33–0.34 | The 17:1 ranking-vs-recall analysis; 95.1 % same-formula gap; negative results; CV traps. **Start here.** |
| `743254` | Lessons from ~30 submissions | 98 % same-formula errors; 8 % tautomer duplicates; mass windows; curated-DB coverage; noise floor. |
| `741745` | [0.339 Top 1 Solution] 4-Channel … | The public top architecture: FPNet, analog propagation, 31 features, HistGBM. |
| `748524` | Validation strategy | Class shares 28/20/52; seed noise; ICEBERG +0.02→0; PubChem insertion 0.295→0.288. |
| `748068` | Public notebooks' 0.43+ LB has never been validated by CV | 130-molecule LB; 0.008/molecule; popularity prior +0.50→+0.075; fpnet_full1 unfalsifiable. |
| `745952` | Looking to merge: validation harness + FPNet pipeline | Three-mask 0.933/0.630/0.347; FPNet recipe; locking picks 0.63→0.89 CV but 0.413→0.409 LB. |
| `741597` | Using 250 enveda-np-examples structures as CV doesn't seem very helpful | 16/45/39 class shares; 55,599 leaked spectra; library-only 0.151. |
| `749251` | Stuck in 0.440 and teammate hiring | Fingerprint models offline vs LB; fragmentation 0.502→0.627; S1/S2 splits. |
| `748029` | Adding valid guesses at ranks 2–25 lowers the public score? | Metric mechanics; first-match MRR; public LB noise. |
| `742088` | Dissecting the 0.32–0.34 plateau | Four probe submissions; 0.40 ceiling; filler 0.000; seed ±0.006. |
| `741659` | Possible incentive gap in MRR@25 | Generation can hurt before it helps; safe augmentation. |
| `745468` | Problem is Isomer identification + LLMs bad | Formula correct >95 %; isomers are the failure; fragment-formula idea. |

### Data / EDA / chemistry

| ID | Title | Why read it |
|---|---|---|
| `741404` | A complete statistical tour of the data (EDA) | Column-by-column; class 1 nearly solved; classes 2/3 are the score. |
| `741423` | 10 things the CASMI 2026 data told me | Peak noise floor, precursor missing, multi-adduct 1.3 ppm, CE, polarity. |
| `741641` | 230 peaks, 18 that matter | Adduct table, neutral losses, CE, stereo-free metric. |
| `744011` | Test is 100 % timsTOF, train is 46 % | Instrument/adduct/dimer/CE shift checklist. |
| `742087` | Adduct labels: Enveda's are clean, the library's are not | 6.9 % library rows; formate mislabels; 0/250 enveda-np. |
| `754140` | A picture book of the training data | Formula-isomer examples; 2D/3D browsing. |
| `755965` | Five open mass-spec libraries as public S3 mirrors | Enveda-180/COCONUT/LOTUS/MassBank/GNPS mirrors; cross-instrument check. |
| `741471` | Please re-download training data | Water-loss adducts added 2026-09-15. |
| `746954` | Rerun test dataset updated | Electron-mass fix, hidden test only. |
| `746878` | Metric Update (RemoveStereochemistry) | Metric change 2026-10-07. |

### Models / training / forward models

| ID | Title | Why read it |
|---|---|---|
| `741666` | My Learning Roadmap | Generic but a useful MS/ML resource list; MIST, SIRIUS, Spec2Vec. |
| `742171` | What architecture is working best here? | Only a weak answer (random forest), but a marker of how few teams shared architectures. |
| `741984` | Why no one is using encoder-decoder models? | Spectrum→SMILES validity issues; ranker vs generator. |
| `744343` | Trick: train a smiles to MS/MS predictor too! | Forward model idea; GrAFF-MS, CFM-ID, MS2Mol, dMSdMol. |
| `744893` | DreaMS: structure from spectrum retrieval demo | DreaMS embeddings; coverage probes. |
| `745134` | ICEBERG PubChem atlas | 91.4 M predicted spectra; allowed as model output. |
| `751373` | Which third-party pretrained spectrum models are OK? | MS2DeepScore/FIORA/MIST allowed; SIRIUS not; PubChemLite fine. |
| `749217` | Compliance: approved-dataset list, NIST-derived corpora | DreaMS/GLACIER/ICEBERG official checkpoints allowed; third-party derivatives need training code. |
| `755089` | Class 3 graph generation: MC isomers | MC rewiring +0.006–0.011 local, 0 LB; stability is not the issue. |
| `753400` | Negative result: rule-based biotransformation | 46/32/16.5 % generation; class-3 MRR 0.03–0.08; net +0.004–0.010. |
| `750241` | New molecules' prediction | 130 DB-absent molecules; generator reached 100/130 with suitable parents; 6 parents default. |
| `755961` | A 250-molecule stage audit | Fusion can demote correct top-1s; second engine must score generated candidates. Exposed proxy, caveat. |
| `755962` | Why 0.94 local MRR did not transfer | Validation-domain audit; 60 structures, 5 spectra median, timsTOF subset 0.9595. Username is offensive; content is a useful caution. |
| `755964` | GNPS2 N-ACYL-DIPEPTIDES: one spectrum ID → multiple structures | Benchmark construction caution; 1,731 IDs with >1 structure. |

### Public pipeline / leaderboard mechanics

| ID | Title | Why read it |
|---|---|---|
| `752356` | Three changes on top of the 0.464 base | Simulator coverage +0.008 steps; gen20 −0.010; genfix; gold ~0.471. |
| `749412` | A safe +0.002: engine-2 top-1 at rank 2 | The cheapest validated trick; hardcoded-400 warning. |
| `749908` | Your notebook of a 0.44+ notebook scores lower? | Docker pinning; 0.446+ differences; lower PubChem threshold. |
| `748270` | Table: which datasets/data/models can be used | Host-answer summary; check the linked threads. |
| `745072` | Prize eligibility: public Kaggle datasets with non-commercial licence | train.parquet open; third-party weights need training code. |
| `745841` | Are non-commercial public datasets/weights allowed? | GLACIER/ICEBERG/DreaMS/ChEBI/LIPID MAPS fine; compliance can remove. |
| `741857` | External structure databases for Class 2 retrieval | Host: PubChem structures/subset fine for prize-eligible solutions. |
| `742274` | Scoring edge cases and novelty-class definitions | Class 3 absent from both; empty entries; metric code. |
| `748552` | Are the four approved sources meant to extend coverage beyond PubChem? | Union adds 2/2,000; ~11 % not in PubChem; LOTUS/NP Atlas marginal. |
| `752735` | What building an honest retrieval baseline taught me | Zero-able split; blind control; failure ledger; channel fusion; local vs LB. |
| `753511` | How much of a +0.005 public gain is real? | Noise model for a ~125-molecule board; best-of-7 +0.019. |
| `753525` | What is 0.005 MRR worth? Public LB noise with ~125 molecules | Companion noise discussion. |
| `745715` | Visible test.parquet molecule_ids appear to leak class-1 answers | Placeholder test; do not calibrate on it. |
| `755039` | A meta-level reading of the private test set | Speculation that class 3 may be concentrated in the private split; treat as a guess. |
| `752297` | Rules question: automated processing across test spectra | Host: unsupervised test-time networking allowed. |

---

## Appendix A — how to re-pull the forum

The official Kaggle v1 API has no discussion endpoints. The website uses an
internal JSON API (undocumented; may change). As of 2026-11-05 it worked
without authentication:

```python
import json, urllib.request

BASE = "https://www.kaggle.com/api/i/discussions.DiscussionsService"

def post(method, body):
    req = urllib.request.Request(
        f"{BASE}/{method}", json.dumps(body).encode(),
        {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))

FORUM_ID = 11293906          # enveda-CASMI26-molecule-id-mass-spectra

# 1) all topics, 20 per page
page, topics = 1, []
while True:
    d = post("GetTopicListByForumId", {"forumId": FORUM_ID, "page": page})
    ts = d.get("topics", [])
    if not ts:
        break
    topics += ts
    page += 1

# 2) full thread: OP + nested comments
for t in topics:
    d = post("GetForumMessagesInTopic",
             {"topicId": t["id"], "includeFirstForumMessage": True,
              "filter": "ALL"})
    # d["comments"] is a nested tree: replies are in c["replies"]
```

Notes:
- `forumId` comes from
  `POST /api/i/competitions.CompetitionService/GetCompetition`
  with `{"competitionName": "enveda-CASMI26-molecule-id-mass-spectra"}`
  → `"forumId": 11293906`.
- `includeFirstForumMessage: true` is the key to getting the OP; without it,
  `GetForumMessagesInTopic` returns only comments.
- Rate-limit yourself (~1 request/s); anonymous access returned HTTP 429 when
  hit too fast.
- The Kaggle sidecar in this omac session only proxies `/api/v1/...`, so these
  calls were made directly against `www.kaggle.com`.

The scraped corpus used for this report was 149 markdown files
(one per thread) plus a 150-row topic index; it is not committed to the repo.
Re-running the snippet above regenerates it.

---

## Appendix B — open questions the forum does not answer

1. **True class shares.** Hosts will not disclose; 16/45/39 vs 28/20/52 are
   both estimates. Our 60 % assumption is the highest.
2. **How much of class 2 is "PubChem-only" vs "COCONUT".** `748552` says ~11 %
   of answers are outside PubChem; `742088` says almost no test truths are
   PubChem-only. These are not fully reconciled.
3. **Whether the private split is class-3-heavy.** `755039` is speculation.
4. **Whether same-formula separation can be improved at all beyond the current
   ranker.** Seven attempts in `742055` were ~0 marginal; the strongest result
   (fragmentation + analog features) is still modest on the LB.
5. **How to make generation safe under MRR@25.** `741659`, `752356`, `753400`,
   `755089` all point to the same problem: generation can hurt before it helps.
6. **A public, leak-free class-2 benchmark** at the scale and chemistry of the
   hidden test. Everyone is using variants of `enveda-np-examples` /
   `enveda-180`; `755962` shows a well-isolated split can still be a poor proxy.
7. **Whether test-time spectral networking actually helps** at ~400 molecules.
   It is allowed; nobody reported a measured gain.

---

## Appendix C — one-paragraph summary for a teammate

Class 2 is "structure known, spectrum never measured". The forum's best
evidence says the bottleneck is **ranking same-formula isomers**, not
retrieval: 17:1 ranking-vs-recall headroom, and removing same-formula blockers
closes 95 % of the gap. Analog propagation is a strong but saturated baseline;
the useful additions are simple in-silico fragmentation features, a
spectrum→fingerprint model (FPNet), and a GBDT/LambdaRank ranker with
list-relative features. Blind PubChem expansion and popularity priors are
consistently harmful. Validation must hold out structures by InChIKey14 across
every library, and pretrained models must be scored on structures with no
public spectra / not in MassSpecGym, or they look better than they are. For our
experiment: add the same-formula panel and failure ledger, stratify by public
spectra, add a blind control and 5-seed error bars, add the simple MetFrag-lite
feature block, switch to 8.5 ppm / 0.01 Da, and keep COCONUT as the only
external pool. Do not add PubChem or popularity features.
