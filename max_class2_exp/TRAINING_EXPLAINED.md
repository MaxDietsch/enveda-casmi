# What `max_class2_exp/` trains, in plain words

This page explains **what is trained** in the difficulty-2 experiment:
which models exist, what goes into each one, which molecules each one
sees, and which molecules it is never allowed to see.

It does not replace the other two documents:

- [`README.md`](README.md) is the experiment plan (why, risks, evaluation protocol).
- [`PIPELINE.md`](PIPELINE.md) is the code walkthrough (function by function).

If you only read one section, read [§3 The big picture](#3-the-big-picture).

---

## 1. Vocabulary

Each term below has one meaning on this page.

### Data

| term | meaning |
|---|---|
| **spectrum** / **row** | One MS/MS measurement: a list of peaks (m/z, intensity), plus a precursor m/z, an adduct, an ionisation mode, a collision energy and an instrument. One row of `data/train_clean.parquet` is one spectrum. |
| **peak** | One (m/z, intensity) pair in a spectrum. |
| **precursor m/z** | The m/z of the ion that was fragmented. Together with the adduct it gives the neutral mass of the molecule. |
| **adduct** | The ion form, e.g. `[M+H]+` (molecule plus a proton) or `[M-H]-`. The competition uses ten adducts. |
| **CE** | Collision energy in eV, the energy used to fragment the ion. |
| **molecule** / **structure** | One chemical compound. It is identified by its **key**. A structure is the molecule's graph (its SMILES string), with or without any spectra. |
| **key** | Tautomer-canonical InChIKey, first 14 characters. Two rows with the same key are spectra of the same molecule. |
| **fingerprint (FP)** | A 2048-bit Morgan fingerprint (radius 2) of a structure. Each bit roughly says "this small substructure is present". Computed from the SMILES with RDKit, never from a spectrum. |
| **Tanimoto (Tc)** | Similarity of two fingerprints, from 0 (nothing shared) to 1 (identical). |
| **`enveda-180`** | The organisers' own timsTOF library. 1.15 M of the 2.54 M training spectra. Nearly all timsTOF data, and nearly all eval molecules, come from here. |
| **public libraries** | Every other `ingest_lib` (GNPS, MassBank, MoNA, RIKEN, …). Mostly Orbitrap and Q-TOF spectra. |
| **eligible row** | A row that may be used as an eval spectrum: timsTOF, one of the ten test adducts, CE known, base peak ≥ 1000, precursor within 10 ppm of the structure, ≥ 1 peak. Same rule as the hidden test. |

### The difficulty-2 setting

| term | meaning |
|---|---|
| **class 2** | A test molecule whose structure is known (in PubChem / COCONUT) but which was **never measured**: no spectrum of it exists anywhere in the data. |
| **difficulty 2** | Our local imitation of class 2: the file `dataset-creation/out/difficulty2.parquet`. |
| **split** | One such file, with its masks. It says which molecules are evaluated and which rows are hidden. |
| **eval molecule** / **truth** | A molecule in the split. Its correct structure is what we try to rank first. |
| **query** | The 1–16 spectra of one eval molecule that we are given. The input at prediction time. |
| **hidden row** | A training row that we act as if it did not exist. **Every** row of every eval molecule is hidden. |
| **visible row** | A training row that is not hidden. Only visible rows may be used for anything. |
| **visible molecule** | A molecule with at least one visible row. |
| **library addition** | An eval molecule put back into the candidate pool **as a structure only** (key + SMILES, no spectra). Without this, the right answer could not be found at all. |

### Ranking

| term | meaning |
|---|---|
| **candidate pool** | Every structure that could be an answer: all visible molecules, plus the library additions, optionally plus an external list (COCONUT), sorted by mass. |
| **candidate** | One structure in the pool. |
| **mass window** / **shortlist** | The candidates whose exact mass lies within ±8.5 ppm of the query's neutral mass. Usually 50–300 structures. The truth is one of them (if it is in the window at all). |
| **decoy** | Any candidate in the shortlist that is not the truth. |
| **isomer** / **same-formula decoy** | A decoy with exactly the same molecular formula as the truth. Mass cannot tell them apart. A median of ~26 per shortlist. This is the hard part of class 2. |
| **MRR@25** | Mean reciprocal rank: 1/rank of the truth if it ranks in the top 25, else 0, averaged over molecules. The competition metric. |

### Model parts

| term | meaning |
|---|---|
| **channel** | One independent way to score how well a candidate structure fits the query spectra. Each channel produces a few numbers (**features**) per (query, candidate) pair. |
| **fusion model** / **re-ranker** | The small classifier (logistic regression or gradient-boosted trees) that takes all channel features of a candidate and outputs one score: P(this candidate is the truth). |
| **pseudo-truth** | A visible molecule that we *pretend* is a class-2 molecule, so we can make training examples for the fusion model. See [§5.3](#53-the-fusion-model-re-ranker). |
| **pseudo-query** | The spectra of a pseudo-truth used as its query. |
| **zero-shot** | Used with frozen weights from someone else. No training on our data. |
| **leak** | Training data that contains information about an eval molecule. A leak makes local scores look better than they will be on the hidden test. |

---

## 2. The task in one paragraph

We get 1–16 timsTOF spectra of an unknown molecule. Its structure is in
the candidate pool, but **no spectrum of it exists anywhere**. Mass
narrows the pool to a shortlist of 50–300 structures, about 26 of which
have the exact same formula. Most decoys have measured spectra; the truth
never has. So we cannot look for "the candidate whose reference spectrum
looks like the query". We must answer a different question:

> **How plausibly would this candidate *structure* produce the observed peaks?**

Every trained model on this page exists to answer that question.

---

## 3. The big picture

```
                        query: 1–16 timsTOF spectra of an unknown molecule
                                          │
                     median neutral mass  │  (from precursor m/z + adduct)
                                          ▼
         candidate pool  ──── ±8.5 ppm ────►  shortlist of 50–300 structures
                                          │
           for every (query, candidate) pair, each channel computes features
                                          │
   ┌──────────────┬──────────────┬────────┴───────┬──────────────┬──────────────┐
   │ channel P    │ channel F    │ channel F′     │ channel F2   │ channel M    │
   │ spectrum →   │ Fragmentor:  │ MetFrag-lite:  │ ICEBERG:     │ mass error,  │
   │ predicted FP │ theoretical  │ 1- and 2-bond  │ predicted    │ same-formula │
   │ vs candidate │ fragments vs │ cuts vs        │ spectrum vs  │ competition  │
   │ FP           │ peaks        │ peaks          │ query        │              │
   │ TRAINED      │ no training  │ no training    │ zero-shot    │ no training  │
   └──────┬───────┴──────┬───────┴───────┬────────┴──────┬───────┴──────┬───────┘
          └──────────────┴───────────────┼───────────────┴──────────────┘
                                         ▼
                      fusion model (logistic / HGB) — TRAINED
                                         │
                                         ▼
                            ranked list of candidates (top 25)
```

(Channel A, "analogues", also exists but is **off by default**. See [§6](#6-the-channels-that-are-not-trained).)

### What is actually trained

There are **four** trained models. Two are always trained; two are optional.

| # | model | trained by | when | where it runs | used in |
|---|---|---|---|---|---|
| 1 | **Channel-P v1**: linear spectrum → fingerprint | `fpchannel.py` (called by `stage2.py`) | inside every `stage2.py` run | CPU, a few minutes | channel P (default) |
| 2 | **Fusion model**: logistic + HGB re-ranker | `features.fit_models` (called by `stage2.py`) | inside every `stage2.py` run | CPU, seconds | final ranking |
| 3 | **Channel-P v2 encoder**: spectrum → SMILES transformer | `train_denovo.py` | once, on a GPU box (`train_all.py`) | GPU, hours | channel P (replaces #1 if present) |
| 4 | **Channel-P v2 FP head**: encoder embedding → fingerprint | `train_fp_head.py` | once, after #3 (`train_all.py`) | CPU or GPU, minutes | channel P (replaces #1 if present) |

Channel F2 (ICEBERG) is a neural network too, but we do **not** train it.
We use public weights as they are.

---

## 4. The molecules: who is where

There are three groups of molecules. Keeping them apart is the whole leakage story.

### 4.1 Eval molecules (the difficulty-2 split)

File: `dataset-creation/out/difficulty2.parquet`, made by
`dataset-creation/make_difficulty2.py`.

| property | value |
|---|---|
| number of molecules | 1,000 |
| eval spectra | 2,751 in total; 1–10 per molecule, median 3 (drawn from a geometric distribution, p = 0.25) |
| source of the eval spectra | 98 % `enveda-180`, 2 % `enveda-np-examples` (all timsTOF, all eligible rows) |
| ion mode | 75 % positive, 25 % negative |
| adducts of the eval spectra | mostly `[M+H]+` (67 %) and `[M-H]-` (26 %); the rest `[M+CH2O2-H]-`, `[M+Na]+`, `[M+NH4]+`, … |
| exact mass | 212–954 Da, median 335 Da |
| how they were picked | weighted sampling toward the test's mass and ion-mode mix; no two eval molecules with Tc ≥ 0.6 to each other |
| hidden rows | **all** rows of every eval molecule (12,031 rows), from every library and instrument |
| neighbourhood removal | for 100 of the 1,000, every molecule with Tc ≥ 0.5 to them is hidden as well ("nothing similar left in train") |
| nearest visible relative | median Tc 0.58 (`max_visible_tc`; range 0.39–1.0) |
| in the candidate pool? | **yes**, as library additions: structure only, no spectra |

### 4.2 Visible molecules (what the models may learn from)

Everything in `data/train_clean.parquet` (2.54 M spectra, ~276 k keys)
**minus** the hidden rows above. That leaves roughly 270 k visible molecules.
They are:

- the material for every trained model (§5),
- most of the candidate pool, so most decoys are visible molecules, and
- the source of the pseudo-truths (§4.3).

### 4.3 Pseudo-truths (training examples for the fusion model)

Inside each `stage2.py` run, `sample_training_examples` draws **2,000**
visible molecules (`--n-train`, seed `--train-seed 2027`) that:

- have at least one eligible row (timsTOF, test adduct, …) and a mass of 157–1159 Da,
- are **not** eval molecules.

Each pseudo-truth then goes through what the split did to the eval
molecules:

1. 1–16 of its eligible rows become its pseudo-query (same geometric draw).
2. **All** its rows are hidden for this run.
3. It is added back to the training pool as a structure only.

This way, the fusion model learns in the same situation it will be tested
in: the right answer has no spectra, and the decoys do.

Two differences from the eval sampling: pseudo-truths are drawn
**uniformly** (no mass/ion-mode weighting, no diversity rule), and there
is **no neighbourhood removal** for them.

### 4.4 Which model may see which molecules

| model | eval molecules (§4.1) | pseudo-truths of this run (§4.3) | other visible molecules |
|---|---|---|---|
| 1 channel-P v1 | never (asserted) | never (asserted) | yes, up to 200 k random spectra |
| 2 fusion model | never (asserted) | yes: these **are** its training examples, but only as structures + their pseudo-query | yes, as decoys in the shortlists |
| 3 + 4 channel-P v2 | never: every molecule of **every** split in `dataset-creation/out/` is excluded (2,890 keys today), checked by stage2 at load time | possibly (trained once, before any run draws its pseudo-truths) | yes, except `enveda-180` (see §5.2) |
| F2 ICEBERG (not trained) | never: public weights, no Enveda data | never: no Enveda data | never: no Enveda data |

---

## 5. The trained models, one by one

### 5.1 Model 1: channel-P v1, linear spectrum → fingerprint

**Idea (from CSI:FingerID).** We cannot compare the query to the truth's
spectrum, because it does not exist. But the truth's *structure* exists,
so its fingerprint exists. If we can predict the fingerprint from the
spectrum, we can compare the predicted fingerprint with every candidate's
real fingerprint.

**Input** (one spectrum → one vector of 511 numbers, `fpchannel._tabular`):

| part | size | what it holds |
|---|---|---|
| m/z bins | 500 | m/z from 0 to 2000 split into 500 bins of 4 Da each; each bin holds the summed intensity of its peaks. Intensities sum to 1 per spectrum. |
| adduct one-hot | 10 | which of the ten competition adducts (`[2M+H]+` counts as `[M+H]+`, and so on) |
| precursor | 1 | precursor m/z ÷ 1000 |

Spectra are cleaned first, exactly like a query: drop the precursor
region, merge peaks closer than 0.01 Da, drop peaks below 1 % of the base
peak, normalise to sum 1.

**Output:** 2048 probabilities, one per fingerprint bit.

**Target:** the 2048-bit Morgan fingerprint of the spectrum's own molecule.

**Model and training:** one `torch.nn.Linear(511 → 2048)` with a sigmoid
per bit; binary cross-entropy loss; AdamW, learning rate 0.02, batch
8,192, 2 epochs.

**Training data:** visible rows of this run (split mask minus the
pseudo-truths' rows), a random sample of at most 200,000 spectra, all
libraries and instruments mixed.

**Not saved to disk on purpose.** It is retrained in every run, so it can
never be a stale model from a run with a different split.

**How it scores a candidate** (`features.channel_block`):

1. Predict a probability vector for every query spectrum and average them → `p` (2048 numbers).
2. For each candidate with real bits `c`:
   - `fp_ll` = Σ over bits of `c·log(p/prior) + (1−c)·log((1−p)/(1−prior))`.
     `prior` is how often the bit is on across the training molecules. A
     bit that is nearly always on gives almost no points, so "predict
     everything as 1" does not win.
   - `fp_tanimoto` = Tanimoto between `p > 0.5` and `c`.
   - `fp_available` = 1 if a prediction was made.

### 5.2 Models 3 + 4: channel-P v2, transformer encoder + fingerprint head

A stronger replacement for model 1, trained in two steps.
`train_all.py` runs both steps and checks the result.

**Step 1: the encoder (`train_denovo.py`, model 3).** A transformer
(`DeNovoModel` in `src/enveda_casmi26/fp_model.py`) learns to *write
the SMILES string* of a molecule from its spectrum. We do not use the
generated SMILES. We train this task only so that the encoder learns a
good internal summary of a spectrum.

| | |
|---|---|
| input | the 127 most intense peaks of a spectrum (intensity scaled to max 1), plus the precursor m/z as an extra "peak" with intensity 2.0. No adduct, no CE. Each m/z is turned into sine/cosine features, then into a 512-number vector per peak. |
| architecture | 4-layer transformer encoder over the peaks; 4-layer transformer decoder that writes SMILES tokens. 512 dimensions, 8 heads, feed-forward 2048. |
| target | the molecule's SMILES, split into tokens by a byte-pair tokenizer (512 tokens) trained on the training SMILES |
| loss | next-token cross-entropy |
| training | 500,000 sampled spectra, 2 epochs, batch 64 × 2 gradient-accumulation steps, AdamW lr 1e-4, bf16 on GPU |
| validation | 500 held-out structures (one spectrum each); the checkpoint with the best "exact structure among 8 generated SMILES" rate is kept |
| output file | `out/denovo_v1.pt` + `out/denovo_tokenizer.json` |

**Step 2: the fingerprint head (`train_fp_head.py`, model 4).** The
encoder is frozen. For each spectrum, its per-peak vectors are averaged
into one 512-number vector (the **embedding**). A small network maps
embedding → fingerprint.

| | |
|---|---|
| input | the 512-number embedding of one spectrum (frozen encoder) |
| architecture | Linear(512 → 1024) → ReLU → Dropout 0.2 → Linear(1024 → 2048) |
| target | the 2048-bit Morgan fingerprint of the spectrum's molecule (same bits as the candidate pool) |
| loss | binary cross-entropy per bit |
| training | all spectra that the encoder could have trained on (no 500 k cap), 4 epochs, batch 4,096, AdamW lr 1e-3; best validation loss kept |
| also saved | the bit priors used by `fp_ll` |
| output file | `out/denovo_fp_head.pt` |

**Training data for both steps:** `data/train_clean.parquet` with two
groups of rows removed:

1. **all `enveda-180` rows** (as in the organisers' tutorial), so v2 learns from
   the ~1.39 M public-library spectra of ~95 k molecules;
2. **every molecule of every split in `dataset-creation/out/`** (difficulty 1,
   difficulty 2, the adduct variant; 2,890 keys today). These keys are stored in the
   checkpoint as `meta.excluded_keys`.

**Scoring:** identical to v1 (`fp_ll`, `fp_tanimoto`, `fp_available`),
only the predicted probabilities come from encoder + head.

**When is v2 used instead of v1?** Automatically, when both
`out/denovo_v1.pt` and `out/denovo_fp_head.pt` exist **and** their
`meta.excluded_keys` contain every eval molecule of the split being run.
If not, `stage2.py` stops with an error. It does not silently fall back:
retrain the artifacts, or move them away to use v1.

> **Worth knowing.** Because `enveda-180` is excluded, v2 never sees a
> timsTOF spectrum during training, but every query it scores is a
> timsTOF spectrum from `enveda-180`. v1 has no such gap (about 45 % of
> its training spectra are `enveda-180`). Keep this in mind when you
> compare v1 and v2.

**Final Kaggle model only:** `--include-eval-molecules` turns the
exclusion off. `stage2.py` will then refuse those artifacts locally, as
intended.

### 5.3 The fusion model (re-ranker)

This is the model that produces the final ranking.

**What one training example is:** one (pseudo-query, candidate) pair.
Label 1 if the candidate is the pseudo-truth, else 0.

**How the training set is built** (all inside `stage2.py`, all in memory):

1. Draw 2,000 pseudo-truths (§4.3) and hide all their rows.
2. Build a training candidate pool: visible molecules (now without the
   pseudo-truths' rows) + the pseudo-truths as structure-only additions.
3. For each pseudo-truth, take the ±8.5 ppm shortlist around its pseudo-query's mass.
4. Drop (and count) shortlists whose truth fell outside the window or
   whose spectra were empty after cleaning. Every remaining shortlist has
   exactly one positive.
5. Compute **the same features with the same code** as at prediction time.

A typical run: ~2,000 shortlists, ~190,000 rows, ~1 % positive.

**Inputs (features), per candidate:**

| channel | features | trained? |
|---|---|---|
| P (§5.1 / §5.2) | `fp_ll`, `fp_tanimoto`, `fp_available` | yes (model 1, or 3 + 4) |
| F (Fragmentor) | `frag_matched_peaks`, `frag_explained_intensity`, `frag_mean_ppm`, `n_theo_frags`, `frag_available` | no |
| F′ (MetFrag-lite) | `mtf_matched_peaks`, `mtf_explained_linear`, `mtf_rarity_explained`, `mtf_gap_best`, `mtf_mean_ppm`, `mtf_available` | no |
| F2 (ICEBERG) | `f2_cosine_best`, `f2_cosine_mean`, `f2_available` | no (zero-shot) |
| M (mass) | `mass_error_ppm`, `mass_error_ppm_max`, `cand_n_same_formula`, `cand_is_external` | no |
| A (analogues, off) | `entropy_best`, `entropy_composite`, `cosine_best`, `nl_entropy_best`, `analogue_best_tc`, `n_analogue_molecules`, `n_analogue_spectra` | no; all 0 when off |

Every channel feature is averaged over the query's spectra.

**Removed from the model on purpose** (still computed and saved):
`cand_is_addition`, `cand_visible_rows`, `cand_tims_rows`,
`cand_public_rows`. Locally, the truth is *always* the one candidate
without spectra, so these four flags alone find it (MRR ≈ 1.0). The model
would learn how the simulator was built, not chemistry. On Kaggle, most
decoys also have no spectra. `--keep-source-features` turns them back on
to measure this.

**Scaling:** each continuous feature is z-scored **within its shortlist**,
so the model sees "how does this candidate compare to its competitors",
not absolute values. The `*_available` and `cand_is_external` flags stay
as 0/1.

**Models fitted** (`features.fit_models`) on 90 % of the shortlists:

- logistic regression (`class_weight="balanced"`, max_iter 5000), and
- gradient-boosted trees (`HistGradientBoostingClassifier`, max_depth 3,
  up to 300 trees, early stopping, balanced classes), fitted
  `--model-seeds` times with different seeds and averaged.

The other 10 % of shortlists are the **internal validation set**. The
model with the higher MRR there becomes the primary model and is saved
to `out/<split>_stage2_model.joblib`.

**Prediction:** score every candidate, sort by score (ties: smaller ppm
error first), report the top 25.

---

## 6. The channels that are not trained

They still matter, because they are inputs to the fusion model. For each
one: what goes in, and what comes out.

| channel | input | what it does | output features |
|---|---|---|---|
| **F, Fragmentor** (`insilico.py`) | candidate SMILES + the query spectrum's adduct | rule-based fragmentation (side-chain cuts, ring fissions, neutral losses) → list of theoretical fragment m/z; observed peaks matched within 10 ppm | matched peaks, explained intensity share, mean ppm, number of theoretical fragments |
| **F′, MetFrag-lite** (`metfrag.py`) | candidate SMILES + ion mode | all 1-bond cuts (weight 1.0) and 2-bond cuts (weight 0.6), each shifted by −2…+3 H and ± a proton; peaks matched within 0.01 Da | matched peaks, explained intensity, the same with rare peaks weighted up, gap to the best candidate in the shortlist, mean ppm |
| **F2, ICEBERG** (`f2_bridge.py` → `f2_batch.py`) | candidate SMILES + adduct + CE (rounded to 5 eV) + instrument | a public neural network (ICEBERG 2.1, MassSpecGym weights) predicts a full spectrum with intensities; compared to the query by matched cosine | best and mean cosine over the query's spectra |
| **M, mass** | candidate exact mass + formula | how well the mass fits every query spectrum; how many other shortlist candidates share its formula | `mass_error_ppm`, `mass_error_ppm_max`, `cand_n_same_formula` |
| **A, analogues** (off; `--enable-analogues`) | spectra of the candidate's 8 most similar *visible* molecules (Tc ≥ 0.3) | compares the query to these relatives' spectra, as a stand-in for the candidate's own spectrum | entropy / cosine similarities, Tc of the best relative |

F2 runs in its own Python environment (`extern/ms-pred/.venv`). Its
predictions are cached in `out/cache/f2_predictions.parquet`. If the
environment or the weights are missing, all `f2_*` features are 0 and
everything else still works.

---

## 7. The leakage rules, short version

These are checked with `assert` in the code, not just promised.

| rule | in words |
|---|---|
| **R1** | No eval molecule is ever a training example, for any model. No hidden row is ever read. |
| **R2** | A pseudo-truth's rows are hidden from everything in that run, including channel-P v1. Its truth is reachable only as a structure. |
| **R2b** | A training shortlist without its truth inside the window is dropped, never padded. |
| **R3** | Simulator bookkeeping (`difficulty`, `neighbours_removed`, `max_visible_tc`) is never a feature. It is only used to split the results. |
| **R4** | Training and prediction use the same feature code. |
| **R5** | All sampling comes from one seeded random generator (`--train-seed`). |
| **v2 guard** | Channel-P v2 checkpoints must list every eval key in `meta.excluded_keys`, or `stage2.py` refuses them. |

---

## 8. Commands

```bash
# the experiment: trains model 1 + the fusion model, evaluates on difficulty 2
uv run python max_class2_exp/stage2.py --blind-control --model-seeds 3

# quick check on 30 eval molecules
uv run python max_class2_exp/stage2.py --limit 30

# models 3 + 4 (channel-P v2), on a GPU box; next stage2 run picks them up
python max_class2_exp/train_all.py --no-mlflow
python max_class2_exp/train_all.py --smoke          # 1-minute end-to-end check
```

Outputs land in `max_class2_exp/out/`. See README §7.1 and §7.2 for all flags.
