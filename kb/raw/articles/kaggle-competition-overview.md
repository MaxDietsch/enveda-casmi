# Enveda CASMI 2026 — Competition Overview

Help uncover the chemical diversity of nature.

In the Critical Assessment of Small Molecule Identification (CASMI) 2026 challenge, participants build models that predict natural product structures from mass spectra.

Modern mass spectrometry can detect thousands of molecules in nature, but identifying them all remains a challenge. Researchers must predict chemical structures from tandem mass spectra, even for molecules never seen before.

Current methods mostly compare spectra against reference libraries. They work for known compounds but struggle with the many unknown molecules found in real biological samples. Many still require slow, expensive lab work to identify.

In this competition, you’ll build machine learning models that predict 2D chemical structures (a SMILES string) from LC-MS/MS spectra. Your goal is to generate accurate SMILES representations for both known and novel molecules.

Your solution could help researchers discover new medicines and identify disease biomarkers.

## Evaluation

Submissions are evaluated using Mean Reciprocal Rank @ 25 (MRR@25):

$$\text{MRR@25} = \frac{1}{U} \sum_{u=1}^{U} \frac{1}{rank_u}$$

where \\( U \\) is the number of molecules and \\( rank_u \\) is the position of the first correct structure in your ranked list for molecule \\( u \\). A molecule scores 0 if none of your guesses is correct. Each molecule has exactly one correct structure, so only your **first** correct guess counts. A correct guess at position 1 scores 1.0, at position 2 scores 0.5, at position 25 scores 0.04.

## Matching

A prediction is correct when it describes the same **atom connectivity** as the answer. Both your SMILES and the answer are passed through RDKit's tautomer canonicalization (pinned at 2026.03.3) and reduced to the first block of their InChIKey (the InChIKey14), then compared. The prediction is correct when the two first-block keys match.

This means you are not penalized for getting stereocenters or tautomer forms wrong. For example, both of the following score identically against an answer of glucose, because they reduce to the same InChIKey14 (`WQZGKKKJIJFFOK`):

    OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O
    OCC1OC(O)C(O)C(O)C1O

This is the link to the [metric notebook](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank).

## Submission File

For each `molecule_id` in the test set, predict up to 25 candidate structures as SMILES, **best guess first**, joined by semicolons in a single field.

Every `molecule_id` must appear exactly once. The file should contain a header and have the following format:

    molecule_id,smiles
    m_0014ef,CC1=CC(=O)C=CC1=O;OC(=O)c1ccccc1O;CN1C=NC2=C1C(=O)N(C)C(=O)N2C
    m_004d06,NCCc1ccc(O)cc1;CC(=O)Nc1ccc(O)cc1;OCC(O)CO
    ... 

Fewer than 25 guesses is allowed but there is no penalty for a wrong guess beyond the rank it occupies. A submission is rejected if it is missing the `molecule_id` or `smiles` column, is empty, contains nulls in either column, repeats a `molecule_id`, or gives more than 25 semicolon-separated guesses for any molecule.

## Background

**What is an MS/MS mass spectrum?**

You don't need a chemistry background to compete, but it helps to understand the basics of how mass spectra are generated in a mass spectrometer:

- **Ionization:** The instrument can only detect and measure charged ions, so first a molecule is ionized, picking up or losing charged species. In the instrument’s **positive ion mode** (more common for small molecules), the resulting ion is positively charged, in **negative ion mode** (less common but still used), it is negatively charged. The kind of charged ion the molecule acquires is called an adduct.  For some common examples, the molecule may acquire a proton (positive), acquire an ammonium ion (positive), or lose a proton (negative). We would write these **adduct** forms respectively as [M+H]+, [M+NH4]+, or [M-H]-.

- **Precursor mass detection:** The instrument measures the ion's **precursor m/z** (mass-to-charge ratio) with high accuracy, which strongly constrains the molecular formula. For small molecules, the charge (z) is usually +1 or -1, so m/z corresponds to the mass of the ion, and sometimes we will refer to m/z simply as “mass”.

- **Fragmentation:** The precursor ion (actually many individual ions of the same molecule+adduct) is then selected and fragmented by collision with neutral gas at a given **collision energy,** and the instrument records the **m/z and intensity (abundance) of each fragment**. The histogram of intensities across all detected fragments is the molecule’s mass spectrum. Intensities are typically normalized. A mass spectrum looks like this:

Figure 1: MS/MS spectrum of Chrysin [M-H]- adduct at around 50eV, timsTOF (Source: Enveda)

Each m/z is called a **fragment ion or peak**. The **base peak** is the highest-intensity fragment ion. The **precursor peak** is the m/z corresponding to the intact, unfragmented ion. It may or may not be present, depending on how thoroughly the ion was fragmented.