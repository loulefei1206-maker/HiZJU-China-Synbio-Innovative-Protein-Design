# Tutorial: Computational Design of Bright and Thermostable avGFP Variants

## Overview

This project aims to computationally engineer enhanced variants of **Aequorea victoria Green Fluorescent Protein (avGFP)** with improved fluorescence intensity and thermal stability.

Rather than relying on experimental screening, we integrate protein language models, directed evolution algorithms, structure prediction tools, and molecular dynamics simulations into a unified computational pipeline.

The workflow progressively narrows a large candidate pool into a small number of high-confidence variants for final submission.

---

# Project Objective

The primary design target is:

**avGFP (Aequorea victoria GFP)**

To improve sequence diversity and broaden exploration of the GFP fitness landscape, four additional GFP homologs are incorporated as reference proteins.

| Protein | Role                        | PDB  |
| ------- | --------------------------- | ---- |
| avGFP   | Primary optimization target | 2WUR |
| sfGFP   | Reference sequence          | 2B3P |
| amacGFP | Reference sequence          | 7LG4 |
| cgreGFP | Reference sequence          | 2HPW |
| ppluGFP | Reference sequence          | 2G6X |

These homologs provide evolutionary diversity during sequence generation while all downstream evaluations focus on identifying improved avGFP variants.

---

# Workflow

```text
Reference GFP Sequences
        │
        ▼
Sequence Pool Construction
        │
        ▼
Top100 Selection (per model)
        │
        ▼
CD-HIT Deduplication
        │
        ▼
ESM Fluorescence Prediction
        │
        ▼
FoldX Stability Prediction
        │
        ▼
ESMFold Structural Screening
        │
        ▼
Top20 Candidates
        │
        ▼
AlphaFold2 Validation
        │
        ▼
Top15 Candidates
        │
        ▼
AlphaFold3 Structure Generation
        │
        ▼
GROMACS MD Simulation
        │
        ▼
Final Top6 Variants
```

---

# 1. Reference Sequence Preparation

Five GFP-family proteins are used as starting references.

These sequences serve two purposes:

1. Expanding the searchable sequence space.
2. Providing evolutionary information for sequence generation models.

Although multiple GFP families are included, all final candidates are evaluated as avGFP variants.

---

# 2. Sequence Pool Construction

Candidate sequences are generated using multiple complementary design strategies.

| Method                    | Generated Sequences |
| ------------------------- | ------------------: |
| EVOLVEpro + CLADE         |               3,503 |
| EvoProtGrad               |              30,000 |
| ESM-based generation      |               4,761 |
| ProteinNPT                |               5,000 |
| MULTI-evolve              |               2,161 |
| Other GFP-derived sources |             ~10,000 |

Total candidate pool:

≈55,000 sequences

These approaches provide different exploration behaviors:

* EVOLVEpro focuses on active-learning-guided optimization.
* EvoProtGrad performs gradient-guided directed evolution.
* ESM explores protein language model sequence space.
* ProteinNPT introduces structure-aware sequence variations.
* MULTI-evolve captures beneficial mutation combinations.

---

# 3. Initial Candidate Selection

To reduce computational cost while maintaining diversity, each generation model is processed independently.

For every model:

1. Rank generated sequences using the model-specific score.
2. Select the top 100 candidates.
3. Export selected sequences in FASTA format.

The selected candidates are then merged into a unified sequence pool.

---

# 4. Sequence Deduplication

Redundant candidates are removed using CD-HIT.

Filtering includes:

* Exact duplicate removal.
* Highly similar sequence clustering.
* Competition exclusion list filtering.

This step prevents overrepresentation of similar variants in downstream analyses.

---

# 5. Fluorescence Prediction

Fluorescence performance is evaluated using ESM-based scoring.

The objective is to identify variants that remain compatible with the functional landscape learned by large protein language models.

Evaluation considers:

* Sequence likelihood
* Mutation-effect prediction
* Evolutionary plausibility

Output:

**Fluorescence Score**

Sequences with poor fluorescence predictions are removed from further consideration.

---

# 6. Thermal Stability Prediction

Thermal stability is evaluated using FoldX.

For each candidate:

1. Generate a structural model.
2. Repair structural geometry.
3. Calculate energetic properties.
4. Estimate relative stability.

Output:

**FoldX Stability Score**

Variants exhibiting unfavorable stability profiles are excluded.

---

# 7. Structural Screening Using ESMFold

The remaining candidates are subjected to structure prediction using ESMFold.

## pTM Filtering

Only structures satisfying:

```text
pTM > 0.51
```

are retained.

This threshold removes candidates with low structural confidence.

## pLDDT Ranking

Candidates passing the pTM filter are ranked according to predicted local confidence.

The highest-confidence structures are selected.

Output:

**Top 20 Candidates**

---

# 8. AlphaFold2 Structural Validation

The Top20 candidates are re-evaluated using AlphaFold2.

Assessment includes:

* Global fold quality
* pLDDT confidence
* β-barrel integrity
* Chromophore environment preservation
* Consistency with known GFP structures

Reference structures:

* avGFP (2WUR)
* sfGFP (2B3P)
* amacGFP (7LG4)
* cgreGFP (2HPW)
* ppluGFP (2G6X)

After structural inspection and ranking:

**Top 15 Candidates**

are selected for molecular dynamics preparation.

---

# 9. AlphaFold3 Structure Generation

AlphaFold3 is used to generate high-quality structural models for molecular dynamics simulations.

These structures serve as the initial conformations for GROMACS.

Output:

**Top15 AlphaFold3 PDB Structures**

---

# 10. Molecular Dynamics Simulation

Molecular dynamics simulations are performed using GROMACS.

The objective is to evaluate structural robustness under dynamic conditions.

Key metrics include:

## RMSD

Measures global structural deviation throughout the simulation.

## RMSF

Measures residue-level flexibility.

## Radius of Gyration

Evaluates overall compactness.

## Hydrogen-Bond Network

Monitors preservation of critical stabilizing interactions.

## Chromophore Environment

Evaluates structural stability surrounding the fluorophore-forming residues.

Candidates exhibiting significant structural instability are removed.

---

# 11. Final Ranking

Final ranking integrates information from all evaluation stages.

Ranking criteria include:

* ESM fluorescence score
* FoldX stability score
* AlphaFold2 confidence metrics
* Structural integrity assessment
* Molecular dynamics stability metrics

Candidates are ranked comprehensively to identify the most promising variants.

Output:

**Final Top 6 avGFP Variants**

---

# Repository Structure

```text
project/
│
├── data/
│   ├── reference_sequences
│   ├── generated_sequences/
│   └── reference/
│
├── scripts/
│   ├── sequence_processing/
│   ├── esm_scoring/
│   ├── foldx/
│   ├── esmfold/
│   ├── alphafold/
│   └── gromacs/
│
├── results/
│   ├── pool_clean/
│   ├── top15/
│   └── final_top6/
│
├── tutorial.md
│
└── README.md
```

---

# Final Deliverables

## Sequence Submission

A CSV file containing the final six avGFP variants.

## Technical Report

The report should include:

* Sequence generation methodology
* Fluorescence prediction strategy
* FoldX stability evaluation
* Structural screening pipeline
* AlphaFold validation
* Molecular dynamics analysis
* Final ranking methodology

## Reproducibility

All scripts, configuration files, software versions, and model parameters should be included to ensure complete reproducibility of the workflow.

---

# Expected Outcome

Successful candidates should satisfy the following criteria:

* High predicted fluorescence
* Favorable FoldX stability
* Reliable ESMFold confidence
* High AlphaFold2 confidence
* Stable molecular dynamics trajectory
* Preserved GFP β-barrel architecture
* Intact chromophore environment

These criteria collectively define the selection of the final avGFP variants for submission.
