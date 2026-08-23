# Paper Reproduction

This document gives the exact chain from inputs to manuscript tables.

## Which pipeline generated the reported values

The manuscript reports values from the **v2 pipeline** (exhaustive n-gram search,
per-corpus coverage-probe k_max, Laplace smoothing). All canonical outputs are
preserved in `results_canonical_snapshot/`.

The repository also contains `run_experiment_v3.py`, which uses a per-character
adaptive k-search and produces materially different β₃ values. It is a subsequent
updated pipeline and was **not** used for any reported result.

## Exact command sequence

```bash
pip install numpy pandas scipy matplotlib

# Step 1 — NL1/NL2 panels and peaks
python run_experiment_v2.py
# Output: results_v2/panel_v2_shakespeare.csv
#         results_v2/panel_v2_pride_prej.csv
#         results_v2/peaks_v2_*.csv

# Step 2 — Robustness: β₃ at τ = 0.25/0.50/0.75, smoother comparison,
#           permutation test, functional form
python run_validations.py
# Output: results_robustness/robustness_b3.csv   ← NL1/NL2 headline β₃
#         results_robustness/permutation_test.csv
#         results_robustness/functional_form.csv

# Step 3 — Code1 with coverage probe extended to k ≤ 10
python code1_coverage_sensitivity.py
# Output: results_robustness/code1_coverage_sensitivity.csv ← Code1 headline β₃
```

## Where each reported number comes from

| Paper table / figure | Source file | Key column / row |
|---|---|---|
| NL1 β₃ = +0.551 | `results_canonical_snapshot/robustness_b3.csv` | shakespeare, τ=0.50, laplace |
| NL2 β₃ = +1.131 | `results_canonical_snapshot/robustness_b3.csv` | pride_prej, τ=0.50, laplace |
| Code1 β₃ = −0.024 | `results_canonical_snapshot/code1_coverage_sensitivity.csv` | τ=0.50 row |
| Coverage sensitivity tables | `results_canonical_snapshot/robustness_b3.csv` | all τ rows |
| Permutation p values | `results_robustness/permutation_test.csv` | — |
| Functional form (AIC) | `results_robustness/functional_form.csv` | — |
| Per-character k_peak / CG_peak | `results_canonical_snapshot/peaks_v2_*.csv` | — |

## Corpus IDs

| ID | Corpus | k_max |
|----|--------|-------|
| NL1 | tinyshakespeare | 7 |
| NL2 | Pride & Prejudice | 8 |
| Code1 | Python 3.12 stdlib (163 files from reported run) | 10 |

Note: the Python stdlib is sourced from the local Python 3.12 installation at
runtime. A different installation may contain a different file count; the
reported run used 163 files (4,577,353 characters).

## Parameters

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| Train/val/test split | 80/10/10 on raw characters |
| Smoothing | Laplace (add-1) |
| Coverage threshold τ | 0.50 (per-symbol) |
| Min test instances to report | 30 |
| Bootstrap resamples (n < 30) | 1,000 |
| SE type | Cluster-robust (clustered by character) |
