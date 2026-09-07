# Which Characters Need Context?
### Measuring Character-Specific Context Gain in Natural Language and Source Code

**Author:** Amar Bassan  
**Preprint:** https://doi.org/10.5281/zenodo.22074823

---

## What This Is

This repo contains the full experiment code, corpora, and results for a study of how much
**preceding context** reduces surprisal for individual characters — and whether **structural**
characters (`.`, `,`, `!`, `(`, `:`, …) benefit more from context than **lexical** ones
(letters, digits).

The key metric is **Context Gain**:

> CG_x(k; D) = S_x(1; D) − S_x(k; D)

where S_x(k) is the mean surprisal of character x given k preceding characters.
The main hypothesis: in natural language, structural characters show a steeper CG curve
(larger β₃ on the log₂(k) × Structural interaction term).

---

## Reproducing the Paper

All pre-computed results are already in the repo. The commands below regenerate them from scratch.

### Step 1 — Canonical baseline (NL1, NL2, Code1)

```bash
pip install numpy pandas scipy matplotlib

python run_experiment_v2.py          # panels → results_canonical_snapshot/panel_*.csv
python run_validations.py            # β₃ robustness, permutation → results_canonical_snapshot/robustness_b3.csv
python code1_coverage_sensitivity.py # Code1 coverage sweep → results_canonical_snapshot/code1_coverage_sensitivity.csv
```

### Step 2 — P0 kill tests (LOO + KT smoothing)

```bash
python tmlr_experiments/run_p0.py
# → results_tmlr/leave_one_structural_out.csv
# → results_tmlr/smoothing_robustness.csv
```

### Step 3 — Phase 2 replication (5 NL + 3 code corpora)

```bash
# Download corpora (Reuters, Bible KJV, WikiText-103, Node.js, Commons Lang Java)
python phase2_download_corpora.py
# → corpora_phase2/*.txt  (also already committed to the repo)

python tmlr_experiments/run_phase2.py --max-k 8
# → results_tmlr/phase2/panel_*.csv  (one per corpus)

python tmlr_experiments/make_forest_plot.py
# → results_tmlr/phase2/forest_plot.png
# → results_tmlr/phase2/consolidated_results.csv
```

**Parameters:** seed = 42 | Laplace (add-1) smoothing | τ = 0.50 | min n = 30 | cluster-robust SEs

**Where each reported number comes from:**

| Reported value | Source file | Row/note |
|---|---|---|
| NL1 β₃ = +0.551 | `results_canonical_snapshot/robustness_b3.csv` | shakespeare, τ=0.50, laplace |
| NL2 β₃ = +1.131 | `results_canonical_snapshot/robustness_b3.csv` | pride_prej, τ=0.50, laplace |
| Code1 β₃ = −0.024 | `results_canonical_snapshot/code1_coverage_sensitivity.csv` | τ=0.50 row |
| Permutation p values | `results_robustness/permutation_test.csv` | — |
| Functional form (AIC) | `results_robustness/functional_form.csv` | — |
| LOO β₃ range | `results_tmlr/leave_one_structural_out.csv` | all 5 NL2 exclusions |
| KT smoothing shift | `results_tmlr/smoothing_robustness.csv` | NL2 Δβ₃ = 0.017 |
| Phase 2 β₃ (8 corpora) | `results_tmlr/phase2/consolidated_results.csv` | — |
| Forest plot | `results_tmlr/phase2/forest_plot.png` | — |

**Note:** `run_experiment_v3.py` uses a per-character adaptive k-search and produces
different β₃ values (NL1: +0.101; NL2: +0.500; Code1: +0.079). It is a subsequent
updated pipeline, not the one used for the reported results.

---

## Running the v3 Pipeline (updated; differs from paper values)

The SQLite cache is included. To run the regression and plot with the updated pipeline:

```bash
pip install numpy pandas scipy matplotlib
python run_experiment_v3.py --skip-compute
```

This loads cached surprisals from `experiment_cache.db` and produces
`results_v3/` in a few seconds. Note: `results_v3/` is the output of the updated
pipeline and does **not** reproduce the manuscript headline β₃ values.

---

## Full Replication of v3 Pipeline (Laplace smoothing, ~15 min)

```bash
pip install numpy pandas scipy matplotlib
python run_experiment_v3.py
```

Deletes nothing — completed (corpus, k) pairs are skipped on restart.
Results written to `results_v3/`. See `results_canonical_snapshot/` for paper values.

### Requirements

Python 3.10+ with:
```
numpy pandas scipy matplotlib
```
No GPU required. The Python stdlib corpus is sourced from your local Python 3.12
installation automatically.

---

## KenLM Backend (Modified Kneser-Ney, requires Docker)

For large corpora (>100 M chars) the in-memory Laplace backend becomes
memory-prohibitive. KenLM (Heafield, 2011) uses a compressed trie under a fixed
memory budget. A Docker image compiles KenLM and runs the experiment inside the
container while keeping all output files on your local filesystem.

**Build image and run (first time ~10 min, subsequent runs skip model build):**

```powershell
# Windows PowerShell
.\docker_run.ps1
```

```bash
# Linux / macOS
docker build -t kenlm-experiment .
docker run --rm -v "$PWD:/workspace" kenlm-experiment \
    run_experiment_v3.py --backend kenlm \
    --db /workspace/experiment_cache_kenlm.db
```

Pre-built KenLM binary models (`.bin`) are **not** included in the repo — they are
architecture-specific (Linux x86_64). They are rebuilt automatically on first run and
cached in `kenlm_models/` on your local machine.

---

## Repository Layout

```
# Experiment scripts
run_experiment_v2.py               Canonical pipeline (used for paper values)
run_validations.py                 β₃ robustness, permutation, functional form tests
code1_coverage_sensitivity.py      Code1 β₃ sweep at τ = 0.25/0.50/0.75
character_context_profile.py       Standalone per-character CG profiler (any corpus)
phase2_download_corpora.py         Download Phase 2 corpora from NLTK / HuggingFace / GitHub
audit_k1.py                        Verify k=1 baseline is identical across pipeline versions

tmlr_experiments/
  run_p0.py                        P0 kill tests: LOO regression + KT smoothing robustness
  run_phase2.py                    Phase 2 replication across 8 corpora (--max-k 8)
  make_forest_plot.py              Forest plot + consolidated_results.csv
  code_classification_assumptions.md  Notes on cross-language code classification choices

# Corpora
corpus_shakespeare.txt             tinyshakespeare — ~1.1 M chars (public domain)
corpus_pride_prejudice.txt         Pride and Prejudice — Gutenberg #1342 (~694 K chars)
corpora_phase2/
  reuters.txt                      Reuters-21578 (research use)
  bible_kjv.txt                    King James Bible (public domain)
  wikitext_103.txt                 WikiText-103 sample (CC-BY-SA)
  nodejs_js.txt                    Node.js v22 stdlib JS source (MIT)
  commons_lang_java.txt            Apache Commons Lang (Apache-2.0)
  # Python 3.12 stdlib sourced from local install at runtime by run_experiment_v2.py

# Pre-computed results — paper source of truth
results_canonical_snapshot/        ← NL1/NL2/Code1 canonical values (Steps 1–2 above)
  CANONICAL_VALUES.md              β₃ table + per-character reference points
  robustness_b3.csv                β₃ at τ = 0.25/0.50/0.75, Laplace + KT
  code1_coverage_sensitivity.csv   Code1 β₃ at three coverage thresholds
  panel_*.csv                      Regression panels per corpus
  cross_corpus_v2.csv              Cross-corpus β₃ summary (canonical 3 corpora)

results_robustness/
  permutation_test.csv             Permutation p-values (10 000 shuffles)
  functional_form.csv              AIC comparison: log, linear, categorical k
  coverage_sensitivity.csv         Per-corpus coverage at each k

results_tmlr/
  leave_one_structural_out.csv     LOO β₃ for each structural character excluded (NL2, NL1, Code1)
  smoothing_robustness.csv         KT α=0.5 vs Laplace Δβ₃ per corpus
  phase2/
    consolidated_results.csv       β₃, SE, CI, p for all 8 corpora
    forest_plot.png                Publication figure (Figure 1 in paper)
    panel_*.csv                    Per-corpus regression panels (one per Phase 2 corpus)

# TMLR Beyond PDF submission
tmlr_submission/
  submission_folder/
    submission.md                  Full paper in Jekyll/Distill format
    assets/bibliography/
      submission.bib               BibTeX references (12 entries)
    assets/img/submission/
      forest_plot.png              Forest plot for web rendering
  compile_submission.py            Merge files + launch Docker Jekyll server
  README.md                        TMLR Beyond PDF build instructions

# Infrastructure
Dockerfile                         KenLM Docker build (for large-corpus KenLM backend)
docker_run.ps1                     PowerShell helper: build / run / shell
requirements.txt                   Python dependencies
```

---

## Key Results (Laplace, τ = 0.50)

Values are from `results_canonical_snapshot/` (the source used in the manuscript).
The v3 pipeline produces different numbers; see the Paper Results section above.

| Corpus | k_max | β₃ | SE | 95 % CI | p |
|---|---|---|---|---|---|
| NL1 — tinyshakespeare | 7 | +0.551 | 0.250 | [+0.049, +1.053] | 0.0319 |
| NL2 — Pride & Prejudice | 8 | **+1.131** | 0.337 | [+0.452, +1.810] | 0.0016 |
| Code1 — Python stdlib | 10 | −0.024 | 0.262 | [−0.481, +0.561] | 0.9304 |

β₃ > 0 in both NL corpora (NL2 survives 3-test Bonferroni at α = 0.017; NL1 is directional).
β₃ ≈ 0 for code — structural characters in Python are not more context-dependent
than lexical ones.

---

## Corpora

| Corpus | Domain | Source |
|---|---|---|
| Shakespeare | Natural language | Complete Works (public domain) |
| Pride and Prejudice | Natural language | Project Gutenberg #1342 |
| Python 3.12 stdlib | Source code | CPython local installation (163 files in the reported experiment; replication on a different Python 3.12 installation may yield a different file count) |

---

## Metrics

| Symbol | Definition |
|---|---|
| S_x(k; D) | Mean surprisal of char x given k preceding chars (Laplace-smoothed) |
| CG_x(k; D) | S_x(1; D) − S_x(k; D) — Context Gain relative to the one-character-context (bigram) baseline |
| k_peak | argmax_k CG_x(k) within the reliable k range |
| β₃ | Coefficient on log₂(k) × Structural in the panel regression |

---

## Citation

```
@misc{bassan2026context,
  title  = {Which Characters Need Context? Measuring Character-Specific
             Context Gain in Natural Language and Source Code},
  author = {Bassan, Amar},
  year   = {2026},
  note   = {Preprint. \url{https://github.com/asbassan/char-context-gain}}
}
```

---

