# Which Characters Need Context?
### Measuring Character-Specific Context Gain in Natural Language and Source Code

**Author:** Amar Bassan  
**Preprint:** [Zenodo DOI — to be assigned]  
**Companion book:** *Transformers From Scratch* (Leanpub)

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

## Quickest Path: Pre-Computed Results

The SQLite cache is included. If you only want to run the regression and plot:

```bash
pip install numpy pandas scipy matplotlib
python run_experiment_v3.py --skip-compute
```

This loads cached surprisals from `experiment_cache.db` and produces
`results_v3/` in a few seconds — no recomputation needed.

---

## Full Replication (Laplace smoothing, ~15 min)

```bash
pip install numpy pandas scipy matplotlib
python run_experiment_v3.py
```

Deletes nothing — completed (corpus, k) pairs are skipped on restart.
Results written to `results_v3/`.

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
run_experiment_v3.py          Main experiment (Laplace + KenLM backends)
Dockerfile                    Builds KenLM on Linux, runs experiment in container
docker_run.ps1                PowerShell helper: build image / run / open shell

corpus_shakespeare.txt        Complete Works of Shakespeare (public domain)
corpus_pride_prejudice.txt    Pride and Prejudice — Gutenberg #1342 (public domain)
                              Python 3.12 stdlib sourced from local install at runtime

experiment_cache.db           Pre-computed Laplace surprisals (restartable cache)
experiment_cache_kenlm.db     Pre-computed KenLM surprisals

results_v3/
  cross_corpus_v3.csv         Per-corpus β₃, SE, CI, p, n
  panel_v3_*.csv              Per-character (char, k, CG, type, …) panel data
  peaks_v3_*.csv              Per-character k_peak, CG_peak, sym_cov
  robustness_b3_v3.csv        Coverage sensitivity sweep
  context_curves_v3.png       Publication figure

results_canonical_snapshot/   Locked reference values (pre-adaptive-k rerun)
  CANONICAL_VALUES.md         β₃ table + key per-character reference points
  *.csv                       Snapshot CSVs
```

---

## Key Results (Laplace, τ = 0.50)

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
| Python 3.12 stdlib | Source code | CPython local installation (163 files) |

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

*Part of "Transformers From Scratch: A Complete Manual" — Amar Bassan, 2026*
