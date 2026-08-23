# Which Characters Need Context?
## Measuring Character-Specific Context Horizons in Natural Language and Source Code

**Author:** Amar Bassan  
**Status:** v2 complete. PARTIAL GO — Mann-Whitney significant in NL1 + Code1; β₃ > 0 all three corpora.  
**Preprint:** [Zenodo DOI — to be assigned]  
**Companion book:** Transformers From Scratch (Leanpub)

---

## Reproducing the Experiment

### Requirements
```
pip install torch numpy matplotlib scipy pandas
```

### Run
```
python run_experiment_v2.py
```
No GPU required. Estimated runtime: ~15 minutes. Results saved to `results_v2/`.

---

## Files

| File | Description |
|---|---|
| `run_experiment_v2.py` | Full per-symbol experiment with Python tokenization — **run this** |
| `experiment.ipynb` | Notebook version of experiment |
| `run_experiment.py` | v1 experiment (per-class, superseded by v2) |
| `ngram_go_nogo.py` | Quick GO/NO-GO on Shakespeare only |
| `requirements.txt` | Python dependencies |
| `results_v2/` | v2 outputs (auto-generated) |
| `results/` | v1 outputs (archived) |

### Results directory — v2

| File | Description |
|---|---|
| `peaks_v2_*.csv` | Per-symbol CG_peak, CI, k_peak, coverage, type for each corpus |
| `cross_corpus_v2.csv` | Mann-Whitney summary + regression β₃ per corpus |
| `context_curves_v2.png/pdf` | Publication figures (shaded sparsity region) |
| `config_v2.json` | Full reproducibility config |

---

## Corpora

| Corpus | Domain | Source |
|---|---|---|
| tinyshakespeare | Natural language | github.com/karpathy/char-rnn |
| Pride and Prejudice | Natural language | Project Gutenberg #1342 |
| Python 3.12 stdlib | Source code | CPython local installation |

---

## Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| S_x(k; D) | E[-log₂ P(X_t\|context) \| X_t = x] | Mean surprisal of char x at context k |
| CG_x(k; D) | S_x(1; D) − S_x(k; D) | Context gain over bigram baseline |
| k_peak(x; D) | argmax CG within reliable k range | Context length of peak gain |
| LS_x(D) | 1 − CG_peak / S_x(1) | Local sufficiency (0=fully context-dependent) |
| β₃ | regression: log₂(k) × Structural | Structural benefit per unit log-context, controlling for frequency |

---

## Result Summary (v2 — PARTIAL GO)

| Corpus | Struct. n | Lex. n | Median CG ratio | Mann-Whitney p | β₃ | Decision |
|--------|-----------|--------|-----------------|---------------|----|----------|
| NL1 shakespeare | 7 | 49 | 2.52× | 0.013 | +0.514 | GO |
| NL2 pride_prej | 5 | 40 | 1.18× | 0.148 | +1.142 | NO-GO (low power) |
| Code1 python | 20 | 58 | 1.23× | 0.022 | +0.088 | GO |

β₃ > 0 across all three corpora — primary finding.

---

*Part of "Transformers From Scratch: A Complete Manual" — Amar Bassan, 2026*
