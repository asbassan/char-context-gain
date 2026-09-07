---
layout: distill
title: "Which Characters Need Context? Measuring Character-Specific Context Gain in Natural Language and Source Code"
description: "We decompose context dependence per individual target character using n-gram models, finding a trajectory/magnitude dissociation between natural language and source code: structural characters accumulate predictive benefit faster per unit of additional context in prose, but not in code."
htmlwidgets: false

authors:
  - name: Amarpreet Singh Bassan
    affiliations:
      name: IEEE, Microsoft

bibliography: submission.bib

toc:
  - name: Abstract
  - name: Introduction
  - name: Background and Related Work
  - name: Methodology
    subsections:
      - name: Notation and Terminology
      - name: Target-Character Mean Surprisal
      - name: Per-Symbol Analysis
      - name: Python Tokenization
      - name: Coverage Diagnostic and Reliable k Range
      - name: Confidence Intervals
      - name: Statistical Tests
      - name: Corpora
  - name: Results
    subsections:
      - name: Regression with Cluster-Robust Standard Errors
      - name: Peak Context-Gain Comparison
      - name: Selected Per-Symbol Results
      - name: Python Code1 Structural Characters
      - name: The Colon Cross-Corpus Comparison
      - name: Robustness P0 Kill Tests
      - name: Replication Across Corpora
  - name: Discussion
    subsections:
      - name: The Strongest Finding and the Test Divergence
      - name: Implications for Context Engineering
      - name: Corpus-Specific Variation
      - name: Estimator-Dependence of Context Gain
      - name: Python Tokenization as a Required Preprocessing Step
      - name: Future Direction
      - name: Limitations and Future Work
  - name: Conclusion
  - name: Appendix Reproducibility
---

## Abstract

A character bigram model conditions on only one preceding character, making its contextual limitation unusually transparent: any predictive information available from earlier characters is necessarily unavailable to the bigram. This paper asks not whether longer context helps in aggregate — that is already established — but *which individual characters benefit, by how much, and at what context length?*

Beyond character-level language modelling, this question is relevant to context engineering: if the marginal predictive value of additional context differs across targets, then context requirements may be content-dependent rather than uniform across a sequence.

Results reveal a dissociation between two separable components of context dependence. In natural-language prose, structural characters — punctuation and delimiters — accumulate predictive benefit *faster per unit of additional context* (trajectory) than lexical characters do, after controlling for frequency: $$\beta_3=+1.131$$ in Pride and Prejudice (cluster-robust $$p=0.002$$, permutation $$p=0.001$$), directional in Shakespeare ($$\beta_3=+0.551$$, $$p=0.033$$). In Python source code, this trajectory effect is absent ($$\beta_3=-0.024$$, $$p=0.930$$); structural characters show higher *total* context gain (Mann-Whitney $$p=0.022$$, exploratory) but accumulate it at the same rate as lexical characters — high magnitude, flat trajectory. This trajectory/magnitude dissociation is the paper's sharpest conceptual contribution: the same structural/lexical distinction that predicts steeper accumulation in prose does not predict steeper accumulation in code, even though structural characters remain absolutely more context-dependent in both domains.

We define **Character Context Gain** $$CG_x(k; D) = S_x(1; D) - S_x(k; D)$$, where $$S_x(k; D)$$ is the **target-character mean surprisal** — the expected negative log-probability of character $$x$$ given $$k$$ characters of preceding context in corpus $$D$$. We measure per individual character, then classify afterward. Across three corpora we fit:

$$CG_x(k; D) = \beta_0 + \beta_1 \log_2(k) + \beta_2 \text{Structural}_x + \beta_3 [\log_2(k) \times \text{Structural}_x] + \beta_4 \log_2(\text{Freq}_x) + \varepsilon$$

In Code1, $$\beta_3=-0.024$$ ($$p=0.930$$); structural symbols show higher absolute peak context gain (Mann-Whitney $$p=0.022$$, exploratory) but accumulate it at the same rate as lexical characters. This slope/peak divergence is interpretively informative: the regression (trajectory) and Mann-Whitney (magnitude) measure different components of context dependence, and those components dissociate across domains.

The structural-trajectory pattern (positive $$\beta_3$$) replicates across five diverse natural-language corpora and is absent in all three source-code corpora tested (Section [Replication Across Corpora]). Two P0 kill tests verify the NL2 finding: leave-one-structural-out regression confirms no single structural character drives $$\beta_3$$, and replacing Laplace with KT smoothing ($$\alpha=0.5$$) shifts the estimate by $$0.017$$ (Section [Robustness P0 Kill Tests]).

---

## Introduction

The history of language modelling begins at the character level. Shannon (1951) <d-cite key="shannon1951prediction"></d-cite> estimated the entropy of English by asking human subjects to predict successive letters. Modern large language models report aggregate perplexity, which compresses all character and symbol types into a single number. That compression hides a question worth asking directly: do all individual characters benefit equally from longer context, or do structural symbols — punctuation, delimiters, sentence boundaries — benefit disproportionately from additional context compared with ordinary letters?

**Contributions:**

1. A **reusable per-character context profiling method and reference implementation** — `character_context_profile.py` takes any corpus file, estimates $$S_x(k; D)$$ and $$CG_x(k; D)$$ per individual character across n-gram orders, and reports context-gain trajectories with per-symbol coverage diagnostics. Measurements are corpus-conditional and estimator-dependent.

2. A **coverage-based reliability diagnostic** — per-symbol and global, turning the n-gram sparsity problem from a silent artifact into a reported finding.

3. A **regression finding** — $$\beta_3$$ is positive and permutation-validated in NL2 (cluster-robust $$p=0.002$$, permutation $$p=0.001$$), positive and directional in NL1 ($$p=0.033$$, permutation $$p=0.093$$), and null in Code1 ($$\beta_3=-0.024$$, $$p=0.930$$). Code1 shows an exploratory peak-gain difference (Mann-Whitney $$p=0.022$$). Standard errors are cluster-robust.

4. A **tokenization preprocessing requirement** — characters inside string literals and comments are not executable syntax operators; we apply Python's tokenizer to restrict the structural analysis to syntax-stratum characters only. In this corpus (Python 3.12 stdlib), 42% of characters fall in string or comment tokens.

5. A **trajectory/magnitude dissociation** — context dependence has two separable components: how steeply gain accumulates with context (trajectory, measured by $$\beta_3$$) and how much total gain is achieved (magnitude, measured by $$CG_{\text{peak}}$$). In source code, structural characters show high magnitude but flat trajectory; in natural-language prose, they show both. These components dissociate across domains in a way not captured by aggregate perplexity. This is the paper's central conceptual contribution.

6. A **corpus-specificity finding** — the reliable $$k$$ range, per-symbol $$k_{\text{peak}}$$, and context gain magnitudes all vary by corpus structure, not just corpus size.

7. A **cross-corpus replication** showing positive $$\beta_3$$ in all five natural-language corpora tested and null $$\beta_3$$ in all three code corpora, consistent with a domain-specific rather than universal effect.

8. Two **P0 kill tests**: leave-one-structural-out (LOO) regression shows the NL2 finding is not driven by any single structural character; KT smoothing ($$\alpha=0.5$$) shifts the NL2 $$\beta_3$$ estimate by $$0.017$$.

---

## Background and Related Work

**Character-level entropy.** Shannon (1951) <d-cite key="shannon1951prediction"></d-cite> measured $$H(k)$$ for English prose as a whole. Brown et al. (1992) <d-cite key="brown1992estimate"></d-cite> estimated entropy decay with context. Scheibner et al. (2025) <d-cite key="scheibner2025large"></d-cite> showed character-level conditional entropy continues declining with very long context in large LLMs. These works primarily characterize aggregate entropy or surprisal rather than context-gain trajectories for individual target characters.

**Punctuation and context in neural models.** Razzhigaev et al. (2025) <d-cite key="razzhigaev2025llm"></d-cite> showed punctuation carries disproportionate contextual information in Transformer attention. Chauhan et al. (2026) <d-cite key="chauhan2026punctuations"></d-cite> demonstrated model-dependent computational roles for punctuation. Our work measures context dependency at the raw character level, using only count-based n-gram models without neural model involvement.

**Code naturalness and localness.** Hindle et al. (2012) <d-cite key="hindle2012naturalness"></d-cite> established that source code is highly predictable under n-gram models. Tu et al. (2014) <d-cite key="tu2014localness"></d-cite> showed local cache statistics account for much of this predictability — and that syntax tokens are particularly predictable locally. Our work extends this: not just *how predictable* but *how much does additional context reduce surprisal*, and whether that marginal gain differs between syntax and identifier characters.

**Surprisal reduction curves.** Measuring how surprisal or entropy changes as context length grows has a long history <d-cite key="shannon1951prediction"></d-cite><d-cite key="brown1992estimate"></d-cite> and continues in recent large-model work <d-cite key="scheibner2025large"></d-cite>. Studying surprisal reduction as a function of context length is therefore established methodology, not a novel contribution of the present work.

**Variable-order Markov models and adaptive context.** Variable-order Markov models (VOM; Ron et al. (1996) <d-cite key="ron1996power"></d-cite>) and the related Prediction by Partial Matching (PPM) algorithm (Cleary & Witten (1984) <d-cite key="cleary1984data"></d-cite>; Bell et al. (1990) <d-cite key="bell1990text"></d-cite>) address a question superficially similar to this paper: how much context should be used when predicting a character? VOM and PPM answer by selecting context length *conditioned on the observed history*: use the longest available context that has sufficient frequency statistics, backing off to shorter contexts when the long history was unseen in training. Context length is therefore a function of what appears *before* the prediction target. The present work asks a complementary but distinct question: conditioned on the *target character being* $$x$$, how much does context of length $$k$$ reduce $$x$$'s surprisal on average, across all corpus positions where $$x$$ occurs? This is **target-conditioned context benefit** rather than history-conditioned context selection — the independent variable is the identity of the character being predicted, not the composition of the preceding history. VOM addresses *when to use more context* (a history-dependent decision made at each position); $$CG_x(k)$$ addresses *which targets benefit more from more context* (a target-dependent measurement aggregated over all positions). These are orthogonal decompositions of the same prediction problem: a VOM model can decide to use $$k=5$$ at a given position without revealing whether the benefit of that longer context is distributed uniformly across all possible next characters or concentrated in structural ones.

**Gap.** The present work differs in three ways not found together in prior work: (1) decomposition at the level of individual characters rather than word categories, (2) a frequency-controlled regression comparing structural vs. lexical character types across prose and source code, and (3) Python tokenization stratification to separate syntax operators from identical characters in strings and comments. $$CG_x(k; D)$$ is defined for this study as an operationalization of these measurements; the underlying operation of subtracting surprisal estimates is not claimed as a new mathematical quantity.

---

## Methodology

### Notation and Terminology

| Symbol | Definition |
|--------|-----------|
| $$x$$ | Individual character (e.g., `,`, `a`, `(`) |
| $$D$$ | Corpus; $$D_{\text{train}}$$ / $$D_{\text{test}}$$ non-overlapping splits |
| $$k$$ | Context length (number of preceding characters) |
| $$S_x(k; D)$$ | Target-character **mean surprisal** of $$x$$ at context $$k$$ in $$D$$ |
| $$CG_x(k; D)$$ | Context gain $$= S_x(1; D) - S_x(k; D)$$ |
| $$k_{\text{peak}}(x; D)$$ | $$\arg\max_k CG_x(k; D)$$ within reliable $$k$$ range |
| $$\text{Coverage}_x(k; D)$$ | Fraction of test positions for $$x$$ where $$k$$-char context was seen in training |

*Notation note.* Surprisal is the negative log-probability assigned to a particular observed outcome. Entropy is the probability-weighted expected surprisal over the distribution of possible outcomes. In this study, $$S_x(k; D)$$ averages surprisal only over positions where the observed target character is $$x$$ — not over the full distribution of next characters. We therefore write $$S$$ rather than $$H$$ to make explicit that this is a per-target-outcome average, not an entropy.

### Target-Character Mean Surprisal

For a Laplace-smoothed n-gram trained on $$D_{\text{train}}$$ with vocabulary size $$|V|$$:

$$P(x_t | x_{t-k:t-1}) = \frac{\text{count}(x_{t-k:t-1}, x_t) + 1}{\text{count}(x_{t-k:t-1}) + |V|}$$

$$S_x(k; D) = \mathbb{E}[-\log_2 P(x_t | x_{t-k:t-1}) \mid x_t = x]$$

### Per-Symbol Analysis

We measure $$S_x(k; D)$$ and $$CG_x(k; D)$$ for **every individual character** $$x$$ in the corpus vocabulary, then classify characters as structural or lexical *after* measurement. This avoids the apples-to-oranges problem of comparing singleton punctuation characters (`,`, `.`) against broad aggregates (all vowels, all consonants).

**Natural language classification:**
- Structural: `,` `.` `;` `:` `?` `!` `'` `"` `(` `)` `[` `]` `{` `}`
- Lexical: alphabetic characters (a–z, A–Z), digits, space
- Ambiguous: newline (excluded from primary comparison)

**Source code classification (Code1):**
- Structural: Python syntax operators — *only* when the character appears in Python's `OP` token category as determined by the tokenizer
- Lexical: alphabetic and digit characters in `NAME` or `NUMBER` tokens
- Excluded: characters inside `STRING` or `COMMENT` tokens; whitespace tokens

### Python Tokenization

We apply Python's `tokenize` module to each `.py` file, assigning each character position a stratum: `syntax`, `keyword`, `identifier`, `string`, `comment`, `numeric`, `whitespace`, `other`. Only `syntax`-stratum characters are candidates for the structural category in Code1.

Stratum distribution in Code1 test set:

| Stratum | % of characters |
|---------|----------------|
| string | 33.8% |
| identifier | 25.3% |
| other | 13.7% |
| comment | 8.6% |
| whitespace | 8.1% |
| syntax | 5.4% |
| keyword | 4.6% |
| numeric | 0.4% |

In this corpus, 42% of characters fall inside STRING or COMMENT tokens. Strings may contain paths, regex, serialised data, SQL, or program constants — not necessarily natural language. When interpreting punctuation specifically as executable syntax operators, tokenizer stratification is required to separate them from identical characters appearing in string literals and comments.

### Coverage Diagnostic and Reliable k Range

At high $$k$$, most test contexts were never seen in training. Laplace smoothing then produces loss near $$-\log_2(1/|V|)$$ regardless of the true dependency — a measurement artifact, not a signal.

$$\text{Coverage}_x(k; D) = \frac{\text{test positions for } x \text{ where } k\text{-char context was in } D_{\text{train}}}{\text{total test positions for } x}$$

We report the **reliable $$k$$ range** as $$k$$ where global $$\text{Coverage}(k; D) \geq 50\%$$. We additionally enforce per-symbol coverage $$\geq 50\%$$ for each $$(x, k)$$ observation in the regression.

| Corpus | Reliable $$k$$ range | $$k=8$$ coverage |
|--------|---------------------|-----------------|
| NL1 tinyshakespeare | $$k \leq 7$$ | 43.9% — sparse |
| NL2 Pride & Prejudice | $$k \leq 8$$ | 61.6% |
| Code1 Python stdlib | $$k \leq 10$$ | 63.1% |

### Confidence Intervals

For $$n \geq 30$$ test instances: 95% CI on mean surprisal using the CLT. For $$n < 30$$: bootstrap 95% CI (1,000 resamples). All symbols with $$n < 30$$ are excluded from primary analysis.

*$$CG_{\text{peak}}$$ CI.* Since $$CG_x(k) = S_x(1) - S_x(k)$$, the 95% CI is computed as a conservative bound treating the two surprisal estimates as independent. Additionally, $$CG_{\text{peak}} = \max_k CG_x(k)$$ is a selected maximum from noisy estimates, introducing selection optimism. These CIs should be interpreted as approximate.

### Statistical Tests

**Mann-Whitney U**: one-sided test (structural > lexical) on per-symbol $$CG_{\text{peak}}$$ distributions. Tests whether structural characters have stochastically higher peak context gain than lexical characters.

**Regression**: for each corpus, pooling all $$(x, k)$$ observations for $$k \in \{2, 3, \ldots, k_{\max}\}$$ within the reliable range:

$$CG_x(k; D) = \beta_0 + \beta_1 \log_2(k) + \beta_2 \text{Structural}_x + \beta_3 [\log_2(k) \times \text{Structural}_x] + \beta_4 \log_2(\text{Freq}_x) + \varepsilon$$

Note: $$k=1$$ serves as the baseline for computing $$CG_x(k) = S_x(1; D) - S_x(k; D)$$ but is **not itself a regression observation**. By definition $$CG_x(1) = 0$$ for every character; including it would constrain the fit at a deterministically zero outcome. The key coefficient $$\beta_3$$ is unaffected by this centering choice.

### Corpora

| ID | Corpus | Domain | Size | Char vocab |
|----|--------|--------|------|-----------|
| NL1 | tinyshakespeare | Natural language | ~1.1M chars | 65 |
| NL2 | Pride and Prejudice | Natural language | ~694K chars | 87 |
| Code1 | Python 3.12 stdlib (163 files) | Source code | ~4.6M chars | 164 |

All splits: 80% train / 10% test on raw characters. Phase 2 replication corpora are described in Section [Replication Across Corpora].

---

## Results

### Regression with Cluster-Robust Standard Errors

OLS with standard errors clustered by character ($$G$$ = number of unique characters), enforcing per-symbol coverage $$\geq 50\%$$.

| Corpus | $$n$$ obs | $$G$$ (chars) | $$R^2$$ | $$\beta_3$$ | SE | 95% CI | $$p$$ |
|--------|----------|--------------|---------|------------|-----|--------|-------|
| NL1 shakespeare | 272 | 56 | 0.393 | **+0.551** | 0.250 | [+0.049, +1.053] | **0.033** |
| NL2 pride_prej | 310 | 45 | 0.701 | **+1.131** | 0.337 | [+0.452, +1.810] | **0.002** |
| Code1 python | 456 | 77 | 0.571 | −0.024 | 0.262 | [−0.481, +0.561] | 0.930 |

The negative $$\beta_1$$ values reflect that high-frequency lexical characters achieve most of their context gain at $$k=2$$–$$3$$ and then plateau. For structural characters in NL2: effective slope $$= \beta_1 + \beta_3 = -0.856 + 1.131 = +0.275$$ (positive). In NL1: $$-0.242 + 0.551 = +0.309$$ (positive). Structural characters' CG continues to grow with $$\log_2(k)$$ within the reliable range.

**Coverage threshold sensitivity** — $$\beta_3$$ at per-symbol coverage thresholds $$\tau = 0.25, 0.50, 0.75$$:

| Corpus | $$\tau=0.25$$ | $$\tau=0.50$$ (main) | $$\tau=0.75$$ |
|--------|-------------|---------------------|-------------|
| NL1 shakespeare | +0.636** | +0.551* | +0.337 ($$p=0.18$$) |
| NL2 pride_prej | +1.142** | +1.131** | +0.982** |
| Code1 python | +0.008 (n.s.) | −0.024 (n.s.) | −0.059 (n.s.) |

NL2 is stable across all three thresholds. NL1 provides threshold-sensitive evidence. Code1 is consistently not significant.

**Character-label permutation test** (10,000 permutations, structural/lexical labels shuffled, count fixed):

| Corpus | $$\beta_3$$ | $$p$$ cluster-robust | $$p$$ permutation |
|--------|------------|--------------------|--------------------|
| NL1 shakespeare | +0.551 | 0.033 | 0.093 |
| NL2 pride_prej | **+1.131** | **0.002** | **0.001** |
| Code1 python | −0.024 | 0.930 | 0.532 |

NL2's permutation $$p = 0.001$$: approximately 10 of 10,000 random label assignments produced $$\beta_3^{\text{perm}} \geq \beta_3^{\text{obs}}$$. NL1's cluster-robust $$p = 0.033$$ does not survive permutation ($$p = 0.093$$) and should be treated as directional.

**Functional form comparison** — M1: $$\log_2(k)$$, M2: linear $$k$$, M3: categorical $$k$$:

| Corpus | Model | AIC | $$\beta_{\text{interaction}}$$ | $$p$$ |
|--------|-------|-----|-------------------------------|-------|
| NL2 | M1 $$\log_2(k)$$ | −186.6 | **+1.131** | **0.002** |
| NL2 | M2 linear $$k$$ | −219.8 | +0.375 | 0.001 |
| NL2 | M3 categorical | −241.4 | +1.398 (mean) | — |

For NL2, all six $$k$$-specific interaction coefficients in M3 are positive and increase monotonically ($$k=3$$: +0.40; $$k=8$$: +2.19), confirming the finding is robust to functional form.

**Primary vs exploratory tests.** We designate $$\beta_3$$ in NL2 as the strongest inferential result. Under Bonferroni correction for three simultaneous regression tests ($$\alpha \approx 0.017$$), NL2 survives ($$p=0.002$$); NL1 does not ($$p=0.033$$).

### Peak Context-Gain Comparison

| Corpus | Structural $$n$$ | Lexical $$n$$ | Struct median CG | Lex median CG | Ratio | $$p$$-value | Interpretation |
|--------|----------------|-------------|-----------------|--------------|-------|------------|----------------|
| NL1 shakespeare | 7 | 49 | 1.810 bits | 0.717 bits | 2.52× | 0.013 | survives 3-test Bonferroni |
| NL2 pride_prej | 5 | 40 | 1.933 bits | 1.639 bits | 1.18× | 0.148 | not significant |
| Code1 python | 20 | 57 | 2.158 bits | 1.692 bits | 1.28× | 0.022 | exploratory; does not survive correction |

NL2 is not significant under Mann-Whitney: only 5 structural characters meet $$n \geq 30$$. However, the regression shows $$\beta_3=+1.131$$ ($$p=0.002$$). These tests answer different questions: Mann-Whitney asks whether $$CG_{\text{peak}}$$ is higher for structural chars; regression asks whether structural chars have a steeper slope of gain with $$\log(k)$$. In NL2 the slopes differ substantially even though the peak values don't separate cleanly.

Code1 shows the reverse pattern: exploratory peak-gain difference ($$p=0.022$$, structural median 2.158 vs lexical 1.692 bits) but null regression $$\beta_3=-0.024$$ ($$p=0.930$$). Structural chars in Python reach their peak quickly at $$k=2$$ or $$k=3$$ at roughly the same pace as lexical chars.

### Selected Per-Symbol Results

Structural characters with $$n \geq 30$$ in NL1 (tinyshakespeare):

| Char | $$S_x(1)$$ | $$CG_{\text{peak}}$$ | 95% CI | $$k_{\text{peak}}$$ | sym_cov |
|------|-----------|---------------------|--------|--------------------|----|
| `!` | 8.225 | 2.481 | [2.27, 2.70] | 6 | 79.8% |
| `?` | 8.027 | 2.183 | [1.99, 2.38] | 6 | 83.1% |
| `'` | 7.149 | 1.880 | [1.68, 2.08] | 4 | 95.2% |
| `;` | 7.495 | 1.810 | [1.61, 2.01] | 5 | 90.9% |
| `.` | 6.336 | 1.347 | [1.21, 1.48] | 4 | 96.5% |
| `,` | 4.957 | 1.183 | [1.08, 1.29] | 3 | 99.6% |
| `:` | 4.651 | 0.820 | [0.56, 1.08] | 2 | 99.9% |

Note that rare lexical characters (e.g., `v`, `g`) can have higher $$CG_{\text{peak}}$$ than common structural characters. The regression controls for this frequency effect; the raw ratio does not.

### Python Code1 Structural Characters

Key structural characters with $$n \geq 30$$ in syntax stratum:

| Char | $$S_x(1)$$ | $$CG_{\text{peak}}$$ | $$k_{\text{peak}}$$ | sym_cov |
|------|-----------|---------------------|--------------------|----|
| `&` | 12.35 | 5.73 | 6 | 56.1% |
| `%` | 9.05 | 5.60 | 2 | 100% |
| `(` | 5.77 | 1.67 | 3 | 97.2% |
| `.` | 4.31 | 1.54 | 3 | 96.7% |
| `:` | 4.40 | 0.78 | 3 | 98.5% |
| `,` | 5.01 | 0.65 | 2 | 99.9% |
| `)` | 5.16 | 0.55 | 2 | 99.9% |

`:` and `,` in Python have *lower* $$CG_{\text{peak}}$$ than in Shakespeare — consistent with grammar-enforced placement making them more locally predictable.

### The Colon Cross-Corpus Comparison

| Corpus | $$S_x(1)$$ | $$CG_{\text{peak}}$$ | $$k_{\text{peak}}$$ | $$n_{\text{test}}$$ |
|--------|-----------|---------------------|--------------------|----|
| NL1 shakespeare | 4.651 | 0.820 | 2 | 1272 |
| Code1 python (syntax only) | 4.400 | 0.781 | 3 | 2816 |

Python colon (syntax stratum, $$n=2816$$) has *slightly lower* $$CG_{\text{peak}}$$ than Shakespeare colon — the opposite of what a naive grammar-complexity argument would suggest. This is consistent with the hypothesis that grammar-enforced placement after short keywords (`if`, `def`, `for`) makes the 2–3 character context highly informative.

### Robustness P0 Kill Tests

**Leave-one-structural-out (LOO) — NL2 (Pride & Prejudice), 5 structural characters:**

| Excluded | $$\beta_3$$ | 95% CI | $$p$$ |
|----------|-----------|--------|-------|
| `!` | +0.913 | [+0.242, +1.585] | 0.009 |
| `'` | +1.318 | [+0.598, +2.037] | 0.001 |
| `.` | +1.343 | [+0.662, +2.024] | <0.001 |
| `;` | +1.124 | [+0.287, +1.961] | 0.010 |
| `?` | +0.957 | [+0.221, +1.692] | 0.012 |

All five exclusions keep $$\beta_3 > +0.90$$ with $$p < 0.013$$. The smallest LOO estimate (+0.913, excluding `!`) lies well within the original 95% CI ([+0.452, +1.810]). NL1 (7 structural characters): all seven exclusions produce positive $$\beta_3$$ (range +0.45 to +0.73). Code1 (20 structural characters): all 20 exclusions produce $$\beta_3$$ between −0.184 and +0.066.

**KT smoothing ($$\alpha=0.5$$):**

| Corpus | Laplace $$\beta_3$$ | KT $$\beta_3$$ | $$\Delta\beta_3$$ | $$p$$ (KT) |
|--------|-------------------|--------------|----------------|------------|
| NL1 shakespeare | +0.549 | +0.510 | −0.039 | 0.039 |
| NL2 pride_prej | **+1.131** | **+1.114** | −0.017 | **0.002** |
| Code1 python | −0.024 | −0.011 | +0.013 | 0.968 |

The NL2 estimate shifts by 0.017 (1.5%) and remains highly significant. The structural/lexical separation is not an artifact of the Laplace pseudocount choice.

### Replication Across Corpora

**Phase 2 pipeline.** All five NL corpora use the same character classification as the canonical analysis. All three code corpora use a shared `CODE_STRUCTURAL_CHARS` set with character-identity-only classification — no per-language tokenizer strata — to ensure cross-language comparability.

**Methodological note on Phase 2 code classification.** Section [Python Tokenization] established that Python tokenizer stratification is required when interpreting punctuation specifically as executable syntax operators. The Phase 2 code corpora (Node.js, Commons Lang Java) use character-identity-only classification for cross-language comparability: building accurate tokenizers for JavaScript and Java at the same fidelity as Python's built-in `tokenize` module is a substantial separate engineering effort. This introduces a **conservative bias toward null**: punctuation inside strings and comments is more locally predictable — it follows no grammar and its context is often formulaic — giving it shorter acquisition range and lower CG than syntax-stratum punctuation. Contaminating the structural category with these lower-CG characters attenuates the structural/lexical gap rather than inflating it. A null $$\beta_3$$ under character-identity-only classification therefore represents a stronger null than a null obtained with full tokenization stratification.

**Phase 2 corpora:**

| ID | Corpus | Domain | Size | Source |
|----|--------|--------|------|--------|
| NL1 | tinyshakespeare | Drama | ~1.1M chars | canonical |
| NL2 | Pride and Prejudice | 19c novel | ~694K chars | canonical |
| NL3 | Reuters-21578 | News wire | ~9.0M chars | NLTK |
| NL4 | Bible KJV | Religious prose | ~4.6M chars | Project Gutenberg |
| NL5 | WikiText-103 | Wikipedia | ~2.0M chars | HuggingFace |
| Code1 | Python 3.12 stdlib | Source code | ~4.6M chars | canonical |
| Code2 | Node.js v22 stdlib | Source code | ~3.3M chars | GitHub |
| Code3 | Apache Commons Lang | Source code | ~8.7M chars | GitHub |

**Consolidated results ($$\beta_3$$, cluster-robust OLS):**

| Corpus | Type | $$\beta_3$$ | SE | 95% CI | $$p$$ | $$n$$ obs |
|--------|------|------------|-----|--------|-------|----------|
| Pride & Prejudice | NL | **+1.131** | 0.337 | [+0.452, +1.810] | **0.002** | 310 |
| Reuters | NL | **+0.712** | 0.348 | [+0.018, +1.405] | **0.044** | 490 |
| Bible KJV | NL | +0.693 | 0.391 | [−0.087, +1.474] | 0.081 | 440 |
| WikiText-103 | NL | +0.676 | 0.404 | [−0.131, +1.482] | 0.099 | 470 |
| Shakespeare | NL | **+0.549** | 0.250 | [+0.047, +1.050] | **0.033** | 272 |
| Python stdlib | Code | −0.024 | 0.278 | [−0.577, +0.528] | 0.930 | 456 |
| Node.js stdlib | Code | −0.321 | 0.192 | [−0.703, +0.060] | 0.098 | 547 |
| Commons Lang (Java) | Code | +0.075 | 0.264 | [−0.451, +0.601] | 0.777 | 556 |

{% include figure.html path="assets/img/submission/forest_plot.png" style="max-width:90%;height:auto;" class="img-fluid rounded" caption="Figure 1: β₃ estimates with 95% CIs across all eight corpora. Diamonds: p < 0.05; circles: p ≥ 0.05. NL panel (blue, left), code panel (red-orange, right)." %}

All five NL corpora show positive $$\beta_3$$ (range +0.55 to +1.13). Two reach $$p < 0.05$$; the remaining three are positive and directional. All three code corpora show $$\beta_3$$ near zero or slightly negative (range −0.32 to +0.08); none is significant.

**Statistical note.** Under Bonferroni correction for eight simultaneous tests, $$\alpha \approx 0.006$$; only NL2 survives ($$p=0.002$$). A one-sided sign test over the five NL results — five positives out of five — yields $$p = (0.5)^5 = 0.031$$ under the null that each corpus's $$\beta_3$$ sign is random, providing a conservative non-parametric replication summary.

---

## Discussion

### The Strongest Finding and the Test Divergence

The central claim is not "structural symbols have 3× higher CG." The defensible claim, with cluster-robust inferential statistics, is:

> **NL2 (strongest robust finding):** Structural symbols have a robustly steeper rate of context-gain per unit $$\log$$-context than lexical symbols after controlling for character frequency ($$\beta_3=+1.131$$, cluster-robust $$p=0.002$$, permutation $$p=0.001$$; stable across coverage thresholds and functional forms).
>
> **NL1 (directional):** Same sign ($$\beta_3=+0.551$$, cluster-robust $$p=0.033$$) but does not survive the character-label permutation test ($$p=0.093$$) or the 75% coverage threshold. Treated as corroborating NL2, not an independent replication.
>
> **Code1 (domain contrast, exploratory):** Structural symbols have higher peak context gain (Mann-Whitney $$p=0.022$$, exploratory; structural median 2.158 vs lexical 1.692 bits) but the rate of accumulation with $$\log$$-context is not steeper ($$\beta_3=-0.024$$, $$p=0.930$$). This motivates a distinction between the magnitude and trajectory of context gain.

The Mann-Whitney and regression divergences are not contradictions — they measure different things. NL2 reversal (Mann-Whitney not significant, regression significant): the 5 structural chars have steep $$k$$-slopes but similar $$CG_{\text{peak}}$$ to the field of 40 lexical chars. Code1 reversal (Mann-Whitney exploratory, regression null): structural chars in Python reach their $$CG_{\text{peak}}$$ at $$k=2$$ or $$k=3$$, as do most lexical chars. The *absolute level* is higher for structural chars but the *rate of ascent* is not steeper.

### Implications for Context Engineering

Context engineering typically treats context as a budget to be selected, retrieved, compressed, or truncated at the sequence level. Our results suggest that the predictive value of additional context can vary substantially at a finer granularity: different target characters exhibit different context-gain trajectories, and the structural-versus-lexical pattern itself varies across corpora and domains. The broader implication is that context demand may be content-dependent rather than solely sequence-length-dependent.

Character Context Gain does not by itself prescribe an adaptive context policy, but it provides a transparent measurement framework for studying context sufficiency and saturation at the character level. We make no claim that the current n-gram measurements translate directly to neural model behaviour; the relationship between n-gram CG and transformer-level context dependency is an open empirical question. Whether analogous saturation signals exist at the token or semantic level, and whether they transfer from n-gram to neural estimators, are open empirical questions — and testing them is the natural next step for this measurement framework.

### Corpus-Specific Variation

Every measurement carries an implicit $$(D)$$. The reliable $$k$$ range, $$S_x(1; D)$$, and $$CG_x(k_{\text{peak}}; D)$$ all vary by corpus structure. NL2 stays reliable to $$k=8$$ despite being the smallest corpus — a structure effect, not a size effect. The coverage diagnostic must be computed per corpus; it cannot be read off from corpus size alone.

### Estimator-Dependence of Context Gain

Because $$CG_x(k; D)$$ is defined as $$S_x(1; D) - S_x(k; D)$$ using a Laplace-smoothed n-gram, the measured values are estimator-dependent. A model with higher capacity would produce different surprisal estimates, and the difference at $$k=1$$ vs $$k=k^*$$ could be larger or smaller. No upper or lower bound relationship between n-gram CG and Transformer CG follows from either model being a better estimator of the true distribution. The n-gram measurements describe what a Laplace-smoothed count model finds in these corpora within the reliable $$k$$ range. Whether a more expressive model would show larger or smaller structural/lexical separation is an open empirical question.

### Python Tokenization as a Required Preprocessing Step

The Python stratum distribution (33.8% string, 8.6% comment) means that 42% of characters in Code1 appear inside STRING or COMMENT tokens rather than executable syntax. Character-level analyses that classify `?`, `!`, or `-` as structural code operators without tokenization stratification are measuring an uninterpretable mix of syntactic and non-syntactic occurrence contexts. When interpreting punctuation specifically as executable syntax operators, tokenizer stratification is required to separate them from identical characters appearing in strings and comments.

### Future Direction

A natural follow-on question is whether the context-dependency differences identified here are preserved or altered when those characters are absorbed into subword tokens by BPE merging. A subword model may represent the same underlying dependency at a coarser granularity. Whether it does is empirically testable using the $$S_x(k; D)$$ framework as a character-level baseline.

### Limitations and Future Work

1. **Alternative smoothing**: Laplace smoothing is suboptimal at high $$k$$. Per-symbol coverage filtering and frequency control address the primary bias mechanisms, and the coverage sensitivity analysis shows the NL2 finding is robust across all three thresholds. A true interpolated Kneser-Ney test would require recursive backoff, which is beyond the scope of this study.

2. **NL1 directional, not robust**: The NL1 $$\beta_3$$ finding is positive and cluster-robust at $$\tau=0.50$$ ($$p=0.033$$) but does not survive the character-label permutation test ($$p=0.093$$). With only $$G=7$$ structural character clusters, the cluster-robust $$t$$-approximation is optimistic. NL1 should be read as directionally consistent with NL2, not as independent replication.

3. **NL2 power**: Only 5 structural characters cleared $$n \geq 30$$ in Pride and Prejudice, making Mann-Whitney underpowered. The regression is validated by the permutation test ($$p=0.001$$): despite only 5 structural clusters, the observed $$\beta_3=+1.131$$ is in the top 0.1% of the permutation null distribution.

4. **$$CG_{\text{peak}}$$ selection optimism**: $$CG_{\text{peak}} = \max_k CG_x(k)$$ is a maximum selected from noisy estimates, so $$\mathbb{E}[\max(\widehat{CG})] > \max(\mathbb{E}[\widehat{CG}])$$. This affects the Mann-Whitney test more than the regression. For Code1, whose only positive inferential signal is the exploratory Mann-Whitney peak-gain finding, this limitation is especially relevant.

5. **$$k$$ range and corpus scale**: The reliable range is corpus-specific: $$k \leq 7$$ for NL1, $$k \leq 8$$ for NL2, and $$k \leq 10$$ for Code1. For corpora exceeding ~100M characters, the in-memory n-gram frequency tables become memory-prohibitive; KenLM (Heafield, 2011) <d-cite key="heafield2011kenlm"></d-cite> is the natural replacement.

6. **Regression random effects**: The regression uses cluster-robust standard errors (clustered by character). A character random-intercept model would additionally capture character-level variance in baseline surprisal.

7. **Classification sensitivity**: Alternative classification schemes — for example, restricting the lexical category to alphabetic characters only — could test how sensitive the structural/lexical interaction is to category definitions.

---

## Conclusion

Context dependence is not a single quantity. This paper's central finding is a dissociation between two components — *trajectory* (how steeply context gain accumulates per unit log-context, measured by $$\beta_3$$) and *magnitude* (the total gain achieved, measured by $$CG_{\text{peak}}$$) — that behave differently across domains. In natural-language prose, structural characters show both: steeper trajectory than lexical characters in Pride and Prejudice ($$\beta_3=+1.131$$, cluster-robust $$p=0.002$$, permutation $$p=0.001$$) and directionally in Shakespeare ($$\beta_3=+0.551$$, $$p=0.033$$). In Python source code, structural characters show high magnitude (Mann-Whitney $$p=0.022$$, exploratory) but flat trajectory ($$\beta_3=-0.024$$, $$p=0.930$$) — they reach a higher context-gain peak than lexical characters but not through a steeper accumulation rate. This dissociation is not predicted by aggregate perplexity measures and is not captured by any single statistic; it requires decomposing context dependence per target character.

Python tokenization reveals that 42% of Python stdlib characters are inside strings or comments; tokenizer stratification is required when interpreting punctuation specifically as executable syntax operators. The colon cross-corpus result shows grammar-enforced placement can produce lower context gain than discourse-governed usage, potentially because short repeated keywords are more informative than varied speaker names.

The reliable n-gram range is corpus-specific and must be measured per corpus. All measurements are finite-data estimates subject to n-gram sparsity and smoothing bias; the n-gram framework is chosen because it provides a transparent count-based estimate of corpus-local dependency structure without conflating model capacity with corpus properties.

A cross-corpus replication extending the analysis to five NL and three code corpora shows consistent positive $$\beta_3$$ across all five NL domains tested (range +0.55 to +1.13; two of five $$p<0.05$$), and null or near-null $$\beta_3$$ in all three code corpora (range −0.32 to +0.08). Two P0 kill tests additionally establish that the NL2 finding is not driven by any single structural character (LOO, min LOO $$\beta_3=+0.913$$) and is not an artifact of Laplace smoothing (KT $$\alpha=0.5$$, $$\Delta\beta_3=0.017$$). Taken together, the evidence supports the structural-trajectory distinction as a consistent property of natural-language corpora rather than an artefact of a particular text or estimator.

More broadly, these results suggest that context requirement is not necessarily uniform across prediction targets, motivating future work on context-sufficiency signals for adaptive retrieval, compression, and context allocation.

---

## Appendix Reproducibility

**Exact pipeline for manuscript values (inputs → code → outputs):**

| Step | Command | Output |
|------|---------|--------|
| 1 | `python run_experiment_v2.py` | `results_canonical_snapshot/panel_*.csv` — regression panels for NL1/NL2 |
| 2 | `python run_validations.py` | `results_canonical_snapshot/robustness_b3.csv` — NL1/NL2 $$\beta_3$$ at all $$\tau$$ |
| 3 | `python code1_coverage_sensitivity.py` | `results_canonical_snapshot/code1_coverage_sensitivity.csv` |
| 4 | `python tmlr_experiments/run_p0.py` | `results_tmlr/leave_one_structural_out.csv`, `results_tmlr/smoothing_robustness.csv` |
| 5 | `python phase2_download_corpora.py` | `corpora_phase2/*.txt` |
| 6 | `python tmlr_experiments/run_phase2.py --max-k 8` | `results_tmlr/phase2/panel_*.csv` |
| 7 | `python tmlr_experiments/make_forest_plot.py` | `results_tmlr/phase2/forest_plot.png`, `consolidated_results.csv` |

All canonical outputs are preserved in `results_canonical_snapshot/`. The repository also contains `run_experiment_v3.py`, which uses a per-character adaptive $$k$$-search and produces different $$\beta_3$$ values — it is a subsequent updated pipeline, not the one used for reported results.

**Dependencies:** numpy, matplotlib, pandas, scipy, nltk, datasets (Phase 2 corpus download)

**Seed:** 42 | **Smoothing:** Laplace (add-1) | **Min $$n$$ to report:** 30

The experiment script is idempotent: completed (corpus, $$k$$) pairs are stored in an SQLite cache and skipped on restart. The $$k$$ range for each corpus is derived automatically from a coverage probe rather than hardcoded; this produced $$k \leq 7$$ for NL1, $$k \leq 8$$ for NL2, and $$k \leq 10$$ for Code1.

**Standalone profiler.** `character_context_profile.py` is a self-contained tool that takes any corpus and produces per-character context-gain trajectories with coverage diagnostics:

{% highlight bash %}
# Natural language corpus
python character_context_profile.py corpus.txt --max-k 8 --output profile.csv

# Python source code (with tokenizer stratification)
python character_context_profile.py src/ --python --max-k 8
{% endhighlight %}

**Scalability note.** For corpora exceeding ~100M characters, the in-memory n-gram counting backend should be replaced with KenLM <d-cite key="heafield2011kenlm"></d-cite>, which builds a compressed trie under a configurable memory ceiling. Note that KenLM uses Modified Kneser-Ney smoothing by default, so CG values under a KenLM backend would differ numerically from those reported here.
