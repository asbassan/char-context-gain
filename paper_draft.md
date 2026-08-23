# Which Characters Need Context?
## Measuring Character-Specific Context Gain in Natural Language and Source Code

**Author:** Amar Bassan  
**Affiliation:** Independent Researcher  
**Date:** 2026-08-21  
**Version:** v3 — adaptive k, coverage-based k_max, KenLM backend  
**Repository:** github.com/asbassan/char-context-gain  
**Zenodo DOI:** [to be assigned]

---

## Abstract

A character bigram model conditions on only one preceding character, making its contextual limitation unusually transparent: any predictive information available from earlier characters is necessarily unavailable to the bigram. This paper asks not whether longer context helps in aggregate — that is already established — but *which individual characters benefit, by how much, and at what context length?*

Beyond character-level language modelling, this question is relevant to context engineering: if the marginal predictive value of additional context differs across targets, then context requirements may be content-dependent rather than uniform across a sequence.

The strongest result occurs in natural-language prose: in Pride and Prejudice, structural characters — punctuation and delimiters — accumulate significantly more predictive benefit from increasing context than lexical characters do, after controlling for character frequency. Shakespeare shows the same directional pattern but weaker robustness. Python source code does not reproduce this trajectory effect; instead it shows an exploratory difference in peak context gain without a structural-vs-lexical difference in context-gain trajectory, suggesting that context dependency differs across corpora and domains.

We define for this study **Character Context Gain** CG_x(k; D) = S_x(1; D) − S_x(k; D), where S_x(k; D) is the **target-character mean surprisal** — the expected negative log-probability of character x given k characters of preceding context in corpus D. The underlying quantity is corpus-conditional; the present study estimates it using count-based n-grams. We measure it per individual character, then classify characters afterward, avoiding the apples-to-oranges comparison that arises from mixing singleton structural symbols with broad alphabetic aggregates.

Across three corpora — tinyshakespeare (NL1), Pride and Prejudice (NL2), Python 3.12 stdlib (Code1) — we fit the regression:

CG_x(k; D) = β₀ + β₁ log₂(k) + β₂ Structural_x + **β₃ [log₂(k) × Structural_x]** + β₄ log₂(Freq_x) + ε

The central quantity is β₃: a positive value means structural symbols accumulate CG faster per doubling of context than lexical symbols do, after controlling for frequency. **NL2 is the primary robust finding**: β₃=+1.131 (cluster-robust p=0.002, permutation p=0.001, stable across coverage thresholds and all functional forms). NL1 is directional but not permutation-significant: β₃=+0.551 (cluster-robust p=0.033, permutation p=0.093). In Code1 β₃=−0.024 (p=0.930) is null for the slope; Code1 shows an exploratory peak-gain difference — structural symbols have higher peak context gain (Mann-Whitney p=0.022, exploratory; does not survive Bonferroni), motivating a distinction between the *trajectory* and *magnitude* of context gain. Mann-Whitney tests are also significant in NL1 (p=0.013). This divergence between slope and peak tests is interpretively informative: they measure different aspects of context dependence.

Python tokenization of Code1 reveals that 33.8% of characters are inside string literals and 8.6% inside comments — only 5.4% are actual Python syntax operators. The primary structural/lexical comparison in Code1 uses only syntax-stratum characters, excluding contaminated punctuation from strings and comments.

The reliable n-gram range is corpus-specific: k ≤ 7 for NL1, k ≤ 8 for NL2, and k ≤ 10 for Code1, determined by a per-symbol coverage diagnostic. All CG values are finite-data estimates subject to n-gram sparsity and smoothing bias.

---

## 1. Introduction

The history of language modelling begins at the character level. Shannon (1951) estimated the entropy of English by asking human subjects to predict successive letters. Modern large language models report aggregate perplexity, which compresses all character and symbol types into a single number. That compression hides a question worth asking directly: do all individual characters benefit equally from longer context, or do structural symbols — punctuation, delimiters, sentence boundaries — require disproportionately more context than ordinary letters?

**Contributions:**

1. A **reusable per-character context profiling method and reference implementation** — `character_context_profile.py` takes any corpus file, estimates S_x(k; D) and CG_x(k; D) per individual character across n-gram orders, and reports context-gain trajectories with per-symbol coverage diagnostics identifying the reliable context range. Measurements are corpus-conditional and estimator-dependent; the tool reports what a Laplace-smoothed n-gram finds in the supplied corpus, not a universal character property.
2. A **coverage-based reliability diagnostic** — per-symbol and global, turning the n-gram sparsity problem from a silent artifact into a reported finding.
3. A **regression finding** — β₃ is positive and permutation-validated in NL2 (cluster-robust p=0.002, permutation p=0.001), positive and directional in NL1 (cluster-robust p=0.033, permutation p=0.093, does not survive the stricter 75% coverage threshold), and null in Code1 (β₃=−0.024, p=0.930). Code1 shows an exploratory peak-gain difference (Mann-Whitney p=0.022, does not survive Bonferroni). Standard errors are cluster-robust; residual independence within clusters is not assumed.
4. A **tokenization preprocessing requirement** — characters inside string literals and comments are not executable syntax operators; we apply Python's tokenizer to restrict the structural analysis to syntax-stratum characters only. Without this step, punctuation in strings and comments contaminates the structural category. In this corpus (Python 3.12 stdlib), 42% of characters fall in string or comment tokens — a corpus-specific figure, not a general Python claim.
5. A **corpus-specificity finding** — the reliable k range, per-symbol k_peak, and context gain magnitudes all vary by corpus structure, not just corpus size.

---

## 2. Background and Related Work

**Overview.** *Prior work measures context dependence either in aggregate — Shannon entropy over all characters, perplexity over all tokens — or studies punctuation's role inside neural attention weights. No prior work combines per-character decomposition, frequency-controlled regression, and corpus comparison across prose and source code. This section situates the paper within that landscape and identifies the specific gap the present study addresses.*

**Character-level entropy.** Shannon (1951) measured H(k) for English prose as a whole. Brown et al. (1992) estimated entropy decay with context. Scheibner et al. (2025) showed character-level conditional entropy continues declining with very long context in large LLMs. These works primarily characterize aggregate entropy or surprisal rather than context-gain trajectories for individual target characters.

**Punctuation and context in neural models.** Razzhigaev et al. (2025) showed punctuation carries disproportionate contextual information in Transformer attention. Chauhan et al. (2026) demonstrated model-dependent computational roles for punctuation. Our work measures context dependency at the raw character level, using only count-based n-gram models without neural model involvement.

**Code naturalness and localness.** Hindle et al. (2012) established that source code is highly predictable under n-gram models. Tu et al. (2014) showed local cache statistics account for much of this predictability — and that syntax tokens (separators, brackets) are particularly predictable locally. Our work extends this: not just *how predictable* but *how much does additional context reduce surprisal* at each position, and whether that marginal gain differs between syntax and identifier characters.

**Surprisal reduction curves.** Measuring how surprisal or entropy changes as context length grows has a long history (Shannon, 1951; Brown et al., 1992) and continues in recent large-model work (Scheibner et al., 2025). Studying surprisal reduction as a function of context length — rather than reporting a single aggregate perplexity — is therefore established methodology, not a novel contribution of the present work.

**Gap.** The present work differs in three ways not found together in prior work: (1) decomposition at the level of individual characters rather than word categories, (2) a frequency-controlled regression comparing structural vs. lexical character types across prose and source code, and (3) Python tokenization stratification to separate syntax operators from identical characters in strings and comments. CG_x(k; D) is defined for this study as an operationalization of these measurements; the underlying operation of subtracting surprisal estimates is not claimed as a new mathematical quantity.

---

## 3. Methodology

**Overview.** *This section defines the measurement framework precisely. For each character in a corpus we estimate how surprising it is at each context length k using a count-based n-gram model, then compute Context Gain as the reduction in surprisal relative to a one-character baseline. A per-symbol coverage diagnostic identifies when n-gram estimates become unreliable at high k, gating which observations enter the regression. The regression itself tests whether structural characters — punctuation and delimiters — accumulate context gain faster than lexical characters after controlling for frequency. All design choices that affect the reported numbers are stated here explicitly.*

### 3.1 Notation and terminology

| Symbol | Definition |
|--------|-----------|
| x | Individual character (e.g., `,`, `a`, `(`) |
| D | Corpus; D_train / D_test non-overlapping splits |
| k | Context length (number of preceding characters) |
| S_x(k; D) | Target-character **mean surprisal** of x at context k in D |
| CG_x(k; D) | Context gain = S_x(1; D) − S_x(k; D) |
| k_peak(x; D) | argmax CG_x(k; D) within reliable k range |
| Coverage_x(k; D) | Fraction of test positions for x where k-char context was seen in training |

**Notation note.** We write S (surprisal) rather than H (entropy). H_x(k) would denote the Shannon entropy of the distribution over next characters given x occurred — a different quantity. S_x(k; D) is the expected negative log-probability *at positions where x occurs*, which is mean surprisal, not entropy. This distinction matters: entropy is over the full outcome distribution; surprisal is conditioned on the observed outcome.

### 3.2 Target-character mean surprisal

For a Laplace-smoothed n-gram trained on D_train with vocabulary size |V|:

```
P(x_t | x_{t-k:t-1}) = (count(x_{t-k:t-1}, x_t) + 1) / (count(x_{t-k:t-1}) + |V|)

S_x(k; D) = E[-log₂ P(x_t | x_{t-k:t-1}) | x_t = x]
           = mean over test positions where x_t = x
```

### 3.3 Per-symbol analysis: individual characters, then classification

**Overview.** *We measure surprisal for every character separately, then assign each a type — structural or lexical — after measurement, not before. This avoids a common aggregation error: comparing a single punctuation character against dozens of letters lumped together would confuse frequency effects with type effects. Measuring first, classifying after, is what makes the regression's frequency control meaningful.*

We measure S_x(k; D) and CG_x(k; D) for **every individual character x** in the corpus vocabulary. We then classify characters as structural or lexical *after* measurement. This avoids the apples-to-oranges problem of comparing singleton punctuation characters (`,`, `.`) against broad aggregates (all vowels, all consonants).

**Natural language classification:**
- Structural: `,` `.` `;` `:` `?` `!` `'` `"` `(` `)` `[` `]` `{` `}`
- Lexical: alphabetic characters (a-z, A-Z), digits, space
- Ambiguous: `\n` (excluded from primary comparison)

**Source code classification (Code1):**
- Structural: Python syntax operators `( ) [ ] { } , : ; = . @ + - * / % & | ^ ~ < > @` — *only* when the character appears in Python's `OP` token category, as determined by the tokenizer
- Lexical: alphabetic and digit characters in `NAME` or `NUMBER` tokens
- Excluded from primary comparison: characters inside `STRING` or `COMMENT` tokens; `NEWLINE`/`INDENT`/`DEDENT` whitespace (ambiguous)

This means `?` and `!` in Python are excluded from the structural set entirely — they are not Python syntax operators and appear almost exclusively inside strings and comments.

### 3.4 Python tokenization

**Overview.** *Python source files contain punctuation in two very different roles: as executable syntax operators (`:` in `if x:`) and as characters inside string literals and comments (`:` in `"hello: world"`). These have different statistical properties and should not be compared as if they were the same. We use Python's built-in tokenizer to assign each character position a stratum so that only genuine syntax-stratum characters enter the structural category in the Code1 analysis.*

We apply Python's `tokenize` module to each `.py` file individually before concatenation, assigning each character position a stratum: `syntax`, `keyword`, `identifier`, `string`, `comment`, `numeric`, `whitespace`, `other`. Only `syntax`-stratum characters are candidates for the structural category in Code1.

Stratum distribution in Code1 test set:
- string: 33.8%
- identifier: 25.3%
- other: 13.7%
- comment: 8.6%
- whitespace: 8.1%
- syntax: 5.4%
- keyword: 4.6%
- numeric: 0.4%

In this corpus, 42% of characters fall inside STRING or COMMENT tokens — a corpus-specific figure that reflects the stdlib's heavy use of docstrings and string constants, not a general property of Python codebases. Strings may contain paths, regex, serialised data, SQL, or program constants — not necessarily natural language. When interpreting punctuation characters specifically as executable syntax operators, tokenizer stratification is required to separate them from identical characters appearing in string literals and comments.

### 3.5 Coverage diagnostic and reliable k range

**Overview.** *At long context lengths, most test sequences were never seen during training. The n-gram then produces near-uniform smoothed probabilities regardless of the true dependency — an artifact of data sparsity, not a genuine signal. This section defines a per-symbol coverage metric that flags when estimates become unreliable, and uses it to determine the maximum k we trust for each corpus. The reliable range turns out to be corpus-specific and cannot be read off from corpus size alone.*

At high k, most test contexts were never seen in training. Laplace smoothing then produces loss near −log₂(1/|V|) regardless of the true dependency — a measurement artifact, not a signal.

```
Coverage_x(k; D) = (test positions for x where k-char context was in D_train)
                   / (total test positions for x)

Global Coverage(k; D) = same but over all characters
```

We report the **reliable k range** as k where global Coverage(k; D) ≥ 50%. We additionally enforce per-symbol coverage ≥ 50% for each (character, k) observation in the regression. Sensitivity to thresholds of 25%, 50%, and 75% is reported in Section 4.1. The reliable range is corpus-specific:

| Corpus | Reliable k range | k=8 coverage |
|--------|-----------------|-------------|
| NL1 tinyshakespeare | k ≤ 7 | 43.9% — sparse |
| NL2 Pride & Prejudice | k ≤ 8 | 61.6% |
| Code1 Python stdlib | k ≤ 10 | 63.1% |

NL2 stays reliable at k=8 despite being the smallest corpus. NL1 breaks at k=8 despite being larger. This is an observed coverage difference between the two corpora; the mechanism — whether it reflects authorial consistency, genre, or something else — is not directly measured here and should not be interpreted as a claim about Austen's prose style.

### 3.6 Confidence intervals

**Overview.** *We report uncertainty on each surprisal estimate using standard intervals. However, peak context gain — the maximum CG across all k values for a character — carries an additional upward bias because we are selecting the largest of several noisy estimates. The more k values a character qualifies for, the more chances it has to record an accidentally high peak. This section explains why peak CG confidence intervals are approximate and why the regression, which avoids peak selection entirely, is the more reliable inferential test.*

For n ≥ 30 test instances: 95% CI on mean surprisal using the CLT (mean ± 1.96 × SE).
For n < 30: bootstrap 95% CI (1,000 resamples). All symbols with n < 30 are excluded from primary analysis and flagged if reported.

**CG_peak CI.** Since CG_x(k) = S_x(1) − S_x(k), the 95% CI on CG is computed as a conservative bound: CI_lo = S_x(1)_lo − S_x(k_peak)_hi and CI_hi = S_x(1)_hi − S_x(k_peak)_lo. This treats the two surprisal estimates as if they were independent, producing CIs that are wider than necessary. Additionally, CG_peak = max_k CG_x(k) is a selected maximum from noisy estimates, introducing selection optimism: E[max(CG_hat)] > max(E[CG_hat]). Characters with more eligible k values have more chances to record an extreme maximum. These CG_peak CIs should therefore be interpreted as approximate and likely anti-conservative for the peak estimate specifically. The regression analysis (Section 4.1), which uses CG at each k independently rather than the selected peak, is not affected by this selection issue.

### 3.7 Statistical tests

**Overview.** *Two tests answer two different questions. Mann-Whitney asks whether structural characters reach a higher maximum context gain than lexical characters — a comparison of peaks. The regression asks whether structural characters accumulate context gain faster per unit of additional context — a comparison of slopes. Both are reported throughout the paper; they are not interchangeable and can legitimately give different answers for the same corpus. Understanding this distinction is essential for interpreting the results in Section 4.*

**Mann-Whitney U**: one-sided test (structural > lexical) on per-symbol CG_peak distributions. Tests whether structural characters have stochastically higher peak context gain than lexical characters.

**Regression**: for each corpus, pooling all (x, k) observations for k ∈ {2, 3, ..., k_max} within the reliable range:

```
CG_x(k; D) = β₀ + β₁ log₂(k) + β₂ Structural_x + β₃ [log₂(k) × Structural_x] + β₄ log₂(Freq_x) + ε
```

Note: k=1 serves as the baseline for computing CG_x(k) = S_x(1; D) − S_x(k; D) but is **not itself a regression observation**. By definition CG_x(1) = 0 for every character; including it would constrain the fit at a deterministically zero outcome. The intercept and β₂ therefore represent extrapolated values at log₂(k)=0 (i.e., k=1), outside the regression's observed range. The key coefficient β₃ is unaffected by this centering choice.

The key coefficient is **β₃**: the additional context-gain rate for structural symbols per unit log₂(k), after controlling for frequency and baseline context growth. β₃ > 0 means structural symbols benefit more per doubling of context, after controlling for log character frequency.

### 3.8 Corpora

| ID | Corpus | Domain | Size | Vocab |
|----|--------|--------|------|-------|
| NL1 | tinyshakespeare | Natural language | ~1.1M chars | 65 |
| NL2 | Pride and Prejudice | Natural language | ~694K chars | 87 |
| Code1 | Python 3.12 stdlib (163 files) | Source code | ~4.6M chars | 164 |

All splits: 80% train / 10% test on raw characters. (A 10% validation split was created but not used in any reported computation; all surprisal estimates and regression observations are from the held-out test portion only.) Context lengths: k ∈ {1, 2, 3, 4, 5, 6, 7, 8}; k=1 serves as the CG baseline; regression observations use k ∈ {2, ..., k_max}.

---

## 4. Results

**Overview.** *The primary result is in Pride and Prejudice (NL2): structural characters have a significantly steeper context-gain trajectory than lexical characters (β₃=+1.131, p=0.002), confirmed by permutation test and stable across coverage thresholds and functional forms. Shakespeare (NL1) shows the same direction but does not survive the permutation test. Python source code (Code1) shows a null trajectory effect but an exploratory peak-gain difference. This section reports the full evidence chain in order: main regression, coverage sensitivity, permutation test, functional form robustness, multiple-testing hierarchy, peak context-gain comparison, and selected per-character profiles. Readers who want only the headline result can read Section 4.1 and the summary paragraph at the end of Section 4.2.*

### 4.1 Regression with cluster-robust standard errors

**Overview.** *This is the primary inferential section. The main regression result is reported for all three corpora, followed by four robustness checks in sequence: coverage threshold sensitivity (does the result hold if we tighten or loosen the coverage filter?), character-label permutation test (is β₃ an artifact of which characters happen to be labelled structural?), functional form comparison (does the result depend on the log₂(k) parametrisation?), and the multiple-testing hierarchy (which results survive Bonferroni correction?). NL2 survives all four checks; NL1 does not; Code1 is null throughout.*

This section reports: (a) the main regression results, (b) coverage threshold sensitivity, (c) character-label permutation test, (d) functional form robustness, and (e) the multiple-testing hierarchy.

OLS with standard errors clustered by character (G = number of unique characters), enforcing per-symbol coverage ≥ 50% for each (character, k) observation included.

| Corpus | n obs | G (chars) | R² | β₃ | SE | 95% CI | p |
|--------|-------|-----------|----|----|-----|--------|---|
| NL1 shakespeare | 272 | 56 | 0.393 | **+0.551** | 0.250 | [+0.049, +1.053] | **0.033** |
| NL2 pride_prej | 310 | 45 | 0.701 | **+1.131** | 0.337 | [+0.452, +1.810] | **0.002** |
| Code1 python | 456 | 77 | 0.571 | −0.024 | 0.262 | [−0.481, +0.561] | 0.930 |

Full coefficient table — NL1 (shakespeare):

| Coefficient | β | SE | 95% CI | p |
|-------------|---|-----|--------|---|
| intercept | −1.300 | 0.486 | [−2.275, −0.326] | 0.010 |
| log₂(k) | −0.242 | 0.145 | [−0.533, +0.048] | 0.101 |
| Structural | −0.369 | 0.490 | [−1.351, +0.614] | 0.455 |
| **log₂(k)×Structural** | **+0.551** | **0.250** | **[+0.049, +1.053]** | **0.033** |
| log₂(Freq) | −0.315 | 0.051 | [−0.416, −0.213] | <0.001 |

Full coefficient table — NL2 (pride_prej):

| Coefficient | β | SE | 95% CI | p |
|-------------|---|-----|--------|---|
| intercept | −0.034 | 0.231 | [−0.499, +0.432] | 0.886 |
| log₂(k) | −0.856 | 0.094 | [−1.046, −0.666] | <0.001 |
| Structural | −2.086 | 0.667 | [−3.430, −0.743] | 0.003 |
| **log₂(k)×Structural** | **+1.131** | **0.337** | **[+0.452, +1.810]** | **0.002** |
| log₂(Freq) | −0.378 | 0.024 | [−0.427, −0.330] | <0.001 |

Full coefficient table — Code1 (python):

| Coefficient | β | SE | 95% CI | p |
|-------------|---|-----|--------|---|
| intercept | −2.315 | 0.551 | [−3.413, −1.217] | <0.001 |
| log₂(k) | −0.133 | 0.153 | [−0.438, +0.171] | 0.386 |
| Structural | +0.202 | 0.565 | [−0.924, +1.327] | 0.722 |
| **log₂(k)×Structural** | **−0.024** | **0.262** | **[−0.481, +0.561]** | **0.930** |
| log₂(Freq) | −0.467 | 0.045 | [−0.557, −0.377] | <0.001 |

The negative β₁ values reflect that high-frequency lexical characters — which dominate the panel — achieve most of their context gain at k=2–3 and then plateau or slightly reverse; the regression fits this average downward trend. This does not mean context is unhelpful: for structural characters in NL2 and NL1, the effective slope is β₁ + β₃. In NL2, β₁ + β₃ = −0.856 + 1.131 = +0.275 (positive), and in NL1, −0.242 + 0.551 = +0.309 (positive) — structural characters' CG continues to grow with log₂(k) within the reliable range, consistent with the observed k_peak values of 4–6 for most structural characters. For Code1, β₃ is null (−0.024, p=0.930), so no differential slope interpretation applies; structural characters in Python reach their CG peak at k=2–3 at roughly the same rate as lexical characters. The negative β₂ (Structural main effect) in NL2 represents the structural-vs-lexical difference extrapolated at log₂(k)=0, outside the regression's observed range; its sign should not be interpreted substantively. The meaningful coefficient is β₃.

**Coverage threshold sensitivity.** The table below shows β₃ at per-symbol coverage thresholds τ = 0.25, 0.50, and 0.75 (Laplace smoothing, cluster-robust SEs throughout). A result is marked significant if p < 0.05.

| Corpus | τ=0.25 | τ=0.50 (main) | τ=0.75 |
|--------|--------|---------------|--------|
| NL1 shakespeare | +0.636** | +0.551* | +0.337 (p=0.18) |
| NL2 pride_prej | +1.142** | +1.131** | +0.982** |
| Code1 python | +0.008 (n.s., p=0.973) | −0.024 (n.s., p=0.930) | −0.059 (n.s., p=0.867) |

NL2 is stable across all three thresholds, providing the most robust evidence for a structural-context interaction. NL1 provides threshold-sensitive evidence: significant at τ=0.25 and τ=0.50 but not at τ=0.75, where the stricter filter removes the higher-k observations for rare structural characters (!, ? with sym_cov 79–83% at k=6). Code1 is consistently not significant at any threshold (all three values from the canonical Python 3.12 pipeline). These findings should not be treated as equally robust: NL2's result survives coverage restriction, NL1's does not.

**Character-label permutation test.** To verify that β₃ is not an artifact of which characters happen to be labelled structural, we ran 10,000 permutations of the structural/lexical label across characters within each corpus, holding the count of structural characters fixed (7 structural out of 56 for NL1; 5 out of 45, i.e., 5-to-40 structural-to-lexical, for NL2; 20 out of 78 for Code1). For each permutation we refitted OLS and recorded β₃. The one-tailed permutation p is the fraction of permutations where β₃_perm ≥ β₃_observed.

| Corpus | β₃ | p cluster-robust | p permutation (N=10,000) |
|--------|----|-----------------|--------------------------|
| NL1 shakespeare | +0.551 | 0.033 | 0.093 |
| NL2 pride_prej | **+1.131** | **0.002** | **0.001** |
| Code1 python | −0.024 | 0.930 | 0.532 |

NL2's permutation p = 0.001: approximately 10 of 10,000 random label assignments produced β₃_perm ≥ β₃_obs (at fixed 5-to-40 structural-to-lexical proportions). This provides additional robustness evidence for the NL2 finding despite having only five structural character clusters. NL1's cluster-robust p = 0.033 does not survive permutation (p = 0.093): with G = 7 structural clusters, the cluster-robust t-distribution approximation is optimistic. NL1 evidence should be treated as directional, not primary. Code1 is null under all specifications (β₃=−0.024, p=0.930, permutation p=0.532).

**Numerical note.** The permutation loop uses OLS point estimates; OLS and cluster-robust OLS produce identical coefficients (they differ only in standard errors). All permutation and functional form checks load the regression panel saved by run_experiment_v3.py (results_v3/panel_v3_*.csv), ensuring exact numerical agreement with the main regression table.

**Functional form comparison.** We fit three versions of the regression, varying only how k enters the model: M1 (log₂k, the main model), M2 (linear k), and M3 (categorical k, one dummy per k value with k=2 as reference, interaction terms likewise). AIC is computed as n·ln(RSS/n) + 2p; lower is better.

| Corpus | Model | n params | AIC | b\_interaction | p\_interaction |
|--------|-------|----------|-----|----------------|----------------|
| NL1 | M1 log₂(k) | 5 | −11.8 | +0.551 | 0.032\* |
| NL1 | M2 linear k | 5 | −14.9 | +0.207 | 0.032\* |
| NL1 | M3 categorical k | 13 | −13.3 | +0.575 (mean) | — |
| NL2 | M1 log₂(k) | 5 | −186.6 | **+1.131** | **0.002\*\*** |
| NL2 | M2 linear k | 5 | −219.8 | +0.375 | 0.001\*\* |
| NL2 | M3 categorical k | 15 | −241.4 | +1.398 (mean) | — |
| Code1 | M1 log₂(k) | 5 | +3.3 | −0.024 | 0.930 |
| Code1 | M2 linear k | 5 | 0.0 | −0.001 | 0.988 |
| Code1 | M3 categorical k | 15 | +5.5 | −0.119 (mean) | — |

For NL2, all six k-specific interaction coefficients in M3 are positive and increase monotonically (k=3: +0.40, k=4: +0.86, k=5: +1.29, k=6: +1.67, k=7: +1.99, k=8: +2.19), and the categorical model fits substantially better than log₂(k) (ΔAIC = −54.8). Log₂(k) is therefore not the best-fitting specification: categorical k and linear k have lower AIC in most corpora, and the log-linear parametrisation should not be interpreted as a validated functional law. What the functional form comparison does confirm is that the structural interaction effect for NL2 is robust to how k is specified: positive and significant under both continuous parameterizations (log₂(k) and linear k); the categorical M3 shows all six interaction estimates positive and increasing monotonically. For NL1, the interaction is positive and significant under both continuous models; the categorical M3 has 4 of 5 coefficients positive (k=4: +0.442, k=5: +0.695, k=6: +0.846, k=7: +0.944) with k=3 slightly negative (−0.053), meaning the structural advantage is negligible at the shortest context and emerges at k=4. Code1 is null under all three functional forms and its categorical M3 interactions are all negative. The choice of log₂(k) in the primary model reflects parsimony (fewer parameters than categorical k, better BIC in NL1), not a claim about the shape of the context-gain curve.

**Primary vs exploratory tests.** This paper reports three regression tests (one per corpus), three Mann-Whitney tests, coverage sensitivity at three thresholds, permutation tests, and functional form comparisons. We designate the regression interaction β₃ in NL2 as the primary inferential analysis. This designation is informed by post-hoc robustness — NL2 shows the most consistent pattern across all checks — and we acknowledge it is not pre-registered. Independent of the robustness results, NL2 also has the widest reliable k range (k≤8 vs NL1's k≤7), uses cohesive prose rather than dramatic dialogue, and was selected as the second natural-language corpus to test generalization beyond Shakespeare. A Bonferroni-adjusted threshold for three simultaneous regression tests is α = 0.05/3 ≈ 0.017; NL2 survives (p=0.002), NL1 does not (p=0.033). Under the same three-test correction for Mann-Whitney: NL1 survives (p=0.013), Code1 does not (p=0.022). All remaining analyses — coverage sensitivity, permutation, functional form — are robustness and exploratory checks, not additional primary hypotheses.

### 4.2 Peak context-gain comparison

**Overview.** *The Mann-Whitney test asks whether structural characters reach a higher peak context gain than lexical characters. The result differs by corpus: significant in NL1, not significant in NL2 (only 5 structural characters, severely underpowered), and exploratory in Code1. The divergence between this test and the regression β₃ result is not a contradiction — they measure different properties — and is explained fully at the end of this section.*

| Corpus | Structural n | Lexical n | Struct median CG | Lex median CG | Ratio | p-value | Interpretation |
|--------|-------------|----------|-----------------|--------------|-------|---------|----------------|
| NL1 shakespeare | 7 | 49 | 1.810 bits | 0.717 bits | 2.52× | 0.013 | survives 3-test Bonferroni |
| NL2 pride_prej | 5 | 40 | 1.933 bits | 1.639 bits | 1.18× | 0.148 | not significant |
| Code1 python | 20 | 57 | 2.158 bits | 1.692 bits | 1.28× | 0.022 | exploratory; does not survive correction |

NL2 is not significant under Mann-Whitney: only 5 structural characters meet n≥30, and their median CG_peak (1.933) is only modestly above the lexical median (1.639). With only five structural characters, the test has limited power; the nonsignificant result therefore does not establish absence of a peak-gain difference. However, the regression shows β₃=+1.131 (p=0.002) for NL2 — the steepest interaction of all three corpora. These tests answer different questions: Mann-Whitney asks whether CG_peak is higher for structural chars; regression asks whether structural chars have a steeper slope of gain with log(k). In NL2 the slopes differ substantially even though the peak values don't separate cleanly.

Code1 shows the reverse pattern: Mann-Whitney gives an exploratory peak-gain difference (p=0.022, structural median 2.158 vs lexical 1.692 bits; does not survive Bonferroni α≈0.017) but regression β₃=−0.024 is null (p=0.930). Structural chars in Python have higher absolute CG but their rate of gain per doubling of context is not steeper than lexical chars — structural chars in Python reach their peak quickly (often at k=2 or k=3) at roughly the same pace as lexical chars.

β₃ is permutation-validated and Bonferroni-significant in NL2 (p=0.002); cluster-robust-significant but not permutation-significant in NL1 (p=0.033); null in Code1 (β₃=−0.024, p=0.930). Mann-Whitney survives Bonferroni in NL1 (p=0.013); in Code1 it is exploratory (p=0.022, does not survive Bonferroni).

### 4.3 Selected per-symbol results — NL1 (tinyshakespeare)

Structural characters with n ≥ 30:

| Char | S_x(1) | CG_peak | 95% CI | k_peak | sym_cov |
|------|--------|---------|--------|--------|---------|
| `!` | 8.225 | 2.481 | [2.27, 2.70] | 6 | 79.8% |
| `?` | 8.027 | 2.183 | [1.99, 2.38] | 6 | 83.1% |
| `'` | 7.149 | 1.880 | [1.68, 2.08] | 4 | 95.2% |
| `;` | 7.495 | 1.810 | [1.61, 2.01] | 5 | 90.9% |
| `.` | 6.336 | 1.347 | [1.21, 1.48] | 4 | 96.5% |
| `,` | 4.957 | 1.183 | [1.08, 1.29] | 3 | 99.6% |
| `:` | 4.651 | 0.820 | [0.56, 1.08] | 2 | 99.9% |

Comparable high-frequency lexical characters:

| Char | S_x(1) | CG_peak | 95% CI | k_peak | sym_cov |
|------|--------|---------|--------|--------|---------|
| `v` | 5.707 | 2.536 | [2.29, 2.79] | 3 | 99.9% |
| `g` | 4.813 | 1.314 | [1.11, 1.52] | 3 | 99.9% |
| `o` | 3.444 | 0.933 | [0.86, 1.01] | 2 | 100% |
| `e` | 2.537 | 0.296 | [0.22, 0.38] | 3 | 98.9% |

Note that rare lexical characters (`v`, `g`) can have higher CG_peak than common structural characters (`,`, `:`). The regression controls for this frequency effect; the raw ratio does not. This is why the regression β₃ is the primary reported finding. *CG_peak CIs in both tables are conservative bounds subject to selection optimism; see Section 3.6.*

### 4.4 Python Code1 — structural chars (syntax stratum only)

Key structural characters with n ≥ 30 in syntax stratum:

| Char | S_x(1) | CG_peak | k_peak | sym_cov |
|------|--------|---------|--------|---------|
| `&` | 12.35 | 5.73 | 6 | 56.1% |
| `%` | 9.05 | 5.60 | 2 | 100% |
| `@` | 11.34 | 5.26 | 7 | 90.8% |
| `{` | 9.78 | 3.35 | 3 | 98.3% |
| `+` | 9.16 | 3.03 | 3 | 94.6% |
| `[` | 8.75 | 2.48 | 4 | 91.4% |
| `=` | 5.99 | 1.93 | 3 | 98.3% |
| `(` | 5.77 | 1.67 | 3 | 97.2% |
| `.` | 4.31 | 1.54 | 3 | 96.7% |
| `:` | 4.40 | 0.78 | 3 | 98.5% |
| `,` | 5.01 | 0.65 | 2 | 99.9% |
| `)` | 5.16 | 0.55 | 2 | 99.9% |

`:` and `,` in Python have *lower* CG_peak than in Shakespeare — consistent with grammar-enforced placement making them more locally predictable. `&`, `%`, `@` have very high CG but low sym_cov at k_peak (56-90%) — treat with caution.

### 4.5 The colon cross-corpus comparison

**Overview.** *The colon `:` appears in both Shakespeare (following speaker names, e.g. `HAMLET:`) and Python (following keywords, e.g. `if x:`). Comparing its context-gain profile across corpora illustrates how the same character can have very different context dependency depending on the surrounding grammar. The result is counterintuitive: the Python colon — despite being grammar-enforced — has slightly lower peak context gain than the Shakespeare colon, consistent with short repeated keyword patterns making it locally predictable without requiring long-range context.*

| Corpus | S_x(1) | CG_peak | k_peak | n_test |
|--------|--------|---------|--------|--------|
| NL1 shakespeare | 4.651 | 0.820 | 2 | 1272 |
| Code1 python (syntax only) | 4.400 | 0.781 | 3 | 2816 |

Python colon (syntax stratum, n=2816) has *slightly lower* CG_peak than Shakespeare colon — the opposite of what a naive grammar-complexity argument would suggest. This is consistent with the hypothesis that grammar-enforced placement after short keywords (`if`, `def`, `for`) makes the 2–3 character context highly informative; however, preceding-context distributions were not directly measured. Shakespeare colons follow varied speaker names: the bigram provides some signal, but the 2-char window is already saturated.

This is a methodologically important result: grammar enforcement does not necessarily produce *longer* context horizons at the character level. The result is consistent with short repeated syntax patterns making some structural symbols more locally predictable than discourse-governed usage.

---

## 5. Discussion

**Overview.** *This section interprets what the results do and do not establish. Section 5.1 explains why the regression and Mann-Whitney tests give divergent signals for the same corpus — they measure different aspects of context dependence — and states the defensible claims precisely. Section 5.2 draws out implications for context engineering. Sections 5.3–5.5 address estimator dependence, tokenization, and future directions. Section 5.7 lists the limitations a reviewer should weigh: small structural sample in NL2, Laplace smoothing, k ≤ 8 range, and selection optimism in CG_peak.*

### 5.1 The primary finding and the test divergence

**Overview.** *This section states the defensible claims precisely and explains why the regression and Mann-Whitney tests give divergent signals for NL2 and Code1 — not because one test is wrong, but because they measure different properties of context dependence. The trajectory-vs-magnitude distinction introduced here is the paper's central conceptual contribution beyond the empirical finding: a character can have high absolute context gain (magnitude) without accumulating that gain more steeply than lexical characters (trajectory), and these two properties can dissociate across corpora.*

The central claim is not "structural symbols have 3× higher CG." That claim was produced by an unweighted mean over singletons vs. broad aggregates, and it does not survive the per-symbol analysis. The defensible claim, with cluster-robust inferential statistics, is:

> **NL2 (primary robust finding):** Structural symbols have a robustly steeper rate of context-gain per unit log-context than lexical symbols after controlling for character frequency (β₃=+1.131, cluster-robust p=0.002, permutation p=0.001; stable across coverage thresholds and functional forms).
>
> **NL1 (directional):** Same sign (β₃=+0.551, cluster-robust p=0.033) but does not survive the character-label permutation test (p=0.093) or the 75% coverage threshold. Treated as corroborating NL2, not an independent replication.
>
> **Code1 (domain contrast, exploratory):** Structural symbols have higher peak context gain (Mann-Whitney p=0.022, exploratory; structural median 2.158 vs lexical 1.692 bits) but the rate of accumulation with log-context is not steeper (β₃=−0.024, p=0.930). This motivates a distinction between the magnitude and trajectory of context gain.

The Mann-Whitney and regression divergences are not contradictions — they measure different things. Mann-Whitney on CG_peak asks: does the maximum context gain achieved differ between types? Regression on (character, k) pairs asks: does the *slope* of gain with log₂(k) differ between types?

NL2 reversal (Mann-Whitney not significant, regression significant): the 5 structural chars have steep k-slopes but similar CG_peak to the field of 40 lexical chars. The peak distribution is not separable at n=5 vs n=40. The regression interaction is identified using only 5 structural character clusters; the 310 (character, k) observations provide more observations per cluster but do not increase the number of independent structural units.

The character-label permutation test (Section 4.1) gives p = 0.001 for NL2: approximately 10 of 10,000 random label shuffles produced β₃_perm ≥ β₃_obs (at fixed 5-to-40 structural-to-lexical proportions). The monotone increase in categorical-k interaction coefficients (k=3: +0.40 to k=8: +2.19) further confirms the finding does not depend on the log₂(k) functional form. The NL2 regression result is the paper's strongest supported finding.

Code1 reversal (Mann-Whitney exploratory, regression null): structural chars in Python reach their CG_peak at k=2 or k=3, as do most lexical chars. The *absolute level* is higher for structural chars (median 2.158 vs 1.692 bits) but the *rate of ascent* is not steeper (β₃=−0.024, p=0.930). This suggests a useful distinction: context dependence has both a *magnitude* component (total gain achieved) and an *acquisition-range* component (how steeply gain accumulates with additional context). Python structural symbols show high magnitude with short acquisition range; natural-language structural symbols show high magnitude with longer acquisition range. Whether a symbol needs more context is therefore a two-dimensional question.

### 5.2 Implications for context engineering

**Overview.** *Context engineering treats context as a budget — what to retrieve, retain, compress, or truncate. Our results suggest that the useful amount of context may itself depend on what is being predicted, not just on sequence length. This section draws out that implication, connects it to the trajectory-vs-magnitude distinction, and describes how the character_context_profile.py tool operationalizes the measurement. All claims are framed as motivating future work, not as demonstrated engineering outcomes.*

Context engineering typically treats context as a budget to be selected, retrieved, compressed, or truncated at the sequence level. Our results suggest that the predictive value of additional context can vary substantially at a finer granularity: different target characters exhibit different context-gain trajectories, and the structural-versus-lexical pattern itself varies across corpora and domains. The broader implication is that context demand may be content-dependent rather than solely sequence-length-dependent — moving the question from *how much context can the model accept?* toward *how much context does this prediction benefit from?*

Character Context Gain does not by itself prescribe an adaptive context policy, but it provides a transparent measurement framework for studying context sufficiency and saturation at the character level. Future systems could test whether analogous signals at token, syntactic, or semantic levels can inform adaptive context allocation, retrieval depth, or context compression. We make no claim that the current n-gram measurements translate directly to neural model behaviour; the relationship between n-gram CG and transformer-level context dependency is an open empirical question noted in Section 5.4 (Estimator-dependence).

The trajectory-vs-magnitude distinction from Section 5.1 has a direct analogue in context engineering terms. A character whose CG saturates at k=3 signals that, for that prediction position and corpus, context beyond three characters provides no additional measured benefit — a context-sufficiency signal. A character whose CG continues growing through k=7 signals that longer context remains informative for that position. The `character_context_profile.py` tool operationalizes this directly: given any corpus it produces per-character saturation curves that could, in principle, inform position-level context budgeting. Whether analogous saturation signals exist at the token or semantic level, and whether they transfer from n-gram to neural estimators, are open empirical questions — but the measurement pattern is defined and reproducible.

### 5.3 Corpus-specific variation

**Overview.** *Every number in this paper is implicitly tagged with a specific corpus. The reliable k range, baseline surprisal, and context gain magnitudes all vary across corpora in ways that cannot be predicted from corpus size alone — NL2 is the smallest corpus but has the widest reliable range. This section reinforces that the findings describe these three corpora, not natural language or source code in general, and that the coverage diagnostic must be recomputed for any new corpus.*

Every measurement carries an implicit (D). The reliable k range, S_x(1; D), and CG_x(k_peak; D) all vary by corpus structure. NL2 stays reliable to k=8 despite being the smallest corpus — a structure effect, not a size effect. The coverage diagnostic must be computed per corpus; it cannot be read off from corpus size alone.

### 5.4 Estimator-dependence of context gain

**Overview.** *All CG values in this paper are estimates from a Laplace-smoothed n-gram model. A different estimator — a Transformer, a Kneser-Ney model — would produce different numbers. This section explains why no guaranteed relationship holds between n-gram CG and neural CG, and why the n-gram framework was chosen despite this: it is transparent, reproducible, and does not conflate model capacity with corpus properties. The structural/lexical separation found here may be larger or smaller under a neural estimator — that is an open empirical question.*

Because CG_x(k; D) is defined as S_x(1; D) − S_x(k; D) using a Laplace-smoothed n-gram, the measured values are estimator-dependent. A model with higher capacity would produce different surprisal estimates, and the difference at k=1 vs k=k* could be larger or smaller. No upper or lower bound relationship between n-gram CG and Transformer CG follows from either model being a better estimator of the true distribution. The n-gram measurements describe what a Laplace-smoothed count model finds in these corpora at k ≤ 8. Whether a more expressive model would show larger or smaller structural/lexical separation is an open empirical question.

### 5.5 Python tokenization as a required preprocessing step

The Python stratum distribution (33.8% string, 8.6% comment) means that 42% of characters in Code1 appear inside STRING or COMMENT tokens rather than executable syntax. Strings may contain paths, regex, serialised data, SQL, or program constants — not necessarily natural language. Character-level analyses that classify `?`, `!`, or `-` as structural code operators without tokenization stratification are measuring an uninterpretable mix of syntactic and non-syntactic occurrence contexts. When interpreting punctuation specifically as executable syntax operators, tokenizer stratification is required to separate them from identical characters appearing in strings and comments.

### 5.6 Future direction: tokenizer context interaction

A natural follow-on question is whether the context-dependency differences identified here — higher CG_x for structural symbols, controlling for frequency — are preserved or altered when those characters are absorbed into subword tokens by BPE merging. A subword model may represent the same underlying dependency at a coarser granularity. Whether it does is empirically testable using the S_x(k; D) framework as a character-level baseline. We make no claim here about any tokenizer design decision; the present work provides the measurement apparatus for that future study.

### 5.7 Limitations and future work

**Overview.** *Six limitations are disclosed explicitly: Laplace smoothing is suboptimal at high k; NL1 is directional not primary; NL2 has only five structural characters making Mann-Whitney underpowered; CG_peak has selection optimism that particularly affects the Code1 exploratory result; the reliable range ends at k=8; and the regression does not model character-level random intercepts. These are known limitations of the current study scope, not hidden failures — each is assessed for its impact on the reported findings.*

1. **Alternative smoothing**: Laplace smoothing is suboptimal at high k. Per-symbol coverage filtering (≥50%) and frequency control (β₄) address the primary Laplace bias mechanisms, and the coverage sensitivity analysis (Section 4.1) shows the NL2 finding is robust across all three thresholds. A true interpolated Kneser-Ney test would require recursive backoff (k → k-1 → ... → unigram), which is beyond the scope of this study. Flat-to-unigram discounting, tested but not reported as a primary result, is not a valid KN surrogate because at k ≥ 5 the unigram backoff dominates and collapses context-dependent surprisal estimates.

2. **NL1 directional, not primary**: The NL1 β₃ finding is positive and cluster-robust at τ=0.50 (p=0.033) but does not survive the character-label permutation test (p=0.093, Section 4.1). With only G=7 structural character clusters, the cluster-robust t-approximation is optimistic. Coverage sensitivity adds to this picture: significance disappears at τ=0.75 (p=0.18). NL1 should be read as directionally consistent with NL2, not as independent replication.

3. **NL2 power**: Only 5 structural characters cleared n≥30 in Pride and Prejudice, making Mann-Whitney underpowered. The regression is validated by the permutation test (p=0.001): despite only 5 structural clusters, the observed β₃=+1.131 is in the top 0.1% of the permutation null distribution. A larger corpus would provide a cleaner Mann-Whitney test.

4. **CG_peak selection optimism**: CG_peak = max_k CG_x(k) is a maximum selected from noisy estimates, so E[max(CG_hat)] > max(E[CG_hat]) — the peak is upward-biased relative to the true maximum. Characters with more eligible k values (higher coverage) have more opportunities to attain an extreme peak. This selection optimism affects the Mann-Whitney test (which uses CG_peak) more than the regression (which uses CG at each k directly). For Code1, whose primary positive result is the Mann-Whitney peak-gain finding (p=0.022), this limitation is especially relevant. The cleanest resolution would be to bootstrap the entire curve, selecting the peak within each bootstrap replicate to propagate selection uncertainty into the CI and test statistic.

5. **k range and corpus scale**: The reliable range is corpus-specific: k ≤ 7 for NL1, k ≤ 8 for NL2, and k ≤ 10 for Code1, determined by the per-symbol coverage probe (Section 3.5). For corpora exceeding ~100M characters, the in-memory n-gram frequency tables used here become memory-prohibitive; KenLM (Heafield, 2011) is the natural replacement, building a compressed trie under a fixed memory budget — though it uses Modified Kneser-Ney smoothing, so CG values would not be numerically identical to those reported here. A neural character-level model could further extend the reliable range beyond k = 10.

6. **Regression random effects**: The regression uses cluster-robust standard errors (clustered by character), which accounts for within-character correlation across k values. A character random-intercept model would additionally capture character-level variance in baseline surprisal.

---

## 6. Conclusion

We applied a per-symbol, corpus-conditional framework for measuring character-level context dependency across three corpora with Python tokenization stratification. The primary robust finding is in NL2 (Pride and Prejudice): structural symbols have a robustly steeper context-gain trajectory than lexical symbols after controlling for character frequency (β₃=+1.131, cluster-robust p=0.002, permutation p=0.001), stable across coverage thresholds and functional forms. NL1 (Shakespeare) shows the same direction (β₃=+0.549, cluster-robust p=0.033) but does not survive the character-label permutation test (p=0.093) and should be read as directional corroboration, not an independent replication. In Code1, the trajectory effect is null (β₃=−0.024, p=0.930); instead, structural symbols show an exploratory peak-gain difference (Mann-Whitney p=0.022, does not survive Bonferroni), motivating a distinction between the magnitude and trajectory of context dependence.

Python tokenization reveals that 42% of Python stdlib characters are inside strings or comments; tokenizer stratification is required when interpreting punctuation specifically as executable syntax operators. The colon cross-corpus result shows grammar-enforced placement can produce lower context gain than discourse-governed usage, potentially because short repeated keywords are more informative than varied speaker names.

The reliable n-gram range is corpus-specific and must be measured per corpus. All measurements are finite-data estimates subject to n-gram sparsity and smoothing bias; the n-gram framework is chosen because it provides a transparent count-based estimate of corpus-local dependency structure without conflating model capacity with corpus properties.

More broadly, these results suggest that context requirement is not necessarily uniform across prediction targets, motivating future work on context-sufficiency signals for adaptive retrieval, compression, and context allocation.

---

## References

- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*.
- Shannon, C. E. (1951). Prediction and entropy of printed English. *Bell System Technical Journal*.
- Brown, P. F., et al. (1992). An estimate of an upper bound for the entropy of English. *Computational Linguistics*.
- Hindle, A., et al. (2012). On the naturalness of software. *ICSE 2012*.
- Tu, Z., Su, Z., & Devanbu, P. (2014). On the localness of software. *FSE 2014*.
- Scheibner, C., Smith, L. M., & Bialek, W. (2025). Large language models and the entropy of English. *arXiv:2512.24969*.
- Razzhigaev, A., et al. (2025). LLM-Microscope: Uncovering the hidden role of punctuation in context memory of Transformers. *Findings of NAACL 2025*.
- Chauhan, S., et al. (2026). Punctuations and predicates in language models. *Findings of EACL 2026*.
- Heafield, K. (2011). KenLM: Faster and smaller language model queries. *Proceedings of the Sixth Workshop on Statistical Machine Translation*.

---

## Appendix — Reproducibility

**Code:** github.com/asbassan/which-characters-need-context  
**Run:** `python run_experiment_v3.py --db experiment_cache.db` (no GPU, ~15 min) → `cross_corpus_v3.csv`, `panel_v3_*.csv`  
**Validation:** `python run_validations.py` (~15 min) → `results_robustness/permutation_test.csv`, `functional_form.csv`  
**Outputs:** peaks_v3_*.csv, cross_corpus_v3.csv, context_curves_v3.png  
**Seed:** 42 | **Smoothing:** Laplace (add-1) | **Min n to report:** 30  
**Dependencies:** numpy, matplotlib, pandas, scipy

The experiment script is idempotent: completed (corpus, k) pairs are stored in an SQLite cache and skipped on restart, so interrupted runs resume from where they stopped. The k range for each corpus is derived automatically from a coverage probe (Section 3.5) rather than hardcoded; this produced k ≤ 7 for NL1, k ≤ 8 for NL2, and k ≤ 10 for Code1.

**Standalone profiler.** `character_context_profile.py` is a self-contained tool that takes any corpus and produces per-character context-gain trajectories with coverage diagnostics:

```
# Natural language corpus
python character_context_profile.py corpus.txt --max-k 8 --output profile.csv

# Python source code (with tokenizer stratification)
python character_context_profile.py src/ --python --max-k 8
```

Output includes, for each qualifying character: surprisal at k=1 (bigram baseline), context gain CG(k) at each k, per-k coverage, and the k at which peak gain was observed within the reliable range. The tool reports what a Laplace-smoothed n-gram finds in the supplied corpus; it does not establish hard context requirements. Dependencies: numpy only.

**Scalability note.** The n-gram counting step uses in-memory frequency tables sufficient for the corpora here (1–5M characters). For corpora exceeding ~100M characters, the counting backend should be replaced with KenLM (Heafield, 2011), which builds a compressed trie under a configurable memory ceiling and is callable from Python. All downstream steps — SQLite aggregation, coverage probe, adaptive k selection, regression — are unchanged; only the per-(context, character) probability lookup is delegated to KenLM. Note that KenLM uses Modified Kneser-Ney smoothing by default, so CG values under a KenLM backend would differ numerically from those reported here.

**k=1 exclusion audit.** A post-hoc audit (`audit_k1.py`) confirmed that k=1 is absent from all regression panels: min(k)=2 in both NL1 (272 obs) and NL2 (310 obs), with zero k=1 rows. The headline observation counts (272/310/456) therefore reflect k ∈ {2,...,k_max} only, as stated in Section 3.7.
