"""
Option B Full Experiment: Which Characters Need Context?

Key changes from v1:
  1. S_x(k; D) = mean surprisal per INDIVIDUAL CHARACTER, not per broad class
  2. Python tokenization: stratify Code1 into code/string/comment
  3. Per-symbol coverage diagnostic
  4. CLT-based CIs (n>=30) + bootstrap (n<30) on S_x and CG_x
  5. Mann-Whitney U test: structural vs lexical CG_peak distributions
  6. Regression: CG_x,k = b0 + b1*log2(k) + b2*Structural + b3*[log2(k)*Structural] + b4*logFreq
  7. Newline/space moved to 'ambiguous' bin for Code1
  8. ? and ! in Python excluded from syntax structural set
"""

import matplotlib
matplotlib.use('Agg')

import tokenize as py_tokenize
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import urllib.request
import string
import os
import json
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
from scipy import stats

SEED = 42
np.random.seed(SEED)
RESULTS_DIR = Path('results_v2')
RESULTS_DIR.mkdir(exist_ok=True)
MIN_N = 30          # minimum observations to report a symbol
BOOTSTRAP_N = 1000  # resamples for rare symbols (n < MIN_N)
MAX_BOOT_SAMPLES = 5000  # cap stored losses per symbol for memory

print(f'Results dir : {RESULTS_DIR.resolve()}')

# ── Python tokenization ────────────────────────────────────────────────────────

def get_python_char_strata(code):
    """
    Classify each character position in a Python source file as:
      'syntax'     — Python OP token (actual delimiter/operator)
      'keyword'    — Python keyword
      'identifier' — NAME token that is not a keyword
      'string'     — inside a string literal
      'comment'    — inside a comment
      'numeric'    — NUMBER token
      'whitespace' — NEWLINE, NL, INDENT, DEDENT, spaces
      'other'      — anything else
    Returns list of stratum strings, one per character.
    """
    import keyword as kw_mod
    strata = ['other'] * len(code)

    # Build line-start offsets for (row, col) -> flat index
    line_offsets = [0]
    for ch in code:
        if ch == '\n':
            line_offsets.append(line_offsets[-1] + 1)
        else:
            line_offsets[-1] += 1
    # Recompute properly
    line_starts = [0]
    for i, ch in enumerate(code):
        if ch == '\n':
            line_starts.append(i + 1)

    def flat(row, col):
        # row is 1-indexed in tokenize
        idx = line_starts[row - 1] + col if row - 1 < len(line_starts) else len(code)
        return min(idx, len(code))

    try:
        tokens = list(py_tokenize.generate_tokens(io.StringIO(code).readline))
    except py_tokenize.TokenError:
        return strata

    for tok in tokens:
        tok_type   = tok.type
        tok_string = tok.string
        s_row, s_col = tok.start
        e_row, e_col = tok.end
        start = flat(s_row, s_col)
        end   = flat(e_row, e_col)
        end   = min(end, len(code))

        if tok_type == py_tokenize.NAME:
            st = 'keyword' if kw_mod.iskeyword(tok_string) else 'identifier'
        elif tok_type == py_tokenize.NUMBER:
            st = 'numeric'
        elif tok_type == py_tokenize.STRING:
            st = 'string'
        elif tok_type == py_tokenize.COMMENT:
            st = 'comment'
        elif tok_type == py_tokenize.OP:
            st = 'syntax'
        elif tok_type in (py_tokenize.NEWLINE, py_tokenize.NL,
                          py_tokenize.INDENT, py_tokenize.DEDENT):
            st = 'whitespace'
        else:
            st = 'other'

        for i in range(start, end):
            if i < len(code):
                strata[i] = st

    return strata


def load_python_stdlib_with_strata(stdlib_path):
    """Load each .py file separately, tokenize, then concatenate."""
    text_parts   = []
    strata_parts = []
    n_files = 0

    for fname in sorted(os.listdir(stdlib_path)):
        if not fname.endswith('.py'):
            continue
        fpath = os.path.join(stdlib_path, fname)
        try:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                code = f.read()
        except Exception:
            continue
        s = get_python_char_strata(code)
        text_parts.append(code)
        strata_parts.append(s)
        # file separator
        text_parts.append('\n')
        strata_parts.append(['whitespace'])
        n_files += 1

    full_text   = ''.join(text_parts)
    full_strata = [st for part in strata_parts for st in part]
    print(f'  Tokenized {n_files} Python stdlib files ({len(full_text):,} chars)')
    assert len(full_text) == len(full_strata), "strata length mismatch"
    return full_text, full_strata


# ── Corpus loading ─────────────────────────────────────────────────────────────

def download_text(url, filename):
    if not Path(filename).exists():
        print(f'  Downloading {filename}...')
        try:
            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            print(f'  Download failed: {e}'); return None
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()

print('\n=== Loading corpora ===')
nl1 = download_text(
    'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    'corpus_shakespeare.txt')
nl2 = download_text(
    'https://www.gutenberg.org/files/1342/1342-0.txt',
    'corpus_pride_prejudice.txt')
if nl2:
    start = nl2.find('It is a truth')
    end   = nl2.rfind('End of the Project Gutenberg')
    nl2   = nl2[start:end] if start > 0 else nl2

import os as _os
stdlib_path = _os.path.dirname(_os.__file__)
print(f'  Loading Python stdlib from {stdlib_path}...')
code1, code1_strata = load_python_stdlib_with_strata(stdlib_path)

CORPORA = {}
STRATA  = {}
if nl1:   CORPORA['shakespeare']    = nl1;   STRATA['shakespeare']    = None
if nl2:   CORPORA['pride_prej']     = nl2;   STRATA['pride_prej']     = None
if code1: CORPORA['python_stdlib']  = code1; STRATA['python_stdlib']  = code1_strata

print('Corpus sizes:')
for name, text in CORPORA.items():
    print(f'  {name:20s}: {len(text):>10,} chars')


# ── Symbol taxonomy ────────────────────────────────────────────────────────────
# We classify individual characters, not broad classes.
# Classification is corpus-aware for Code1.

# NL structural: punctuation that encodes discourse/sentence structure
NL_STRUCTURAL_CHARS = set(',.;:?!\'"()[]{}')

# Python syntax operators/delimiters (from Python grammar)
# Excludes ? and ! which are not Python operators (they appear in strings/comments)
PY_SYNTAX_CHARS = set('()[]{},:;=.@+-*/%&|^~<>@')

# Chars that are ambiguous (structural in some analyses, not others)
AMBIGUOUS_CHARS = {' ', '\n', '\t'}

def classify_char_nl(ch):
    """Classify for natural language corpora."""
    if ch in NL_STRUCTURAL_CHARS:
        return 'structural'
    if ch in AMBIGUOUS_CHARS:
        return 'ambiguous'
    if ch.isalpha() or ch.isdigit():
        return 'lexical'
    return 'other'

def classify_char_code(ch, stratum):
    """
    Classify for Code1.
    Uses Python tokenizer stratum to distinguish syntax chars from
    the same chars appearing inside strings/comments.
    """
    if stratum in ('string', 'comment'):
        return 'in_string_or_comment'
    if ch in PY_SYNTAX_CHARS and stratum == 'syntax':
        return 'structural'
    if stratum in ('whitespace',) or ch in AMBIGUOUS_CHARS:
        return 'ambiguous'
    if stratum in ('identifier', 'keyword', 'numeric') or ch.isalpha() or ch.isdigit():
        return 'lexical'
    return 'other'


# ── Text splits ────────────────────────────────────────────────────────────────

def make_splits(text, strata=None, train_frac=0.8, val_frac=0.1):
    n    = len(text)
    n_tr = int(train_frac * n)
    n_vl = int(val_frac * n)
    sp = {
        'train'     : text[:n_tr],
        'val'       : text[n_tr:n_tr+n_vl],
        'test'      : text[n_tr+n_vl:],
        'vocab_size': len(set(text)),
    }
    if strata is not None:
        sp['test_strata'] = strata[n_tr+n_vl:]
        sp['train_strata']= strata[:n_tr]
    return sp

TEXT_SPLITS = {
    name: make_splits(text, STRATA[name])
    for name, text in CORPORA.items()
}
print('\nSplits:')
for name, s in TEXT_SPLITS.items():
    print(f'  {name:20s}: vocab={s["vocab_size"]:3d}  '
          f'train={len(s["train"]):>9,}  test={len(s["test"]):>8,}')


# ── N-gram functions ───────────────────────────────────────────────────────────

def build_ngram(text, k):
    counts = defaultdict(Counter)
    for i in range(k, len(text)):
        counts[text[i-k:i]][text[i]] += 1
    return counts

def per_symbol_surprisal(counts, k, test_text, vocab_size,
                         test_strata=None, is_code=False):
    """
    Returns {char: {'losses': list, 'n': int, 'ctx_hit': int}} per individual char.
    ctx_hit = number of positions where the k-char context was seen in training.
    """
    result = defaultdict(lambda: {'losses': [], 'n': 0, 'ctx_hit': 0})

    for i in range(k, len(test_text)):
        ch  = test_text[i]
        ctx = test_text[i-k:i]

        # Determine character type
        if is_code and test_strata:
            st  = test_strata[i]
            typ = classify_char_code(ch, st)
        else:
            typ = classify_char_nl(ch)

        if typ in ('other', 'in_string_or_comment'):
            continue
        if typ == 'ambiguous':
            continue   # exclude ambiguous from primary comparison

        ctx_counts = counts.get(ctx, {})
        ctx_total  = sum(ctx_counts.values())
        prob = (ctx_counts.get(ch, 0) + 1) / (ctx_total + vocab_size)
        surprisal = -np.log2(prob)

        r = result[ch]
        if len(r['losses']) < MAX_BOOT_SAMPLES:
            r['losses'].append(surprisal)
        r['n'] += 1
        if ctx_counts:
            r['ctx_hit'] += 1

    return {ch: dict(v) for ch, v in result.items()}


# ── Coverage (global, for display) ────────────────────────────────────────────

def global_coverage(train_text, test_text, k):
    seen  = set(train_text[i-k:i] for i in range(k, len(train_text)))
    hits  = sum(1 for i in range(k, len(test_text)) if test_text[i-k:i] in seen)
    total = len(test_text) - k
    return hits / total if total > 0 else 0.0


# ── Confidence intervals ───────────────────────────────────────────────────────

def ci_mean(losses, n_full):
    """
    For n_full >= 30: CLT-based 95% CI on mean.
    For n_full <  30: bootstrap 95% CI.
    Returns (mean, lo, hi).
    """
    arr = np.array(losses)
    m   = arr.mean()
    if n_full >= 30:
        se = arr.std(ddof=1) / np.sqrt(len(arr))
        return m, m - 1.96*se, m + 1.96*se
    else:
        boot = [np.random.choice(arr, size=len(arr), replace=True).mean()
                for _ in range(BOOTSTRAP_N)]
        return m, np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def clustered_ols(y, X, group_ids):
    """
    OLS with cluster-robust (sandwich) standard errors.
    Clusters are indexed by integer group_ids (same length as y).
    Returns: coeffs, se_robust, ci_lo, ci_hi, p_values, r2
    """
    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    coeffs  = XtX_inv @ X.T @ y
    resid   = y - X @ coeffs

    unique_groups = np.unique(group_ids)
    G = len(unique_groups)
    B = np.zeros((p, p))
    for g in unique_groups:
        mask   = group_ids == g
        score  = X[mask].T @ resid[mask]
        B     += np.outer(score, score)

    correction = (G / (G - 1)) * ((n - 1) / (n - p))
    V   = correction * XtX_inv @ B @ XtX_inv
    se  = np.sqrt(np.maximum(np.diag(V), 0))

    df_resid = G - 1
    t_stats  = coeffs / np.where(se > 0, se, np.inf)
    p_vals   = 2 * stats.t.sf(np.abs(t_stats), df=df_resid)
    t_crit   = stats.t.ppf(0.975, df=df_resid)
    ci_lo    = coeffs - t_crit * se
    ci_hi    = coeffs + t_crit * se

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return coeffs, se, ci_lo, ci_hi, p_vals, r2


SYM_COV_THRESHOLD = 0.50   # per-symbol coverage required for each (x,k) regression row


# ── Main experiment loop ───────────────────────────────────────────────────────

K_VALUES = [2, 3, 4, 5, 6, 7, 8]
K_ALL    = [1] + K_VALUES

# S[corpus][k_label][char] = {'losses', 'n', 'ctx_hit'}
S   = {name: {} for name in CORPORA}
COV = {}

for corpus_name, splits in TEXT_SPLITS.items():
    print(f'\n=== Corpus: {corpus_name} ===')
    vocab_size   = splits['vocab_size']
    is_code      = (corpus_name == 'python_stdlib')
    test_strata  = splits.get('test_strata')

    # Global coverage
    print('  Global coverage:')
    cov_row = {1: 1.0}
    for k in K_VALUES:
        cov = global_coverage(splits['train'], splits['test'], k)
        cov_row[k] = cov
        flag = ' << sparse' if cov < 0.50 else ''
        print(f'    k={k}: {100*cov:.1f}%{flag}')
    COV[corpus_name] = cov_row

    # Per-symbol surprisal at each k
    for k in K_ALL:
        print(f'  k={k}...', end=' ', flush=True)
        counts = build_ngram(splits['train'], k)
        S[corpus_name][f'k={k}'] = per_symbol_surprisal(
            counts, k, splits['test'], vocab_size,
            test_strata=test_strata, is_code=is_code)
        n_chars = len(S[corpus_name][f'k={k}'])
        print(f'done ({n_chars} chars tracked)')

print('\nSaved raw surprisals in memory.')

# ── Per-symbol CG_peak with CIs ───────────────────────────────────────────────

def compute_symbol_peaks(S_corpus, corpus_name, reliable_kmax):
    """
    For each individual char x, compute:
      S_x(1), CG_x(k) for k in reliable range,
      k_peak, CG_peak with CI, per-symbol coverage at k_peak.
    """
    reliable_ks = [k for k in K_ALL if k <= reliable_kmax]
    is_code     = (corpus_name == 'python_stdlib')
    rows = []

    all_chars = set(S_corpus.get('k=1', {}).keys())

    for ch in sorted(all_chars):
        entry_1 = S_corpus.get('k=1', {}).get(ch)
        if not entry_1 or entry_1['n'] < MIN_N:
            continue

        s1, s1_lo, s1_hi = ci_mean(entry_1['losses'], entry_1['n'])

        best_cg, best_k, best_cg_lo, best_cg_hi = float('-inf'), None, None, None

        for k in reliable_ks[1:]:
            entry_k = S_corpus.get(f'k={k}', {}).get(ch)
            if not entry_k or entry_k['n'] < MIN_N:
                continue
            sk, sk_lo, sk_hi = ci_mean(entry_k['losses'], entry_k['n'])
            cg = s1 - sk
            if cg > best_cg:
                best_cg    = cg
                best_k     = k
                best_cg_lo = s1_lo - sk_hi   # conservative CI
                best_cg_hi = s1_hi - sk_lo

        if best_k is None:
            continue

        # Per-symbol coverage at k_peak
        entry_kpeak = S_corpus.get(f'k={best_k}', {}).get(ch, {})
        n_kpeak     = entry_kpeak.get('n', 0)
        ctx_kpeak   = entry_kpeak.get('ctx_hit', 0)
        sym_cov     = ctx_kpeak / n_kpeak if n_kpeak > 0 else 0.0

        ls = 1 - best_cg / s1 if s1 > 0 else np.nan

        # Classification
        if is_code:
            # For Code1 we need the stratum — use majority stratum at k=1
            entry_code = S_corpus.get('k=1', {}).get(ch, {})
            n_code = entry_code.get('n', 0)
            # Check whether char is in PY_SYNTAX_CHARS
            if ch in PY_SYNTAX_CHARS:
                sym_type = 'structural'
            elif ch.isalpha() or ch.isdigit():
                sym_type = 'lexical'
            else:
                sym_type = 'other'
        else:
            if ch in NL_STRUCTURAL_CHARS:
                sym_type = 'structural'
            elif ch.isalpha() or ch.isdigit() or ch == ' ':
                sym_type = 'lexical'
            else:
                sym_type = 'other'

        if sym_type == 'other':
            continue

        rows.append({
            'char'      : repr(ch),
            'char_raw'  : ch,
            'type'      : sym_type,
            'S_x1'      : round(s1, 4),
            'S_x1_lo'   : round(s1_lo, 4),
            'S_x1_hi'   : round(s1_hi, 4),
            'n_test'    : entry_1['n'],
            'CG_peak'   : round(best_cg, 4),
            'CG_lo'     : round(best_cg_lo, 4),
            'CG_hi'     : round(best_cg_hi, 4),
            'k_peak'    : best_k,
            'sym_cov'   : round(sym_cov, 3),
            'LS'        : round(ls, 4) if not np.isnan(ls) else np.nan,
        })

    return pd.DataFrame(rows).sort_values('CG_peak', ascending=False)


# Reliable K_MAX per corpus
RELIABLE_K_MAX = {}
for name, cov_row in COV.items():
    kmax = 1
    for k in K_VALUES:
        if cov_row.get(k, 0) >= 0.50:
            kmax = k
    RELIABLE_K_MAX[name] = kmax
print(f'\nReliable K_MAX: {RELIABLE_K_MAX}')

PEAKS = {}
for corpus_name in CORPORA:
    print(f'\nComputing peaks: {corpus_name}...')
    PEAKS[corpus_name] = compute_symbol_peaks(
        S[corpus_name], corpus_name, RELIABLE_K_MAX[corpus_name])
    df = PEAKS[corpus_name]
    PEAKS[corpus_name].to_csv(RESULTS_DIR / f'peaks_v2_{corpus_name}.csv', index=False)
    print(df[['char','type','n_test','S_x1','CG_peak','CG_lo','CG_hi','k_peak','sym_cov']].to_string(index=False))


# ── GO / NO-GO with Mann-Whitney ──────────────────────────────────────────────

print('\n' + '='*70)
print('GO / NO-GO -- Mann-Whitney U on structural vs lexical CG_peak distributions')
print('='*70)

all_decisions = []
for corpus_name, df in PEAKS.items():
    s_cg = df[df['type']=='structural']['CG_peak'].dropna().values
    l_cg = df[df['type']=='lexical'   ]['CG_peak'].dropna().values

    if len(s_cg) == 0 or len(l_cg) == 0:
        print(f'  {corpus_name}: insufficient data')
        continue

    u_stat, p_val = stats.mannwhitneyu(s_cg, l_cg, alternative='greater')
    median_s = np.median(s_cg)
    median_l = np.median(l_cg)
    ratio    = median_s / median_l if median_l > 0 else float('inf')
    go       = (p_val < 0.05) and (ratio > 1.2)
    all_decisions.append(go)

    print(f'\n  {corpus_name}:')
    print(f'    Structural (n={len(s_cg)}): median CG = {median_s:.3f} bits')
    print(f'    Lexical    (n={len(l_cg)}): median CG = {median_l:.3f} bits')
    print(f'    Ratio (medians): {ratio:.2f}x  |  Mann-Whitney p = {p_val:.4f}')
    print(f'    -> {"GO" if go else "NO-GO"}')

print()
if all(all_decisions):
    print('>>> FINAL: GO across all corpora (Mann-Whitney p<0.05, ratio>1.2)')
elif any(all_decisions):
    print('>>> FINAL: PARTIAL GO')
else:
    print('>>> FINAL: NO-GO')


# ── Regression: CG_x,k = b0 + b1*log2(k) + b2*Structural + b3*(log2(k)*Structural) + b4*logFreq ──

print('\n' + '='*70)
print('Regression: CG_x,k = b0 + b1*log2(k) + b2*Struct + b3*(log2(k)*Struct) + b4*logFreq')
print('Key coefficient: b3 (structural benefit per unit log-context)')
print('='*70)

for corpus_name, splits in TEXT_SPLITS.items():
    df = PEAKS[corpus_name]
    if df.empty:
        continue

    # Build char frequencies from train
    train_freq = Counter(splits['train'])
    total_train = len(splits['train'])

    rows_reg = []
    reliable_kmax = RELIABLE_K_MAX[corpus_name]
    for _, row in df.iterrows():
        ch  = row['char_raw']
        typ = row['type']
        if typ not in ('structural', 'lexical'):
            continue
        freq      = train_freq.get(ch, 1) / total_train
        is_struct = 1 if typ == 'structural' else 0

        s1_entry = S[corpus_name].get('k=1', {}).get(ch)
        if not s1_entry or s1_entry['n'] < MIN_N:
            continue
        s1 = np.mean(s1_entry['losses'])

        for k in [k_ for k_ in K_VALUES if k_ <= reliable_kmax]:
            sk_entry = S[corpus_name].get(f'k={k}', {}).get(ch)
            if not sk_entry or sk_entry['n'] < MIN_N:
                continue
            # Per-symbol coverage filter: exclude (x,k) pairs below threshold
            sym_cov_k = sk_entry['ctx_hit'] / sk_entry['n'] if sk_entry['n'] > 0 else 0.0
            if sym_cov_k < SYM_COV_THRESHOLD:
                continue
            sk = np.mean(sk_entry['losses'])
            cg = s1 - sk
            rows_reg.append({
                'CG'        : cg,
                'log2k'     : np.log2(k),
                'Structural': is_struct,
                'log2k_x_S' : np.log2(k) * is_struct,
                'logFreq'   : np.log2(max(freq, 1e-10)),
                'char_id'   : ch,
            })

    if not rows_reg:
        continue

    reg_df   = pd.DataFrame(rows_reg)
    reg_df.to_csv(RESULTS_DIR / f'panel_{corpus_name}.csv', index=False)

    X_mat    = reg_df[['log2k','Structural','log2k_x_S','logFreq']].values
    y        = reg_df['CG'].values
    X_mat    = np.column_stack([np.ones(len(X_mat)), X_mat])

    # Integer group IDs for clustering by character
    unique_chars = np.unique(reg_df['char_id'].values)
    char_to_id   = {c: i for i, c in enumerate(unique_chars)}
    group_ids    = np.array([char_to_id[c] for c in reg_df['char_id'].values])
    G            = len(unique_chars)

    coeffs, se_rob, ci_lo, ci_hi, p_vals, r2 = clustered_ols(y, X_mat, group_ids)

    coef_names = ['intercept', 'log2(k)', 'Structural', 'log2(k)*Structural', 'log2(Freq)']
    print(f'\n  {corpus_name}  n={len(y)} obs, G={G} chars (clusters), R2={r2:.3f}')
    print(f'  {"":25s} {"b":>8s}  {"SE":>7s}  {"95% CI":^18s}  {"p":>7s}')
    print(f'  {"-"*72}')
    for nm, b, se, lo, hi, pv in zip(coef_names, coeffs, se_rob, ci_lo, ci_hi, p_vals):
        sig = ' ***' if pv < 0.001 else ' **' if pv < 0.01 else ' *' if pv < 0.05 else ''
        print(f'  {nm:25s} {b:+8.4f}  {se:7.4f}  [{lo:+7.4f}, {hi:+7.4f}]  {pv:7.4f}{sig}')
    b3, se3, lo3, hi3, p3 = coeffs[3], se_rob[3], ci_lo[3], ci_hi[3], p_vals[3]
    print(f'\n  >> b3 = {b3:+.4f}  SE={se3:.4f}  95%CI=[{lo3:+.4f},{hi3:+.4f}]  p={p3:.4f}'
          f'  {"POSITIVE & SIG" if b3 > 0 and p3 < 0.05 else "positive, n.s." if b3 > 0 else "NEGATIVE"}')


# ── Cross-corpus comparison table ─────────────────────────────────────────────

print('\n' + '='*70)
print('Cross-corpus CG_peak and k_peak per individual character')
cross_rows = []
for corpus_name, df in PEAKS.items():
    for _, row in df.iterrows():
        cross_rows.append({
            'corpus' : corpus_name,
            'char'   : row['char'],
            'type'   : row['type'],
            'n_test' : row['n_test'],
            'S_x1'   : row['S_x1'],
            'CG_peak': row['CG_peak'],
            'k_peak' : row['k_peak'],
        })

cross_df = pd.DataFrame(cross_rows)
cross_df.to_csv(RESULTS_DIR / 'cross_corpus_v2.csv', index=False)

print('\nMedian CG_peak by type across corpora:')
print(cross_df.groupby(['corpus','type'])['CG_peak'].agg(['median','mean','count']).round(3).to_string())


# ── Publication figures ────────────────────────────────────────────────────────

STRUCTURAL_CHARS_DISPLAY = list(',.;:?!\'"()[]{}')
LEXICAL_CHARS_DISPLAY    = list('aeiouAEIOUbcdfgBCDF0123456')

def plot_cg_curves(S_corpus, corpus_name, reliable_kmax, target_chars, ax):
    """Plot CG_x(k) curves for selected individual chars, with shaded sparsity."""
    struct_chars = [c for c in target_chars if c in NL_STRUCTURAL_CHARS or c in PY_SYNTAX_CHARS]
    lexical_chars= [c for c in target_chars if c.isalpha()]

    colors_s = cm.Reds( np.linspace(0.35, 0.9, max(len(struct_chars),1)))
    colors_l = cm.Blues(np.linspace(0.35, 0.9, max(len(lexical_chars),1)))

    for ch, col in zip(struct_chars, colors_s):
        e1 = S_corpus.get('k=1',{}).get(ch)
        if not e1 or e1['n'] < MIN_N: continue
        s1 = np.mean(e1['losses'])
        cgs = []
        for k in K_VALUES:
            ek = S_corpus.get(f'k={k}',{}).get(ch)
            cgs.append(s1 - np.mean(ek['losses']) if ek and ek['n']>=MIN_N else np.nan)
        ax.plot(K_VALUES, cgs, '-o', color=col, label=repr(ch), lw=2)

    for ch, col in zip(lexical_chars, colors_l):
        e1 = S_corpus.get('k=1',{}).get(ch)
        if not e1 or e1['n'] < MIN_N: continue
        s1 = np.mean(e1['losses'])
        cgs = []
        for k in K_VALUES:
            ek = S_corpus.get(f'k={k}',{}).get(ch)
            cgs.append(s1 - np.mean(ek['losses']) if ek and ek['n']>=MIN_N else np.nan)
        ax.plot(K_VALUES, cgs, '--s', color=col, label=repr(ch), lw=1.5, alpha=0.8)

    ax.axvspan(reliable_kmax+0.5, max(K_VALUES)+0.5, color='gray', alpha=0.12)
    ax.axvline(reliable_kmax+0.5, color='gray', ls=':', lw=1.5)
    ax.axhline(0, color='black', ls='-', lw=0.5)
    ax.set_xlabel('Context length k')
    ax.set_ylabel('CG_x(k; D) bits')
    ax.set_title(f'{corpus_name}\nPer-symbol context gain (red=structural, blue=lexical)')
    ax.legend(fontsize=6, ncol=3)
    ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(len(CORPORA), 1, figsize=(12, 5*len(CORPORA)))
if len(CORPORA) == 1: axes = [axes]

for ax, corpus_name in zip(axes, CORPORA):
    plot_cg_curves(S[corpus_name], corpus_name,
                   RELIABLE_K_MAX[corpus_name],
                   STRUCTURAL_CHARS_DISPLAY + LEXICAL_CHARS_DISPLAY[:8],
                   ax)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'context_curves_v2.png', dpi=200, bbox_inches='tight')
plt.savefig(RESULTS_DIR / 'context_curves_v2.pdf', bbox_inches='tight')
print(f'\nSaved: context_curves_v2.png / .pdf')


# ── Python stratification summary ──────────────────────────────────────────────

if 'python_stdlib' in CORPORA:
    print('\n' + '='*70)
    print('Python Code1 -- Stratum distribution in test set')
    strat_counter = Counter(code1_strata[len(TEXT_SPLITS['python_stdlib']['train'])+
                                          len(TEXT_SPLITS['python_stdlib']['val']):])
    total = sum(strat_counter.values())
    for st, cnt in sorted(strat_counter.items(), key=lambda x: -x[1]):
        print(f'  {st:20s}: {cnt:>9,}  ({100*cnt/total:.1f}%)')


# ── Save config ────────────────────────────────────────────────────────────────

config = {
    'version'            : 'v2',
    'model'              : 'n-gram count-based + Laplace smoothing',
    'unit'               : 'per-individual-character (not per class)',
    'k_values'           : K_ALL,
    'reliable_k_max'     : RELIABLE_K_MAX,
    'min_n_to_report'    : MIN_N,
    'bootstrap_n'        : BOOTSTRAP_N,
    'python_tokenization': True,
    'python_excludes'    : '? and ! excluded from Python structural set',
    'nl_structural_chars': sorted(NL_STRUCTURAL_CHARS),
    'py_syntax_chars'    : sorted(PY_SYNTAX_CHARS),
    'corpora'            : {
        'shakespeare'  : 'karpathy/char-rnn tinyshakespeare',
        'pride_prej'   : 'Project Gutenberg #1342',
        'python_stdlib': 'Python 3.12 stdlib local',
    }
}
with open(RESULTS_DIR / 'config_v2.json', 'w') as f:
    json.dump(config, f, indent=2)

print('\nResults:')
for f in sorted(RESULTS_DIR.iterdir()):
    print(f'  {f.name:45s} {f.stat().st_size:>8,} bytes')
print('\nDone.')
