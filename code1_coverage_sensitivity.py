"""
Code1 coverage sensitivity at tau=0.25, 0.50, 0.75.
Uses Python 3.12 stdlib (canonical), same MAX_BOOT_SAMPLES and split logic as run_experiment_v2.py.
"""

import numpy as np
import pandas as pd
import os
import io
import tokenize as py_tokenize
import keyword as kw_mod
from collections import defaultdict, Counter
from pathlib import Path
from scipy import stats

SEED = 42
np.random.seed(SEED)
MAX_BOOT_SAMPLES  = 5000
MIN_N             = 30
K_VALUES          = [2, 3, 4, 5, 6, 7, 8]
K_ALL             = [1] + K_VALUES
THRESHOLDS        = [0.25, 0.50, 0.75]
PY_SYNTAX_CHARS   = set('()[]{},:;=.@+-*/%&|^~<>')
AMBIGUOUS_CHARS   = {' ', '\n', '\t'}
RESULTS_DIR       = Path('results_robustness')
RESULTS_DIR.mkdir(exist_ok=True)

def get_python_char_strata(code):
    strata = ['other'] * len(code)
    line_starts = [0]
    for i, ch in enumerate(code):
        if ch == '\n':
            line_starts.append(i + 1)
    def flat(row, col):
        idx = line_starts[row - 1] + col if row - 1 < len(line_starts) else len(code)
        return min(idx, len(code))
    try:
        tokens = list(py_tokenize.generate_tokens(io.StringIO(code).readline))
    except py_tokenize.TokenError:
        return strata
    for tok in tokens:
        tok_type, tok_string = tok.type, tok.string
        start = flat(*tok.start)
        end   = min(flat(*tok.end), len(code))
        if tok_type == py_tokenize.NAME:
            st = 'keyword' if kw_mod.iskeyword(tok_string) else 'identifier'
        elif tok_type == py_tokenize.NUMBER:  st = 'numeric'
        elif tok_type == py_tokenize.STRING:  st = 'string'
        elif tok_type == py_tokenize.COMMENT: st = 'comment'
        elif tok_type == py_tokenize.OP:      st = 'syntax'
        elif tok_type in (py_tokenize.NEWLINE, py_tokenize.NL,
                          py_tokenize.INDENT, py_tokenize.DEDENT): st = 'whitespace'
        else: st = 'other'
        for i in range(start, end):
            if i < len(code):
                strata[i] = st
    return strata

def classify_char_code(ch, stratum):
    if stratum in ('string', 'comment'):         return 'in_string_or_comment'
    if ch in PY_SYNTAX_CHARS and stratum == 'syntax': return 'structural'
    if stratum in ('whitespace',) or ch in AMBIGUOUS_CHARS: return 'ambiguous'
    if stratum in ('identifier', 'keyword', 'numeric') or ch.isalpha() or ch.isdigit(): return 'lexical'
    return 'other'

def build_ngram(text, k):
    counts = defaultdict(Counter)
    for i in range(k, len(text)):
        counts[text[i-k:i]][text[i]] += 1
    return counts

def compute_mean_surprisal_code(counts, k, test_text, vocab_size, test_strata):
    result = defaultdict(lambda: {'sum_surp': 0.0, 'n_sum': 0, 'n': 0, 'ctx_hit': 0})
    for i in range(k, len(test_text)):
        ch  = test_text[i]
        ctx = test_text[i-k:i]
        typ = classify_char_code(ch, test_strata[i])
        if typ in ('other', 'in_string_or_comment', 'ambiguous'):
            continue
        ctx_counts = counts.get(ctx, {})
        ctx_total  = sum(ctx_counts.values())
        prob = (ctx_counts.get(ch, 0) + 1) / (ctx_total + vocab_size)
        r = result[ch]
        if r['n_sum'] < MAX_BOOT_SAMPLES:
            r['sum_surp'] += -np.log2(prob)
            r['n_sum'] += 1
        r['n'] += 1
        if ctx_counts:
            r['ctx_hit'] += 1
    return {ch: dict(v) for ch, v in result.items()}

def global_coverage(train_text, test_text, k):
    seen  = set(train_text[i-k:i] for i in range(k, len(train_text)))
    hits  = sum(1 for i in range(k, len(test_text)) if test_text[i-k:i] in seen)
    total = len(test_text) - k
    return hits / total if total > 0 else 0.0

def clustered_ols(y, X, group_ids):
    n, p    = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    coeffs  = XtX_inv @ X.T @ y
    resid   = y - X @ coeffs
    unique_groups = np.unique(group_ids)
    G = len(unique_groups)
    B = np.zeros((p, p))
    for g in unique_groups:
        mask  = group_ids == g
        score = X[mask].T @ resid[mask]
        B    += np.outer(score, score)
    correction = (G / (G - 1)) * ((n - 1) / (n - p))
    V   = correction * XtX_inv @ B @ XtX_inv
    se  = np.sqrt(np.maximum(np.diag(V), 0))
    df  = G - 1
    t   = coeffs / np.where(se > 0, se, np.inf)
    pv  = 2 * stats.t.sf(np.abs(t), df=df)
    return coeffs, se, pv

# ── Load Code1 corpus ────────────────────────────────────────────────────────
stdlib_path = os.path.dirname(os.__file__)
print(f'Python stdlib: {stdlib_path}')

text_parts, strata_parts, n_files = [], [], 0
for fname in sorted(os.listdir(stdlib_path)):
    if not fname.endswith('.py'):
        continue
    try:
        with open(os.path.join(stdlib_path, fname), 'r', encoding='utf-8', errors='replace') as f:
            code = f.read()
    except Exception:
        continue
    text_parts.append(code)
    strata_parts.append(get_python_char_strata(code))
    text_parts.append('\n')
    strata_parts.append(['whitespace'])
    n_files += 1

full_text   = ''.join(text_parts)
full_strata = [st for part in strata_parts for st in part]
print(f'Tokenized {n_files} files  ({len(full_text):,} chars)')

n    = len(full_text)
n_tr = int(0.8 * n)
n_vl = int(0.1 * n)
train, test = full_text[:n_tr], full_text[n_tr+n_vl:]
train_strata, test_strata = full_strata[:n_tr], full_strata[n_tr+n_vl:]
vocab_size = len(set(full_text))
train_freq = Counter(train)
total_train = len(train)
print(f'train={len(train):,}  test={len(test):,}  vocab={vocab_size}')

# ── Global coverage ──────────────────────────────────────────────────────────
print('\nGlobal coverage:')
K_MAX = 1
for k in K_VALUES:
    cov = global_coverage(train, test, k)
    print(f'  k={k}: {100*cov:.1f}%')
    if cov >= 0.50:
        K_MAX = k
print(f'Reliable K_MAX = {K_MAX}')

# ── Compute mean surprisal per (char, k) ────────────────────────────────────
print('\nComputing S(k) per char...')
S_k = {}
for k in K_ALL:
    print(f'  k={k}...', end=' ', flush=True)
    counts = build_ngram(train, k)
    S_k[k] = compute_mean_surprisal_code(counts, k, test, vocab_size, test_strata)
    print(f'done ({len(S_k[k])} chars)')

# ── Build panel and run regression at each tau ───────────────────────────────
results = []
S1 = S_k[1]

for tau in THRESHOLDS:
    rows = []
    for ch, e1 in S1.items():
        if e1['n'] < MIN_N:
            continue
        if ch in PY_SYNTAX_CHARS:
            sym_type = 'structural'
        elif ch.isalpha() or ch.isdigit():
            sym_type = 'lexical'
        else:
            continue
        s1  = e1['sum_surp'] / e1['n_sum']
        freq = train_freq.get(ch, 1) / total_train
        is_struct = 1 if sym_type == 'structural' else 0
        for k in [kk for kk in K_VALUES if kk <= K_MAX]:
            ek = S_k[k].get(ch)
            if ek is None or ek['n'] < MIN_N:
                continue
            sym_cov = ek['ctx_hit'] / ek['n']
            if sym_cov < tau:
                continue
            sk = ek['sum_surp'] / ek['n_sum']
            rows.append({
                'ch': ch, 'type': sym_type, 'k': k,
                'log2k': np.log2(k),
                'CG': s1 - sk,
                'Structural': is_struct,
                'log2k_x_S': np.log2(k) * is_struct,
                'logFreq': np.log2(max(freq, 1e-10)),
            })

    if not rows:
        print(f'tau={tau}: EMPTY panel')
        continue

    df = pd.DataFrame(rows)
    char_to_id = {c: i for i, c in enumerate(np.unique(df['ch'].values))}
    group_ids  = np.array([char_to_id[c] for c in df['ch'].values])
    y  = df['CG'].values
    X  = np.column_stack([np.ones(len(y)), df['log2k'].values,
                          df['Structural'].values, df['log2k_x_S'].values,
                          df['logFreq'].values])
    coeffs, se, pv = clustered_ols(y, X, group_ids)
    b3, p3 = coeffs[3], pv[3]
    n_struct = int(df['Structural'].sum())
    sig = '**' if p3<0.01 else '*' if p3<0.05 else 'n.s.'
    print(f'tau={tau}: n={len(df)} obs  G={df["ch"].nunique()} chars  n_struct_obs={n_struct}  b3={b3:+.4f}  p={p3:.4f} {sig}')
    results.append({'tau': tau, 'n_obs': len(df), 'n_chars': df['ch'].nunique(),
                    'b3': round(b3, 5), 'p': round(p3, 5)})

out_path = RESULTS_DIR / 'code1_coverage_sensitivity.csv'
pd.DataFrame(results).to_csv(out_path, index=False)
print(f'\nSaved: {out_path}')
