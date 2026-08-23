"""
Validation checks for "Which Characters Need Context?"

CHECK 1: Character-label permutation test (10,000 permutations)
  - Shuffle structural/lexical labels at the character level, keeping ratio fixed
  - Refit OLS b3 each time; compare observed b3 to null distribution
  - Key for NL2: only 5 structural vs 40 lexical chars

CHECK 2: Functional form comparison — log2(k) vs k vs categorical k
  - Fit three models, compare AIC/BIC
  - Verify b3 sign and significance survive across functional forms
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
RESULTS_DIR = Path('results_robustness')
RESULTS_DIR.mkdir(exist_ok=True)

MIN_N             = 30
SYM_COV_THRESHOLD = 0.50
K_VALUES          = [2, 3, 4, 5, 6, 7, 8]
K_ALL             = [1] + K_VALUES
N_PERMUTATIONS    = 10000
MAX_BOOT_SAMPLES  = 5000  # cap matches run_experiment_v2.py for identical accumulation

NL_STRUCTURAL_CHARS = set(',.;:?!\'"()[]{}')
PY_SYNTAX_CHARS     = set('()[]{},:;=.@+-*/%&|^~<>')
AMBIGUOUS_CHARS     = {' ', '\n', '\t'}

print('=== Validation Script ===')
print(f'N_PERMUTATIONS    = {N_PERMUTATIONS}')
print(f'SYM_COV_THRESHOLD = {SYM_COV_THRESHOLD}')
print(f'MIN_N             = {MIN_N}')


# ── Corpus loading ──────────────────────────────────────────────────────────────

def load_file(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()

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

def load_python_stdlib(stdlib_path):
    text_parts, strata_parts, n_files = [], [], 0
    for fname in sorted(os.listdir(stdlib_path)):
        if not fname.endswith('.py'):
            continue
        try:
            with open(os.path.join(stdlib_path, fname), 'r',
                      encoding='utf-8', errors='replace') as f:
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
    print(f'  Tokenized {n_files} Python stdlib files ({len(full_text):,} chars)')
    assert len(full_text) == len(full_strata)
    return full_text, full_strata


# ── Character classification ────────────────────────────────────────────────────

def classify_char_nl(ch):
    if ch in NL_STRUCTURAL_CHARS: return 'structural'
    if ch in AMBIGUOUS_CHARS:     return 'ambiguous'
    if ch.isalpha() or ch.isdigit(): return 'lexical'
    return 'other'

def classify_char_code(ch, stratum):
    if stratum in ('string', 'comment'):         return 'in_string_or_comment'
    if ch in PY_SYNTAX_CHARS and stratum == 'syntax': return 'structural'
    if stratum in ('whitespace',) or ch in AMBIGUOUS_CHARS: return 'ambiguous'
    if stratum in ('identifier', 'keyword', 'numeric') or ch.isalpha() or ch.isdigit(): return 'lexical'
    return 'other'


# ── N-gram and lean surprisal (running sum, no loss storage) ───────────────────

def build_ngram(text, k):
    counts = defaultdict(Counter)
    for i in range(k, len(text)):
        counts[text[i-k:i]][text[i]] += 1
    return counts

def compute_mean_surprisal(counts, k, test_text, vocab_size,
                           test_strata=None, is_code=False):
    """Returns {char: {sum_surp, n_sum, n, ctx_hit}}.
    sum_surp / n_sum is the mean (capped at MAX_BOOT_SAMPLES to match
    run_experiment_v2.py). n and ctx_hit use full counts for coverage.
    """
    result = defaultdict(lambda: {'sum_surp': 0.0, 'n_sum': 0, 'n': 0, 'ctx_hit': 0})
    for i in range(k, len(test_text)):
        ch  = test_text[i]
        ctx = test_text[i-k:i]
        if is_code and test_strata:
            typ = classify_char_code(ch, test_strata[i])
        else:
            typ = classify_char_nl(ch)
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


# ── OLS helpers ─────────────────────────────────────────────────────────────────

def ols_b3(y, X):
    """Fast OLS: return only coefficient index 3 (b3)."""
    try:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        return coeffs[3]
    except Exception:
        return np.nan

def ols_full(y, X):
    """OLS: return (coeffs, RSS)."""
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    resid  = y - X @ coeffs
    return coeffs, float(np.sum(resid ** 2))

def clustered_ols(y, X, group_ids):
    """OLS + cluster-robust sandwich SEs. Returns (coeffs, se, p_vals, r2)."""
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
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return coeffs, se, pv, r2


# ── Build regression panel ──────────────────────────────────────────────────────

def build_panel(S_k, reliable_kmax, is_code, train_freq, total_train):
    """Build (char, k, CG, Structural, logFreq) panel from mean surprisals."""
    rows = []
    S1 = S_k.get(1, {})
    for ch, e1 in S1.items():
        if e1['n'] < MIN_N:
            continue
        s1 = e1['sum_surp'] / e1['n_sum']
        if is_code:
            if   ch in PY_SYNTAX_CHARS:          sym_type = 'structural'
            elif ch.isalpha() or ch.isdigit():    sym_type = 'lexical'
            else:                                  continue
        else:
            if   ch in NL_STRUCTURAL_CHARS:       sym_type = 'structural'
            elif ch.isalpha() or ch.isdigit():    sym_type = 'lexical'
            else:                                  continue
        freq      = train_freq.get(ch, 1) / total_train
        is_struct = 1 if sym_type == 'structural' else 0
        for k in [kk for kk in K_VALUES if kk <= reliable_kmax]:
            ek = S_k.get(k, {}).get(ch)
            if ek is None or ek['n'] < MIN_N:
                continue
            if ek['ctx_hit'] / ek['n'] < SYM_COV_THRESHOLD:
                continue
            sk  = ek['sum_surp'] / ek['n_sum']
            cg  = s1 - sk
            rows.append({
                'ch'        : ch,
                'type'      : sym_type,
                'k'         : k,
                'log2k'     : np.log2(k),
                'CG'        : cg,
                'Structural': is_struct,
                'log2k_x_S' : np.log2(k) * is_struct,
                'logFreq'   : np.log2(max(freq, 1e-10)),
            })
    return pd.DataFrame(rows)


# ── CHECK 1: Permutation test ───────────────────────────────────────────────────

def run_permutation_test(panel, corpus_name):
    if panel.empty:
        print(f'  {corpus_name}: empty panel, skip')
        return None

    char_info = panel.groupby('ch').first()[['type', 'Structural', 'logFreq']].reset_index()
    chars        = char_info['ch'].values
    struct_flags = char_info['Structural'].values
    n_struct     = int(struct_flags.sum())
    n_total      = len(chars)
    n_lex        = n_total - n_struct

    print(f'\n  {corpus_name}: {n_total} chars, {n_struct} structural, {n_lex} lexical')

    # Observed b3 with cluster-robust SE
    char_to_id = {c: i for i, c in enumerate(chars)}
    group_ids  = np.array([char_to_id[c] for c in panel['ch'].values])
    y          = panel['CG'].values
    X_obs = np.column_stack([
        np.ones(len(y)),
        panel['log2k'].values,
        panel['Structural'].values,
        panel['log2k_x_S'].values,
        panel['logFreq'].values,
    ])
    coeffs_obs, se_obs, p_obs, r2_obs = clustered_ols(y, X_obs, group_ids)
    b3_obs = coeffs_obs[3]
    print(f'  Observed b3 = {b3_obs:+.4f}  SE = {se_obs[3]:.4f}  p (cluster-robust) = {p_obs[3]:.4f}')

    # Pre-extract per-row arrays that don't change across permutations
    row_char_ids = np.array([char_to_id[c] for c in panel['ch'].values])
    row_log2k    = panel['log2k'].values
    row_logfreq  = panel['logFreq'].values
    ones         = np.ones(len(y))

    # Permutation loop
    rng     = np.random.default_rng(SEED)
    b3_null = np.zeros(N_PERMUTATIONS)
    base_labels = np.zeros(n_total)
    base_labels[:n_struct] = 1

    for i in range(N_PERMUTATIONS):
        perm_labels = base_labels.copy()
        rng.shuffle(perm_labels)
        row_struct = perm_labels[row_char_ids]
        X_perm = np.column_stack([ones, row_log2k, row_struct, row_log2k * row_struct, row_logfreq])
        b3_null[i] = ols_b3(y, X_perm)

    valid   = b3_null[~np.isnan(b3_null)]
    p_perm  = float(np.mean(valid >= b3_obs))
    print(f'  Permutation p (b3 >= observed, one-tailed): {p_perm:.4f}')
    print(f'  Null b3: mean={np.mean(valid):+.4f}  sd={np.std(valid):.4f}  '
          f'95th pctile={np.percentile(valid, 95):+.4f}')

    return {
        'corpus'           : corpus_name,
        'n_chars'          : n_total,
        'n_structural'     : n_struct,
        'n_lexical'        : n_lex,
        'b3_observed'      : round(b3_obs, 5),
        'p_cluster_robust' : round(p_obs[3], 5),
        'p_permutation'    : round(p_perm, 5),
        'null_b3_mean'     : round(float(np.mean(valid)), 5),
        'null_b3_sd'       : round(float(np.std(valid)), 5),
        'null_b3_95th'     : round(float(np.percentile(valid, 95)), 5),
        'n_permutations'   : N_PERMUTATIONS,
    }


# ── CHECK 2: Functional form comparison ─────────────────────────────────────────

def run_functional_form(panel, corpus_name):
    if panel.empty:
        print(f'  {corpus_name}: empty panel, skip')
        return []

    char_to_id = {c: i for i, c in enumerate(np.unique(panel['ch'].values))}
    group_ids  = np.array([char_to_id[c] for c in panel['ch'].values])
    y          = panel['CG'].values
    n          = len(y)
    struct     = panel['Structural'].values
    logfreq    = panel['logFreq'].values
    ones       = np.ones(n)

    results = []

    # Model 1: log2(k)  ── current model
    log2k = panel['log2k'].values
    X1    = np.column_stack([ones, log2k, struct, log2k * struct, logfreq])
    c1, se1, p1, r2_1 = clustered_ols(y, X1, group_ids)
    _, rss1 = ols_full(y, X1)
    p1_c = X1.shape[1]
    aic1 = n * np.log(rss1 / n) + 2 * p1_c
    bic1 = n * np.log(rss1 / n) + p1_c * np.log(n)
    results.append(dict(corpus=corpus_name, model='log2(k)', n_params=p1_c,
                        AIC=round(aic1, 2), BIC=round(bic1, 2),
                        b_interaction=round(c1[3], 5), se_interaction=round(se1[3], 5),
                        p_interaction=round(p1[3], 5), R2=round(r2_1, 4)))
    sig1 = '***' if p1[3]<0.001 else '**' if p1[3]<0.01 else '*' if p1[3]<0.05 else 'n.s.'
    print(f'  M1 log2(k)    : b_int={c1[3]:+.4f}  SE={se1[3]:.4f}  p={p1[3]:.4f} {sig1}  AIC={aic1:.1f}  BIC={bic1:.1f}')

    # Model 2: k (linear)
    k_raw = panel['k'].values.astype(float)
    X2    = np.column_stack([ones, k_raw, struct, k_raw * struct, logfreq])
    c2, se2, p2, r2_2 = clustered_ols(y, X2, group_ids)
    _, rss2 = ols_full(y, X2)
    p2_c = X2.shape[1]
    aic2 = n * np.log(rss2 / n) + 2 * p2_c
    bic2 = n * np.log(rss2 / n) + p2_c * np.log(n)
    results.append(dict(corpus=corpus_name, model='k_linear', n_params=p2_c,
                        AIC=round(aic2, 2), BIC=round(bic2, 2),
                        b_interaction=round(c2[3], 5), se_interaction=round(se2[3], 5),
                        p_interaction=round(p2[3], 5), R2=round(r2_2, 4)))
    sig2 = '***' if p2[3]<0.001 else '**' if p2[3]<0.01 else '*' if p2[3]<0.05 else 'n.s.'
    print(f'  M2 k (linear) : b_int={c2[3]:+.4f}  SE={se2[3]:.4f}  p={p2[3]:.4f} {sig2}  AIC={aic2:.1f}  BIC={bic2:.1f}')

    # Model 3: categorical k (dummy variables, reference = k_min)
    k_vals_present = sorted(panel['k'].unique())
    k_ref          = k_vals_present[0]
    dummies = np.column_stack([(panel['k'].values == kk).astype(float) for kk in k_vals_present[1:]])
    inter   = dummies * struct.reshape(-1, 1)
    X3      = np.column_stack([ones, dummies, struct, inter, logfreq])
    n_dummies = dummies.shape[1]
    p3_c    = X3.shape[1]

    try:
        c3, se3, p3, r2_3 = clustered_ols(y, X3, group_ids)
        _, rss3 = ols_full(y, X3)
        aic3 = n * np.log(rss3 / n) + 2 * p3_c
        bic3 = n * np.log(rss3 / n) + p3_c * np.log(n)
        # Interaction coefficients: after intercept, n_dummies, Structural = index (1+n_dummies+1)..
        inter_start  = 1 + n_dummies + 1
        inter_coeffs = c3[inter_start: inter_start + n_dummies]
        pos_frac     = float(np.mean(inter_coeffs > 0))
        mean_inter   = float(np.mean(inter_coeffs))
        results.append(dict(corpus=corpus_name, model='categorical_k', n_params=p3_c,
                            AIC=round(aic3, 2), BIC=round(bic3, 2),
                            b_interaction=round(mean_inter, 5), se_interaction=np.nan,
                            p_interaction=np.nan, R2=round(r2_3, 4)))
        k_labels = [str(kk) for kk in k_vals_present[1:]]
        inter_strs = '  '.join(f'k={kl}:{v:+.3f}' for kl, v in zip(k_labels, inter_coeffs))
        print(f'  M3 categ. k   : mean_int={mean_inter:+.4f}  frac_pos={pos_frac:.2f}  AIC={aic3:.1f}  BIC={bic3:.1f}')
        print(f'    interactions: {inter_strs}')
    except Exception as e:
        print(f'  M3 categorical k FAILED: {e}')

    return results


# ── MAIN ────────────────────────────────────────────────────────────────────────

MAIN_RESULTS_DIR = Path('results_v2')
CORPUS_NAMES = ['shakespeare', 'pride_prej', 'python_stdlib']

def load_panels_from_main():
    """Load regression panels saved by run_experiment_v2.py if available."""
    panels = {}
    for name in CORPUS_NAMES:
        panel_path = MAIN_RESULTS_DIR / f'panel_{name}.csv'
        if panel_path.exists():
            df = pd.read_csv(panel_path)
            # Ensure required columns present; add 'ch' alias for 'char_id'
            if 'char_id' in df.columns and 'ch' not in df.columns:
                df = df.rename(columns={'char_id': 'ch'})
            # Reconstruct 'type' column from Structural flag if needed
            if 'type' not in df.columns and 'Structural' in df.columns:
                df['type'] = df['Structural'].map({1: 'structural', 0: 'lexical'})
            # Derive integer k from log2k if missing
            if 'k' not in df.columns and 'log2k' in df.columns:
                df['k'] = np.round(2 ** df['log2k'].values).astype(int)
            panels[name] = df
            n_chars = df['ch'].nunique() if 'ch' in df.columns else '?'
            print(f'  {name:20s}: loaded {len(df)} obs, {n_chars} chars from {panel_path}')
        else:
            print(f'  WARNING: {panel_path} not found — run run_experiment_v2.py first')
    return panels

def rebuild_panels_from_corpora():
    """Fallback: compute panels from scratch (slower, uses current Python stdlib)."""
    import os as _os
    print('\n=== Loading corpora ===')
    corpora, strata_map = {}, {}
    if Path('corpus_shakespeare.txt').exists():
        corpora['shakespeare'] = load_file('corpus_shakespeare.txt')
        strata_map['shakespeare'] = None
        print(f'  shakespeare   : {len(corpora["shakespeare"]):,} chars')
    else:
        print('  WARNING: corpus_shakespeare.txt not found')
    if Path('corpus_pride_prejudice.txt').exists():
        raw = load_file('corpus_pride_prejudice.txt')
        s   = raw.find('It is a truth')
        e   = raw.rfind('End of the Project Gutenberg')
        corpora['pride_prej']  = raw[s:e] if s > 0 else raw
        strata_map['pride_prej'] = None
        print(f'  pride_prej    : {len(corpora["pride_prej"]):,} chars')
    else:
        print('  WARNING: corpus_pride_prejudice.txt not found')
    stdlib_path = _os.path.dirname(_os.__file__)
    print(f'  Python stdlib from {stdlib_path}...')
    code1_text, code1_strata = load_python_stdlib(stdlib_path)
    corpora['python_stdlib'] = code1_text
    strata_map['python_stdlib'] = code1_strata

    print('\n=== Making 80/10/10 splits ===')
    splits_map = {}
    for name, text in corpora.items():
        n    = len(text)
        n_tr = int(0.8 * n)
        n_vl = int(0.1 * n)
        st   = strata_map[name]
        splits_map[name] = {
            'train'       : text[:n_tr],
            'test'        : text[n_tr+n_vl:],
            'vocab_size'  : len(set(text)),
            'test_strata' : st[n_tr+n_vl:] if st else None,
        }
        print(f'  {name:20s}: train={n_tr:,}  test={n-n_tr-n_vl:,}')

    print('\n=== Reliable K_MAX (global coverage >= 50%) ===')
    RELIABLE_K_MAX = {}
    for name, sp in splits_map.items():
        kmax = 1
        for k in K_VALUES:
            if global_coverage(sp['train'], sp['test'], k) >= 0.50:
                kmax = k
        RELIABLE_K_MAX[name] = kmax
        print(f'  {name:20s}: K_MAX = {kmax}')

    print('\n=== Computing mean surprisal per (char, k) ===')
    S_k_all = {}
    for name, sp in splits_map.items():
        print(f'\n  {name}:')
        is_code     = (name == 'python_stdlib')
        test_strata = sp.get('test_strata')
        vocab_size  = sp['vocab_size']
        S_k_all[name] = {}
        for k in K_ALL:
            print(f'    k={k}...', end=' ', flush=True)
            counts = build_ngram(sp['train'], k)
            S_k_all[name][k] = compute_mean_surprisal(
                counts, k, sp['test'], vocab_size,
                test_strata=test_strata, is_code=is_code)
            print(f'done ({len(S_k_all[name][k])} chars)')

    print('\n=== Building regression panels ===')
    panels = {}
    for name, sp in splits_map.items():
        train_freq  = Counter(sp['train'])
        total_train = len(sp['train'])
        is_code     = (name == 'python_stdlib')
        panel       = build_panel(S_k_all[name], RELIABLE_K_MAX[name], is_code, train_freq, total_train)
        panels[name] = panel
        if panel.empty:
            print(f'  {name}: EMPTY panel')
            continue
        ns = (panel['type'] == 'structural').sum()
        nl = (panel['type'] == 'lexical').sum()
        n_chars = panel['ch'].nunique()
        print(f'  {name:20s}: {len(panel)} obs  {n_chars} chars  struct_obs={ns}  lex_obs={nl}')
    return panels


print('\n=== Loading regression panels ===')
panels = load_panels_from_main()
if not panels:
    print('Falling back to corpus re-computation...')
    panels = rebuild_panels_from_corpora()

print('\n' + '='*70)
print(f'CHECK 1: Character-label permutation test  (N={N_PERMUTATIONS})')
print('='*70)
perm_rows = []
for name in ['shakespeare', 'pride_prej', 'python_stdlib']:
    if name not in panels:
        continue
    row = run_permutation_test(panels[name], name)
    if row:
        perm_rows.append(row)

if perm_rows:
    perm_df = pd.DataFrame(perm_rows)
    out_path = RESULTS_DIR / 'permutation_test.csv'
    perm_df.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')
    print(perm_df[['corpus','n_structural','n_lexical','b3_observed',
                   'p_cluster_robust','p_permutation']].to_string(index=False))

print('\n' + '='*70)
print('CHECK 2: Functional form comparison')
print('='*70)
ff_rows = []
for name in ['shakespeare', 'pride_prej', 'python_stdlib']:
    if name not in panels:
        continue
    print(f'\n--- {name} ---')
    rows = run_functional_form(panels[name], name)
    ff_rows.extend(rows)

if ff_rows:
    ff_df = pd.DataFrame(ff_rows)
    out_path = RESULTS_DIR / 'functional_form.csv'
    ff_df.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')
    print(ff_df[['corpus','model','n_params','AIC','BIC',
                 'b_interaction','p_interaction','R2']].to_string(index=False))

print('\n' + '='*70)
print('CHECK 3: Coverage threshold sensitivity')
print('='*70)

COV_THRESHOLDS = [0.25, 0.50, 0.75]
cov_rows = []
for name in ['shakespeare', 'pride_prej', 'python_stdlib']:
    if name not in panels:
        continue
    panel_full = panels[name]
    if panel_full.empty:
        continue
    print(f'\n--- {name} ---')
    for tau in COV_THRESHOLDS:
        # For panels loaded from main experiment, filter by sym_cov at each tau.
        # The main experiment's panel was already filtered at tau=0.50.
        # For tau<0.50 we cannot recover excluded rows; for tau>0.50 we can restrict.
        # So coverage sensitivity at tau != 0.50 requires corpus re-computation.
        # Report tau=0.50 (from panel) and note others need corpus re-run.
        if tau != SYM_COV_THRESHOLD:
            print(f'  tau={tau}: requires corpus re-run (panel filtered at tau={SYM_COV_THRESHOLD})')
            cov_rows.append(dict(corpus=name, tau=tau, b3='n/a (needs re-run)', p='n/a'))
            continue
        panel = panel_full
        if panel.empty:
            continue
        char_to_id = {c: i for i, c in enumerate(np.unique(panel['ch'].values))}
        group_ids  = np.array([char_to_id[c] for c in panel['ch'].values])
        y          = panel['CG'].values
        X = np.column_stack([np.ones(len(y)), panel['log2k'].values,
                             panel['Structural'].values, panel['log2k_x_S'].values,
                             panel['logFreq'].values])
        coeffs, se, pv, r2 = clustered_ols(y, X, group_ids)
        b3, p3 = coeffs[3], pv[3]
        sig = '**' if p3<0.01 else '*' if p3<0.05 else 'n.s.'
        print(f'  tau={tau}: b3={b3:+.4f}  p={p3:.4f} {sig}  (main panel, n={len(y)})')
        cov_rows.append(dict(corpus=name, tau=tau, b3=round(b3, 5), p=round(p3, 5)))

if cov_rows:
    cov_df = pd.DataFrame(cov_rows)
    out_path = RESULTS_DIR / 'coverage_sensitivity.csv'
    cov_df.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')

print('\nDone.')
