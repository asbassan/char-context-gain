"""
run_phase2.py
─────────────
TMLR Phase 2 — Generalization across corpora.

Runs the full CG → regression pipeline on five new corpora using:
  - Shared CODE_STRUCTURAL_CHARS for all code corpora (char-identity-only,
    no Python tokenizer strata — see code_classification_assumptions.md)
  - Same NL_STRUCTURAL_CHARS, clustering regression, and Laplace smoothing
    as the canonical pipeline

Also re-runs Python stdlib with the shared code structural set (for
consistency in the TMLR paper, which reports all code corpora on the same
classification scheme).

Corpora:
  NL   wikitext_103      corpora_phase2/wikitext_103.txt
  NL   reuters           corpora_phase2/reuters.txt
  NL   bible_kjv         corpora_phase2/bible_kjv.txt
  Code nodejs_js         corpora_phase2/nodejs_js.txt
  Code commons_lang_java corpora_phase2/commons_lang_java.txt
  Code python_stdlib_p2  Python 3.12 stdlib (re-run, shared structural set)

Output: results_tmlr/phase2/
Cache:  tmlr_experiments/phase2_cache.db

Usage (from chapter1 directory):
    uv run --python 3.12 --with "numpy,pandas,scipy,matplotlib" \\
        python tmlr_experiments/run_phase2.py
    uv run --python 3.12 --with "numpy,pandas,scipy,matplotlib" \\
        python tmlr_experiments/run_phase2.py --workers 4 --max-k 10
"""

import argparse
import multiprocessing as mp
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Import infrastructure from canonical pipeline ─────────────────────────────
PARENT = Path(__file__).parent.parent
sys.path.insert(0, str(PARENT))

from run_experiment_v3 import (
    ExperimentCache,
    clustered_ols,
    make_splits,
    probe_coverage,
    _mean_ci,
    _corpus_hash,
    load_python_stdlib,
    MIN_N, SYM_COV_THRESHOLD, SEED, MAX_BOOT_SAMPLES,
    NL_STRUCTURAL_CHARS, AMBIGUOUS_CHARS,
)

np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
ROOT_DIR    = SCRIPT_DIR.parent
CORPUS_DIR  = ROOT_DIR / 'corpora_phase2'
RESULTS_DIR = ROOT_DIR / 'results_tmlr' / 'phase2'
CACHE_PATH  = SCRIPT_DIR / 'phase2_cache.db'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared code structural set (all code corpora, char-identity-only) ─────────
# See tmlr_experiments/code_classification_assumptions.md for rationale.
CODE_STRUCTURAL_CHARS = frozenset({
    '(', ')', '{', '}', '[', ']',   # brackets
    ',', '.', ';', ':',              # separators
    '=', '<', '>', '!',             # comparison / assignment
    '+', '-', '*', '/', '%',        # arithmetic
    '&', '|', '^', '~',            # bitwise / logical
    '@', '?',                        # decorators, ternary / optional chaining
})

# ── Corpus manifest ───────────────────────────────────────────────────────────
# is_code=True → CODE_STRUCTURAL_CHARS + char-identity-only classification
# is_code=False → NL_STRUCTURAL_CHARS (same as canonical pipeline)
PHASE2_CORPORA = {
    'wikitext_103':      {'path': CORPUS_DIR / 'wikitext_103.txt',      'is_code': False},
    'reuters':           {'path': CORPUS_DIR / 'reuters.txt',            'is_code': False},
    'bible_kjv':         {'path': CORPUS_DIR / 'bible_kjv.txt',          'is_code': False},
    'nodejs_js':         {'path': CORPUS_DIR / 'nodejs_js.txt',          'is_code': True},
    'commons_lang_java': {'path': CORPUS_DIR / 'commons_lang_java.txt',  'is_code': True},
    # python_stdlib loaded separately below — no plain text file
}


# ═══════════════════════════════════════════════════════════════════════════════
# Classification (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_p2(ch: str, is_code: bool) -> str:
    """Char-identity-only classification for Phase 2.
    Code: uses CODE_STRUCTURAL_CHARS (no strata).
    NL:   uses NL_STRUCTURAL_CHARS (identical to canonical pipeline).
    """
    if is_code:
        if ch in CODE_STRUCTURAL_CHARS:         return 'structural'
        if ch.isalpha() or ch.isdigit() or ch == '_': return 'lexical'
        return 'other'
    else:
        if ch in NL_STRUCTURAL_CHARS:           return 'structural'
        if ch in AMBIGUOUS_CHARS:               return 'ambiguous'
        if ch.isalpha() or ch.isdigit():        return 'lexical'
        return 'other'


# ═══════════════════════════════════════════════════════════════════════════════
# Surprisal worker (Phase 2)
# Module-level so multiprocessing can pickle it.
# ═══════════════════════════════════════════════════════════════════════════════

def _worker_p2(args):
    """Laplace (add-1) surprisal worker with Phase 2 classification."""
    train, test, k, vocab_size, is_code = args

    counts = defaultdict(Counter)
    for i in range(k, len(train)):
        counts[train[i-k:i]][train[i]] += 1

    result = defaultdict(lambda: {
        'sum_surp': 0.0, 'sum_sq': 0.0, 'n_sum': 0, 'n': 0, 'ctx_hit': 0})

    for i in range(k, len(test)):
        ch  = test[i]
        typ = _classify_p2(ch, is_code)
        if typ in ('other', 'ambiguous'):
            continue
        ctx        = test[i-k:i]
        ctx_counts = counts.get(ctx, {})
        ctx_total  = sum(ctx_counts.values())
        prob       = (ctx_counts.get(ch, 0) + 1) / (ctx_total + vocab_size)
        surp       = -np.log2(prob)
        r = result[ch]
        if r['n_sum'] < MAX_BOOT_SAMPLES:
            r['sum_surp'] += surp
            r['sum_sq']   += surp * surp
            r['n_sum']    += 1
        r['n'] += 1
        if ctx_counts:
            r['ctx_hit'] += 1

    results = {ch: dict(v) for ch, v in result.items()}
    seen  = set(train[i-k:i] for i in range(k, len(train)))
    hits  = sum(1 for i in range(k, len(test)) if test[i-k:i] in seen)
    total = len(test) - k
    cov   = hits / total if total else 0.0
    return k, results, cov


# ═══════════════════════════════════════════════════════════════════════════════
# Surprisal computation (Phase 2 — reuses v3 coverage probe, new worker)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_one_k_p2(splits, k, is_code, cache):
    _, res, cov = _worker_p2((
        splits['train'], splits['test'], k,
        splits['vocab_size'], is_code))
    cache.save_k(splits['hash'], k, res, cov)
    return res, cov


def _parallel_batch_p2(splits, k_list, is_code, cache, S_k, workers):
    args = [(splits['train'], splits['test'], k, splits['vocab_size'], is_code)
            for k in k_list]
    if workers > 1 and len(args) > 1:
        with mp.Pool(workers) as pool:
            results = pool.map(_worker_p2, args)
    else:
        results = [_worker_p2(a) for a in args]
    for k, res, cov in sorted(results, key=lambda x: x[0]):
        cache.save_k(splits['hash'], k, res, cov)
        S_k[k] = res
        flag = '' if cov >= SYM_COV_THRESHOLD else '  << sparse'
        print(f'  k={k:>2}  computed  ({len(res):>3} chars)  '
              f'global_cov={100*cov:.1f}%{flag}')


def compute_surprisals_p2(splits, corpus_name, is_code, cache, workers, max_k):
    """Coverage-probe + linear scan surprisal computation (Phase 2 variant)."""
    print(f'\n=== Surprisals: {corpus_name} ===')
    ch_hash = splits['hash']
    done    = cache.progress(ch_hash)
    S_k     = {}

    # Resume path: k_range already decided
    cached_range = cache.load_k_range(ch_hash)
    if cached_range:
        _, _, _, k_values = cached_range
        print(f'  [resumed]  k range: {k_values}')
        for k in k_values:
            if k in done:
                S_k[k] = cache.load_k(ch_hash, k)
                cov = cache.load_coverage(ch_hash, k) or 0
                print(f'  k={k:>2}  CACHED    ({len(S_k[k]):>3} chars)  '
                      f'global_cov={100*cov:.1f}%')
        missing = [k for k in k_values if k not in done]
        if missing:
            _parallel_batch_p2(splits, missing, is_code, cache, S_k, workers)
        return S_k, k_values

    # Coverage probe (imported from v3 — classification-independent)
    probe_ceiling = probe_coverage(
        splits['train'], splits['test'], cache, ch_hash, workers, max_k)

    if probe_ceiling < 2:
        print('  WARNING: coverage insufficient at k=2 — single-k run.')
        k_values = [1]
        cache.save_k_range(ch_hash, 1, 1, 1, k_values)
        if 1 not in done:
            res, cov = _run_one_k_p2(splits, 1, is_code, cache)
            S_k[1]   = res
            print(f'  k= 1  computed  ({len(res):>3} chars)  global_cov=100%')
        else:
            S_k[1] = cache.load_k(ch_hash, 1)
        return S_k, k_values

    # k=1 baseline
    if 1 in done:
        S_k[1] = cache.load_k(ch_hash, 1)
        print(f'  k= 1  CACHED    ({len(S_k[1]):>3} chars)  global_cov=100%')
    else:
        res, cov = _run_one_k_p2(splits, 1, is_code, cache)
        S_k[1]   = res
        print(f'  k= 1  computed  ({len(res):>3} chars)  global_cov=100%')

    # Linear scan k=2..probe_ceiling (stop when coverage drops)
    reliable_kmax = 1
    k_seq = [1]

    for k in range(2, probe_ceiling + 1):
        if k in done:
            S_k[k] = cache.load_k(ch_hash, k)
            cov     = cache.load_coverage(ch_hash, k) or 0
            print(f'  k={k:>2}  CACHED    ({len(S_k[k]):>3} chars)  '
                  f'global_cov={100*cov:.1f}%')
        else:
            res, cov = _run_one_k_p2(splits, k, is_code, cache)
            S_k[k]   = res
            print(f'  k={k:>2}  computed  ({len(res):>3} chars)  '
                  f'global_cov={100*cov:.1f}%')

        if cov < SYM_COV_THRESHOLD:
            print(f'  k={k}: coverage {100*cov:.1f}% < threshold — stopping.')
            del S_k[k]
            break

        reliable_kmax = k
        k_seq.append(k)

    k_values = k_seq
    cache.save_k_range(ch_hash, reliable_kmax, reliable_kmax, reliable_kmax, k_values)
    print(f'  → effective k range: {k_values}')
    return S_k, k_values


# ═══════════════════════════════════════════════════════════════════════════════
# Per-character peaks (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_peaks_p2(S_k: dict, k_values: list, is_code: bool) -> pd.DataFrame:
    """Identify per-character CG peaks using Phase 2 classification."""
    reliable_ks = [k for k in k_values if k > 1]
    rows = []

    for ch in sorted(S_k.get(1, {}).keys()):
        e1 = S_k[1].get(ch)
        if not e1 or e1['n'] < MIN_N:
            continue
        s1, s1_lo, s1_hi = _mean_ci(e1['sum_surp'], e1['sum_sq'], e1['n_sum'])

        best_cg, best_k, best_lo, best_hi = float('-inf'), None, None, None
        for k in reliable_ks:
            ek = S_k.get(k, {}).get(ch)
            if not ek or ek['n'] < MIN_N:
                continue
            sym_cov = ek['ctx_hit'] / ek['n'] if ek['n'] else 0
            if sym_cov < SYM_COV_THRESHOLD:
                continue
            sk, sk_lo, sk_hi = _mean_ci(ek['sum_surp'], ek['sum_sq'], ek['n_sum'])
            cg = s1 - sk
            if cg > best_cg:
                best_cg = cg;  best_k  = k
                best_lo = s1_lo - sk_hi;  best_hi = s1_hi - sk_lo

        if best_k is None:
            continue

        ek_peak  = S_k.get(best_k, {}).get(ch, {})
        sym_cov  = ek_peak['ctx_hit'] / ek_peak['n'] if ek_peak.get('n') else 0.0
        sym_type = _classify_p2(ch, is_code)
        if sym_type in ('other', 'ambiguous'):
            continue

        ls = 1 - best_cg / s1 if s1 > 0 else float('nan')
        rows.append({
            'char':    repr(ch),  'char_raw': ch,     'type':    sym_type,
            'S_x1':    round(s1, 4),          'n_test': e1['n'],
            'CG_peak': round(best_cg, 4),     'k_peak': best_k,
            'sym_cov': round(sym_cov, 3),
        })

    return pd.DataFrame(rows).sort_values('CG_peak', ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Regression panel + OLS (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════

def build_panel_p2(S_k: dict, splits: dict, k_values: list,
                   is_code: bool) -> pd.DataFrame:
    kmax     = max((k for k in k_values if k > 1), default=1)
    panel_ks = [k for k in k_values if k > 1 and k <= kmax]
    freq     = Counter(splits['train'])
    total    = len(splits['train'])
    rows     = []

    peaks_df = compute_peaks_p2(S_k, k_values, is_code)

    for _, row in peaks_df.iterrows():
        ch  = row['char_raw']
        typ = row['type']
        if typ not in ('structural', 'lexical'):
            continue
        e1 = S_k.get(1, {}).get(ch)
        if not e1 or e1['n'] < MIN_N:
            continue
        s1        = e1['sum_surp'] / e1['n_sum']
        f         = freq.get(ch, 1) / total
        is_struct = 1 if typ == 'structural' else 0

        for k in panel_ks:
            ek = S_k.get(k, {}).get(ch)
            if not ek or ek['n'] < MIN_N:
                continue
            sym_cov = ek['ctx_hit'] / ek['n'] if ek['n'] > 0 else 0
            if sym_cov < SYM_COV_THRESHOLD:
                continue
            sk = ek['sum_surp'] / ek['n_sum']
            rows.append({
                'CG':         s1 - sk,
                'log2k':      np.log2(k),
                'Structural': is_struct,
                'log2k_x_S':  np.log2(k) * is_struct,
                'logFreq':    np.log2(max(f, 1e-10)),
                'char_id':    ch,
            })

    return pd.DataFrame(rows)


def regress_panel_p2(panel: pd.DataFrame) -> dict:
    y  = panel['CG'].values
    Xm = np.column_stack([
        np.ones(len(panel)),
        panel[['log2k', 'Structural', 'log2k_x_S', 'logFreq']].values,
    ])
    chars = np.unique(panel['char_id'].values)
    c2i   = {c: i for i, c in enumerate(chars)}
    gids  = np.array([c2i[c] for c in panel['char_id'].values])
    coeffs, se, lo, hi, pv, r2 = clustered_ols(y, Xm, gids)
    n_struct = int(panel[panel['Structural'] == 1]['char_id'].nunique())
    return {
        'beta3': coeffs[3], 'se':      se[3],
        'ci_low': lo[3],    'ci_high': hi[3],
        'p_value': pv[3],   'r2':      r2,
        'n_obs':   len(panel),
        'n_clusters':            len(chars),
        'n_structural_clusters': n_struct,
        'coeffs': coeffs, 'se_all': se, 'lo_all': lo,
        'hi_all': hi,     'pv_all': pv,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Corpus runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_corpus_p2(name: str, text: str, is_code: bool,
                  cache: ExperimentCache, workers: int, max_k: int) -> dict:
    print(f'\n{"="*70}')
    print(f'Corpus: {name}  ({len(text):,} chars)  is_code={is_code}')
    print(f'{"="*70}')

    splits  = make_splits(text, strata=None)
    S_k, k_values = compute_surprisals_p2(
        splits, name, is_code, cache, workers, max_k)

    panel = build_panel_p2(S_k, splits, k_values, is_code)
    if len(panel) == 0:
        print(f'  WARNING: empty panel for {name} — skipping regression.')
        return {'corpus': name, 'beta3': float('nan')}

    panel.to_csv(RESULTS_DIR / f'panel_{name}.csv', index=False)

    r = regress_panel_p2(panel)

    names = ['intercept', 'log2(k)', 'Structural',
             'log2(k)×Structural', 'log2(Freq)']
    print(f'\n  {name}  n={r["n_obs"]}  G={r["n_clusters"]}  R²={r["r2"]:.3f}'
          f'  panel_ks={sorted(panel["log2k"].apply(lambda x: round(2**x)).unique())}')
    print(f'  {"":28s} {"β":>8}  {"SE":>7}  {"95% CI":^20}  {"p":>7}')
    print(f'  {"-"*70}')
    for nm, b, s, l, h, p_ in zip(
            names, r['coeffs'], r['se_all'], r['lo_all'], r['hi_all'], r['pv_all']):
        sig = (' ***' if p_ < 0.001 else ' **' if p_ < 0.01
               else ' *' if p_ < 0.05 else '')
        print(f'  {nm:28s} {b:+8.4f}  {s:7.4f}  [{l:+7.4f}, {h:+7.4f}]  '
              f'{p_:7.4f}{sig}')
    print(f'\n  >> β₃ = {r["beta3"]:+.4f}  SE={r["se"]:.4f}  '
          f'CI=[{r["ci_low"]:+.4f},{r["ci_high"]:+.4f}]  p={r["p_value"]:.4f}')

    return {
        'corpus':     name,
        'is_code':    is_code,
        'beta3':      round(r['beta3'],   5),
        'se':         round(r['se'],      5),
        'ci_low':     round(r['ci_low'],  5),
        'ci_high':    round(r['ci_high'], 5),
        'p_value':    round(r['p_value'], 5),
        'r2':         round(r['r2'],      5),
        'n_obs':      r['n_obs'],
        'n_clusters': r['n_clusters'],
        'n_structural_clusters': r['n_structural_clusters'],
        'k_values':   str(k_values),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='TMLR Phase 2 generalization experiments')
    ap.add_argument('--workers', type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--max-k',   type=int, default=10,
                    help='Upper bound on k (default 10)')
    ap.add_argument('--corpus',  default=None,
                    help='Run one corpus only (default: all)')
    ap.add_argument('--skip-python-rerun', action='store_true',
                    help='Skip Python stdlib re-run with shared structural set')
    args = ap.parse_args()

    os.chdir(ROOT_DIR)   # corpus file paths resolve from ROOT_DIR

    cache = ExperimentCache(str(CACHE_PATH))

    rows = []

    corpora_to_run = (
        {args.corpus: PHASE2_CORPORA[args.corpus]}
        if args.corpus and args.corpus in PHASE2_CORPORA
        else PHASE2_CORPORA
    )

    for name, spec in corpora_to_run.items():
        path = spec['path']
        if not path.exists():
            print(f'\n  MISSING: {path} — run phase2_download_corpora.py first.')
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        row  = run_corpus_p2(
            name, text, spec['is_code'], cache, args.workers, args.max_k)
        rows.append(row)

    # Python stdlib re-run with shared code structural set
    if not args.skip_python_rerun and (
            args.corpus is None or args.corpus == 'python_stdlib_p2'):
        import os as _os
        stdlib_path = _os.path.dirname(__import__('os').__file__)
        from run_experiment_v3 import load_python_stdlib
        text, _ = load_python_stdlib(stdlib_path)  # discard strata
        row = run_corpus_p2(
            'python_stdlib_p2', text, True, cache, args.workers, args.max_k)
        rows.append(row)

    cache.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / 'phase2_results.csv'
    df.to_csv(out, index=False)

    print('\n' + '=' * 70)
    print('PHASE 2 SUMMARY')
    print('=' * 70)
    print(f'\n  {"corpus":25s}  {"type":5s}  {"β₃":>8}  {"SE":>7}  {"p":>7}  '
          f'{"n_obs":>8}  interpretation')
    print(f'  {"-"*80}')
    for _, r in df.iterrows():
        if np.isnan(r['beta3']):
            continue
        typ  = 'code' if r['is_code'] else 'NL'
        dirn = ('+ve (structural > lexical)' if r['beta3'] > 0.05
                else '~0 (null)'             if abs(r['beta3']) <= 0.05
                else '-ve')
        print(f'  {r["corpus"]:25s}  {typ:5s}  {r["beta3"]:+8.4f}  '
              f'{r["se"]:7.4f}  {r["p_value"]:7.4f}  {int(r["n_obs"]):>8}  {dirn}')

    print(f'\n  Results → {RESULTS_DIR}')
    print(f'  Cache   → {CACHE_PATH}')
    print('\nRe-run at any time — completed (corpus, k) pairs are skipped.')


if __name__ == '__main__':
    mp.freeze_support()
    main()
