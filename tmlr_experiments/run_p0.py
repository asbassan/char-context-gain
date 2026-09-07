"""
run_p0.py
─────────
TMLR Phase 1 — P0 Kill Tests

Part A  Leave-one-structural-character-out (LOO)
        Uses existing results_canonical_snapshot/panel_*.csv — no surprisal recomputation.
        For each structural character in each corpus, removes that character's
        panel rows, refits the canonical regression, and records β₃.

Part B  Smoothing robustness (Laplace add-1 vs KT add-0.5)
        Reruns the full CG pipeline with KT smoothing using the same corpus
        text, same train/test split, and same k-range as the canonical run
        (k-range loaded from the canonical SQLite cache).
        Results saved to a separate cache — canonical files never touched.

Validation gate
        Canonical β₃ values are verified from panel CSVs before any test runs.
        Expected:  NL1 ≈ +0.551   NL2 ≈ +1.131   Code1 ≈ -0.024
        If any value deviates by more than TOL=0.01, the script stops.

Output
        results_tmlr/leave_one_structural_out.csv
        results_tmlr/smoothing_robustness.csv

Usage
        # from the chapter1 directory:
        uv run --python 3.12 --with "numpy,pandas,scipy" \\
            python tmlr_experiments/run_p0.py

        # LOO only (fast — no surprisal recomputation):
        uv run --python 3.12 --with "numpy,pandas,scipy" \\
            python tmlr_experiments/run_p0.py --skip-kt
"""

import sys
import os

# Add parent directory so we can import from run_experiment_v3
PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT)

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Import from canonical pipeline ────────────────────────────────────────────
from run_experiment_v3 import (
    ExperimentCache,
    clustered_ols,
    NL_STRUCTURAL_CHARS, PY_SYNTAX_CHARS, AMBIGUOUS_CHARS,
    MIN_N, SYM_COV_THRESHOLD, SEED, MAX_BOOT_SAMPLES,
    make_splits, load_corpora,
    compute_peaks,
    classify_nl, classify_code,
)

np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
ROOT_DIR    = SCRIPT_DIR.parent
RESULTS_DIR = ROOT_DIR / 'results_tmlr'
PANEL_DIR   = ROOT_DIR / 'results_canonical_snapshot'
CANON_CACHE = ROOT_DIR / 'experiment_cache.db'
KT_CACHE    = SCRIPT_DIR / 'kt_cache.db'

RESULTS_DIR.mkdir(exist_ok=True)

# ── Expected canonical values ─────────────────────────────────────────────────
CANONICAL_B3 = {
    'shakespeare':   +0.551,
    'pride_prej':    +1.131,
    'python_stdlib': -0.024,
}
TOL = 0.01   # maximum acceptable deviation from paper values


# ═══════════════════════════════════════════════════════════════════════════════
# Regression helper
# ═══════════════════════════════════════════════════════════════════════════════

def regress_panel(panel: pd.DataFrame) -> dict:
    """Run canonical regression on a panel DataFrame. Returns result dict."""
    y  = panel['CG'].values
    Xm = np.column_stack([
        np.ones(len(panel)),
        panel[['log2k', 'Structural', 'log2k_x_S', 'logFreq']].values,
    ])
    chars = np.unique(panel['char_id'].values)
    c2i   = {c: i for i, c in enumerate(chars)}
    gids  = np.array([c2i[c] for c in panel['char_id'].values])

    coeffs, se, lo, hi, pv, r2 = clustered_ols(y, Xm, gids)

    n_struct = int(
        panel[panel['Structural'] == 1]['char_id'].nunique()
    )
    return {
        'beta3':                 coeffs[3],
        'se':                    se[3],
        'ci_low':                lo[3],
        'ci_high':               hi[3],
        'p_value':               pv[3],
        'r2':                    r2,
        'n_obs':                 len(panel),
        'n_clusters':            len(chars),
        'n_structural_clusters': n_struct,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Validation gate
# ═══════════════════════════════════════════════════════════════════════════════

def validate_canonical():
    print('\n=== Validation: canonical β₃ from panel CSVs ===')
    ok = True
    for corpus in ['shakespeare', 'pride_prej', 'python_stdlib']:
        path = PANEL_DIR / f'panel_{corpus}.csv'
        if not path.exists():
            print(f'  MISSING: {path}')
            ok = False
            continue
        panel = pd.read_csv(path)
        r = regress_panel(panel)
        expected = CANONICAL_B3[corpus]
        diff = abs(r['beta3'] - expected)
        flag = 'OK  ' if diff <= TOL else 'FAIL'
        print(f'  {corpus:20s}  β₃={r["beta3"]:+.5f}  '
              f'expected≈{expected:+.3f}  Δ={diff:.5f}  [{flag}]')
        if diff > TOL:
            ok = False
    if not ok:
        print('\n  *** Canonical values not reproduced. STOPPING. ***')
        print('  Check that results_canonical_snapshot/panel_*.csv files '
              'match the paper values.')
        sys.exit(1)
    print('  Validation passed.\n')


# ═══════════════════════════════════════════════════════════════════════════════
# Part A — Leave-one-structural-character-out
# ═══════════════════════════════════════════════════════════════════════════════

def run_loo() -> pd.DataFrame:
    print('=== Part A: Leave-one-structural-character-out ===')
    rows = []

    for corpus in ['shakespeare', 'pride_prej', 'python_stdlib']:
        path = PANEL_DIR / f'panel_{corpus}.csv'
        if not path.exists():
            print(f'  MISSING: {path} — skipping')
            continue

        panel = pd.read_csv(path)
        r_canon   = regress_panel(panel)
        b3_canon  = r_canon['beta3']
        struct_chars = sorted(
            panel[panel['Structural'] == 1]['char_id'].unique()
        )

        print(f'\n  {corpus}')
        print(f'    canonical β₃ = {b3_canon:+.5f}')
        print(f'    structural chars ({len(struct_chars)}): '
              f'{[repr(c) for c in struct_chars]}')
        print(f'    {"excl":>6}  {"β₃":>8}  {"Δβ₃":>8}  '
              f'{"p":>7}  {"n_struct_clust":>14}  status')
        print(f'    {"-"*62}')

        for excl in struct_chars:
            sub = panel[panel['char_id'] != excl].copy()
            if len(sub) < 10:
                print(f'    {repr(excl):>6}  (too few rows: {len(sub)}, skipped)')
                continue

            try:
                r = regress_panel(sub)
            except Exception as e:
                print(f'    {repr(excl):>6}  regression failed: {e}')
                continue

            delta  = r['beta3'] - b3_canon
            status = 'OK' if r['beta3'] > 0 else '*** COLLAPSED ***'
            print(f'    {repr(excl):>6}  {r["beta3"]:+8.5f}  {delta:+8.5f}  '
                  f'{r["p_value"]:7.4f}  {r["n_structural_clusters"]:14d}  {status}')

            rows.append({
                'corpus':                corpus,
                'excluded_character':    repr(excl),
                'beta3':                 round(r['beta3'],   5),
                'se':                    round(r['se'],      5),
                'ci_low':                round(r['ci_low'],  5),
                'ci_high':               round(r['ci_high'], 5),
                'p_value':               round(r['p_value'], 5),
                'n_obs':                 r['n_obs'],
                'n_clusters':            r['n_clusters'],
                'n_structural_clusters': r['n_structural_clusters'],
                'beta3_canonical':       round(b3_canon, 5),
                'delta_beta3':           round(delta, 5),
            })

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / 'leave_one_structural_out.csv'
    df.to_csv(out, index=False)
    print(f'\n  Saved {len(df)} rows → {out}')
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Part B — KT smoothing robustness
# ═══════════════════════════════════════════════════════════════════════════════

def _worker_kt(args):
    """
    Surprisal worker with Krichevsky-Trofimov (add-0.5) smoothing.
    Drop-in replacement for _worker in run_experiment_v3.py.
    """
    train, test, k, vocab_size, is_code, test_strata = args
    alpha = 0.5   # KT vs Laplace (alpha=1)

    counts = defaultdict(Counter)
    for i in range(k, len(train)):
        counts[train[i-k:i]][train[i]] += 1

    result = defaultdict(lambda: {
        'sum_surp': 0.0, 'sum_sq': 0.0, 'n_sum': 0, 'n': 0, 'ctx_hit': 0})

    for i in range(k, len(test)):
        ch  = test[i]
        typ = (classify_code(ch, test_strata[i]) if is_code and test_strata
               else classify_nl(ch))
        if typ in ('other', 'excluded', 'ambiguous'):
            continue
        ctx        = test[i-k:i]
        ctx_counts = counts.get(ctx, {})
        ctx_total  = sum(ctx_counts.values())
        prob       = (ctx_counts.get(ch, 0) + alpha) / (ctx_total + alpha * vocab_size)
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


def compute_kt_surprisals(splits, corpus_name, canon_cache, kt_cache):
    """
    Compute KT surprisals using the same k-range as the canonical Laplace run.
    k-range is loaded from the canonical SQLite cache — no coverage re-probe needed.
    Returns (S_k, k_values) or (None, None) if canonical k-range is missing.
    """
    is_code     = corpus_name == 'python_stdlib'
    corpus_hash = splits['hash']

    cached_range = canon_cache.load_k_range(corpus_hash)
    if cached_range is None:
        print(f'  {corpus_name}: canonical k-range not found in cache — '
              f'run run_experiment_v3.py first, then retry.')
        return None, None

    _, _, _, k_values = cached_range
    print(f'  {corpus_name}: k_values = {k_values}')

    S_k = {}
    for k in k_values:
        if kt_cache.k_done(corpus_hash, k):
            S_k[k] = kt_cache.load_k(corpus_hash, k)
            cov     = kt_cache.load_coverage(corpus_hash, k) or 0.0
            print(f'    k={k:>2}  CACHED    ({len(S_k[k])} chars)  '
                  f'global_cov={100*cov:.1f}%')
        else:
            _, res, cov = _worker_kt((
                splits['train'], splits['test'], k,
                splits['vocab_size'], is_code, splits.get('test_strata')))
            kt_cache.save_k(corpus_hash, k, res, cov)
            S_k[k] = res
            print(f'    k={k:>2}  computed  ({len(res)} chars)  '
                  f'global_cov={100*cov:.1f}%')

    return S_k, k_values


def build_kt_panel_df(S_k, splits, corpus_name, k_values) -> pd.DataFrame:
    """Build regression panel DataFrame from KT surprisal dicts."""
    kmax     = max((k for k in k_values if k > 1), default=1)
    panel_ks = [k for k in k_values if k > 1 and k <= kmax]
    freq     = Counter(splits['train'])
    total    = len(splits['train'])

    peaks_df = compute_peaks(S_k, k_values, corpus_name)
    rows = []

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


def run_smoothing_robustness(splits_by_corpus) -> pd.DataFrame:
    print('\n=== Part B: Smoothing robustness (Laplace vs KT) ===')

    canon_cache = ExperimentCache(str(CANON_CACHE))
    kt_cache    = ExperimentCache(str(KT_CACHE))

    rows = []

    for corpus in ['shakespeare', 'pride_prej', 'python_stdlib']:
        splits = splits_by_corpus.get(corpus)
        if splits is None:
            print(f'\n  {corpus}: splits not available, skipping')
            continue

        print(f'\n  {corpus}')

        # ── Laplace β₃ from existing canonical panel CSV ──────────────────
        path_lap = PANEL_DIR / f'panel_{corpus}.csv'
        if path_lap.exists():
            panel_lap = pd.read_csv(path_lap)
            r_lap     = regress_panel(panel_lap)
        else:
            print(f'    Laplace panel CSV not found: {path_lap}')
            r_lap = None

        if r_lap:
            print(f'    Laplace  β₃={r_lap["beta3"]:+.5f}  '
                  f'p={r_lap["p_value"]:.4f}')
            rows.append({
                'corpus': corpus, 'smoothing': 'laplace',
                'beta3':   round(r_lap['beta3'],   5),
                'se':      round(r_lap['se'],       5),
                'ci_low':  round(r_lap['ci_low'],   5),
                'ci_high': round(r_lap['ci_high'],  5),
                'p_value': round(r_lap['p_value'],  5),
                'n_obs':   r_lap['n_obs'],
                'n_clusters': r_lap['n_clusters'],
            })

        # ── KT surprisals ─────────────────────────────────────────────────
        S_k_kt, k_values = compute_kt_surprisals(
            splits, corpus, canon_cache, kt_cache)

        if S_k_kt is None:
            continue

        panel_kt = build_kt_panel_df(S_k_kt, splits, corpus, k_values)
        if len(panel_kt) == 0:
            print(f'    KT panel is empty — skipping regression')
            continue

        r_kt = regress_panel(panel_kt)
        print(f'    KT       β₃={r_kt["beta3"]:+.5f}  '
              f'p={r_kt["p_value"]:.4f}')

        if r_lap:
            same_sign = (r_lap['beta3'] > 0) == (r_kt['beta3'] > 0)
            print(f'    sign preserved: {same_sign}  '
                  f'Δβ₃={r_kt["beta3"] - r_lap["beta3"]:+.5f}')

        rows.append({
            'corpus': corpus, 'smoothing': 'kt_add_half',
            'beta3':   round(r_kt['beta3'],   5),
            'se':      round(r_kt['se'],       5),
            'ci_low':  round(r_kt['ci_low'],   5),
            'ci_high': round(r_kt['ci_high'],  5),
            'p_value': round(r_kt['p_value'],  5),
            'n_obs':   r_kt['n_obs'],
            'n_clusters': r_kt['n_clusters'],
        })

    canon_cache.close()
    kt_cache.close()

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / 'smoothing_robustness.csv'
    df.to_csv(out, index=False)
    print(f'\n  Saved {len(df)} rows → {out}')
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(loo_df, smooth_df):
    print('\n' + '=' * 70)
    print('P0 SUMMARY')
    print('=' * 70)

    print('\n─ Leave-one-structural-character-out ─')
    print(f'  {"corpus":20s}  {"β₃ min":>9}  {"β₃ max":>9}  '
          f'{"all > 0":>8}  {"any collapsed":>14}')
    print(f'  {"-"*64}')
    if loo_df is not None and len(loo_df) > 0:
        for corpus in ['shakespeare', 'pride_prej', 'python_stdlib']:
            sub = loo_df[loo_df['corpus'] == corpus]
            if len(sub) == 0:
                continue
            b3_min    = sub['beta3'].min()
            b3_max    = sub['beta3'].max()
            all_pos   = (sub['beta3'] > 0).all()
            collapsed = (sub['beta3'] <= 0).any()
            print(f'  {corpus:20s}  {b3_min:+9.5f}  {b3_max:+9.5f}  '
                  f'{"yes":>8}  {"YES ⚠" if collapsed else "no":>14}')
    else:
        print('  (no LOO results)')

    print('\n─ Smoothing robustness ─')
    print(f'  {"corpus":20s}  {"smoothing":12s}  {"β₃":>9}  {"p":>7}')
    print(f'  {"-"*52}')
    if smooth_df is not None and len(smooth_df) > 0:
        for corpus in ['shakespeare', 'pride_prej', 'python_stdlib']:
            for sm in ['laplace', 'kt_add_half']:
                row = smooth_df[
                    (smooth_df['corpus'] == corpus) &
                    (smooth_df['smoothing'] == sm)
                ]
                if len(row) == 0:
                    continue
                b3 = row['beta3'].values[0]
                p  = row['p_value'].values[0]
                print(f'  {corpus:20s}  {sm:12s}  {b3:+9.5f}  {p:7.4f}')
    else:
        print('  (KT not run — use without --skip-kt)')

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description='TMLR P0 Kill Tests')
    ap.add_argument(
        '--skip-kt', action='store_true',
        help='Run LOO only (fast); skip KT surprisal recomputation')
    args = ap.parse_args()

    # The corpus files live in ROOT_DIR — set CWD so load_corpora() finds them
    os.chdir(ROOT_DIR)

    # Gate: verify canonical values before any test
    validate_canonical()

    # Part A — LOO (reads panel CSVs, no surprisal recomputation)
    loo_df = run_loo()

    # Part B — KT (recomputes surprisals with add-0.5 smoothing)
    smooth_df = None
    if not args.skip_kt:
        print('\nLoading corpora for KT run...')
        corpora, strata = load_corpora()
        splits_by_corpus = {
            name: make_splits(text, strata.get(name))
            for name, text in corpora.items()
        }
        smooth_df = run_smoothing_robustness(splits_by_corpus)

    print_summary(loo_df, smooth_df)

    print('Output files:')
    for f in sorted(RESULTS_DIR.iterdir()):
        print(f'  {f.name:<45s}  {f.stat().st_size:>8,} bytes')
    print(f'\nKT cache: {KT_CACHE}')
    print('Re-run at any time — completed KT (corpus, k) pairs are skipped.')


if __name__ == '__main__':
    main()
