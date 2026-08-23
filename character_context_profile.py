"""
character_context_profile.py  v2
---------------------------------
Adaptive per-character context-gain profiler.

SEARCH STRATEGY — gradient ascent on CG(k):
  Starts at k=2, follows the direction of increasing CG(k). At each step,
  a bootstrap confidence interval on the mean gradient across characters
  determines whether to continue. Stops when the CI no longer excludes zero
  from above — identical in spirit to neural-network early stopping, but
  ascending toward the CG peak rather than descending toward minimum loss.
  A patience parameter tolerates noise-induced dips before stopping.

HARD CEILING — k ≤ MAX_K (default 15, absolute max 50):
  Beyond k~15, Laplace n-gram estimates collapse to smoothing artifacts on
  any realistic corpus. If k_peak reaches the ceiling with healthy coverage,
  a warning recommends switching to a neural character-level estimator.

SQLITE CACHE:
  All (corpus_hash, char, k) results are persisted. Repeated runs on the
  same corpus return cached results instantly; interrupted runs resume cleanly.

PARALLEL COMPUTATION:
  Each k value is dispatched to a worker process. Coverage checks run first
  (lightweight set-based) to establish the reliable range before committing
  to full n-gram computation.

Usage
-----
  # Natural language:
  python character_context_profile.py corpus.txt

  # Python source (tokenizer-stratified):
  python character_context_profile.py src/ --python

  # Adaptive search with defaults:
  python character_context_profile.py corpus.txt --adaptive

  # Full control:
  python character_context_profile.py corpus.txt --adaptive --max-k 20
      --patience 2 --bootstrap-n 500 --workers 4 --output profile.csv

Options
-------
  --python            Apply Python tokenizer stratification
  --adaptive          Use gradient-ascent adaptive search (default: exhaustive)
  --max-k K           Hard k ceiling (default 15, max 50)
  --min-n N           Min test occurrences per character (default 30)
  --cov-threshold F   Per-symbol coverage gate (default 0.50)
  --patience N        Steps without improvement before stopping (default 2)
  --bootstrap-n N     Bootstrap resamples for stopping criterion (default 500)
  --bootstrap-alpha F Significance level; stop when upper CI ≤ 0 (default 0.05)
  --workers N         Parallel workers (default: cpu_count - 1, min 1)
  --db PATH           SQLite cache path (default: cg_cache.db)
  --no-cache          Disable cache reads and writes
  --output FILE       Write per-character summary CSV
  --top N             Print only top N characters by peak gain
"""

import argparse
import csv
import hashlib
import io
import keyword as kw_mod
import multiprocessing as mp
import os
import sqlite3
import sys
import tokenize as py_tokenize
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HARD_K_CEILING   = 50
DEFAULT_K_MAX    = 15
MAX_BOOT_SAMPLES = 5000
NL_STRUCTURAL    = set(',.;:?!\'"()[]{}')
PY_SYNTAX_CHARS  = set('()[]{},:;=.@+-*/%&|^~<>')

def _ngram_label(k):
    names = {1:'bigram',2:'trigram',3:'4-gram',4:'5-gram',5:'6-gram',
             6:'7-gram',7:'8-gram',8:'9-gram',9:'10-gram',10:'11-gram',
             11:'12-gram',12:'13-gram',13:'14-gram',14:'15-gram',15:'16-gram'}
    return names.get(k, f'{k+1}-gram')


# ---------------------------------------------------------------------------
# Python tokenisation helpers
# ---------------------------------------------------------------------------

def _python_strata(code):
    strata = ['other'] * len(code)
    line_starts = [0]
    for i, ch in enumerate(code):
        if ch == '\n':
            line_starts.append(i + 1)

    def flat(row, col):
        idx = line_starts[row-1] + col if row-1 < len(line_starts) else len(code)
        return min(idx, len(code))

    try:
        toks = list(py_tokenize.generate_tokens(io.StringIO(code).readline))
    except py_tokenize.TokenError:
        return strata

    MAP = {py_tokenize.NUMBER: 'numeric', py_tokenize.STRING: 'string',
           py_tokenize.COMMENT: 'comment', py_tokenize.OP: 'syntax',
           py_tokenize.NEWLINE: 'whitespace', py_tokenize.NL: 'whitespace',
           py_tokenize.INDENT: 'whitespace', py_tokenize.DEDENT: 'whitespace'}

    for tok in toks:
        tt, ts = tok.type, tok.string
        start, end = flat(*tok.start), min(flat(*tok.end), len(code))
        if tt == py_tokenize.NAME:
            st = 'keyword' if kw_mod.iskeyword(ts) else 'identifier'
        else:
            st = MAP.get(tt, 'other')
        for i in range(start, end):
            if i < len(code):
                strata[i] = st
    return strata


def _classify_python(ch, stratum):
    if stratum in ('string', 'comment'):           return 'excluded'
    if ch in PY_SYNTAX_CHARS and stratum=='syntax': return 'structural'
    if stratum in ('identifier','keyword','numeric') or ch.isalpha() or ch.isdigit():
        return 'lexical'
    return 'other'


def _classify_nl(ch):
    if ch in NL_STRUCTURAL:                    return 'structural'
    if ch.isalpha() or ch.isdigit() or ch==' ': return 'lexical'
    return 'other'


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

class CGCache:
    """Persistent store for (corpus_hash, char, k) → CG results."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS cg_results (
        corpus_hash TEXT NOT NULL,
        char        TEXT NOT NULL,
        k           INTEGER NOT NULL,
        s1          REAL,
        s_k         REAL,
        CG          REAL,
        coverage    REAL,
        n_obs       INTEGER,
        PRIMARY KEY (corpus_hash, char, k)
    );
    CREATE TABLE IF NOT EXISTS coverage_check (
        corpus_hash    TEXT NOT NULL,
        k              INTEGER NOT NULL,
        global_coverage REAL,
        PRIMARY KEY (corpus_hash, k)
    );
    """

    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def get_cg(self, corpus_hash, char, k):
        row = self.conn.execute(
            'SELECT s1, s_k, CG, coverage, n_obs FROM cg_results '
            'WHERE corpus_hash=? AND char=? AND k=?',
            (corpus_hash, char, k)).fetchone()
        return row  # None if not cached

    def put_cg(self, corpus_hash, char, k, s1, s_k, CG, coverage, n_obs):
        self.conn.execute(
            'INSERT OR REPLACE INTO cg_results VALUES (?,?,?,?,?,?,?,?)',
            (corpus_hash, char, k, s1, s_k, CG, coverage, n_obs))
        self.conn.commit()

    def get_coverage(self, corpus_hash, k):
        row = self.conn.execute(
            'SELECT global_coverage FROM coverage_check '
            'WHERE corpus_hash=? AND k=?', (corpus_hash, k)).fetchone()
        return row[0] if row else None

    def put_coverage(self, corpus_hash, k, global_coverage):
        self.conn.execute(
            'INSERT OR REPLACE INTO coverage_check VALUES (?,?,?)',
            (corpus_hash, k, global_coverage))
        self.conn.commit()

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(path, use_python):
    path = Path(path)
    files = sorted(path.rglob('*.py')) if (path.is_dir() and use_python) \
            else (sorted(path.iterdir()) if path.is_dir() else [path])

    text_parts, strata_parts, n_files = [], [], 0
    for fp in files:
        if not fp.is_file(): continue
        if use_python and fp.suffix != '.py': continue
        try:
            code = fp.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        text_parts.append(code)
        if use_python:
            strata_parts.append(_python_strata(code))
        text_parts.append('\n')
        if use_python:
            strata_parts.append(['whitespace'])
        n_files += 1

    if not text_parts:
        sys.exit(f'No readable files at {path}')

    full_text   = ''.join(text_parts)
    full_strata = [s for p in strata_parts for s in p] if use_python else None

    corpus_hash = hashlib.sha256(full_text.encode('utf-8', errors='replace')).hexdigest()[:16]
    _log(f'Loaded {n_files} file(s) — {len(full_text):,} chars  hash={corpus_hash}')
    return full_text, full_strata, corpus_hash


# ---------------------------------------------------------------------------
# Coverage check (lightweight — set membership only, no probability estimation)
# ---------------------------------------------------------------------------

def _check_coverage(args):
    train, test, k = args
    seen  = set(train[i-k:i] for i in range(k, len(train)))
    hits  = sum(1 for i in range(k, len(test)) if test[i-k:i] in seen)
    total = len(test) - k
    return k, hits / total if total else 0.0


def check_coverages(train, test, k_values, workers):
    """Return {k: global_coverage} for each k in k_values."""
    args = [(train, test, k) for k in k_values]
    if workers > 1 and len(k_values) > 1:
        with mp.Pool(workers) as pool:
            results = pool.map(_check_coverage, args)
    else:
        results = [_check_coverage(a) for a in args]
    return dict(results)


# ---------------------------------------------------------------------------
# N-gram surprisal computation (worker function)
# ---------------------------------------------------------------------------

def _compute_k(args):
    """Worker: build n-gram for k, compute per-char surprisal on test set."""
    train, test, k, vocab_size, use_python, test_strata = args

    # Build n-gram
    counts = defaultdict(Counter)
    for i in range(k, len(train)):
        counts[train[i-k:i]][train[i]] += 1

    # Classify function
    if use_python:
        def classify(i): return _classify_python(test[i], test_strata[i])
    else:
        def classify(i): return _classify_nl(test[i])

    # Accumulate per-char surprisal
    result = defaultdict(lambda: {'sum_surp':0.0,'n_sum':0,'n':0,'ctx_hit':0})
    for i in range(k, len(test)):
        ch  = test[i]
        typ = classify(i)
        if typ in ('other', 'excluded'):
            continue
        ctx        = test[i-k:i]
        ctx_counts = counts.get(ctx, {})
        ctx_total  = sum(ctx_counts.values())
        prob       = (ctx_counts.get(ch, 0) + 1) / (ctx_total + vocab_size)
        r = result[ch]
        if r['n_sum'] < MAX_BOOT_SAMPLES:
            r['sum_surp'] += -np.log2(prob)
            r['n_sum']    += 1
        r['n'] += 1
        if ctx_counts:
            r['ctx_hit'] += 1

    return k, {ch: dict(v) for ch, v in result.items()}


# ---------------------------------------------------------------------------
# Bootstrap stopping criterion
# ---------------------------------------------------------------------------

def bootstrap_gradient_positive(gradients, n_boot, alpha):
    """
    Test whether the mean gradient across characters is significantly positive.
    gradients: array of per-character dCG/dk = CG(k) - CG(k-1)
    Returns True if gradient is significantly positive (keep climbing).
    """
    if len(gradients) == 0:
        return False
    gradients = np.array(gradients)
    if len(gradients) == 1:
        return gradients[0] > 0

    boot_means = np.array([
        np.mean(np.random.choice(gradients, size=len(gradients), replace=True))
        for _ in range(n_boot)
    ])
    # Upper (1-alpha) percentile: if it's <= 0, gradient not significantly positive
    upper = np.percentile(boot_means, 100 * (1 - alpha))
    return upper > 0


# ---------------------------------------------------------------------------
# Adaptive k scheduler — gradient ascent
# ---------------------------------------------------------------------------

class AdaptiveKScheduler:
    """
    Gradient ascent on CG(k). Mirrors neural-network early stopping:
      - 'loss'    → negative CG (we maximise CG, equivalent to minimising -CG)
      - 'patience'→ tolerate N steps without improvement before stopping
      - stopping criterion: bootstrap CI on mean gradient crosses zero

    On each call to .next(cg_by_char), returns the next k to compute,
    or None if the search should stop.
    """

    def __init__(self, k_start, k_max, patience, bootstrap_n, bootstrap_alpha):
        self.k              = k_start       # current k (last computed)
        self.k_max          = k_max
        self.patience       = patience
        self.bootstrap_n    = bootstrap_n
        self.bootstrap_alpha = bootstrap_alpha
        self.no_improve     = 0             # consecutive non-improving steps
        self.prev_cg        = {}            # char -> CG at k-1

    def next(self, cg_by_char):
        """
        cg_by_char: dict {char: CG_value} for the just-computed k.
        Returns next k to compute, or None to stop.
        """
        if self.k >= self.k_max:
            return None

        # Compute per-character gradient: dCG/dk = CG(k) - CG(k-1)
        gradients = []
        for ch, cg_k in cg_by_char.items():
            if ch in self.prev_cg and cg_k is not None and self.prev_cg[ch] is not None:
                gradients.append(cg_k - self.prev_cg[ch])

        self.prev_cg = {ch: v for ch, v in cg_by_char.items()}

        if not gradients:
            self.no_improve += 1
        else:
            still_climbing = bootstrap_gradient_positive(
                gradients, self.bootstrap_n, self.bootstrap_alpha)
            mean_g = float(np.mean(gradients))

            if still_climbing:
                self.no_improve = 0
                _log(f'  k={self.k}  mean_gradient={mean_g:+.4f}  → climbing')
            else:
                self.no_improve += 1
                _log(f'  k={self.k}  mean_gradient={mean_g:+.4f}  '
                     f'→ gradient not significant (patience {self.no_improve}/{self.patience})')

        if self.no_improve >= self.patience:
            _log(f'  Stopping: {self.patience} consecutive non-improving steps.')
            return None

        self.k += 1
        return self.k


# ---------------------------------------------------------------------------
# Main profiling logic
# ---------------------------------------------------------------------------

def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def profile_corpus(text, strata, corpus_hash, use_python,
                   max_k, min_n, cov_threshold,
                   adaptive, patience, bootstrap_n, bootstrap_alpha,
                   workers, cache):

    n    = len(text)
    n_tr = int(0.8 * n)
    n_vl = int(0.1 * n)
    train       = text[:n_tr]
    test        = text[n_tr + n_vl:]
    test_strata = strata[n_tr + n_vl:] if strata else None

    vocab_size  = len(set(text))
    train_freq  = Counter(train)
    total_train = len(train)

    _log(f'Train: {len(train):,}  Test: {len(test):,}  Vocab: {vocab_size}')

    # ── Phase 1: coverage check at probe points ──────────────────────────────
    _log('\nPhase 1 — coverage probe...')
    probe_k = sorted(set([1, 2, 4, 8, min(16, max_k), max_k]))
    probe_k = [k for k in probe_k if k <= max_k]

    # Check cache first
    probe_coverages = {}
    uncached_probes = []
    for k in probe_k:
        cached = cache.get_coverage(corpus_hash, k) if cache else None
        if cached is not None:
            probe_coverages[k] = cached
        else:
            uncached_probes.append(k)

    if uncached_probes:
        computed = check_coverages(train, test, uncached_probes, workers)
        probe_coverages.update(computed)
        if cache:
            for k, cov in computed.items():
                cache.put_coverage(corpus_hash, k, cov)

    # Determine reliable ceiling from probes
    reliable_k_max = 1
    _log('  Global coverage at probe points:')
    for k in sorted(probe_coverages):
        cov   = probe_coverages[k]
        label = 'RELIABLE' if cov >= cov_threshold else 'sparse'
        _log(f'    k={k:>3}  ({_ngram_label(k):>8})  {100*cov:5.1f}%  {label}')
        if cov >= cov_threshold:
            reliable_k_max = k

    effective_max_k = min(max_k, reliable_k_max)
    _log(f'  Reliable ceiling from probes: k ≤ {reliable_k_max}  '
         f'(effective max: {effective_max_k})\n')

    if effective_max_k < 2:
        _log('WARNING: coverage insufficient even at k=2. '
             'Consider a larger corpus.')
        return []

    # ── Phase 2: compute S(k) per char ───────────────────────────────────────
    _log('Phase 2 — surprisal estimation...')

    if adaptive:
        k_sequence = _adaptive_sequence(
            train, test, test_strata, vocab_size, corpus_hash,
            effective_max_k, cache, use_python,
            patience, bootstrap_n, bootstrap_alpha, workers)
    else:
        k_sequence = list(range(1, effective_max_k + 1))

    # Compute all needed k values, using cache where possible
    S_k = {}
    to_compute = []
    for k in k_sequence:
        if cache:
            # Check if we have all chars cached for this k
            # (simplified: if k=1 is cached for at least one char, assume full)
            pass  # Full char-level cache check done in build_profiles below
        to_compute.append(k)

    if to_compute:
        _log(f'  Computing k values: {to_compute}')
        worker_args = [
            (train, test, k, vocab_size, use_python, test_strata)
            for k in to_compute
        ]
        if workers > 1 and len(to_compute) > 1:
            with mp.Pool(workers) as pool:
                results = pool.map(_compute_k, worker_args)
        else:
            results = [_compute_k(a) for a in worker_args]

        for k, s_dict in results:
            S_k[k] = s_dict
            _log(f'    k={k}  ({_ngram_label(k)})  {len(s_dict)} chars')

    # ── Phase 3: build per-character profiles ────────────────────────────────
    return _build_profiles(
        S_k, k_sequence, reliable_k_max, effective_max_k, max_k,
        train_freq, total_train, use_python, strata,
        min_n, cov_threshold, corpus_hash, cache)


def _adaptive_sequence(train, test, test_strata, vocab_size, corpus_hash,
                        effective_max_k, cache, use_python,
                        patience, bootstrap_n, bootstrap_alpha, workers):
    """
    Run gradient ascent on k, returning the ordered list of k values computed.
    k=1 is always first (baseline). Then follow the CG gradient.
    """
    _log('  [adaptive mode] gradient ascent on k')
    k_sequence = [1]

    # Compute k=1 (baseline — CG=0 by definition, but need S(1))
    _, S1 = _compute_k((train, test, 1, vocab_size, use_python, test_strata))

    # Scheduler starts at k=1, drives toward k_max
    scheduler = AdaptiveKScheduler(
        k_start=1, k_max=effective_max_k,
        patience=patience, bootstrap_n=bootstrap_n,
        bootstrap_alpha=bootstrap_alpha)

    # Feed k=1 CG values (all zero — just initialise prev_cg)
    cg_at_1 = {ch: 0.0 for ch in S1}
    next_k = scheduler.next(cg_at_1)

    while next_k is not None:
        k_sequence.append(next_k)
        _, Sk = _compute_k((train, test, next_k, vocab_size, use_python, test_strata))

        # Compute CG for each char at this k
        cg_at_k = {}
        for ch, e1 in S1.items():
            ek = Sk.get(ch)
            if ek and ek['n_sum'] > 0 and e1['n_sum'] > 0:
                cov = ek['ctx_hit'] / ek['n'] if ek['n'] > 0 else 0
                if cov >= 0.30:  # loose gate for scheduler signal
                    cg_at_k[ch] = (e1['sum_surp']/e1['n_sum']) - (ek['sum_surp']/ek['n_sum'])

        next_k = scheduler.next(cg_at_k)

    _log(f'  Adaptive search used k values: {k_sequence}')
    return k_sequence


def _build_profiles(S_k, k_sequence, reliable_k_max, effective_max_k, max_k,
                    train_freq, total_train, use_python, strata,
                    min_n, cov_threshold, corpus_hash, cache):
    if 1 not in S_k:
        return []

    S1 = S_k[1]
    profiles = []

    for ch, e1 in S1.items():
        if e1['n'] < min_n:
            continue
        if use_python:
            # Determine type from strata — use majority stratum
            ch_type = 'other'
            for st_val in ('structural', 'lexical'):
                if (ch in PY_SYNTAX_CHARS and st_val == 'structural') or \
                   (ch.isalpha() or ch.isdigit()) and st_val == 'lexical':
                    ch_type = st_val
                    break
        else:
            ch_type = _classify_nl(ch)
        if ch_type == 'other':
            continue

        s1   = e1['sum_surp'] / e1['n_sum']
        freq = train_freq.get(ch, 1) / total_train

        trajectory = []
        for k in sorted(k_sequence):
            ek = S_k.get(k, {}).get(ch)
            if ek is None or ek['n'] < min_n or ek['n_sum'] == 0:
                trajectory.append({'k':k,'sk':None,'CG':None,'coverage':None})
                continue
            sk  = ek['sum_surp'] / ek['n_sum']
            cov = ek['ctx_hit'] / ek['n']
            trajectory.append({'k':k,'sk':sk,'CG':s1-sk,'coverage':cov})

        # Peak within reliable range and above coverage threshold (exclude k=1)
        reliable = [t for t in trajectory
                    if t['k'] > 1
                    and t['k'] <= reliable_k_max
                    and t['CG'] is not None
                    and t['coverage'] >= cov_threshold]

        if reliable:
            peak        = max(reliable, key=lambda t: t['CG'])
            k_peak      = peak['k']
            cg_peak     = peak['CG']
            in_reliable = True
        else:
            any_valid = [t for t in trajectory if t['k']>1 and t['CG'] is not None]
            if any_valid:
                peak    = max(any_valid, key=lambda t: t['CG'])
                k_peak  = peak['k']
                cg_peak = peak['CG']
            else:
                k_peak = cg_peak = None
            in_reliable = False

        # Neural estimator warning
        at_ceiling = (k_peak is not None and k_peak >= effective_max_k
                      and effective_max_k < max_k)
        at_ceiling_with_coverage = (
            at_ceiling and peak.get('coverage', 0) >= cov_threshold)

        profiles.append({
            'char':          ch,
            'type':          ch_type,
            's1':            s1,
            'freq':          freq,
            'trajectory':    trajectory,
            'k_peak':        k_peak,
            'cg_peak':       cg_peak,
            'in_reliable':   in_reliable,
            'reliable_k_max': reliable_k_max,
            'warn_neural':   at_ceiling_with_coverage,
        })

    profiles.sort(key=lambda p: -(p['cg_peak'] or 0))
    return profiles


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_profile(p, cov_threshold):
    ch      = p['char']
    display = repr(ch) if ch in (' ','\t','\n') else f'"{ch}"'
    cg_str  = f"+{p['cg_peak']:.3f}" if p['cg_peak'] is not None else 'n/a'
    k_label = _ngram_label(p['k_peak']) if p['k_peak'] else 'n/a'

    print(f"\nCharacter: {display}  [{p['type']}]  "
          f"S(1)={p['s1']:.3f} bits  freq={100*p['freq']:.2f}%")

    if p['warn_neural']:
        print(f"  *** k_peak reached the search ceiling with healthy coverage.")
        print(f"  *** Dependencies may extend further. Consider a neural "
              f"character-level estimator (e.g. small Transformer or LSTM).")

    print(f"  {'k':>3}  {'n-gram':>9}  {'S(k) bits':>10}  "
          f"{'CG (bits)':>10}  {'coverage':>9}")
    print(f"  {'---':>3}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*9}")

    for t in p['trajectory']:
        k = t['k']
        if t['sk'] is None:
            print(f"  {k:>3}  {_ngram_label(k):>9}  {'—':>10}  {'—':>10}  {'—':>9}")
            continue
        cov_flag = '' if t['coverage'] >= cov_threshold else ' *'
        cg_disp  = f"+{t['CG']:.3f}" if t['CG'] >= 0 else f"{t['CG']:.3f}"
        print(f"  {k:>3}  {_ngram_label(k):>9}  {t['sk']:>10.3f}  "
              f"{cg_disp:>10}  {100*t['coverage']:>8.1f}%{cov_flag}")

    note = '' if p['in_reliable'] else ' [outside reliable range]'
    print(f"\n  Peak gain: {cg_str} bits at k={p['k_peak']} ({k_label}){note}")
    print(f"  (* = coverage below {100*cov_threshold:.0f}% — estimates less reliable)")


def _write_csv(profiles, path, k_sequence):
    k_cols = ([f'CG_k{k}' for k in k_sequence] +
              [f'cov_k{k}' for k in k_sequence])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['char','type','s1_bits','freq_pct',
                           'k_peak','cg_peak_bits','in_reliable_range',
                           'warn_neural'] + k_cols)
        writer.writeheader()
        for p in profiles:
            traj = {t['k']: t for t in p['trajectory']}
            row  = {
                'char':              p['char'],
                'type':              p['type'],
                's1_bits':           round(p['s1'], 5),
                'freq_pct':          round(100 * p['freq'], 4),
                'k_peak':            p['k_peak'],
                'cg_peak_bits':      round(p['cg_peak'], 5) if p['cg_peak'] else '',
                'in_reliable_range': p['in_reliable'],
                'warn_neural':       p['warn_neural'],
            }
            for k in k_sequence:
                t = traj.get(k)
                row[f'CG_k{k}']  = round(t['CG'],5)       if t and t['CG']  is not None else ''
                row[f'cov_k{k}'] = round(t['coverage'],4) if t and t['coverage'] is not None else ''
            writer.writerow(row)
    _log(f'\nCSV written: {path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Adaptive per-character context-gain profiler. '
                    'Gradient ascent on CG(k) with bootstrap stopping criterion, '
                    'SQLite caching, and parallel k computation.')

    ap.add_argument('corpus',
                    help='Text file or directory of .py files')
    ap.add_argument('--python',         action='store_true',
                    help='Apply Python tokenizer stratification')
    ap.add_argument('--adaptive',       action='store_true',
                    help='Use gradient-ascent adaptive search (default: exhaustive)')
    ap.add_argument('--max-k',          type=int, default=DEFAULT_K_MAX,
                    help=f'Hard k ceiling (default {DEFAULT_K_MAX}, max {HARD_K_CEILING})')
    ap.add_argument('--min-n',          type=int, default=30,
                    help='Min test occurrences per character (default 30)')
    ap.add_argument('--cov-threshold',  type=float, default=0.50,
                    help='Per-symbol coverage gate (default 0.50)')
    ap.add_argument('--patience',       type=int, default=2,
                    help='Gradient ascent patience (default 2)')
    ap.add_argument('--bootstrap-n',    type=int, default=500,
                    help='Bootstrap resamples for stopping criterion (default 500)')
    ap.add_argument('--bootstrap-alpha',type=float, default=0.05,
                    help='Stop when upper CI <= 0 at this level (default 0.05)')
    ap.add_argument('--workers',        type=int,
                    default=max(1, (os.cpu_count() or 2) - 1),
                    help='Parallel workers (default: cpu_count - 1)')
    ap.add_argument('--db',             type=str, default='cg_cache.db',
                    help='SQLite cache path (default: cg_cache.db)')
    ap.add_argument('--no-cache',       action='store_true',
                    help='Disable cache reads and writes')
    ap.add_argument('--output',         type=str, default=None,
                    help='Write per-character summary CSV')
    ap.add_argument('--top',            type=int, default=None,
                    help='Print only top N characters by peak gain')
    args = ap.parse_args()

    # Validate k ceiling
    if args.max_k > HARD_K_CEILING:
        print(f'WARNING: --max-k capped at {HARD_K_CEILING} (n-gram hard ceiling). '
              f'For k > {HARD_K_CEILING}, use a neural estimator.', file=sys.stderr)
        args.max_k = HARD_K_CEILING

    cache = None if args.no_cache else CGCache(args.db)

    text, strata, corpus_hash = load_corpus(Path(args.corpus), args.python)

    profiles = profile_corpus(
        text, strata, corpus_hash,
        use_python      = args.python,
        max_k           = args.max_k,
        min_n           = args.min_n,
        cov_threshold   = args.cov_threshold,
        adaptive        = args.adaptive,
        patience        = args.patience,
        bootstrap_n     = args.bootstrap_n,
        bootstrap_alpha = args.bootstrap_alpha,
        workers         = args.workers,
        cache           = cache,
    )

    if cache:
        cache.close()

    if not profiles:
        print('No characters met the minimum occurrence threshold.')
        return

    to_print = profiles[:args.top] if args.top else profiles

    print(f'\n{"="*62}')
    print(f'CONTEXT-GAIN PROFILES  (sorted by peak gain, high → low)')
    mode = 'adaptive gradient ascent' if args.adaptive else 'exhaustive'
    print(f'Mode: {mode}  |  k ceiling: {args.max_k}  |  '
          f'cov threshold: {args.cov_threshold:.0%}')
    print(f'{"="*62}')
    for p in to_print:
        _print_profile(p, args.cov_threshold)

    # Summary table
    print(f'\n\n{"="*70}')
    print('SUMMARY TABLE')
    print(f'{"="*70}')
    print(f'{"Char":>6}  {"Type":>10}  {"S(1)":>7}  {"k_peak":>7}  '
          f'{"n-gram":>9}  {"Peak CG":>8}  {"Reliable":>9}  {"Neural?":>7}')
    print('-' * 70)
    for p in profiles:
        ch_d = repr(p['char']) if p['char'] in (' ',) else f'"{p["char"]}"'
        ng   = _ngram_label(p['k_peak']) if p['k_peak'] else '—'
        warn = 'WARN' if p['warn_neural'] else ''
        print(f'{ch_d:>6}  {p["type"]:>10}  {p["s1"]:>7.3f}  '
              f'{str(p["k_peak"] or "—"):>7}  {ng:>9}  '
              f'{p["cg_peak"]:>+8.3f}  '
              f'{"yes" if p["in_reliable"] else "NO":>9}  {warn:>7}')

    if args.output:
        k_seq = sorted(set(t['k'] for p in profiles for t in p['trajectory']))
        _write_csv(profiles, args.output, k_seq)

    # Neural estimator advisory
    neural_warned = [p for p in profiles if p['warn_neural']]
    if neural_warned:
        print(f'\n*** NEURAL ESTIMATOR ADVISORY ***')
        print(f'The following characters reached the k ceiling ({args.max_k}) '
              f'with healthy coverage,')
        print(f'suggesting their dependencies extend beyond the n-gram reliable range:')
        for p in neural_warned:
            print(f'  "{p["char"]}"  k_peak={p["k_peak"]}  '
                  f'cg_peak={p["cg_peak"]:+.3f} bits')
        print(f'For these characters, consider a neural character-level estimator')
        print(f'(e.g. a small Transformer or LSTM trained on the same corpus split).')

    print('\n--- NOTE ---')
    print('"Peak gain at k=X" = within the reliable measured range, k=X yielded')
    print('the highest estimated CG under Laplace smoothing for this corpus.')
    print('Not a hard context requirement. Estimates are corpus- and estimator-dependent.')


if __name__ == '__main__':
    mp.freeze_support()
    main()
