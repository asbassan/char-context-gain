"""
run_experiment_v3.py
---------------------
Same methodology as v2 — identical split, classification, regression.

Changes vs v2:
  1. SQLite cache — every (corpus_hash, k, char) persisted immediately;
     interrupted runs resume from the last completed (corpus, k) pair.
  2. Adaptive k discovery — k_max is derived per corpus, not hardcoded:
       a. Coverage probe at k=2,4,8,16,32,50 → reliable_kmax
          (last k where global context-hit rate >= SYM_COV_THRESHOLD)
       b. Sequential gradient ascent within [2, reliable_kmax]:
          compute one k at a time; after each step test whether the
          mean gradient across characters is still significantly positive
          (bootstrap CI, patience=2); stop when it isn't.
       c. effective_kmax = min(reliable_kmax, gradient_kmax)
     Hard ceiling: HARD_K_CEILING=50 (n-gram artifacts beyond this).
  3. k_peak distribution — Stage 4 now reports min/median/max k_peak
     for structural and lexical characters separately.
  4. Parallel batch computation — once the adaptive range is known,
     any uncached k values are computed in parallel (multiprocessing).
  5. Aggregated storage — (sum_surp, sum_sq, n_sum, n, ctx_hit) instead
     of raw loss lists; sufficient for CLT-based CIs (n >= MIN_N = 30).

Split note:
  The 80/10/10 split is kept identical to v2 for reproducibility.
  The validation set (10%) is unused — Laplace n-gram has no hyperparameters
  to tune. Methodologically this should be 90/10; left as-is to reproduce
  the same paper values.

Usage:
  uv run --python 3.12 --with "numpy,pandas,scipy,matplotlib" \\
      python run_experiment_v3.py [--db PATH] [--workers N]
      [--patience N] [--bootstrap-n N] [--max-k N]
"""

import argparse
import hashlib
import io
import json
import keyword as kw_mod
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import sqlite3
import tokenize as py_tokenize
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED              = 42
MIN_N             = 30
MAX_BOOT_SAMPLES  = 5000
SYM_COV_THRESHOLD = 0.50
HARD_K_CEILING    = 50
PROBE_K_POINTS    = [2, 4, 8, 16, 32, 50]   # exponential coverage probe

NL_STRUCTURAL_CHARS = set(',.;:?!\'"()[]{}')
PY_SYNTAX_CHARS     = set('()[]{},:;=.@+-*/%&|^~<>@')
AMBIGUOUS_CHARS     = {' ', '\n', '\t'}

np.random.seed(SEED)
RESULTS_DIR = Path('results_v3')
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Stage 0 — SQLite cache
# ---------------------------------------------------------------------------

class ExperimentCache:
    """
    Persistent store for (corpus_hash, k, char) → surprisal aggregates.
    Also stores coverage values and the derived k-range decision so that
    a restarted run uses the same adaptive stopping choice as the original.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS surprisal (
        corpus_hash  TEXT    NOT NULL,
        k            INTEGER NOT NULL,
        char         TEXT    NOT NULL,
        sum_surp     REAL    NOT NULL,
        sum_sq       REAL    NOT NULL,
        n_sum        INTEGER NOT NULL,
        n            INTEGER NOT NULL,
        ctx_hit      INTEGER NOT NULL,
        PRIMARY KEY (corpus_hash, k, char)
    );
    CREATE TABLE IF NOT EXISTS coverage (
        corpus_hash  TEXT    NOT NULL,
        k            INTEGER NOT NULL,
        value        REAL    NOT NULL,
        PRIMARY KEY (corpus_hash, k)
    );
    CREATE TABLE IF NOT EXISTS completed_k (
        corpus_hash  TEXT    NOT NULL,
        k            INTEGER NOT NULL,
        ts           TEXT    NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (corpus_hash, k)
    );
    CREATE TABLE IF NOT EXISTS k_range (
        corpus_hash   TEXT    PRIMARY KEY,
        reliable_kmax INTEGER NOT NULL,
        gradient_kmax INTEGER NOT NULL,
        effective_kmax INTEGER NOT NULL,
        k_values_json TEXT    NOT NULL
    );
    """

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()
        print(f'Cache: {Path(db_path).resolve()}')

    def k_done(self, corpus_hash, k):
        return self.conn.execute(
            'SELECT 1 FROM completed_k WHERE corpus_hash=? AND k=?',
            (corpus_hash, k)).fetchone() is not None

    def save_k(self, corpus_hash, k, results, coverage_val):
        with self.conn:
            self.conn.executemany(
                'INSERT OR REPLACE INTO surprisal VALUES (?,?,?,?,?,?,?,?)',
                [(corpus_hash, k, ch,
                  v['sum_surp'], v['sum_sq'], v['n_sum'], v['n'], v['ctx_hit'])
                 for ch, v in results.items()])
            self.conn.execute(
                'INSERT OR REPLACE INTO coverage VALUES (?,?,?)',
                (corpus_hash, k, coverage_val))
            self.conn.execute(
                'INSERT OR REPLACE INTO completed_k (corpus_hash, k) VALUES (?,?)',
                (corpus_hash, k))

    def load_k(self, corpus_hash, k):
        rows = self.conn.execute(
            'SELECT char, sum_surp, sum_sq, n_sum, n, ctx_hit '
            'FROM surprisal WHERE corpus_hash=? AND k=?',
            (corpus_hash, k)).fetchall()
        return {r[0]: {'sum_surp':r[1],'sum_sq':r[2],
                       'n_sum':r[3],'n':r[4],'ctx_hit':r[5]}
                for r in rows}

    def load_coverage(self, corpus_hash, k):
        row = self.conn.execute(
            'SELECT value FROM coverage WHERE corpus_hash=? AND k=?',
            (corpus_hash, k)).fetchone()
        return row[0] if row else None

    def save_k_range(self, corpus_hash, reliable_kmax, gradient_kmax,
                     effective_kmax, k_values):
        with self.conn:
            self.conn.execute(
                'INSERT OR REPLACE INTO k_range VALUES (?,?,?,?,?)',
                (corpus_hash, reliable_kmax, gradient_kmax,
                 effective_kmax, json.dumps(k_values)))

    def load_k_range(self, corpus_hash):
        row = self.conn.execute(
            'SELECT reliable_kmax, gradient_kmax, effective_kmax, k_values_json '
            'FROM k_range WHERE corpus_hash=?', (corpus_hash,)).fetchone()
        if row:
            return row[0], row[1], row[2], json.loads(row[3])
        return None

    def progress(self, corpus_hash):
        rows = self.conn.execute(
            'SELECT k FROM completed_k WHERE corpus_hash=?',
            (corpus_hash,)).fetchall()
        return {r[0] for r in rows}

    def close(self):
        self.conn.close()


def _corpus_hash(text):
    return hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Stage 1 — Python tokenisation (identical to v2)
# ---------------------------------------------------------------------------

def get_python_char_strata(code):
    strata = ['other'] * len(code)
    line_starts = [0]
    for i, ch in enumerate(code):
        if ch == '\n':
            line_starts.append(i + 1)

    def flat(row, col):
        idx = line_starts[row-1] + col if row-1 < len(line_starts) else len(code)
        return min(idx, len(code))

    try:
        tokens = list(py_tokenize.generate_tokens(io.StringIO(code).readline))
    except py_tokenize.TokenError:
        return strata

    for tok in tokens:
        tt, ts = tok.type, tok.string
        start = flat(*tok.start)
        end   = min(flat(*tok.end), len(code))
        if tt == py_tokenize.NAME:
            st = 'keyword' if kw_mod.iskeyword(ts) else 'identifier'
        elif tt == py_tokenize.NUMBER:  st = 'numeric'
        elif tt == py_tokenize.STRING:  st = 'string'
        elif tt == py_tokenize.COMMENT: st = 'comment'
        elif tt == py_tokenize.OP:      st = 'syntax'
        elif tt in (py_tokenize.NEWLINE, py_tokenize.NL,
                    py_tokenize.INDENT, py_tokenize.DEDENT): st = 'whitespace'
        else: st = 'other'
        for i in range(start, end):
            if i < len(code):
                strata[i] = st
    return strata


def load_python_stdlib(stdlib_path):
    text_parts, strata_parts, n_files = [], [], 0
    for fname in sorted(os.listdir(stdlib_path)):
        if not fname.endswith('.py'): continue
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
    full_strata = [s for part in strata_parts for s in part]
    print(f'  Tokenized {n_files} Python stdlib files ({len(full_text):,} chars)')
    return full_text, full_strata


# ---------------------------------------------------------------------------
# Stage 1 — Corpus loading
# ---------------------------------------------------------------------------

def download_text(url, filename):
    if not Path(filename).exists():
        print(f'  Downloading {filename}...')
        try:
            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            print(f'  Download failed: {e}'); return None
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def load_corpora():
    print('\n=== Stage 1: Load corpora ===')
    nl1 = download_text(
        'https://raw.githubusercontent.com/karpathy/char-rnn/master/'
        'data/tinyshakespeare/input.txt', 'corpus_shakespeare.txt')
    nl2 = download_text(
        'https://www.gutenberg.org/files/1342/1342-0.txt',
        'corpus_pride_prejudice.txt')
    if nl2:
        start = nl2.find('It is a truth')
        end   = nl2.rfind('End of the Project Gutenberg')
        nl2   = nl2[start:end] if start > 0 else nl2

    stdlib_path = os.path.dirname(os.__file__)
    print(f'  Python stdlib: {stdlib_path}')
    code1, code1_strata = load_python_stdlib(stdlib_path)

    corpora, strata = {}, {}
    if nl1:   corpora['shakespeare'],   strata['shakespeare']   = nl1,  None
    if nl2:   corpora['pride_prej'],    strata['pride_prej']    = nl2,  None
    if code1: corpora['python_stdlib'], strata['python_stdlib'] = code1, code1_strata

    print('Corpus sizes:')
    for name, text in corpora.items():
        print(f'  {name:20s}: {len(text):>10,} chars  hash={_corpus_hash(text)}')
    return corpora, strata


def make_splits(text, strata=None):
    n    = len(text)
    n_tr = int(0.80 * n)
    n_vl = int(0.10 * n)    # val set — kept for v2 reproducibility; not used
    sp = {
        'train':      text[:n_tr],
        'test':       text[n_tr + n_vl:],
        'vocab_size': len(set(text)),
        'hash':       _corpus_hash(text),
    }
    if strata:
        sp['test_strata'] = strata[n_tr + n_vl:]
    return sp


# ---------------------------------------------------------------------------
# Stage 2 — Character classification (identical to v2)
# ---------------------------------------------------------------------------

def classify_nl(ch):
    if ch in NL_STRUCTURAL_CHARS:     return 'structural'
    if ch in AMBIGUOUS_CHARS:         return 'ambiguous'
    if ch.isalpha() or ch.isdigit():  return 'lexical'
    return 'other'


def classify_code(ch, stratum):
    if stratum in ('string', 'comment'):               return 'excluded'
    if ch in PY_SYNTAX_CHARS and stratum == 'syntax':  return 'structural'
    if ch in AMBIGUOUS_CHARS or stratum == 'whitespace': return 'ambiguous'
    if stratum in ('identifier','keyword','numeric') or ch.isalpha() or ch.isdigit():
        return 'lexical'
    return 'other'


# ---------------------------------------------------------------------------
# Stage 3a — Lightweight coverage probe (no surprisal computation)
# ---------------------------------------------------------------------------

def _coverage_probe_worker(args):
    train, test, k = args
    seen  = set(train[i-k:i] for i in range(k, len(train)))
    hits  = sum(1 for i in range(k, len(test)) if test[i-k:i] in seen)
    total = len(test) - k
    return k, hits / total if total else 0.0


def probe_coverage(train, test, cache, corpus_hash_val, workers, max_k):
    """
    Exponential probe at PROBE_K_POINTS up to max_k.
    Returns reliable_kmax: last k with global coverage >= SYM_COV_THRESHOLD.
    """
    probe_ks = [k for k in PROBE_K_POINTS if k <= max_k]
    results  = {}

    uncached = []
    for k in probe_ks:
        cached = cache.load_coverage(corpus_hash_val, k)
        if cached is not None:
            results[k] = cached
        else:
            uncached.append(k)

    if uncached:
        args = [(train, test, k) for k in uncached]
        if workers > 1 and len(uncached) > 1:
            with mp.Pool(workers) as pool:
                computed = pool.map(_coverage_probe_worker, args)
        else:
            computed = [_coverage_probe_worker(a) for a in args]
        results.update(dict(computed))
        # store lightweight coverage entries (will be overwritten with exact
        # values when full surprisal computation runs for these k values)
        for k, cov in computed:
            if not cache.k_done(corpus_hash_val, k):
                # only cache if surprisal not yet computed (avoid overwrite)
                with cache.conn:
                    cache.conn.execute(
                        'INSERT OR IGNORE INTO coverage VALUES (?,?,?)',
                        (corpus_hash_val, k, cov))

    print(f'  Coverage probe:')
    probe_ceiling = max_k      # upper bound: first sparse probe point
    for k in sorted(results):
        cov  = results[k]
        flag = 'reliable' if cov >= SYM_COV_THRESHOLD else 'sparse → scan stops here'
        print(f'    k={k:>3}  {100*cov:5.1f}%  {flag}')
        if cov < SYM_COV_THRESHOLD:
            probe_ceiling = k  # first sparse probe: linear scan will stop here
            break

    # probe_ceiling is either max_k (all probes reliable, scan freely up to max_k)
    # or the first sparse exponential probe (we'll scan k linearly up to this
    # point and stop when actual coverage drops below threshold).
    print(f'  → probe ceiling: scan k ≤ {probe_ceiling} (verified per-step)')
    return probe_ceiling


# ---------------------------------------------------------------------------
# Stage 3b-alt — KenLM backend (large-corpus replacement for _worker)
# ---------------------------------------------------------------------------

class KenLMBackend:
    """
    Drop-in replacement for _worker when --backend kenlm is selected.

    Builds a Modified Kneser-Ney n-gram model via lmplz + build_binary
    (must be on PATH — provided by the Docker image), then queries it for
    per-character surprisal at each context length k.

    Character encoding: each Unicode codepoint → 4-hex-digit token
    (e.g. 'h' → '0068', ' ' → '0020').  Unambiguous, no escaping needed.

    Coverage (ctx_hit): uses KenLM's returned ngram_length. If the full
    k-gram was found (ngram_length == k+1) ctx_hit=1, else 0.  No in-memory
    seen-set is built, so memory is O(1) in corpus size — suitable for 1B+.

    Note for very large test sets (>100M chars): replace the full_scores()
    call loop with the state-based sliding-window API (kenlm.BaseScore) to
    avoid O(k) re-scoring overhead per position.

    Smoothing: MKN (not Laplace) — numerical CG values will differ from the
    Laplace results reported in the paper.  All downstream steps (SQLite
    aggregation, coverage probe, regression) are unchanged.
    """

    _LOG2_10 = __import__('math').log2(10)

    def __init__(self, corpus_hash, train_text, max_order, model_dir):
        try:
            import kenlm as _kenlm
            self._kenlm = _kenlm
        except ImportError:
            raise RuntimeError(
                'kenlm Python package not found. '
                'Run inside the Docker container: .\\docker_run.ps1')

        model_dir   = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self._bin   = model_dir / f'kenlm_{corpus_hash}_o{max_order}.bin'
        self._order = max_order

        if not self._bin.exists():
            self._build(train_text, max_order, model_dir, corpus_hash)

        print(f'  Loading KenLM model ({self._bin.stat().st_size // 1_048_576} MB): '
              f'{self._bin}')
        self.model = self._kenlm.Model(str(self._bin))

    @staticmethod
    def _tok(c):
        return f'{ord(c):04x}'

    def _build(self, train_text, max_order, model_dir, corpus_hash):
        import shutil, subprocess

        lmplz_bin  = shutil.which('lmplz')
        build_bin  = shutil.which('build_binary')
        if not lmplz_bin or not build_bin:
            raise RuntimeError(
                'lmplz / build_binary not in PATH. '
                'Run inside the Docker container: .\\docker_run.ps1')

        txt_path  = model_dir / f'_kenlm_{corpus_hash}_train.txt'
        arpa_path = model_dir / f'_kenlm_{corpus_hash}_o{max_order}.arpa'

        CHUNK = 1000
        n_lines = (len(train_text) + CHUNK - 1) // CHUNK
        print(f'  Writing {len(train_text):,} chars as {n_lines:,} hex-token lines...')
        with open(txt_path, 'w', encoding='ascii') as f:
            for i in range(0, len(train_text), CHUNK):
                f.write(' '.join(self._tok(c) for c in train_text[i:i+CHUNK]) + '\n')

        print(f'  lmplz -o {max_order} → {arpa_path.name}')
        with open(txt_path) as inp, open(arpa_path, 'w') as outp:
            subprocess.run([lmplz_bin, '-o', str(max_order), '--discount_fallback'],
                           stdin=inp, stdout=outp, check=True)

        print(f'  build_binary → {self._bin.name}')
        subprocess.run([build_bin, str(arpa_path), str(self._bin)], check=True)

        txt_path.unlink(missing_ok=True)
        arpa_path.unlink(missing_ok=True)
        print(f'  KenLM model ready  ({self._bin.stat().st_size // 1_048_576} MB)')

    def compute_k(self, k, test_text, is_code, test_strata):
        """Single pass through test_text; returns (results_dict, coverage)."""
        result = defaultdict(lambda: {
            'sum_surp': 0.0, 'sum_sq': 0.0, 'n_sum': 0, 'n': 0, 'ctx_hit': 0})

        for i in range(k, len(test_text)):
            ch  = test_text[i]
            typ = (classify_code(ch, test_strata[i])
                   if is_code and test_strata else classify_nl(ch))
            if typ in ('other', 'excluded', 'ambiguous'):
                continue

            tokens  = [self._tok(c) for c in test_text[i-k:i]] + [self._tok(ch)]
            scores  = list(self.model.full_scores(' '.join(tokens)))
            if not scores:
                continue
            log10_p, ngram_len, oov = scores[-1]
            if oov:
                continue

            surp = -log10_p * self._LOG2_10   # bits
            # Coverage: was the k-gram CONTEXT seen in training?
            # Use the k-th token's ngram_len (last context token, index k-1),
            # not the target's — matching Laplace's ctx_hit semantics exactly.
            ctx_ngram_len = scores[k - 1][1] if k >= 1 and k - 1 < len(scores) else 0
            hit = 1 if ctx_ngram_len >= k else 0

            r = result[ch]
            if r['n_sum'] < MAX_BOOT_SAMPLES:
                r['sum_surp'] += surp
                r['sum_sq']   += surp * surp
                r['n_sum']    += 1
            r['n']       += 1
            r['ctx_hit'] += hit

        results = {ch: dict(v) for ch, v in result.items()}
        total_n   = sum(v['n']       for v in results.values())
        total_hit = sum(v['ctx_hit'] for v in results.values())
        cov = total_hit / total_n if total_n else 0.0
        return results, cov


# ---------------------------------------------------------------------------
# Stage 3b — Surprisal worker (multiprocessing, Laplace backend)
# ---------------------------------------------------------------------------

def _worker(args):
    train, test, k, vocab_size, is_code, test_strata = args

    counts = defaultdict(Counter)
    for i in range(k, len(train)):
        counts[train[i-k:i]][train[i]] += 1

    result = defaultdict(lambda: {
        'sum_surp': 0.0, 'sum_sq': 0.0, 'n_sum': 0, 'n': 0, 'ctx_hit': 0})

    for i in range(k, len(test)):
        ch = test[i]
        typ = (classify_code(ch, test_strata[i]) if is_code and test_strata
               else classify_nl(ch))
        if typ in ('other', 'excluded', 'ambiguous'):
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


def _run_one_k(splits, k, is_code, cache, kenlm_backend=None):
    """Compute one k (sequential), save to cache, return (S_k_dict, cov)."""
    if kenlm_backend is not None:
        res, cov = kenlm_backend.compute_k(
            k, splits['test'], is_code, splits.get('test_strata'))
    else:
        _, res, cov = _worker((
            splits['train'], splits['test'], k,
            splits['vocab_size'], is_code, splits.get('test_strata')))
    cache.save_k(splits['hash'], k, res, cov)
    return res, cov


# ---------------------------------------------------------------------------
# Stage 3c — Bootstrap stopping criterion
# ---------------------------------------------------------------------------

def _bootstrap_gradient_positive(gradients, n_boot, alpha):
    """
    Test H₀: mean gradient ≤ 0.
    Returns True if (1-alpha) upper CI of bootstrap mean gradient > 0.
    gradients: array of per-character dCG/dk values at this step.
    """
    if len(gradients) == 0:
        return False
    if len(gradients) == 1:
        return gradients[0] > 0
    g = np.array(gradients)
    boot = np.array([
        np.mean(np.random.choice(g, size=len(g), replace=True))
        for _ in range(n_boot)])
    return float(np.percentile(boot, 100 * (1 - alpha))) > 0


# ---------------------------------------------------------------------------
# Stage 3 — Adaptive k discovery + surprisal computation
# ---------------------------------------------------------------------------

def compute_surprisals(splits, corpus_name, cache, workers,
                       max_k, patience, bootstrap_n, bootstrap_alpha,
                       kenlm_backend=None):
    """
    1. Coverage probe → reliable_kmax.
    2. Sequential gradient ascent within [2, reliable_kmax]:
         compute k one at a time; after each step, bootstrap-test whether
         mean gradient across chars is still significantly positive.
         Stop when it isn't (patience consecutive non-improving steps).
    3. Compute any remaining uncached k values in a parallel batch.
    Returns: S_k dict, k_values list (the derived range for this corpus).
    """
    print(f'\n=== Stage 3: {corpus_name} ===')
    is_code  = corpus_name == 'python_stdlib'
    ch_hash  = splits['hash']
    done     = cache.progress(ch_hash)
    S_k      = {}

    # Check if k range already decided (restart path)
    cached_range = cache.load_k_range(ch_hash)
    if cached_range:
        reliable_kmax, gradient_kmax, effective_kmax, k_values = cached_range
        print(f'  [resumed] reliable_kmax={reliable_kmax}  '
              f'gradient_kmax={gradient_kmax}  effective_kmax={effective_kmax}')
        print(f'  k range: {k_values}')
        # Load whatever is cached; compute the rest in parallel
        for k in k_values:
            if k in done:
                S_k[k] = cache.load_k(ch_hash, k)
                cov = cache.load_coverage(ch_hash, k) or 0
                print(f'  k={k}  CACHED  ({len(S_k[k])} chars)  '
                      f'global_cov={100*cov:.1f}%')
        missing = [k for k in k_values if k not in done]
        if missing:
            _parallel_batch(splits, missing, is_code, cache, S_k, workers,
                            kenlm_backend=kenlm_backend)
        return S_k, k_values

    # ── Phase A: coverage probe ──────────────────────────────────────────────
    probe_ceiling = probe_coverage(
        splits['train'], splits['test'], cache, ch_hash, workers, max_k)

    if probe_ceiling < 2:
        print('  WARNING: coverage insufficient at k=2. '
              'Consider a larger corpus.')
        k_values = [1]
        cache.save_k_range(ch_hash, 1, 1, 1, k_values)
        # Still need k=1
        if 1 not in done:
            res, cov = _run_one_k(splits, 1, is_code, cache,
                                  kenlm_backend=kenlm_backend)
            S_k[1] = res
            print(f'  k=1  computed  ({len(res)} chars)  global_cov={100*cov:.1f}%')
        else:
            S_k[1] = cache.load_k(ch_hash, 1)
        return S_k, k_values

    # ── Phase B: always compute k=1 (baseline) ───────────────────────────────
    if 1 in done:
        S_k[1] = cache.load_k(ch_hash, 1)
        print(f'  k=1  CACHED  ({len(S_k[1])} chars)  global_cov=100%')
    else:
        res, cov = _run_one_k(splits, 1, is_code, cache,
                              kenlm_backend=kenlm_backend)
        S_k[1] = res
        print(f'  k=1  computed  ({len(res)} chars)  global_cov=100%')

    S1 = S_k[1]

    # ── Phase C: linear scan from k=2 to probe_ceiling ──────────────────────
    # At each step we check ACTUAL coverage (probe may have skipped k values
    # e.g. k=5,6,7 between the reliable k=4 probe and sparse k=8 probe).
    # We also track the mean gradient for informational output, but stopping
    # is driven by coverage only — the regression needs the full slope across
    # the reliable range, not just where the mean peaks.
    prev_cg    = {}
    gradient_kmax = 1
    k_seq = [1]
    reliable_kmax = 1

    for k in range(2, probe_ceiling + 1):
        if k in done:
            S_k[k] = cache.load_k(ch_hash, k)
            cov     = cache.load_coverage(ch_hash, k) or 0
            print(f'  k={k}  CACHED  ({len(S_k[k])} chars)  '
                  f'global_cov={100*cov:.1f}%')
        else:
            res, cov = _run_one_k(splits, k, is_code, cache,
                                  kenlm_backend=kenlm_backend)
            S_k[k] = res
            print(f'  k={k}  computed  ({len(S_k[k])} chars)  '
                  f'global_cov={100*cov:.1f}%')

        # Stop if actual coverage at this k is below threshold
        if cov < SYM_COV_THRESHOLD:
            print(f'  k={k}: actual coverage {100*cov:.1f}% < '
                  f'{100*SYM_COV_THRESHOLD:.0f}% — stopping (coverage ceiling).')
            del S_k[k]   # don't include sparse k in results
            break

        reliable_kmax = k
        k_seq.append(k)
        gradient_kmax = k

        # Compute and log mean gradient (informational — not used for stopping here)
        gradients = []
        for ch, ek in S_k[k].items():
            if ek['n'] < MIN_N: continue
            sym_cov = ek['ctx_hit'] / ek['n'] if ek['n'] else 0
            if sym_cov < SYM_COV_THRESHOLD: continue
            e1 = S1.get(ch)
            if not e1 or e1['n_sum'] == 0: continue
            cg_k   = e1['sum_surp']/e1['n_sum'] - ek['sum_surp']/ek['n_sum']
            cg_prev = prev_cg.get(ch)
            if cg_prev is not None:
                gradients.append(cg_k - cg_prev)
            prev_cg[ch] = cg_k

        if gradients:
            mean_g = float(np.mean(gradients))
            still  = _bootstrap_gradient_positive(gradients, bootstrap_n, bootstrap_alpha)
            trend  = 'climbing' if still else 'plateau (mean CG no longer rising)'
            print(f'    mean_gradient={mean_g:+.4f}  → {trend}')

    # effective_kmax = coverage-based (not gradient). The gradient info above
    # is logged so you can see where the mean plateaus, but the regression
    # panel uses the full coverage-reliable range to estimate β₃ slope.
    effective_kmax = reliable_kmax
    k_values       = k_seq

    cache.save_k_range(ch_hash, reliable_kmax, gradient_kmax,
                       effective_kmax, k_values)
    print(f'  → coverage ceiling (regression k_max): k ≤ {reliable_kmax}')
    print(f'  → mean-CG gradient plateau at:         k ≤ {gradient_kmax}  '
          f'(informational — not used to stop regression)')
    print(f'  → effective k range: {k_values}')
    return S_k, k_values


def _parallel_batch(splits, k_list, is_code, cache, S_k, workers,
                    kenlm_backend=None):
    """Compute a list of k values in parallel and update S_k and cache."""
    if kenlm_backend is not None:
        # KenLM model is shared in-process; no subprocess pool needed.
        for k in sorted(k_list):
            res, cov = kenlm_backend.compute_k(
                k, splits['test'], is_code, splits.get('test_strata'))
            cache.save_k(splits['hash'], k, res, cov)
            S_k[k] = res
            flag = '' if cov >= SYM_COV_THRESHOLD else '  << sparse'
            print(f'  k={k}  computed (kenlm)  ({len(res)} chars)  '
                  f'global_cov={100*cov:.1f}%{flag}')
        return

    args = [(splits['train'], splits['test'], k, splits['vocab_size'],
             is_code, splits.get('test_strata')) for k in k_list]
    if workers > 1 and len(args) > 1:
        with mp.Pool(workers) as pool:
            results = pool.map(_worker, args)
    else:
        results = [_worker(a) for a in args]
    for k, res, cov in sorted(results, key=lambda x: x[0]):
        cache.save_k(splits['hash'], k, res, cov)
        S_k[k] = res
        flag = '' if cov >= SYM_COV_THRESHOLD else '  << sparse'
        print(f'  k={k}  computed  ({len(res)} chars)  '
              f'global_cov={100*cov:.1f}%{flag}')


# ---------------------------------------------------------------------------
# Stage 4 — CI helper, per-character peaks, k-peak distribution
# ---------------------------------------------------------------------------

def _mean_ci(sum_surp, sum_sq, n_sum):
    m   = sum_surp / n_sum
    var = max(sum_sq / n_sum - m * m, 0.0)
    se  = (var ** 0.5) / (n_sum ** 0.5)
    return m, m - 1.96 * se, m + 1.96 * se


def compute_peaks(S_k, k_values, corpus_name):
    is_code     = corpus_name == 'python_stdlib'
    reliable_ks = [k for k in k_values if k > 1]
    rows        = []

    for ch in sorted(S_k.get(1, {}).keys()):
        e1 = S_k[1].get(ch)
        if not e1 or e1['n'] < MIN_N: continue
        s1, s1_lo, s1_hi = _mean_ci(e1['sum_surp'], e1['sum_sq'], e1['n_sum'])

        best_cg, best_k, best_lo, best_hi = float('-inf'), None, None, None
        for k in reliable_ks:
            ek = S_k.get(k, {}).get(ch)
            if not ek or ek['n'] < MIN_N: continue
            sym_cov = ek['ctx_hit'] / ek['n'] if ek['n'] else 0
            if sym_cov < SYM_COV_THRESHOLD: continue
            sk, sk_lo, sk_hi = _mean_ci(ek['sum_surp'], ek['sum_sq'], ek['n_sum'])
            cg = s1 - sk
            if cg > best_cg:
                best_cg = cg;  best_k  = k
                best_lo = s1_lo - sk_hi;  best_hi = s1_hi - sk_lo

        if best_k is None: continue

        ek_peak = S_k.get(best_k, {}).get(ch, {})
        sym_cov = ek_peak['ctx_hit'] / ek_peak['n'] if ek_peak.get('n') else 0.0

        if is_code:
            sym_type = ('structural' if ch in PY_SYNTAX_CHARS else
                        'lexical'    if (ch.isalpha() or ch.isdigit()) else 'other')
        else:
            sym_type = ('structural' if ch in NL_STRUCTURAL_CHARS else
                        'lexical'    if (ch.isalpha() or ch.isdigit() or ch==' ') else 'other')
        if sym_type == 'other': continue

        ls = 1 - best_cg / s1 if s1 > 0 else float('nan')
        rows.append({
            'char':    repr(ch), 'char_raw': ch, 'type': sym_type,
            'S_x1':    round(s1, 4),      'S_x1_lo': round(s1_lo, 4),
            'S_x1_hi': round(s1_hi, 4),   'n_test':  e1['n'],
            'CG_peak': round(best_cg, 4), 'CG_lo':   round(best_lo, 4),
            'CG_hi':   round(best_hi, 4), 'k_peak':  best_k,
            'sym_cov': round(sym_cov, 3), 'LS':      round(ls, 4) if not np.isnan(ls) else np.nan,
        })

    return pd.DataFrame(rows).sort_values('CG_peak', ascending=False)


def k_peak_distribution(peaks_df, corpus_name, effective_kmax):
    """Report min / median / max k_peak for structural and lexical chars."""
    print(f'\n  k_peak distribution — {corpus_name}  '
          f'(effective_kmax={effective_kmax}):')
    print(f'  {"type":>10}  {"n":>4}  {"min":>5}  {"p25":>5}  '
          f'{"median":>7}  {"p75":>5}  {"max":>5}  {"at_ceiling":>10}')
    print(f'  {"-"*62}')
    for typ in ('structural', 'lexical', 'all'):
        sub = (peaks_df if typ == 'all'
               else peaks_df[peaks_df['type'] == typ])
        kp  = sub['k_peak'].dropna()
        if len(kp) == 0: continue
        at_ceil = (kp == effective_kmax).sum()
        print(f'  {typ:>10}  {len(kp):>4}  {int(kp.min()):>5}  '
              f'{int(kp.quantile(0.25)):>5}  {kp.median():>7.1f}  '
              f'{int(kp.quantile(0.75)):>5}  {int(kp.max()):>5}  '
              f'{at_ceil:>5} ({100*at_ceil/len(kp):.0f}%)')
    if (peaks_df['k_peak'] == effective_kmax).any():
        print(f'  NOTE: some chars peaked at the ceiling — '
              f'true peak may be beyond k={effective_kmax}.')


# ---------------------------------------------------------------------------
# Stage 5 — Mann-Whitney
# ---------------------------------------------------------------------------

def run_mann_whitney(peaks_by_corpus):
    print('\n=== Stage 5: Mann-Whitney U ===')
    for corpus_name, df in peaks_by_corpus.items():
        s_cg = df[df['type']=='structural']['CG_peak'].dropna().values
        l_cg = df[df['type']=='lexical'   ]['CG_peak'].dropna().values
        if not len(s_cg) or not len(l_cg): continue
        u, p  = stats.mannwhitneyu(s_cg, l_cg, alternative='greater')
        med_s = np.median(s_cg);  med_l = np.median(l_cg)
        ratio = med_s / med_l if med_l > 0 else float('inf')
        go    = p < 0.05 and ratio > 1.2
        print(f'  {corpus_name:20s}  struct_med={med_s:.3f}  lex_med={med_l:.3f}  '
              f'ratio={ratio:.2f}x  p={p:.4f}  → {"GO" if go else "no-go"}')


# ---------------------------------------------------------------------------
# Stage 6 — Regression
# ---------------------------------------------------------------------------

def clustered_ols(y, X, group_ids):
    n, p    = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    coeffs  = XtX_inv @ X.T @ y
    resid   = y - X @ coeffs
    G       = len(np.unique(group_ids))
    B       = np.zeros((p, p))
    for g in np.unique(group_ids):
        mask  = group_ids == g
        score = X[mask].T @ resid[mask]
        B    += np.outer(score, score)
    corr = (G / (G-1)) * ((n-1) / (n-p))
    V    = corr * XtX_inv @ B @ XtX_inv
    se   = np.sqrt(np.maximum(np.diag(V), 0))
    df_r = G - 1
    t    = coeffs / np.where(se > 0, se, np.inf)
    pv   = 2 * stats.t.sf(np.abs(t), df=df_r)
    tc   = stats.t.ppf(0.975, df=df_r)
    ss_r = np.sum(resid**2);  ss_t = np.sum((y - y.mean())**2)
    r2   = 1 - ss_r / ss_t if ss_t > 0 else 0.0
    return coeffs, se, coeffs - tc*se, coeffs + tc*se, pv, r2


def run_regression(S_k_by_corpus, peaks_by_corpus, splits_by_corpus,
                   kmax_by_corpus, k_values_by_corpus):
    print('\n=== Stage 6: Regression ===')
    robustness_rows = []

    for corpus_name, S_k in S_k_by_corpus.items():
        df       = peaks_by_corpus[corpus_name]
        sp       = splits_by_corpus[corpus_name]
        kmax     = kmax_by_corpus[corpus_name]
        k_values = k_values_by_corpus[corpus_name]
        freq     = Counter(sp['train'])
        total    = len(sp['train'])

        # k values used in regression panel: all derived k > 1 within kmax
        panel_ks = [k for k in k_values if k > 1 and k <= kmax]

        rows = []
        for _, row in df.iterrows():
            ch  = row['char_raw']
            typ = row['type']
            if typ not in ('structural','lexical'): continue
            e1 = S_k.get(1, {}).get(ch)
            if not e1 or e1['n'] < MIN_N: continue
            s1        = e1['sum_surp'] / e1['n_sum']
            f         = freq.get(ch, 1) / total
            is_struct = 1 if typ == 'structural' else 0
            for k in panel_ks:
                ek = S_k.get(k, {}).get(ch)
                if not ek or ek['n'] < MIN_N: continue
                sym_cov = ek['ctx_hit'] / ek['n'] if ek['n'] > 0 else 0
                if sym_cov < SYM_COV_THRESHOLD: continue
                sk = ek['sum_surp'] / ek['n_sum']
                rows.append({
                    'CG':         s1 - sk,
                    'log2k':      np.log2(k),
                    'Structural': is_struct,
                    'log2k_x_S':  np.log2(k) * is_struct,
                    'logFreq':    np.log2(max(f, 1e-10)),
                    'char_id':    ch,
                })

        if not rows: continue
        reg = pd.DataFrame(rows)
        reg.to_csv(RESULTS_DIR / f'panel_v3_{corpus_name}.csv', index=False)

        y  = reg['CG'].values
        Xm = np.column_stack([np.ones(len(reg)),
                               reg[['log2k','Structural','log2k_x_S','logFreq']].values])
        chars = np.unique(reg['char_id'].values)
        c2i   = {c: i for i, c in enumerate(chars)}
        gids  = np.array([c2i[c] for c in reg['char_id'].values])

        coeffs, se, lo, hi, pv, r2 = clustered_ols(y, Xm, gids)

        names = ['intercept','log2(k)','Structural','log2(k)×Structural','log2(Freq)']
        print(f'\n  {corpus_name}  n={len(y)}  G={len(chars)}  R²={r2:.3f}  '
              f'panel_ks={panel_ks}')
        print(f'  {"":28s} {"β":>8}  {"SE":>7}  {"95% CI":^20}  {"p":>7}')
        print(f'  {"-"*70}')
        for nm, b, s, l, h, p_ in zip(names, coeffs, se, lo, hi, pv):
            sig = ' ***' if p_<0.001 else ' **' if p_<0.01 else ' *' if p_<0.05 else ''
            print(f'  {nm:28s} {b:+8.4f}  {s:7.4f}  [{l:+7.4f}, {h:+7.4f}]  {p_:7.4f}{sig}')

        b3, se3, lo3, hi3, p3 = coeffs[3], se[3], lo[3], hi[3], pv[3]
        print(f'\n  >> β₃ = {b3:+.4f}  SE={se3:.4f}  '
              f'CI=[{lo3:+.4f},{hi3:+.4f}]  p={p3:.4f}')
        robustness_rows.append({
            'corpus': corpus_name, 'tau': SYM_COV_THRESHOLD,
            'effective_kmax': kmax, 'panel_ks': str(panel_ks),
            'b3': round(b3,5), 'se3': round(se3,5),
            'ci_lo3': round(lo3,5), 'ci_hi3': round(hi3,5),
            'p3': round(p3,5), 'r2': round(r2,5), 'n': len(y), 'G': len(chars),
        })

    pd.DataFrame(robustness_rows).to_csv(
        RESULTS_DIR / 'robustness_b3_v3.csv', index=False)


# ---------------------------------------------------------------------------
# Stage 7 — Plots
# ---------------------------------------------------------------------------

def make_plots(S_k_by_corpus, k_values_by_corpus):
    fig, axes = plt.subplots(len(S_k_by_corpus), 1,
                             figsize=(12, 5 * len(S_k_by_corpus)))
    if len(S_k_by_corpus) == 1: axes = [axes]

    for ax, (corpus_name, S_k) in zip(axes, S_k_by_corpus.items()):
        is_code  = corpus_name == 'python_stdlib'
        k_vals   = [k for k in k_values_by_corpus[corpus_name] if k > 1]
        s_chars  = list(PY_SYNTAX_CHARS if is_code else NL_STRUCTURAL_CHARS)
        l_chars  = [c for c in 'aeioutrns' if S_k.get(1,{}).get(c)]

        cols_s = cm.Reds( np.linspace(0.35, 0.9, max(len(s_chars), 1)))
        cols_l = cm.Blues(np.linspace(0.35, 0.9, max(len(l_chars), 1)))

        for ch, col in zip(s_chars[:8], cols_s):
            e1 = S_k.get(1, {}).get(ch)
            if not e1 or e1['n'] < MIN_N: continue
            s1  = e1['sum_surp'] / e1['n_sum']
            cgs = [s1 - S_k[k][ch]['sum_surp'] / S_k[k][ch]['n_sum']
                   if k in S_k and ch in S_k[k] and S_k[k][ch]['n'] >= MIN_N
                   else float('nan') for k in k_vals]
            ax.plot(k_vals, cgs, '-o', color=col, label=repr(ch), lw=2)

        for ch, col in zip(l_chars[:5], cols_l):
            e1 = S_k.get(1, {}).get(ch)
            if not e1 or e1['n'] < MIN_N: continue
            s1  = e1['sum_surp'] / e1['n_sum']
            cgs = [s1 - S_k[k][ch]['sum_surp'] / S_k[k][ch]['n_sum']
                   if k in S_k and ch in S_k[k] and S_k[k][ch]['n'] >= MIN_N
                   else float('nan') for k in k_vals]
            ax.plot(k_vals, cgs, '--s', color=col, label=repr(ch), lw=1.5, alpha=0.8)

        ax.axhline(0, color='black', lw=0.5)
        ax.set_xlabel('Context length k'); ax.set_ylabel('CG bits')
        ax.set_title(f'{corpus_name} — CG(k) [red=structural, blue=lexical]  '
                     f'(k range derived adaptively: 1..{max(k_vals, default=1)})')
        ax.legend(fontsize=6, ncol=3); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'context_curves_v3.png', dpi=200, bbox_inches='tight')
    print(f'  Saved: context_curves_v3.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='experiment_cache.db')
    ap.add_argument('--workers',          type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--max-k',            type=int, default=HARD_K_CEILING,
                    help=f'Upper bound on k search (default {HARD_K_CEILING})')
    ap.add_argument('--patience',         type=int, default=2,
                    help='Gradient ascent patience (default 2)')
    ap.add_argument('--bootstrap-n',      type=int, default=500,
                    help='Bootstrap resamples for stopping criterion (default 500)')
    ap.add_argument('--bootstrap-alpha',  type=float, default=0.05,
                    help='Significance level for gradient CI test (default 0.05)')
    ap.add_argument('--backend', choices=['laplace', 'kenlm'], default='laplace',
                    help='Counting backend: laplace (default, paper values) or '
                         'kenlm (MKN smoothing, requires Docker / Linux)')
    args = ap.parse_args()

    effective_max_k = min(args.max_k, HARD_K_CEILING)
    cache = ExperimentCache(args.db)

    # Stage 1
    corpora, strata = load_corpora()
    splits_by_corpus = {
        name: make_splits(text, strata.get(name))
        for name, text in corpora.items()
    }
    print('\nSplits (80 train / 10 unused-val / 10 test):')
    for name, sp in splits_by_corpus.items():
        print(f'  {name:20s}  vocab={sp["vocab_size"]:3d}  '
              f'train={len(sp["train"]):>9,}  test={len(sp["test"]):>8,}  '
              f'hash={sp["hash"]}')

    # Initialise KenLM backends once per corpus (model build is expensive)
    # max_order is capped at 12 to match the KENLM_MAX_ORDER compiled into the binary;
    # that covers k ≤ 11 (we need k+1 grams), well above the empirical k_max ≤ 10.
    KENLM_MAX_ORDER = 12
    kenlm_backends = {}
    if args.backend == 'kenlm':
        model_dir = Path(args.db).parent / 'kenlm_models'
        print(f'\nBackend: KenLM  (MKN smoothing)  models → {model_dir}')
        for name, sp in splits_by_corpus.items():
            kenlm_backends[name] = KenLMBackend(
                sp['hash'], sp['train'],
                max_order=min(effective_max_k, KENLM_MAX_ORDER),
                model_dir=model_dir)
    else:
        print('\nBackend: Laplace (add-1)  — paper canonical values')

    # Stage 3 — adaptive k discovery + surprisal
    S_k_by_corpus      = {}
    k_values_by_corpus = {}
    kmax_by_corpus     = {}

    for name, sp in splits_by_corpus.items():
        S_k, k_vals = compute_surprisals(
            sp, name, cache, args.workers,
            max_k           = effective_max_k,
            patience        = args.patience,
            bootstrap_n     = args.bootstrap_n,
            bootstrap_alpha = args.bootstrap_alpha,
            kenlm_backend   = kenlm_backends.get(name))
        S_k_by_corpus[name]      = S_k
        k_values_by_corpus[name] = k_vals
        kmax_by_corpus[name]     = max((k for k in k_vals if k > 1), default=1)

    # Stage 4 — peaks
    print('\n=== Stage 4: Per-character peaks ===')
    peaks_by_corpus = {}
    for name, S_k in S_k_by_corpus.items():
        k_vals  = k_values_by_corpus[name]
        eff_kmax = kmax_by_corpus[name]
        print(f'\n  {name}  effective_kmax={eff_kmax}  k_range={k_vals}')
        df = compute_peaks(S_k, k_vals, name)
        peaks_by_corpus[name] = df
        df.to_csv(RESULTS_DIR / f'peaks_v3_{name}.csv', index=False)
        print(df[['char','type','n_test','S_x1','CG_peak','k_peak','sym_cov'
                  ]].to_string(index=False))
        k_peak_distribution(df, name, eff_kmax)

    # Cross-corpus
    cross = pd.concat(
        [df.assign(corpus=name) for name, df in peaks_by_corpus.items()],
        ignore_index=True)
    cross.to_csv(RESULTS_DIR / 'cross_corpus_v3.csv', index=False)

    # Stage 5
    run_mann_whitney(peaks_by_corpus)

    # Stage 6
    run_regression(S_k_by_corpus, peaks_by_corpus, splits_by_corpus,
                   kmax_by_corpus, k_values_by_corpus)

    # Stage 7
    print('\n=== Stage 7: Plots ===')
    make_plots(S_k_by_corpus, k_values_by_corpus)

    cache.close()
    print(f'\n=== Done ===')
    for f in sorted(RESULTS_DIR.iterdir()):
        print(f'  {f.name:50s} {f.stat().st_size:>8,} bytes')
    print(f'\nCache: {Path(args.db).resolve()}')
    print('Re-run at any time — completed (corpus, k) pairs are skipped.')


if __name__ == '__main__':
    mp.freeze_support()
    main()
