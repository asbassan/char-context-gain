"""
run_split_robustness.py  —  P1: split robustness for NL1 and NL2.

Runs the canonical regression on 5 different contiguous 80/10 splits
of tinyshakespeare and Pride & Prejudice, varying the split start offset.
Each split takes a non-overlapping 80% train / 10% test slice starting
at a different character position.

Output: results_tmlr/split_robustness.csv
"""
import sys, os, csv, random, math
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from scipy import stats as scipy_stats

# ─── Minimal n-gram machinery (same as canonical pipeline) ────────────────────

def load_text(path):
    with open(path, encoding='utf-8') as f:
        return f.read()

def build_ngram(text, k):
    """Return count dict for k+1-grams (context of length k → next char)."""
    counts = {}
    for i in range(k, len(text)):
        ctx = text[i-k:i]
        ch  = text[i]
        if ctx not in counts:
            counts[ctx] = {}
        counts[ctx][ch] = counts[ctx].get(ch, 0) + 1
    return counts

def laplace_surprisal(ctx, ch, counts, vocab_size):
    """Laplace-smoothed surprisal of ch given ctx."""
    ctx_counts = counts.get(ctx, {})
    ctx_total  = sum(ctx_counts.values())
    ch_count   = ctx_counts.get(ch, 0)
    p = (ch_count + 1) / (ctx_total + vocab_size)
    return -math.log2(p)

def compute_panel(text, train_frac=0.8, test_frac=0.1, max_k=8, min_n=30,
                  tau=0.5, seed=42, offset=0):
    """
    Build regression panel for this text with the given split.
    offset: starting character for the split (so we can vary splits).
    Returns list of (char, k, CG, is_structural, log_freq) rows.
    """
    n = len(text)
    # Rotate text by offset to get different splits
    text_rot = text[offset:] + text[:offset]

    train_end = int(n * train_frac)
    test_end  = train_end + int(n * test_frac)
    train_text = text_rot[:train_end]
    test_text  = text_rot[train_end:test_end]

    vocab = sorted(set(text_rot))
    vocab_size = len(vocab)

    # Per-character test positions
    char_positions = {}
    for i, ch in enumerate(test_text):
        char_positions.setdefault(ch, []).append(i)

    # Character frequency in train
    char_freq = {}
    for ch in train_text:
        char_freq[ch] = char_freq.get(ch, 0) + 1
    total_chars = len(train_text)

    # NL classification (same as canonical)
    STRUCTURAL = set('.,;:?!\'"()[]{}')
    LEXICAL    = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')

    # Build n-gram counts for each k needed
    ngram_counts = {}
    for k in range(1, max_k + 1):
        ngram_counts[k] = build_ngram(train_text, k)

    # Compute surprisal at k=1 and each k for each character
    panel = []
    for ch in vocab:
        positions = char_positions.get(ch, [])
        if len(positions) < min_n:
            continue
        if ch not in STRUCTURAL and ch not in LEXICAL:
            continue
        is_struct = 1 if ch in STRUCTURAL else 0
        freq = char_freq.get(ch, 0) / total_chars
        if freq <= 0:
            continue
        log_freq = math.log2(freq)

        # S_x(1) — surprisal at k=1
        s1_vals = []
        for pos in positions:
            if pos < 1:
                continue
            ctx = test_text[pos-1:pos]
            s1_vals.append(laplace_surprisal(ctx, ch, ngram_counts[1], vocab_size))
        if not s1_vals:
            continue
        s1 = np.mean(s1_vals)

        for k in range(2, max_k + 1):
            # Coverage check
            covered = sum(1 for pos in positions if pos >= k
                          and test_text[pos-k:pos] in ngram_counts[k])
            cov = covered / len(positions) if positions else 0
            if cov < tau:
                continue

            sk_vals = []
            for pos in positions:
                if pos < k:
                    continue
                ctx = test_text[pos-k:pos]
                sk_vals.append(laplace_surprisal(ctx, ch, ngram_counts[k], vocab_size))
            if len(sk_vals) < min_n:
                continue

            sk = np.mean(sk_vals)
            cg = s1 - sk
            panel.append({
                'char': ch, 'k': k, 'CG': cg,
                'structural': is_struct, 'log_freq': log_freq,
                'log2k': math.log2(k),
            })

    return panel


def cluster_robust_b3(panel):
    """Cluster-robust OLS, return b3, se3, ci_lo, ci_hi, p."""
    if not panel:
        return None
    chars = sorted(set(r['char'] for r in panel))
    if len(chars) < 3:
        return None

    Y = np.array([r['CG']         for r in panel])
    X = np.column_stack([
        np.ones(len(panel)),
        [r['log2k']      for r in panel],
        [r['structural'] for r in panel],
        [r['log2k'] * r['structural'] for r in panel],
        [r['log_freq']   for r in panel],
    ])

    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except Exception:
        return None

    resid = Y - X @ beta
    n, p  = X.shape
    G     = len(chars)

    # Cluster sandwich
    char_idx = {c: i for i, c in enumerate(chars)}
    B = np.zeros((p, p))
    for ch in chars:
        idx = [i for i, r in enumerate(panel) if r['char'] == ch]
        Xi = X[idx]
        ei = resid[idx]
        s  = Xi.T @ ei
        B += np.outer(s, s)

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None

    V = XtX_inv @ B @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - p))
    se = np.sqrt(np.diag(V))

    b3  = beta[3]
    se3 = se[3]
    df  = G - 1
    t   = b3 / se3
    p_val = 2 * scipy_stats.t.sf(abs(t), df=df)
    ci_lo = b3 - scipy_stats.t.ppf(0.975, df=df) * se3
    ci_hi = b3 + scipy_stats.t.ppf(0.975, df=df) * se3

    return {'b3': b3, 'se3': se3, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'p': p_val, 'n': len(panel), 'G': G}


# ─── Run 5 splits ─────────────────────────────────────────────────────────────

CORPORA = {
    'NL1_shakespeare':  'corpus_shakespeare.txt',
    'NL2_pride_prej':   'corpus_pride_prejudice.txt',
}

N_SPLITS  = 5
MAX_K     = 8
MIN_N     = 30
TAU       = 0.5

results = []

for corpus_id, path in CORPORA.items():
    print(f"\n{'='*60}")
    print(f"Corpus: {corpus_id}")
    text = load_text(path)
    n    = len(text)
    # Space offsets evenly across the first 50% of text
    offsets = [int(n * i / N_SPLITS / 2) for i in range(N_SPLITS)]
    print(f"  n={n:,}  offsets={offsets}")

    for split_i, offset in enumerate(offsets, 1):
        print(f"  Split {split_i} (offset={offset:,})...", end=' ', flush=True)
        panel = compute_panel(text, max_k=MAX_K, min_n=MIN_N,
                              tau=TAU, offset=offset)
        res = cluster_robust_b3(panel)
        if res:
            print(f"b3={res['b3']:+.4f}  p={res['p']:.4f}  n={res['n']}  G={res['G']}")
            results.append({
                'corpus': corpus_id,
                'split': split_i,
                'offset': offset,
                'b3': res['b3'],
                'se': res['se3'],
                'ci_lo': res['ci_lo'],
                'ci_hi': res['ci_hi'],
                'p': res['p'],
                'n_obs': res['n'],
                'G': res['G'],
            })
        else:
            print("FAILED (insufficient data)")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n\n=== Split Robustness Summary ===")
for corpus_id in CORPORA:
    corp_res = [r for r in results if r['corpus'] == corpus_id]
    b3s = [r['b3'] for r in corp_res]
    ps  = [r['p']  for r in corp_res]
    print(f"\n{corpus_id}")
    print(f"  b3 range:  {min(b3s):+.3f} to {max(b3s):+.3f}")
    print(f"  b3 mean:   {sum(b3s)/len(b3s):+.3f}")
    print(f"  b3 sd:     {np.std(b3s, ddof=1):.3f}")
    print(f"  all positive: {all(b > 0 for b in b3s)}")
    print(f"  p values:  {[f'{p:.3f}' for p in ps]}")
    for r in corp_res:
        print(f"  Split {r['split']}: b3={r['b3']:+.4f}  SE={r['se']:.4f}  "
              f"CI=[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  p={r['p']:.4f}  n={r['n_obs']}")

# ─── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("results_tmlr", exist_ok=True)
output_path = "results_tmlr/split_robustness.csv"
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['corpus','split','offset','b3','se',
                                       'ci_lo','ci_hi','p','n_obs','G'])
    w.writeheader()
    w.writerows(results)
print(f"\nSaved to {output_path}")
