"""
run_classification_sensitivity.py  —  P1: lexical category sensitivity.

Tests whether β₃ in NL1 and NL2 is sensitive to the lexical category definition.

Definition A (canonical): lexical = alphabetic + digits + space
Definition B (alpha-only): lexical = alphabetic only (a-z, A-Z)
                           digits and space are excluded from both categories

This checks the reviewer concern that including space/digits in lexical
creates a cross-domain asymmetry vs code (which excludes whitespace tokens).

Output: results_tmlr/classification_sensitivity.csv
"""
import sys, os, csv, math
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from scipy import stats as scipy_stats


def load_text(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def build_ngram(text, k):
    counts = {}
    for i in range(k, len(text)):
        ctx = text[i-k:i]
        ch  = text[i]
        if ctx not in counts:
            counts[ctx] = {}
        counts[ctx][ch] = counts[ctx].get(ch, 0) + 1
    return counts


def laplace_surprisal(ctx, ch, counts, vocab_size):
    ctx_counts = counts.get(ctx, {})
    ctx_total  = sum(ctx_counts.values())
    ch_count   = ctx_counts.get(ch, 0)
    p = (ch_count + 1) / (ctx_total + vocab_size)
    return -math.log2(p)


def run_regression(text, structural_set, lexical_set,
                   train_frac=0.8, test_frac=0.1, max_k=8,
                   min_n=30, tau=0.5):
    n = len(text)
    train_text = text[:int(n * train_frac)]
    test_text  = text[int(n * train_frac):int(n * (train_frac + test_frac))]

    vocab      = sorted(set(text))
    vocab_size = len(vocab)

    char_positions = {}
    for i, ch in enumerate(test_text):
        char_positions.setdefault(ch, []).append(i)

    char_freq = {}
    for ch in train_text:
        char_freq[ch] = char_freq.get(ch, 0) + 1
    total_chars = len(train_text)

    ngram_counts = {k: build_ngram(train_text, k) for k in range(1, max_k + 1)}

    panel = []
    for ch in vocab:
        positions = char_positions.get(ch, [])
        if len(positions) < min_n:
            continue
        if ch not in structural_set and ch not in lexical_set:
            continue
        is_struct = 1 if ch in structural_set else 0
        freq = char_freq.get(ch, 0) / total_chars
        if freq <= 0:
            continue
        log_freq = math.log2(freq)

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

            panel.append({
                'char': ch, 'k': k,
                'CG': s1 - np.mean(sk_vals),
                'structural': is_struct,
                'log2k': math.log2(k),
                'log_freq': log_freq,
            })

    if not panel:
        return None

    chars = sorted(set(r['char'] for r in panel))
    G = len(chars)
    if G < 3:
        return None

    Y = np.array([r['CG']   for r in panel])
    X = np.column_stack([
        np.ones(len(panel)),
        [r['log2k']                     for r in panel],
        [r['structural']                for r in panel],
        [r['log2k'] * r['structural']   for r in panel],
        [r['log_freq']                  for r in panel],
    ])
    n_obs, p = X.shape
    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except Exception:
        return None

    resid = Y - X @ beta
    B = np.zeros((p, p))
    for ch in chars:
        idx = [i for i, r in enumerate(panel) if r['char'] == ch]
        Xi = X[idx]; ei = resid[idx]; s = Xi.T @ ei
        B += np.outer(s, s)

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None

    V   = XtX_inv @ B @ XtX_inv * (G/(G-1)) * ((n_obs-1)/(n_obs-p))
    se  = np.sqrt(np.diag(V))
    b3  = beta[3]; se3 = se[3]; df = G - 1
    t   = b3 / se3
    p_v = 2 * scipy_stats.t.sf(abs(t), df=df)
    ci_lo = b3 - scipy_stats.t.ppf(0.975, df=df) * se3
    ci_hi = b3 + scipy_stats.t.ppf(0.975, df=df) * se3

    n_struct = sum(1 for ch in chars if any(r['structural'] == 1 and r['char'] == ch for r in panel))
    n_lex    = G - n_struct

    return {
        'b3': b3, 'se': se3, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'p': p_v, 'n_obs': len(panel), 'G': G,
        'n_structural': n_struct, 'n_lexical': n_lex,
    }


# ─── Definitions ──────────────────────────────────────────────────────────────

STRUCTURAL = set('.,;:?!\'"()[]{}')

LEX_A = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')  # canonical
LEX_B = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')              # alpha-only

CORPORA = {
    'NL1_shakespeare': 'corpus_shakespeare.txt',
    'NL2_pride_prej':  'corpus_pride_prejudice.txt',
}

results = []

for corpus_id, path in CORPORA.items():
    print(f"\n{'='*50}  {corpus_id}")
    text = load_text(path)

    for lex_name, lex_set in [('canonical (alpha+digit+space)', LEX_A),
                               ('alpha_only', LEX_B)]:
        res = run_regression(text, STRUCTURAL, lex_set)
        if res:
            print(f"  Lex={lex_name:35s}  b3={res['b3']:+.4f}  SE={res['se']:.4f}  "
                  f"p={res['p']:.4f}  n={res['n_obs']}  G={res['G']} "
                  f"(S={res['n_structural']}, L={res['n_lexical']})")
            results.append({
                'corpus':          corpus_id,
                'lexical_def':     lex_name.split(' ')[0],
                'n_structural':    res['n_structural'],
                'n_lexical':       res['n_lexical'],
                'b3':              res['b3'],
                'se':              res['se'],
                'ci_lo':           res['ci_lo'],
                'ci_hi':           res['ci_hi'],
                'p':               res['p'],
                'n_obs':           res['n_obs'],
                'G':               res['G'],
            })
        else:
            print(f"  Lex={lex_name:35s}  FAILED")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n\n=== Classification Sensitivity Summary ===")
print(f"{'Corpus':<25} {'Lex def':<12} {'b3':>7} {'SE':>6} {'p':>7} {'G':>4}")
for r in results:
    print(f"{r['corpus']:<25} {r['lexical_def']:<12} {r['b3']:>+7.4f} "
          f"{r['se']:>6.4f} {r['p']:>7.4f} {r['G']:>4}")

# ─── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("results_tmlr", exist_ok=True)
out = "results_tmlr/classification_sensitivity.csv"
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=[
        'corpus','lexical_def','n_structural','n_lexical',
        'b3','se','ci_lo','ci_hi','p','n_obs','G'])
    w.writeheader()
    w.writerows(results)
print(f"\nSaved to {out}")
