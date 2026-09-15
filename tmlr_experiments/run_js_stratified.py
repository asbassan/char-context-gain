"""Re-run Node.js analysis with approximate string/comment stratification.

Uses regex to identify characters inside:
  - Line comments: // ... newline
  - Block comments: /* ... */
  - String literals: "...", '...', `...` (with basic backslash escape handling)

Characters in these strata are excluded from the structural category,
mirroring the Python tokenizer stratification used for Code1.
"""
import re, sys, csv, math
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
    from scipy import stats
except ImportError:
    sys.exit("Run with: uv run --with numpy --with scipy python run_js_stratified.py")

BASE = Path(__file__).parent.parent
CORPUS = BASE / "corpora_phase2/nodejs_js.txt"
OUT_PANEL = BASE / "results_tmlr/panel_nodejs_stratified.csv"
OUT_RESULT = BASE / "results_tmlr/nodejs_stratified_result.csv"

# Characters in JS OP token category (operators/punctuation as syntax)
JS_STRUCTURAL = set("()[]{}.,;:=<>!&|^~%+-*/\\?@#")

def classify_js_strata(text):
    """Return array of stratum labels: 'code', 'string', 'comment'."""
    n = len(text)
    strata = ['code'] * n
    i = 0
    while i < n:
        c = text[i]
        # Line comment
        if c == '/' and i + 1 < n and text[i+1] == '/':
            j = i
            while j < n and text[j] != '\n':
                strata[j] = 'comment'
                j += 1
            i = j
        # Block comment
        elif c == '/' and i + 1 < n and text[i+1] == '*':
            strata[i] = 'comment'
            strata[i+1] = 'comment'
            i += 2
            while i < n:
                if text[i] == '*' and i + 1 < n and text[i+1] == '/':
                    strata[i] = 'comment'
                    strata[i+1] = 'comment'
                    i += 2
                    break
                strata[i] = 'comment'
                i += 1
        # String literals (double, single, backtick)
        elif c in ('"', "'", '`'):
            quote = c
            strata[i] = 'string'
            i += 1
            while i < n:
                if text[i] == '\\' and i + 1 < n:
                    strata[i] = 'string'
                    strata[i+1] = 'string'
                    i += 2
                elif text[i] == quote:
                    strata[i] = 'string'
                    i += 1
                    break
                else:
                    strata[i] = 'string'
                    i += 1
        else:
            i += 1
    return strata

def cluster_robust_ols(rows):
    n = len(rows)
    if n < 10:
        return None
    X = np.array([[1, r["log2k"], r["Structural"], r["log2k_x_S"], r["logFreq"]] for r in rows])
    y = np.array([r["CG"] for r in rows])
    chars = [r["char_id"] for r in rows]
    unique_chars = list(set(chars))
    G = len(unique_chars)
    if G < 5:
        return None
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    B = np.zeros((5, 5))
    for ch in unique_chars:
        idx = [i for i, c in enumerate(chars) if c == ch]
        Xc = X[idx]; ec = e[idx]
        score = Xc.T @ ec
        B += np.outer(score, score)
    B *= G / (G - 1)
    V = XtX_inv @ B @ XtX_inv
    se = np.sqrt(np.diag(V))
    b3 = beta[3]
    se3 = se[3]
    t3 = b3 / se3
    p3 = 2 * stats.t.sf(abs(t3), df=G - 1)
    ci_lo = b3 - 1.96 * se3
    ci_hi = b3 + 1.96 * se3
    # R²
    ss_res = float(e @ e)
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"b3": b3, "se": se3, "ci_lo": ci_lo, "ci_hi": ci_hi, "p": p3, "n": n, "G": G, "r2": r2}

# ── 1. Load and classify corpus ──────────────────────────────────────────────
print("Loading Node.js corpus...")
text = CORPUS.read_text(encoding="utf-8", errors="replace")
print(f"  Total chars: {len(text):,}")

print("Classifying strata (this may take a moment)...")
strata = classify_js_strata(text)
n_code    = strata.count('code')
n_string  = strata.count('string')
n_comment = strata.count('comment')
pct_str = 100 * n_string / len(text)
pct_com = 100 * n_comment / len(text)
print(f"  code={n_code:,} ({100*n_code/len(text):.1f}%)  "
      f"string={n_string:,} ({pct_str:.1f}%)  "
      f"comment={n_comment:,} ({pct_com:.1f}%)")

# ── 2. Build character frequency table on full text (for logFreq) ────────────
from collections import Counter
freq = Counter(text)
total = len(text)

# ── 3. Build surprisal estimates via n-gram counting ─────────────────────────
# Use 80/10 split on raw text
split_80 = int(0.8 * len(text))
split_90 = int(0.9 * len(text))
train_text = text[:split_80]
test_text  = text[split_80:split_90]
test_strata = strata[split_80:split_90]

VOCAB = sorted(set(text))
V = len(VOCAB)
MAX_K = 8
COVERAGE_THRESH = 0.50
MIN_N = 30

print(f"\nVocab size: {V},  train: {len(train_text):,}  test: {len(test_text):,}")

# NL structural classification for JS: JS_STRUCTURAL chars that appear in 'code' stratum
print("Building n-gram counts...")
# Count (k-gram context → next char) in training
from collections import defaultdict
ngram_counts = [defaultdict(Counter) for _ in range(MAX_K + 1)]
unigram = Counter(train_text)
for k in range(1, MAX_K + 1):
    for i in range(k, len(train_text)):
        ctx = train_text[i-k:i]
        ch  = train_text[i]
        ngram_counts[k][ctx][ch] += 1

def laplace_prob(ctx, ch, k):
    ctx_count = sum(ngram_counts[k][ctx].values())
    ch_count  = ngram_counts[k][ctx].get(ch, 0)
    return (ch_count + 1) / (ctx_count + V)

# ── 4. Per-character surprisal at each k on test set ─────────────────────────
# Only process 'code'-stratum test positions for structural classification
print("Computing per-character surprisal...")
# For each char, collect surprisal at k=1..MAX_K for code-stratum positions
surp = defaultdict(lambda: defaultdict(list))  # char -> k -> [surprisals]
cov  = defaultdict(lambda: defaultdict(int))   # char -> k -> seen_count
tot  = defaultdict(int)                        # char -> total test positions

for i in range(MAX_K, len(test_text)):
    ch = test_text[i]
    s = test_strata[i]
    # Only include positions that are 'code' stratum for structural chars
    # For lexical chars (alpha/digit) include all strata
    for k in range(1, MAX_K + 1):
        # Check if enough context is available
        if i - k < 0:
            continue
        ctx = test_text[i-k:i]
        seen = sum(ngram_counts[k].get(ctx, {}).values()) > 0
        if seen:
            cov[ch][k] += 1
        p = laplace_prob(ctx, ch, k)
        surp[ch][k].append(-math.log2(p))
    tot[ch] += 1

# Global coverage at k=1 (bigram baseline)
print("Computing CG and building panel...")
# Identify character classifications
# Structural: JS_STRUCTURAL chars that appear predominantly in code stratum
# Lexical: alpha/digit/space

# Check stratum purity for structural candidate chars
struct_code_frac = {}
for i, (ch, s) in enumerate(zip(test_text, test_strata)):
    if ch in JS_STRUCTURAL:
        if ch not in struct_code_frac:
            struct_code_frac[ch] = [0, 0]
        struct_code_frac[ch][1] += 1
        if s == 'code':
            struct_code_frac[ch][0] += 1

print("\nStructural char stratum purity (code / total):")
for ch in sorted(struct_code_frac.keys()):
    code_n, tot_n = struct_code_frac[ch]
    if tot_n > 0:
        print(f"  '{ch}': {code_n}/{tot_n} = {100*code_n/tot_n:.1f}% in code stratum")

# Build panel: only include chars with sufficient n at k=1
panel_rows = []
surp_k1 = {}
for ch in sorted(set(test_text)):
    if len(surp[ch][1]) < MIN_N:
        continue
    s1 = float(np.mean(surp[ch][1]))
    surp_k1[ch] = s1
    lf = math.log2(freq.get(ch, 1) / total)
    is_struct = 1 if (ch in JS_STRUCTURAL and
                      struct_code_frac.get(ch, [0,1])[1] > 0 and
                      struct_code_frac.get(ch, [0,1])[0] / struct_code_frac.get(ch, [0,1])[1] >= 0.50
                      ) else 0
    is_lex = 1 if (ch.isalpha() or ch.isdigit() or ch == ' ') else 0
    if not is_struct and not is_lex:
        continue
    for k in range(2, MAX_K + 1):
        if len(surp[ch][k]) < MIN_N:
            continue
        # Coverage check
        cov_frac = cov[ch][k] / max(tot[ch], 1)
        if cov_frac < COVERAGE_THRESH:
            continue
        sk = float(np.mean(surp[ch][k]))
        cg = s1 - sk
        panel_rows.append({
            "CG": cg, "log2k": math.log2(k), "Structural": is_struct,
            "log2k_x_S": math.log2(k) * is_struct, "logFreq": lf, "char_id": ch
        })

print(f"\nPanel: {len(panel_rows)} rows")
struct_chars = sorted(set(r["char_id"] for r in panel_rows if r["Structural"] == 1))
lex_chars    = sorted(set(r["char_id"] for r in panel_rows if r["Structural"] == 0))
print(f"Structural chars ({len(struct_chars)}): {struct_chars}")
print(f"Lexical chars ({len(lex_chars)}): {lex_chars[:10]}...")

# ── 5. Run OLS ────────────────────────────────────────────────────────────────
result = cluster_robust_ols(panel_rows)
if result:
    print(f"\n=== Node.js (stratified) ===")
    print(f"b3={result['b3']:+.3f}  SE={result['se']:.3f}  "
          f"CI=[{result['ci_lo']:+.3f},{result['ci_hi']:+.3f}]  "
          f"p={result['p']:.4f}  n={result['n']}  G={result['G']}  R²={result['r2']:.3f}")
    print(f"\nFor comparison:")
    print(f"  Node.js (unstratified, Phase 2): b3=-0.321  SE=0.192  p=0.098")

# ── 6. Save panel and result ──────────────────────────────────────────────────
with open(OUT_PANEL, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["CG","log2k","Structural","log2k_x_S","logFreq","char_id"])
    w.writeheader()
    w.writerows(panel_rows)

if result:
    with open(OUT_RESULT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["corpus","b3","se","ci_lo","ci_hi","p","n","G","r2","note"])
        w.writerow(["nodejs_stratified", f"{result['b3']:.4f}", f"{result['se']:.4f}",
                    f"{result['ci_lo']:.4f}", f"{result['ci_hi']:.4f}",
                    f"{result['p']:.6f}", result['n'], result['G'], f"{result['r2']:.3f}",
                    "regex-based string/comment stratification"])
print(f"\nSaved panel to {OUT_PANEL}")
print(f"Saved result to {OUT_RESULT}")
