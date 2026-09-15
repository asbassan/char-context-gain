"""Reuters (NL3) LOO robustness — elevate to co-primary candidate."""
import csv, sys, math
from pathlib import Path

try:
    import numpy as np
    from scipy import stats
except ImportError:
    sys.exit("Run with: uv run --with numpy --with scipy python run_reuters_robustness.py")

PANEL = Path(__file__).parent.parent / "results_tmlr/phase2/panel_reuters.csv"
OUT   = Path(__file__).parent.parent / "results_tmlr/reuters_loo.csv"

def load_panel(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "CG": float(r["CG"]),
                "log2k": float(r["log2k"]),
                "Structural": int(r["Structural"]),
                "log2k_x_S": float(r["log2k_x_S"]),
                "logFreq": float(r["logFreq"]),
                "char_id": r["char_id"],
            })
    return rows

def cluster_robust_ols(rows):
    """OLS with cluster-robust (sandwich) SEs, clustered by char_id."""
    n = len(rows)
    if n < 10:
        return None
    # Design matrix: intercept, log2k, Structural, log2k_x_S, logFreq
    X = np.array([[1, r["log2k"], r["Structural"], r["log2k_x_S"], r["logFreq"]] for r in rows])
    y = np.array([r["CG"] for r in rows])
    chars = [r["char_id"] for r in rows]
    unique_chars = list(set(chars))
    G = len(unique_chars)
    if G < 5:
        return None
    # OLS coefficients
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    # Sandwich variance: V = (X'X)^-1 * B * (X'X)^-1
    B = np.zeros((5, 5))
    for c in unique_chars:
        idx = [i for i, ch in enumerate(chars) if ch == c]
        Xc = X[idx]; ec = e[idx]
        score = Xc.T @ ec
        B += np.outer(score, score)
    B *= G / (G - 1)  # small-sample correction
    V = XtX_inv @ B @ XtX_inv
    se = np.sqrt(np.diag(V))
    b3 = beta[3]
    se3 = se[3]
    t3 = b3 / se3
    p3 = 2 * stats.t.sf(abs(t3), df=G - 1)
    ci_lo = b3 - 1.96 * se3
    ci_hi = b3 + 1.96 * se3
    return {"b3": b3, "se": se3, "ci_lo": ci_lo, "ci_hi": ci_hi, "p": p3, "n": n, "G": G}

rows = load_panel(PANEL)
struct_chars = sorted(set(r["char_id"] for r in rows if r["Structural"] == 1))
print(f"Reuters panel: {len(rows)} obs, structural chars: {struct_chars}")

# Full fit
full = cluster_robust_ols(rows)
print(f"\nFull: b3={full['b3']:+.3f}  SE={full['se']:.3f}  p={full['p']:.4f}  n={full['n']}  G={full['G']}")

# LOO
print("\n=== LOO (exclude one structural character) ===")
loo_results = []
for excl in struct_chars:
    sub = [r for r in rows if r["char_id"] != excl]
    res = cluster_robust_ols(sub)
    if res:
        print(f"  Exclude '{excl}': b3={res['b3']:+.3f}  SE={res['se']:.3f}  "
              f"CI=[{res['ci_lo']:+.3f},{res['ci_hi']:+.3f}]  p={res['p']:.4f}  n={res['n']}")
        loo_results.append({"excluded": excl, **res})

b3s = [r["b3"] for r in loo_results]
print(f"\nLOO summary: min b3={min(b3s):+.3f}  max b3={max(b3s):+.3f}  all positive={all(b>0 for b in b3s)}")

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["excluded", "b3", "se", "ci_lo", "ci_hi", "p", "n", "G"])
    for r in loo_results:
        w.writerow([r["excluded"], f"{r['b3']:.4f}", f"{r['se']:.4f}",
                    f"{r['ci_lo']:.4f}", f"{r['ci_hi']:.4f}", f"{r['p']:.6f}",
                    r["n"], r["G"]])
print(f"\nSaved to {OUT}")
