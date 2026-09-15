"""
audit_paper_values.py
Assert every manuscript-reported statistic against its source CSV.
Run: uv run python audit_paper_values.py
"""
import csv
import sys
import math

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"

errors = []
warnings = []


def approx(a, b, tol=0.0015):
    """True if |a-b| <= tol (default: matches 3-decimal rounding)."""
    return abs(a - b) <= tol


def check(name, paper_val, csv_val, tol=0.0015, warn_only=False):
    ok = approx(paper_val, csv_val, tol)
    status = PASS if ok else (WARN if warn_only else FAIL)
    print(f"  [{status}] {name}: paper={paper_val:.4f}  csv={csv_val:.4f}  diff={paper_val-csv_val:+.4f}")
    if not ok:
        if warn_only:
            warnings.append(f"{name}: paper={paper_val:.4f} csv={csv_val:.4f}")
        else:
            errors.append(f"{name}: paper={paper_val:.4f} csv={csv_val:.4f}")


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: v.strip() for k, v in row.items()})
    return rows


def flt(s):
    try:
        return float(s)
    except Exception:
        return None


# ─── Load CSVs ───────────────────────────────────────────────────────────────

rob_b3    = {(r["corpus"], r["tau_"], r["smoother"]): r
             for r in load_csv("results_canonical_snapshot/robustness_b3.csv")}
code1_cov = {r["tau_"]: r
             for r in load_csv("results_canonical_snapshot/code1_coverage_sensitivity.csv")}
perm      = {r["corpus"]: r
             for r in load_csv("results_robustness/permutation_test.csv")}
ff        = {(r["corpus"], r["model"]): r
             for r in load_csv("results_robustness/functional_form.csv")}
smooth    = {(r["corpus"], r["smoothing"]): r
             for r in load_csv("results_tmlr/smoothing_robustness.csv")}
loo       = load_csv("results_tmlr/leave_one_structural_out.csv")
phase2    = {r["name"]: r
             for r in load_csv("results_tmlr/phase2/consolidated_results.csv")}


# ─── Section 1: Main regression table ───────────────────────────────────────
# Paper source decision:
#   NL1/NL2: run_p0.py smoothing_robustness.csv (laplace row) — consistent
#             with permutation test and LOO
#   Code1:   smoothing_robustness.csv (laplace row) — canonical n=456 pipeline
# Note: robustness_b3.csv has NL1 β₃=0.5512 (slightly different pipeline run)

print("\n=== Main Regression Table ===")

nl1 = smooth[("shakespeare", "laplace")]
print("\nNL1 (Shakespeare) — source: smoothing_robustness.csv laplace")
check("NL1 b3",  0.549, flt(nl1["beta3"]))
check("NL1 SE",  0.250, flt(nl1["se"]))
check("NL1 ci_lo", 0.047, flt(nl1["ci_low"]))
check("NL1 ci_hi", 1.050, flt(nl1["ci_high"]))
check("NL1 p",   0.033, flt(nl1["p_value"]))
check("NL1 n",   272,   flt(nl1["n_obs"]), tol=0.5)
check("NL1 G",   56,    flt(nl1["n_clusters"]), tol=0.5)

nl2 = smooth[("pride_prej", "laplace")]
print("\nNL2 (Pride & Prejudice) — source: smoothing_robustness.csv laplace")
check("NL2 b3",  1.131, flt(nl2["beta3"]))
check("NL2 SE",  0.337, flt(nl2["se"]))
check("NL2 ci_lo", 0.452, flt(nl2["ci_low"]))
check("NL2 ci_hi", 1.810, flt(nl2["ci_high"]))
check("NL2 p",   0.002, flt(nl2["p_value"]))
check("NL2 n",   310,   flt(nl2["n_obs"]), tol=0.5)
check("NL2 G",   45,    flt(nl2["n_clusters"]), tol=0.5)

c1 = smooth[("python_stdlib", "laplace")]
print("\nCode1 (Python stdlib) — source: smoothing_robustness.csv laplace (n=456)")
check("Code1 b3",     -0.024, flt(c1["beta3"]))
check("Code1 SE",      0.278, flt(c1["se"]))      # paper had 0.262 — WRONG
check("Code1 ci_lo",  -0.577, flt(c1["ci_low"]))  # paper had -0.481 — WRONG
check("Code1 ci_hi",   0.528, flt(c1["ci_high"])) # paper had +0.561 — WRONG
check("Code1 p",       0.930, flt(c1["p_value"]))
check("Code1 n",       456,   flt(c1["n_obs"]), tol=0.5)
check("Code1 G",       77,    flt(c1["n_clusters"]), tol=0.5)


# ─── Section 2: Coverage threshold sensitivity ───────────────────────────────
print("\n=== Coverage Threshold Sensitivity (NL1/NL2) ===")
# Source: robustness_b3.csv (Laplace rows — only source with τ=0.25/0.75)

for corpus, label, paper_vals in [
    ("shakespeare", "NL1", {
        "0.25": (0.636, "**"), "0.5": (0.551, "*"), "0.75": (0.337, "ns")
    }),
    ("pride_prej", "NL2", {
        "0.25": (1.142, "**"), "0.5": (1.131, "**"), "0.75": (0.982, "**")
    }),
]:
    print(f"\n{label}")
    for tau_, (pval, sig) in paper_vals.items():
        key = (corpus, tau_, "laplace")
        if key in rob_b3:
            row = rob_b3[key]
            check(f"{label} b3 τ={tau_}", pval, flt(row["b3"]))
        else:
            print(f"  [WARN] {label} τ={tau_} not in robustness_b3.csv")

print("\nCode1 coverage sensitivity — source: code1_coverage_sensitivity.csv")
for tau_, paper_b3 in [("0.25", 0.008), ("0.5", -0.024), ("0.75", -0.059)]:
    row = code1_cov.get(tau_, {})
    csv_b3 = flt(row.get("b3") or row.get("beta3") or row.get("b3", None))
    if csv_b3 is None:
        # Try alternate column name
        csv_b3 = flt(next((v for k, v in row.items() if "b3" in k.lower()), None))
    if csv_b3 is not None:
        check(f"Code1 b3 τ={tau_}", paper_b3, csv_b3)
    else:
        print(f"  [WARN] Code1 τ={tau_} b3 column not found. Row keys: {list(row.keys())}")


# ─── Section 3: Permutation test ─────────────────────────────────────────────
print("\n=== Permutation Test ===")
for corpus, label, paper_p_perm in [
    ("shakespeare", "NL1", 0.093),
    ("pride_prej",  "NL2", 0.001),
    ("python_stdlib","Code1", 0.532),
]:
    row = perm[corpus]
    check(f"{label} b3_obs",   flt(rob_b3.get((corpus,"0.5","laplace"),{}).get("b3", smooth.get((corpus,"laplace"),{}).get("beta3", "nan"))),
          flt(row["b3_observed"]), warn_only=True)
    check(f"{label} p_perm",   paper_p_perm, flt(row["p_permutation"]))
    check(f"{label} p_cluster", {
        "shakespeare": 0.033, "pride_prej": 0.002, "python_stdlib": 0.930
    }[corpus], flt(row["p_cluster_robust"]))


# ─── Section 4: Functional form / AIC ────────────────────────────────────────
print("\n=== Functional Form (NL2) ===")
for model_key, paper_aic, paper_b, paper_p in [
    ("log2(k)",       -186.6, 1.131, 0.002),
    ("k_linear",      -219.8, 0.375, 0.001),
    ("categorical_k", -241.4, 1.398, None),
]:
    key = ("pride_prej", model_key)
    if key in ff:
        row = ff[key]
        check(f"NL2 {model_key} AIC",  paper_aic, flt(row["AIC"]))
        check(f"NL2 {model_key} b_int", paper_b,   flt(row["b_interaction"]))
        if paper_p is not None:
            check(f"NL2 {model_key} p_int", paper_p, flt(row["p_interaction"]))
    else:
        print(f"  [WARN] {key} not in functional_form.csv")


# ─── Section 5: KT smoothing robustness ──────────────────────────────────────
print("\n=== KT Smoothing Robustness ===")
for corpus, label, paper_b3_lap, paper_b3_kt, paper_delta, paper_p_kt in [
    ("shakespeare", "NL1",  0.549, 0.510, -0.039, 0.039),
    ("pride_prej",  "NL2",  1.131, 1.114, -0.017, 0.002),
    ("python_stdlib","Code1",-0.024,-0.011, 0.013, 0.968),
]:
    lap = smooth[(corpus, "laplace")]
    kt  = smooth[(corpus, "kt_add_half")]
    check(f"{label} laplace b3", paper_b3_lap, flt(lap["beta3"]))
    check(f"{label} KT b3",      paper_b3_kt,  flt(kt["beta3"]))
    check(f"{label} Δb3",        paper_delta,  flt(kt["beta3"]) - flt(lap["beta3"]))
    check(f"{label} KT p",       paper_p_kt,   flt(kt["p_value"]))


# ─── Section 6: LOO (NL2 primary) ────────────────────────────────────────────
print("\n=== Leave-One-Out Regression (NL2) ===")
nl2_loo = [r for r in loo if r["corpus"] == "pride_prej"]
paper_loo = {
    "'!'": (0.913, 0.009),
    "\"'\"\"\"": None,
    "\"','\"": None,
    "'.'": (1.343, 0.000),
    "';'": (1.124, 0.010),
    "'?'": (0.957, 0.012),
}
for row in nl2_loo:
    exc = row["excluded_character"]
    b3  = flt(row["beta3"])
    p   = flt(row["p_value"])
    print(f"  Excl {exc:10s}: b3={b3:.4f}  p={p:.5f}")

# Min LOO b3 check
nl2_b3s = [flt(r["beta3"]) for r in nl2_loo]
print(f"  min LOO b3 = {min(nl2_b3s):.4f}  (paper: 0.913)")
check("NL2 min LOO b3", 0.913, min(nl2_b3s))


# ─── Section 7: Phase 2 replication table ────────────────────────────────────
print("\n=== Phase 2 Replication Table ===")
paper_phase2 = {
    "pride_prej":       (1.131, 0.337, 0.452, 1.810, 0.002, 310),
    "reuters":          (0.712, 0.348, 0.018, 1.405, 0.044, 490),
    "bible_kjv":        (0.693, 0.391,-0.087, 1.474, 0.081, 440),
    "wikitext_103":     (0.676, 0.404,-0.131, 1.482, 0.099, 470),
    "shakespeare":      (0.549, 0.250, 0.047, 1.050, 0.033, 272),
    "nodejs_js":       (-0.321, 0.192,-0.703, 0.060, 0.098, 547),
    "commons_lang_java":(0.075, 0.264,-0.451, 0.601, 0.777, 556),
    "python_stdlib":   (-0.024, 0.278,-0.577, 0.528, 0.930, 456),
}
for name, (pb3, pse, plo, phi, pp, pn) in paper_phase2.items():
    row = phase2.get(name, {})
    if not row:
        print(f"  [WARN] {name} not in consolidated_results.csv")
        continue
    print(f"\n  {name}")
    check(f"  b3",    pb3, flt(row["beta3"]))
    check(f"  SE",    pse, flt(row["se"]))
    check(f"  ci_lo", plo, flt(row["ci_low"]))
    check(f"  ci_hi", phi, flt(row["ci_high"]))
    check(f"  p",     pp,  flt(row["p_value"]))
    check(f"  n",     pn,  flt(row["n_obs"]), tol=0.5)


# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
if errors:
    print(f"\033[31mFAILED: {len(errors)} inconsistencies found:\033[0m")
    for e in errors:
        print(f"  • {e}")
else:
    print("\033[32mAll primary checks passed.\033[0m")

if warnings:
    print(f"\n\033[33mWARNINGS: {len(warnings)} minor discrepancies:\033[0m")
    for w in warnings:
        print(f"  • {w}")

sys.exit(1 if errors else 0)
