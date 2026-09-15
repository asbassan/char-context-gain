"""
run_domain_permutation.py  —  P0 kill test: formal NL vs code domain contrast.

Uses the 8 β₃ estimates from consolidated_results.csv.
Exhaustive permutation over all C(8,5)=56 ways of assigning
5 corpora to "NL" and 3 to "code", then checks where the
observed NL/code partition ranks.

Output: results_tmlr/domain_permutation.csv
"""
import csv
import itertools
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

INPUT  = "results_tmlr/phase2/consolidated_results.csv"
OUTPUT = "results_tmlr/domain_permutation.csv"

# ─── Load β₃ values ──────────────────────────────────────────────────────────
rows = []
with open(INPUT, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({
            "name":    r["name"].strip(),
            "is_code": r["is_code"].strip().lower() == "true",
            "beta3":   float(r["beta3"]),
            "label":   r["label"].split("\n")[0].strip(),
        })

names  = [r["name"]  for r in rows]
betas  = [r["beta3"] for r in rows]
is_code = [r["is_code"] for r in rows]

n_total = len(rows)   # 8
n_nl    = sum(1 for x in is_code if not x)   # 5
n_code  = sum(1 for x in is_code if x)       # 3

print(f"Corpora: {n_total} total ({n_nl} NL, {n_code} code)")
for r in rows:
    tag = "code" if r["is_code"] else "NL"
    print(f"  [{tag:4s}] {r['name']:25s}  beta3={r['beta3']:+.4f}")

# ─── Observed statistic: mean(NL β₃) - mean(code β₃) ─────────────────────────
nl_betas   = [b for b, c in zip(betas, is_code) if not c]
code_betas = [b for b, c in zip(betas, is_code) if c]

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

obs_stat = mean(nl_betas) - mean(code_betas)
print(f"\nObserved: mean(NL)={mean(nl_betas):+.4f}  mean(code)={mean(code_betas):+.4f}")
print(f"Observed statistic (mean NL - mean code) = {obs_stat:+.4f}")

# ─── Exhaustive permutation ───────────────────────────────────────────────────
indices = list(range(n_total))
all_nl_assignments = list(itertools.combinations(indices, n_nl))
n_perms = len(all_nl_assignments)
print(f"\nTotal permutations C({n_total},{n_nl}) = {n_perms}")

perm_stats = []
for nl_idx in all_nl_assignments:
    code_idx = [i for i in indices if i not in nl_idx]
    perm_nl   = [betas[i] for i in nl_idx]
    perm_code = [betas[i] for i in code_idx]
    perm_stats.append(mean(perm_nl) - mean(perm_code))

perm_stats.sort(reverse=True)

# One-sided p: fraction of permutations >= observed
n_ge = sum(1 for s in perm_stats if s >= obs_stat - 1e-10)
p_one_sided = n_ge / n_perms

# Rank of observed partition (1-based, 1 = largest)
rank = sum(1 for s in perm_stats if s > obs_stat - 1e-10)

print(f"\nPermutation null distribution (sorted, top 10):")
for i, s in enumerate(perm_stats[:10]):
    marker = " <-- observed" if abs(s - obs_stat) < 1e-8 else ""
    print(f"  [{i+1:2d}] {s:+.4f}{marker}")

print(f"\nResults:")
print(f"  Observed statistic rank: {rank} / {n_perms}")
print(f"  One-sided p (>= obs):    {n_ge}/{n_perms} = {p_one_sided:.4f}")
print(f"  Two-sided p (|stat|>=obs): {sum(1 for s in perm_stats if abs(s) >= abs(obs_stat)-1e-10)/n_perms:.4f}")

# ─── Write output ─────────────────────────────────────────────────────────────
os.makedirs("results_tmlr", exist_ok=True)
with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["stat", "value"])
    w.writerow(["n_corpora", n_total])
    w.writerow(["n_nl", n_nl])
    w.writerow(["n_code", n_code])
    w.writerow(["n_permutations", n_perms])
    w.writerow(["mean_nl_beta3", f"{mean(nl_betas):.6f}"])
    w.writerow(["mean_code_beta3", f"{mean(code_betas):.6f}"])
    w.writerow(["observed_stat", f"{obs_stat:.6f}"])
    w.writerow(["rank", rank])
    w.writerow(["p_one_sided", f"{p_one_sided:.6f}"])
    w.writerow(["n_ge_observed", n_ge])
    w.writerow(["p_interpretation",
                f"Observed NL/code partition has the {rank}-largest mean-difference "
                f"among all {n_perms} C({n_total},{n_nl}) assignments. "
                f"One-sided exact p = {n_ge}/{n_perms} = {p_one_sided:.4f}."])

print(f"\nSaved to {OUTPUT}")
