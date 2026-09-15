"""AUC magnitude: mean CG over reliable k range — avoids CG_peak selection bias."""
import csv, sys
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
    from scipy import stats
except ImportError:
    sys.exit("Run with: uv run --with numpy --with scipy python run_auc_magnitude.py")

BASE = Path(__file__).parent.parent

PANELS = {
    "NL1_shakespeare":    BASE / "results_canonical_snapshot/panel_shakespeare.csv",
    "NL2_pride_prej":     BASE / "results_canonical_snapshot/panel_pride_prej.csv",
    "Code1_python":       BASE / "results_canonical_snapshot/panel_python_stdlib.csv",
    "NL3_reuters":        BASE / "results_tmlr/phase2/panel_reuters.csv",
}
OUT = BASE / "results_tmlr/auc_magnitude.csv"

def load_panel(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "CG": float(r["CG"]),
                "Structural": int(r["Structural"]),
                "char_id": r["char_id"],
            })
    return rows

results = []
for corpus, path in PANELS.items():
    if not path.exists():
        print(f"  SKIP {corpus}: {path} not found")
        continue
    rows = load_panel(path)

    # Compute per-character mean CG (AUC proxy)
    cg_by_char = defaultdict(list)
    struct_by_char = {}
    for r in rows:
        cg_by_char[r["char_id"]].append(r["CG"])
        struct_by_char[r["char_id"]] = r["Structural"]

    structural_auc = []
    lexical_auc = []
    for ch, cgs in cg_by_char.items():
        mean_cg = float(np.mean(cgs))
        if struct_by_char[ch] == 1:
            structural_auc.append(mean_cg)
        else:
            lexical_auc.append(mean_cg)

    if not structural_auc or not lexical_auc:
        print(f"  SKIP {corpus}: missing structural or lexical chars")
        continue

    u, p = stats.mannwhitneyu(structural_auc, lexical_auc, alternative="greater")
    med_s = float(np.median(structural_auc))
    med_l = float(np.median(lexical_auc))

    print(f"\n{corpus}")
    print(f"  Structural n={len(structural_auc)}  median_AUC={med_s:.3f}")
    print(f"  Lexical    n={len(lexical_auc)}  median_AUC={med_l:.3f}")
    print(f"  Mann-Whitney (AUC): U={u:.0f}  p={p:.4f}")
    results.append({
        "corpus": corpus, "n_struct": len(structural_auc), "n_lex": len(lexical_auc),
        "median_struct_auc": f"{med_s:.4f}", "median_lex_auc": f"{med_l:.4f}",
        "mw_p": f"{p:.6f}"
    })

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["corpus","n_struct","n_lex","median_struct_auc","median_lex_auc","mw_p"])
    w.writeheader()
    w.writerows(results)
print(f"\nSaved to {OUT}")
