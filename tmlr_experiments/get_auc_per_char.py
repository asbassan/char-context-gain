"""Per-character AUC (mean CG) for NL2 demo."""
import csv, sys
from collections import defaultdict
try:
    import numpy as np
except ImportError:
    sys.exit("uv run --with numpy")
from pathlib import Path

BASE = Path(__file__).parent.parent
panel = BASE / "results_canonical_snapshot/panel_pride_prej.csv"

rows = []
with open(panel, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append({"CG": float(r["CG"]), "Structural": int(r["Structural"]), "char_id": r["char_id"]})

cg_by = defaultdict(list)
s_by = {}
for r in rows:
    cg_by[r["char_id"]].append(r["CG"])
    s_by[r["char_id"]] = r["Structural"]

print("NL2 (Pride & Prejudice) per-character AUC:")
print(f"  {'Char':<6} {'Type':<10} {'mean CG':>8}  {'n k-vals':>8}")
for ch in sorted(cg_by.keys(), key=lambda c: -np.mean(cg_by[c])):
    if len(cg_by[ch]) < 2:
        continue
    t = "struct" if s_by[ch] == 1 else "lexical"
    print(f"  {repr(ch):<6} {t:<10} {np.mean(cg_by[ch]):>8.3f}  {len(cg_by[ch]):>8}")
