"""
make_forest_plot.py
───────────────────
Generate the Phase 2 forest plot and consolidated results table.

NL panel  (top):  5 corpora, sorted by β₃ descending
Code panel (bot): 3 corpora (canonical Python strata only)

Reads panel CSVs from:
  results_canonical_snapshot/  — canonical paper corpora
  results_tmlr/phase2/         — Phase 2 new corpora

Outputs:
  results_tmlr/phase2/forest_plot.png
  results_tmlr/phase2/consolidated_results.csv

Usage (from chapter1 directory):
    uv run --python 3.12 --with "numpy,pandas,scipy,matplotlib" \\
        python tmlr_experiments/make_forest_plot.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PARENT = Path(__file__).parent.parent
sys.path.insert(0, str(PARENT))

from run_experiment_v3 import clustered_ols

ROOT        = PARENT
CANON_DIR   = ROOT / 'results_canonical_snapshot'
PHASE2_DIR  = ROOT / 'results_tmlr' / 'phase2'
OUT_DIR     = PHASE2_DIR


# ── Regression helper ─────────────────────────────────────────────────────────

def regress(panel_path: Path) -> dict:
    df   = pd.read_csv(panel_path)
    y    = df['CG'].values
    Xm   = np.column_stack([
        np.ones(len(df)),
        df[['log2k', 'Structural', 'log2k_x_S', 'logFreq']].values,
    ])
    chars = np.unique(df['char_id'].values)
    c2i   = {c: i for i, c in enumerate(chars)}
    gids  = np.array([c2i[c] for c in df['char_id'].values])
    coeffs, se, lo, hi, pv, r2 = clustered_ols(y, Xm, gids)
    n_struct = int(df[df['Structural'] == 1]['char_id'].nunique())
    return {
        'beta3':   coeffs[3], 'se':      se[3],
        'ci_low':  lo[3],     'ci_high': hi[3],
        'p_value': pv[3],     'r2':      r2,
        'n_obs':   len(df),   'n_clusters': len(chars),
        'n_structural_clusters': n_struct,
    }


# ── Corpus manifest ───────────────────────────────────────────────────────────
# (label, panel_path, is_code, display_name)
CORPORA = [
    # ── Natural language ──────────────────────────────────────────────────────
    ('pride_prej',   CANON_DIR  / 'panel_pride_prej.csv',
     False, 'Pride & Prejudice\n(Austen, 19c novel)'),
    ('reuters',      PHASE2_DIR / 'panel_reuters.csv',
     False, 'Reuters\n(news wire)'),
    ('bible_kjv',    PHASE2_DIR / 'panel_bible_kjv.csv',
     False, 'Bible KJV\n(religious prose)'),
    ('wikitext_103', PHASE2_DIR / 'panel_wikitext_103.csv',
     False, 'WikiText-103\n(Wikipedia)'),
    ('shakespeare',  CANON_DIR  / 'panel_shakespeare.csv',
     False, 'Shakespeare\n(drama)'),
    # ── Source code ───────────────────────────────────────────────────────────
    ('nodejs_js',         PHASE2_DIR / 'panel_nodejs_js.csv',
     True, 'Node.js stdlib\n(JavaScript)'),
    ('commons_lang_java', PHASE2_DIR / 'panel_commons_lang_java.csv',
     True, 'Commons Lang\n(Java)'),
    ('python_stdlib',     CANON_DIR  / 'panel_python_stdlib.csv',
     True, 'Python stdlib\n(canonical strata)'),
]


# ── Compute regression for every corpus ───────────────────────────────────────

def build_results() -> pd.DataFrame:
    rows = []
    for name, path, is_code, label in CORPORA:
        if not path.exists():
            print(f'  MISSING: {path}')
            continue
        r = regress(path)
        rows.append({
            'name':     name,
            'label':    label,
            'is_code':  is_code,
            'beta3':    round(r['beta3'],   4),
            'se':       round(r['se'],      4),
            'ci_low':   round(r['ci_low'],  4),
            'ci_high':  round(r['ci_high'], 4),
            'p_value':  round(r['p_value'], 4),
            'r2':       round(r['r2'],      4),
            'n_obs':    r['n_obs'],
            'n_clusters':            r['n_clusters'],
            'n_structural_clusters': r['n_structural_clusters'],
        })
    return pd.DataFrame(rows)


# ── Forest plot ───────────────────────────────────────────────────────────────

def make_forest_plot(df: pd.DataFrame, out_path: Path) -> None:
    nl_df   = df[~df['is_code']].sort_values('beta3', ascending=False).reset_index(drop=True)
    code_df = df[ df['is_code']].sort_values('beta3', ascending=False).reset_index(drop=True)

    NL_COL   = '#2166ac'   # blue
    CODE_COL = '#d6604d'   # red-orange
    ZERO_COL = '#888888'

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5),
        gridspec_kw={'width_ratios': [5, 3]},
    )

    def _draw_panel(ax, sub_df, color, title):
        n = len(sub_df)
        ys = np.arange(n)

        ax.axvline(0, color=ZERO_COL, lw=1.0, ls='--', zorder=1)

        for i, row in sub_df.iterrows():
            y    = n - 1 - i   # top-to-bottom order
            b3   = row['beta3']
            lo   = row['ci_low']
            hi   = row['ci_high']
            p    = row['p_value']
            sig  = '**' if p < 0.01 else '*' if p < 0.05 else ''

            # CI whisker
            ax.plot([lo, hi], [y, y], color=color, lw=1.8, zorder=2)
            # Point estimate
            ax.scatter([b3], [y], color=color, s=60, zorder=3,
                       marker='D' if sig else 'o')

            # Right-side annotation
            ax.text(
                ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 2.0,
                y,
                f'  β₃={b3:+.3f}{sig}  p={p:.3f}',
                va='center', fontsize=8, color='#333333',
            )

        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [sub_df.loc[n - 1 - i, 'label'] for i in range(n)],
            fontsize=9,
        )
        ax.set_xlabel('β₃  (log2(k) × Structural interaction)', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.set_ylim(-0.7, n - 0.3)
        ax.grid(axis='x', alpha=0.25, lw=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ── Determine shared x-axis limits ────────────────────────────────────────
    all_lo  = min(df['ci_low'].min(),  -0.2)
    all_hi  = max(df['ci_high'].max(),  0.2)
    pad     = (all_hi - all_lo) * 0.45   # room for annotation

    for ax in axes:
        ax.set_xlim(all_lo - 0.05, all_hi + pad)

    _draw_panel(axes[0], nl_df,   NL_COL,   'Natural language (5 corpora)')
    _draw_panel(axes[1], code_df, CODE_COL, 'Source code (3 corpora)')

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=NL_COL,   label='Natural language'),
        mpatches.Patch(color=CODE_COL, label='Source code'),
        plt.Line2D([0], [0], marker='D', color='grey', linestyle='None',
                   markersize=7, label='p < 0.05 (filled diamond)'),
        plt.Line2D([0], [0], marker='o', color='grey', linestyle='None',
                   markersize=7, label='p ≥ 0.05 (circle)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        'Context-gain interaction (β₃): structural vs. lexical characters\n'
        'Clustered OLS with sandwich SE, clusters = characters',
        fontsize=11, y=1.02,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'  Saved: {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Computing regressions...')
    df = build_results()

    # ── Print table ───────────────────────────────────────────────────────────
    print('\nConsolidated results:')
    print(f'  {"corpus":25s}  {"type":5s}  {"β₃":>8}  {"SE":>7}  '
          f'{"CI low":>8}  {"CI high":>8}  {"p":>7}  {"n_obs":>6}')
    print(f'  {"-"*82}')
    for _, r in df.iterrows():
        typ = 'code' if r['is_code'] else 'NL'
        sig = ('**' if r['p_value'] < 0.01 else '*' if r['p_value'] < 0.05
               else '  ')
        print(f'  {r["name"]:25s}  {typ:5s}  {r["beta3"]:+8.4f}  '
              f'{r["se"]:7.4f}  {r["ci_low"]:+8.4f}  {r["ci_high"]:+8.4f}  '
              f'{r["p_value"]:7.4f}{sig}  {r["n_obs"]:>6}')

    csv_out = OUT_DIR / 'consolidated_results.csv'
    df.to_csv(csv_out, index=False)
    print(f'\n  Saved: {csv_out}')

    print('\nGenerating forest plot...')
    make_forest_plot(df, OUT_DIR / 'forest_plot.png')


if __name__ == '__main__':
    main()
