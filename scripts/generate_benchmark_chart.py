"""Generate benchmark performance charts for README."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Color palette ──────────────────────────────────────────────
C_PHASE1  = '#9DB2C9'   # muted blue-grey  (baseline)
C_PHASE2  = '#2E86AB'   # strong blue       (Phase 2)
C_LOCOMO  = '#A23B72'   # purple-red        (LoCoMo)
C_BG      = '#F8F9FA'
C_GRID    = '#E0E4EA'
C_TEXT    = '#2C3E50'

fig = plt.figure(figsize=(14, 9), facecolor=C_BG)
fig.suptitle(
    'Cliriux Agent - Benchmark Performance',
    fontsize=17, fontweight='bold', color=C_TEXT, y=0.97
)

gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.35,
                      left=0.07, right=0.97, top=0.90, bottom=0.08)

# ── Helper ──────────────────────────────────────────────────────
def style_ax(ax, title):
    ax.set_facecolor(C_BG)
    ax.set_title(title, fontsize=12, fontweight='bold', color=C_TEXT, pad=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)

categories = ['Short', 'Medium', 'Long']
x = np.arange(len(categories))
w = 0.35

# ── 1. Task Completion Rate ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
p1 = [94.3, 66.7, 21.4]
p2 = [100,  100,  100]
bars1 = ax1.bar(x - w/2, p1, w, color=C_PHASE1, label='Phase 1 (Baseline)', zorder=3)
bars2 = ax1.bar(x + w/2, p2, w, color=C_PHASE2, label='Phase 2', zorder=3)
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f'{bar.get_height():.0f}%', ha='center', va='bottom',
             fontsize=8, color=C_TEXT)
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f'{bar.get_height():.0f}%', ha='center', va='bottom',
             fontsize=8, color=C_PHASE2, fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels(categories)
ax1.set_ylabel('Completion Rate (%)', color=C_TEXT, fontsize=9)
ax1.set_ylim(0, 115)
ax1.legend(fontsize=8, frameon=False)
style_ax(ax1, 'Task Completion Rate')

# ── 2. Response Quality ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
q1 = [4.45, 3.85, 2.17]
q2 = [4.80, 4.775, 4.65]
bars3 = ax2.bar(x - w/2, q1, w, color=C_PHASE1, label='Phase 1 (Baseline)', zorder=3)
bars4 = ax2.bar(x + w/2, q2, w, color=C_PHASE2, label='Phase 2', zorder=3)
for bar in bars3:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{bar.get_height():.2f}', ha='center', va='bottom',
             fontsize=8, color=C_TEXT)
for bar, v1, v2 in zip(bars4, q1, q2):
    pct = (v2 - v1) / v1 * 100
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{bar.get_height():.2f}', ha='center', va='bottom',
             fontsize=8, color=C_PHASE2, fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels(categories)
ax2.set_ylabel('Quality Score (/ 5.0)', color=C_TEXT, fontsize=9)
ax2.set_ylim(0, 5.8)
ax2.axhline(5.0, color=C_GRID, linestyle='--', linewidth=0.8)
ax2.legend(fontsize=8, frameon=False)
style_ax(ax2, 'Response Quality (5-pt Scale)')

# ── 3. LoCoMo Category Accuracy ─────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
locomo_cats = ['Cat 1\n(Identity)', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5', 'Overall']
locomo_vals = [71, 87, 85, 91, 88, 85.2]
colors = [C_LOCOMO if c == 'Overall' else '#C47DAA' for c in locomo_cats]
colors[0] = '#E8A0CC'   # Cat 1 lighter (lower score)
bars5 = ax3.bar(range(len(locomo_cats)), locomo_vals, color=colors, zorder=3)
for bar, v in zip(bars5, locomo_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{v:.1f}%', ha='center', va='bottom',
             fontsize=8.5, color=C_TEXT,
             fontweight='bold' if v == 85.2 else 'normal')
ax3.set_xticks(range(len(locomo_cats)))
ax3.set_xticklabels(locomo_cats, fontsize=8.5)
ax3.set_ylabel('Accuracy (%)', color=C_TEXT, fontsize=9)
ax3.set_ylim(0, 105)
overall_patch = mpatches.Patch(color=C_LOCOMO, label='Overall: 85.2%')
ax3.legend(handles=[overall_patch], fontsize=8, frameon=False)
style_ax(ax3, 'LoCoMo Cross-Session Memory Recall\n(19 sessions / 419 turns)')

# ── 4. Phase 2 Summary ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(C_BG)
ax4.spines[:].set_visible(False)
ax4.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax4.set_title('Phase 2 Highlights', fontsize=12, fontweight='bold',
              color=C_TEXT, pad=8)

metrics = [
    ('Tool-Call Accuracy', '100%',  'Short / Medium / Long'),
    ('Long-context Quality', '+114.3%', 'vs Phase 1 baseline'),
    ('LoCoMo Recall',       '85.2%', '19 sessions / 419 turns'),
    ('Memory System',       'Vector Store\n+ ChromaDB', '1536-dim OpenAI Embeddings'),
    ('Context Compression', '≤ 14 turns', 'Auto-summarise old messages'),
]

for i, (label, value, sub) in enumerate(metrics):
    y_pos = 0.88 - i * 0.19
    ax4.text(0.04, y_pos,      label, transform=ax4.transAxes,
             fontsize=9, color='#666', va='top')
    ax4.text(0.04, y_pos-0.06, value, transform=ax4.transAxes,
             fontsize=13, fontweight='bold', color=C_PHASE2, va='top')
    ax4.text(0.04, y_pos-0.12, sub,   transform=ax4.transAxes,
             fontsize=8, color='#888', va='top')

out_path = REPO_ROOT / 'docs' / 'benchmark_performance.png'
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=C_BG)
print(f'Saved: {out_path}')
