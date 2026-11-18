"""Feasible configurations vs minimum support, from the 576-per-cell sweep."""
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURF = '#fcfcfb'
INK = '#0b0b0b'
MUTED = '#898781'
BLUE = '#2a78d6'
ORANGE = '#eb6834'
AQUA = '#1baf7a'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7.5,
    'axes.facecolor': SURF, 'figure.facecolor': SURF,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
    'axes.linewidth': 0.6,
})

rows = []
for s in (0, 1):
    rows += [json.loads(l) for l in open('outputs/nmin_big_%d.jsonl' % s)]

fig, ax = plt.subplots(figsize=(3.4, 2.1), constrained_layout=True)
nms = [1, 2, 3, 4, 5]
for m, col, lab in (('A', ORANGE, 'presence'), ('B', BLUE, 'injection'),
                    ('C', AQUA, 'suppression')):
    y = [sum(1 for r in rows if r['mode'] == m and r['n_min'] == nm
             and r['feas_seg']) for nm in nms]
    ax.plot(nms, y, '-o', color=col, lw=1.6, ms=5, label=lab,
            markeredgecolor=SURF, markeredgewidth=1.0)
    for x, v in zip(nms, y):
        if v > 0:
            ax.text(x, v + 1.2, str(v), ha='center', fontsize=6.5, color=col)
ax.set_xticks(nms)
ax.set_xlabel(r'minimum support $n_{\min}$')
ax.set_ylabel('feasible configurations')
ax.set_ylim(-2, 38)
ax.grid(True, color='#e1e0d9', lw=0.4)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.5, loc='upper right')
fig.savefig('outputs/fig_nmin.png', dpi=300, facecolor=SURF,
            bbox_inches='tight')
print('saved')
