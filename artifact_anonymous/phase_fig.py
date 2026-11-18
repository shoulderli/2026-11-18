"""Figure 1: phase diagram of colluder feasibility.

x = r = mass-weighted clean mean speed / tau   (distance of the state from
    the decision threshold, dimensionless)
y = f = K / n_clean                            (colluder share of the honest
    reporters inside the decision segment)

Necessary condition for flipping a mean-speed decision:
  theta_B = r - 1        (inject K zero-speed phantoms)
  theta_A = (r - 1) / r  (K present vehicles report zero)
  theta_C = 1 - r        (suppress the K slowest reports, r < 1)
"""
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURF = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BLUE = '#2a78d6'
ORANGE = '#eb6834'
AQUA = '#1baf7a'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.facecolor': SURF, 'figure.facecolor': SURF,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
    'axes.linewidth': 0.6,
})

YMIN, YMAX = 0.03, 60.0
rows = [json.loads(l) for l in open('outputs/phase.jsonl')]
for r in rows:
    # r is the physical state of the segment: mean honest-reported speed
    # weighted by presence, i.e. total speed mass over total report count.
    # (n_min is a defender design choice and does not enter the state.)
    nm = r.get('n_mass', 0.0)
    r['r'] = (r.get('v_mass', 0.0) / nm) * 3.6 / r['thr'] if nm > 0 and r['thr'] else np.nan
    r['f'] = r['K'] / r['n_clean_mean'] if r['n_clean_mean'] > 0 else np.nan

THETA = {'A': lambda x: (x - 1.0) / np.maximum(x, 1e-3),
         'B': lambda x: x - 1.0,
         'C': lambda x: 1.0 - x}
COL = {'A': ORANGE, 'B': BLUE, 'C': AQUA}
LABEL = {'A': 'A  present / falsify', 'B': 'B  phantom inject',
         'C': 'C  suppress'}
TITLE = {'A': '(a) present, falsify', 'B': '(b) phantom inject',
         'C': '(c) suppress'}

def _arith(x):
    if x['mode'] == 'C':
        return (x['clean_rate'] >= 0.5) and (x['atk_fs'] <= 0.1)
    return bool(x['succ'])


fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), constrained_layout=True)

for ax, m in zip(axes.flat[[0, 1, 2]], ('A', 'B', 'C')):
    sub = [r for r in rows if r['mode'] == m and np.isfinite(r['r'])
           and np.isfinite(r['f']) and YMIN <= r['f'] <= YMAX
           and 0.25 <= r['r'] <= 2.45]
    neg = [r for r in sub if not r['feas_seg']]
    pos = [r for r in sub if r['feas_seg']]
    ax.scatter([r['r'] for r in neg], [r['f'] for r in neg], s=7,
               facecolors='none', edgecolors=MUTED, linewidths=0.4,
               alpha=0.30, label='blocked')
    ax.scatter([r['r'] for r in pos], [r['f'] for r in pos], s=18,
               color=COL[m], edgecolors=SURF, linewidths=0.5,
               alpha=0.9, label='feasible', zorder=3)
    xx = np.linspace(0.25, 2.45, 600)
    yy = THETA[m](xx)
    yy = np.ma.masked_where(yy <= YMIN, yy)
    ax.plot(xx, yy, color=INK, lw=1.6, label=r'$\theta_{%s}(r)$' % m, zorder=4)
    ax.axvline(1.0, color=MUTED, lw=0.6, ls=(0, (3, 3)), zorder=1)
    ax.set_xlim(0.25, 2.45)
    ax.set_yscale('log')
    ax.set_ylim(YMIN, YMAX)
    ax.set_title(TITLE[m], fontsize=8.5, color=INK, loc='left')
    ax.set_xlabel(r'$r=\bar v_{\mathrm{clean}}/\tau$')
    ax.set_ylabel(r'$f=K/n$')
    ax.grid(True, color=GRID, lw=0.4, which='major')
    ax.set_axisbelow(True)
    below = sum(1 for r in pos if r['f'] <= THETA[m](r['r']))
    ax.text(0.97, 0.95, 'feasible %d, below $\\theta$ %d' % (len(pos), below),
            transform=ax.transAxes, ha='right', va='top', fontsize=6.5,
            color=INK2)

# empty-segment corner (n = 0) cannot be placed on the f axis; state it.
nz = [r for r in rows if r['mode'] == 'B' and r['n_clean_mean'] <= 0
      and r['feas_seg']]
if nz:
    axes.flat[1].text(0.97, 0.05, 'n=0 (empty segment): %d feasible' % len(nz),
                      transform=axes.flat[1].transAxes, ha='right', va='bottom',
                      fontsize=6.5, color=INK2)

ax = axes.flat[3]
bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50)]
cent = [1, 3.2, 7, 14, 32]
for m in ('B', 'A', 'C'):
    fr = []
    for lo, hi in bins:
        v = [r for r in rows if r['mode'] == m and lo <= r['n_clean_mean'] < hi]
        fr.append(np.mean([_arith(r) for r in v]) if v else np.nan)
    ax.plot(cent, fr, '-o', color=COL[m], lw=1.8, ms=5.5, label=LABEL[m],
            markeredgecolor=SURF, markeredgewidth=1.0)
ax.set_xscale('log')
ax.set_xlabel(r'honest reporters in segment, $n$')
ax.set_ylabel('arithmetically feasible')
ax.set_ylim(-0.02, 0.50)
ax.grid(True, color=GRID, lw=0.4)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.5, loc='upper right')
ax.set_title('(d) complementarity (arithmetic only)', fontsize=8.5, color=INK, loc='left')

fig.savefig('outputs/fig1_phase.png', dpi=300, facecolor=SURF)
print('saved outputs/fig1_phase.png')

print('\n=== hard-floor check ===')
for m in ('A', 'B', 'C'):
    sub = [r for r in rows if r['mode'] == m and np.isfinite(r['r'])
           and np.isfinite(r['f'])]
    pos = [r for r in sub if r['feas_seg']]
    below = [r for r in pos if r['f'] <= THETA[m](r['r'])]
    print(' %s: feasible %3d  below theta %2d  r %.2f-%.2f'
          % (m, len(pos), len(below),
             min(r['r'] for r in pos), max(r['r'] for r in pos)))
