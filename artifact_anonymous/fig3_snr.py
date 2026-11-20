"""Figure 3: predicted vs measured detection probability of segment
conservation, as a function of the SNR of Proposition 1.

SNR = K / sqrt(n (1-p_eff)); prediction P = Phi(SNR-z) + Phi(-SNR-z)
with z = z_{0.995} = 2.576 for the two-sided 99th-percentile rule.
"""
import json
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURF = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
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

rows = [json.loads(l) for l in open('outputs/phase.jsonl')]
pts = []
for r in rows:
    n = r['n_clean_mean']
    if n <= 0 or r['mode'] == 'A':
        continue
    p = 0.98 * r['p']
    snr = r['K'] / math.sqrt(n * (1 - p)) if p < 1 else float('inf')
    if not np.isfinite(snr):
        continue
    pts.append((snr, bool(r['det_seg']), r['mode']))

edges = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 30])
cent, rate, cnt = [], [], []
for a, b in zip(edges[:-1], edges[1:]):
    v = [x for x in pts if a <= x[0] < b]
    if len(v) >= 20:
        cent.append((a + b) / 2)
        rate.append(np.mean([x[1] for x in v]))
        cnt.append(len(v))

fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
ax.scatter(cent, rate, s=[max(12, min(60, c / 20)) for c in cnt], color=BLUE,
           edgecolors=SURF, linewidths=0.8, zorder=3, label='measured')
xx = np.linspace(0, 8, 300)
zz = 2.576
ax.plot(xx, [0.5 * (1 + math.erf((x - zz) / math.sqrt(2)))
             + 0.5 * (1 + math.erf((-x - zz) / math.sqrt(2))) for x in xx],
        color=INK, lw=1.5, label='two-sided Gaussian')
ax.axvline(zz, color=MUTED, lw=0.6, ls=(0, (3, 3)))
ax.set_xlim(0, 8)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel(r'$\mathrm{SNR}=K/\sqrt{n(1-p_{\mathrm{eff}})}$')
ax.set_ylabel('detection rate')
ax.grid(True, color='#e1e0d9', lw=0.4)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.5, loc='lower right')
fig.savefig('outputs/fig3_snr.png', dpi=300, facecolor=SURF)
print('saved outputs/fig3_snr.png')

for a, b, c, n in zip(cent, rate, cnt, cnt):
    print(' SNR %.2f  rate %.2f  n %d' % (a, b, c))
