"""Figure 2: the two gates above the arithmetic one.

Bar height = configurations passing the arithmetic gate (f > theta), of 540.
Stack      = blocked by spatial conflict / feasible.
Conservation never fires on an arithmetic-passing configuration: its signal is
K/p while its noise is sqrt(N(1-p))/p, so SNR = K/sqrt(N(1-p)) < 1 throughout.
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
BLUE = '#2a78d6'
G2 = '#b5b3aa'
G3 = '#7d7b74'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.facecolor': SURF, 'figure.facecolor': SURF,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK,
    'axes.linewidth': 0.6,
})

rows = [json.loads(l) for l in open('outputs/phase.jsonl')]
DS = ['h2_train', 'h2_test', 'h7_train', 'h7_test']
SHORT = {'h2_train': 'h2 train\nsparse', 'h2_test': 'h2 test\nsparse',
         'h7_train': 'h7 train\nmid', 'h7_test': 'h7 test\ndense'}
MODES = [('A', '(a) present, falsify'), ('B', '(b) phantom inject'),
         ('C', '(c) suppress')]


def arith(x):
    if x['mode'] == 'C':
        return (x['clean_rate'] >= 0.5) and (x['atk_fs'] <= 0.1)
    return bool(x['succ'])


fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), constrained_layout=True)
x = np.arange(len(DS))
for ax, (m, title) in zip(axes, MODES):
    feas, conf, cons, tot = [], [], [], []
    for ds in DS:
        r = [q for q in rows if q['mode'] == m and q['dataset'] == ds and arith(q)]
        tot.append(len(r))
        feas.append(sum(1 for q in r if q['feas_seg']))
        conf.append(sum(1 for q in r if not q['feas_seg'] and q['conf'] >= 0))
        cons.append(sum(1 for q in r if not q['feas_seg'] and q['conf'] < 0 and q['det_seg']))
    b = np.array(feas, float)
    ax.bar(x, feas, color=BLUE, width=0.6, edgecolor=SURF, linewidth=1.2,
           label='feasible')
    ax.bar(x, cons, bottom=b, color=G3, width=0.6, edgecolor=SURF,
           linewidth=1.2, label='blocked: conservation')
    b = b + np.array(cons, float)
    ax.bar(x, conf, bottom=b, color=G2, width=0.6, edgecolor=SURF,
           linewidth=1.2, label='blocked: spatial conflict')
    for j in range(len(DS)):
        if tot[j]:
            ax.text(x[j], tot[j] + 6, '%d' % tot[j], ha='center', va='bottom',
                    fontsize=7, color=INK)
        if feas[j]:
            ax.text(x[j], feas[j] / 2, '%d' % feas[j], ha='center', va='center',
                    fontsize=7, color='#ffffff')
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[d] for d in DS], fontsize=6.5)
    ax.set_ylim(0, 320)
    ax.set_title(title, fontsize=8.5, loc='left', color=INK, pad=10)
    ax.grid(True, axis='y', color='#e1e0d9', lw=0.4)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel('pass arithmetic (of 540)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, ncol=3, loc='upper center',
           bbox_to_anchor=(0.5, 1.09))
fig.savefig('outputs/fig2_attrib.png', dpi=300, facecolor=SURF,
            bbox_inches='tight')
print('saved outputs/fig2_attrib.png')
for m, _ in MODES:
    t = [sum(1 for q in rows if q['mode'] == m and q['dataset'] == ds and arith(q))
         for ds in DS]
    f = [sum(1 for q in rows if q['mode'] == m and q['dataset'] == ds
             and q['feas_seg']) for ds in DS]
    print(m, 'arith', t, 'feas', f)
