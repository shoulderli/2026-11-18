"""Figure 1: system and threat model (three panels, print-ready)."""
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

SURF = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
ROAD = '#e8e7e1'
LANE = '#ffffff'
BLUE = '#2a78d6'
ORANGE = '#eb6834'
AQUA = '#1baf7a'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7,
    'axes.facecolor': SURF, 'figure.facecolor': SURF,
    'text.color': INK, 'axes.edgecolor': MUTED,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.linewidth': 0.6,
})

fig = plt.figure(figsize=(7.2, 3.05))
gs = fig.add_gridspec(2, 2, height_ratios=[0.86, 1.14], width_ratios=[1.12, 1.0],
                      hspace=0.26, wspace=0.16,
                      left=0.03, right=0.985, top=0.95, bottom=0.10)

# ---------------------------------------------------------------- (a) road
axa = fig.add_subplot(gs[0, :])
axa.set_xlim(0, 1); axa.set_ylim(0, 1); axa.axis('off')
axa.set_title('(a) one decision segment, three perturbations', loc='left',
              fontsize=7.8, color=INK, pad=10)

x0, x1 = 0.07, 0.93
axa.add_patch(Rectangle((x0, 0.34), x1 - x0, 0.48, fc=ROAD, ec='none', zorder=1))
axa.plot([x0, x1], [0.58, 0.58], color=LANE, lw=1.0, ls=(0, (6, 5)), zorder=2)
axa.plot([x0, x1], [0.34, 0.34], color=MUTED, lw=0.7, zorder=2)
axa.plot([x0, x1], [0.82, 0.82], color=MUTED, lw=0.7, zorder=2)

for x in (x0, x1):
    axa.plot([x, x], [0.26, 0.90], color=INK2, lw=1.1, zorder=4)
    axa.plot([x - 0.013, x + 0.013], [0.90, 0.90], color=INK2, lw=1.1, zorder=4)
axa.text(x0, 0.17, 'passage counter', ha='left', fontsize=6.0, color=INK2)
axa.text(x1, 0.17, 'passage counter', ha='right', fontsize=6.0, color=INK2)

def car(ax, x, y, fc=INK2, ec=None, dashed=False, w=0.034, h=0.105):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle='round,pad=0.002,rounding_size=0.011',
                                fc=fc, ec=ec if ec else fc, lw=1.0,
                                ls=(0, (2.2, 1.6)) if dashed else '-', zorder=5))

for x in (0.15, 0.27, 0.40, 0.76, 0.87):
    car(axa, x, 0.70)
for x in (0.19, 0.31, 0.64, 0.88):
    car(axa, x, 0.46)

car(axa, 0.52, 0.70, fc=SURF, ec=BLUE, dashed=True)
car(axa, 0.64, 0.70, fc=SURF, ec=BLUE, dashed=True)
car(axa, 0.31, 0.46, fc=ORANGE, ec=ORANGE)
axa.add_patch(Rectangle((0.31 - 0.030, 0.46 - 0.075), 0.060, 0.150, fc='none',
                        ec=ORANGE, lw=1.0, ls=(0, (2, 2)), zorder=4))
car(axa, 0.76, 0.46, fc=SURF, ec=AQUA, dashed=True)

def tag(text, xy, xytext, col):
    axa.annotate(text, xy=xy, xytext=xytext, fontsize=6.6, color=col,
                 ha='center', va='center',
                 arrowprops=dict(arrowstyle='-', color=col, lw=0.6,
                                 shrinkA=1, shrinkB=1))

tag('A  present: claims 0', (0.31, 0.55), (0.18, 0.955), ORANGE)
tag('B  inject: $K$ phantoms at 0', (0.58, 0.76), (0.55, 0.955), BLUE)
tag('C  suppress: no report', (0.76, 0.55), (0.855, 0.955), AQUA)
axa.text(0.5, 0.045, r'decision $=\ \bar v_J<\tau$', ha='center',
         fontsize=6.6, color=MUTED)

# ---------------------------------------------------------------- (b) ledger
axb = fig.add_subplot(gs[1, 0])
axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.axis('off')
axb.set_title('(b) two ledgers, two checks', loc='left', fontsize=7.8,
              color=INK, pad=1)

for lab, y, col in (('A', 0.83, ORANGE), ('B', 0.50, BLUE), ('C', 0.17, AQUA)):
    axb.add_patch(FancyBboxPatch((0.015, y - 0.072), 0.095, 0.144,
                                 boxstyle='round,pad=0.008', fc='white',
                                 ec=col, lw=1.1))
    axb.text(0.0625, y, lab, ha='center', va='center', fontsize=9.0, color=col)

def ledger(y, label):
    axb.add_patch(FancyBboxPatch((0.285, y - 0.060), 0.155, 0.120,
                                 boxstyle='round,pad=0.006', fc='#f2f1ec',
                                 ec=MUTED, lw=0.55))
    axb.text(0.3625, y, label, ha='center', va='center', fontsize=6.5)

ledger(0.73, 'position')
ledger(0.27, 'count')

def arrow(x1, y1, x2, y2, col):
    axb.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                  mutation_scale=7.0, lw=0.9, color=col,
                                  shrinkA=1, shrinkB=1))

arrow(0.11, 0.84, 0.285, 0.755, ORANGE)
arrow(0.11, 0.52, 0.285, 0.705, BLUE)
arrow(0.11, 0.48, 0.285, 0.295, BLUE)
arrow(0.11, 0.16, 0.285, 0.245, AQUA)
axb.text(0.20, 0.615, 'claims', fontsize=6.2, color=BLUE, ha='center')
axb.text(0.20, 0.395, '+$K$', fontsize=6.2, color=BLUE, ha='center')
axb.text(0.195, 0.135, '−$K$', fontsize=6.2, color=AQUA, ha='center')

axb.add_patch(FancyBboxPatch((0.565, 0.665), 0.42, 0.135,
                             boxstyle='round,pad=0.008', fc='white',
                             ec=INK2, lw=0.95))
axb.text(0.775, 0.7325, 'spatial conflict', ha='center', va='center', fontsize=6.4)
axb.add_patch(FancyBboxPatch((0.565, 0.205), 0.42, 0.135,
                             boxstyle='round,pad=0.008', fc='white',
                             ec=INK2, lw=0.95))
axb.text(0.775, 0.2725, 'conservation', ha='center', va='center', fontsize=6.4)
arrow(0.44, 0.73, 0.565, 0.73, INK2)
arrow(0.44, 0.27, 0.565, 0.27, INK2)
axb.text(0.503, 0.762, r'$r_c$', fontsize=6.2, color=INK2, ha='center')
axb.text(0.503, 0.302, r'$M$', fontsize=6.2, color=INK2, ha='center')

# ---------------------------------------------------------------- (c) series
axc = fig.add_subplot(gs[1, 1])
axc.set_title('(c) what it does to the decision', loc='left', fontsize=7.8,
              color=INK, pad=1)
d = json.load(open('outputs/series.json'))
clean = np.array([np.nan if v is None else v for v in d['clean']])
atk = np.array([np.nan if v is None else v for v in d['atk']])
t = d['t_start'] + np.arange(len(clean))
tau = d['tau']
axc.axvspan(d['t0'], d['t0'] + d['win'], color='#eef3fb', zorder=0)
axc.axhline(tau, color=ORANGE, lw=1.0, ls=(0, (4, 3)), zorder=2)
axc.text(t[0] + 0.5, tau + 3.5, r'$\tau$', color=ORANGE, fontsize=7.2)
axc.plot(t, clean, color=INK2, lw=1.5, marker='o', ms=2.0, zorder=3)
axc.plot(t, atk, color=BLUE, lw=1.5, marker='o', ms=2.0, zorder=4)
axc.set_xlabel('time (s)', fontsize=6.6, labelpad=1)
axc.set_ylabel('mean speed (km/h)', fontsize=6.6, labelpad=1)
axc.tick_params(labelsize=6)
axc.set_xlim(139.0, 154.8)
axc.set_ylim(0, 120)
axc.grid(True, color='#e8e7e0', lw=0.4)
axc.set_axisbelow(True)
for s in ('top', 'right'):
    axc.spines[s].set_visible(False)
axc.text(0.975, 0.90, 'clean', color=INK2, fontsize=6.4, ha='right',
         transform=axc.transAxes)
axc.text(0.975, 0.10, 'attacked', color=BLUE, fontsize=6.4, ha='right',
         transform=axc.transAxes)
axc.text(d['t0'] + d['win'] / 2, 6, 'attack window', ha='center', fontsize=5.6,
         color=MUTED)

fig.savefig('outputs/fig0_model.png', dpi=300, facecolor=SURF,
            bbox_inches='tight', pad_inches=0.02)
print('saved')
