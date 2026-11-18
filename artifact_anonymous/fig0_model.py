"""Figure 1: system and threat model.

(a) the same decision segment under the three attacks
(b) attacks -> two ledgers -> two checks
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

SURF = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
ROAD = '#e6e5de'
BLUE = '#2a78d6'
ORANGE = '#eb6834'
AQUA = '#1baf7a'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7,
    'axes.facecolor': SURF, 'figure.facecolor': SURF,
    'text.color': INK,
})

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.7),
                               gridspec_kw={'width_ratios': [1.5, 1]})
for ax in (axa, axb):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# ---------------- (a) three lanes ----------------
axa.set_title('(a) one segment, three perturbations', loc='left',
              fontsize=7.5, color=INK, pad=2)
lanes = [('B  inject', 0.80, BLUE), ('A  present', 0.50, ORANGE),
         ('C  suppress', 0.20, AQUA)]
x0, x1 = 0.24, 0.96
for name, y, col in lanes:
    axa.add_patch(Rectangle((x0, y - 0.05), x1 - x0, 0.10, color=ROAD, zorder=1))
    for x in (x0, x1):
        axa.plot([x, x], [y - 0.075, y + 0.075], color=MUTED, lw=1.0, zorder=2)
    axa.text(0.01, y, name, fontsize=8, color=col, va='center')
    for x in (0.32, 0.42, 0.54, 0.66, 0.78, 0.90):
        axa.add_patch(Circle((x, y), 0.014, color=INK2, zorder=3))
    if col == BLUE:
        for x in (0.48, 0.60):
            axa.add_patch(Circle((x, y), 0.015, facecolor=SURF, edgecolor=BLUE,
                                 lw=1.3, ls=(0, (2, 1.4)), zorder=4))
        axa.annotate('phantoms', xy=(0.54, y + 0.02), xytext=(0.55, y + 0.12),
                     fontsize=6, color=BLUE, ha='center',
                     arrowprops=dict(arrowstyle='-', color=BLUE, lw=0.6))
    elif col == ORANGE:
        axa.add_patch(Circle((0.42, y), 0.026, facecolor='none',
                             edgecolor=ORANGE, lw=1.3, zorder=4))
        axa.annotate('claims 0', xy=(0.42, y - 0.026), xytext=(0.40, y - 0.13),
                     fontsize=6, color=ORANGE, ha='center',
                     arrowprops=dict(arrowstyle='-', color=ORANGE, lw=0.6))
    else:
        axa.add_patch(Circle((0.66, y), 0.014, facecolor=SURF, edgecolor=AQUA,
                             lw=1.1, ls=(0, (1, 1)), zorder=4))
        axa.annotate('no report', xy=(0.66, y + 0.02), xytext=(0.67, y + 0.12),
                     fontsize=6, color=AQUA, ha='center',
                     arrowprops=dict(arrowstyle='-', color=AQUA, lw=0.6))
axa.text((x0 + x1) / 2, 0.03, 'boundary counters', fontsize=6, color=MUTED,
         ha='center')

# ---------------- (b) block diagram ----------------
axb.set_title('(b) two ledgers, two checks', loc='left', fontsize=7.5,
              color=INK, pad=2)
for lab, y, col in (('B', 0.88, BLUE), ('A', 0.68, ORANGE), ('C', 0.12, AQUA)):
    axb.add_patch(FancyBboxPatch((0.02, y - 0.055), 0.11, 0.11,
                                 boxstyle='round,pad=0.01', fc='white',
                                 ec=col, lw=1.1))
    axb.text(0.075, y, lab, ha='center', va='center', fontsize=9, color=col)

axb.add_patch(FancyBboxPatch((0.30, 0.625), 0.19, 0.11,
                             boxstyle='round,pad=0.005', fc='#f0efec',
                             ec=MUTED, lw=0.6))
axb.text(0.395, 0.68, 'position', ha='center', va='center', fontsize=6.5)
axb.add_patch(FancyBboxPatch((0.30, 0.265), 0.19, 0.11,
                             boxstyle='round,pad=0.005', fc='#f0efec',
                             ec=MUTED, lw=0.6))
axb.text(0.395, 0.32, 'count', ha='center', va='center', fontsize=6.5)

def arrow(x1, y1, x2, y2, col):
    axb.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                  mutation_scale=7, lw=0.9, color=col,
                                  shrinkA=1, shrinkB=1))

arrow(0.13, 0.90, 0.30, 0.72, BLUE)
arrow(0.13, 0.84, 0.30, 0.35, BLUE)
arrow(0.13, 0.68, 0.30, 0.68, ORANGE)
arrow(0.13, 0.14, 0.30, 0.29, AQUA)
axb.text(0.185, 0.90, '+K', fontsize=6, color=BLUE, ha='center')
axb.text(0.225, 0.585, 'claims', fontsize=6, color=BLUE, ha='center')
axb.text(0.19, 0.11, '−K', fontsize=6, color=AQUA, ha='center')

axb.add_patch(FancyBboxPatch((0.62, 0.61), 0.34, 0.14,
                             boxstyle='round,pad=0.01', fc='white',
                             ec=INK2, lw=1.0))
axb.text(0.79, 0.68, 'conflict', ha='center', va='center', fontsize=6.5)
axb.add_patch(FancyBboxPatch((0.62, 0.25), 0.34, 0.14,
                             boxstyle='round,pad=0.01', fc='white',
                             ec=INK2, lw=1.0))
axb.text(0.79, 0.32, 'conservation', ha='center', va='center', fontsize=6.5)
arrow(0.49, 0.68, 0.62, 0.68, INK2)
arrow(0.49, 0.32, 0.62, 0.32, INK2)
axb.text(0.555, 0.705, '$r_c$', fontsize=6, color=INK2, ha='center')
axb.text(0.555, 0.345, '$M$', fontsize=6, color=INK2, ha='center')

fig.savefig('outputs/fig0_model.png', dpi=300, facecolor=SURF,
            bbox_inches='tight')
print('saved')
