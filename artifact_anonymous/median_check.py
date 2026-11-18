"""Median vs mean: does a robust statistic remove the r -> 1 vulnerability?

Standalone check on the NextGen slices.  For each time step we take the
reported speeds inside the decision segment, apply the attack, and ask whether
the median-based decision flips.  No simulator needed.
"""
import sys

import numpy as np

sys.path.insert(0, 'src')
from nextgen_load import load

DATASETS = [
    ('h2_train', 'highway_2', 'Train', (452150, 452300), (5100.0, 5400.0)),
    ('h7_train', 'highway_7', 'Train', (452150, 452300), (25200.0, 25500.0)),
]


def runs_for(sim, K, p, t0, win, thr, seed=0):
    rng = np.random.default_rng(seed)
    X, V, AL = sim['X'], sim['V'], sim['AL']
    T, N = V.shape
    rep = AL & (rng.random((T, N)) < p * 0.98)
    cell = np.clip((X / 100.0).astype(int), 0, 19)
    seg = rep & (cell >= 0) & (cell <= 4)
    out = {}
    for stat in ('mean', 'median'):
        flip = {'B': 0, 'A': 0, 'C': 0}
        nstep = {'B': 0, 'A': 0, 'C': 0}
        for t in range(t0, min(t0 + win, T)):
            vals = V[t][seg[t]]
            n = len(vals)
            if n == 0:
                continue
            f_mean = lambda a: np.mean(a) if len(a) else 0.0
            f_med = lambda a: np.sort(a)[(len(a) - 1) // 2] if len(a) else 0.0  # lower median, as in Prop. 4
            f = f_mean if stat == 'mean' else f_med
            clean = f(vals) * 3.6
            # B: add K zeros
            if n + K >= 3:
                nstep['B'] += 1
                if clean >= thr and f(np.concatenate([vals, np.zeros(K)])) * 3.6 < thr:
                    flip['B'] += 1
            # A: the K fastest report zero
            if n >= K + 1:
                nstep['A'] += 1
                idx = np.argsort(vals)[-K:]
                a = vals.copy()
                a[idx] = 0.0
                if clean >= thr and f(a) * 3.6 < thr:
                    flip['A'] += 1
            # C: remove the K slowest (hide congestion)
            if n >= max(K + 1, 3):
                nstep['C'] += 1
                rest = np.sort(vals)[K:]
                if clean < thr and f(rest) * 3.6 >= thr:
                    flip['C'] += 1
        out[stat] = {m: (flip[m] / nstep[m] if nstep[m] else 0.0) for m in flip}
    return out


print('%-9s %-4s %-6s %-4s  %-16s %-16s' %
      ('dataset', 'mode', 'K', 'p', 'mean flip rate', 'median flip rate'))
for name, tag, split, band, t_win in DATASETS:
    sim = load(split=split, band=band, t_win=t_win, tag=tag)
    vref = float(np.percentile(sim['V_rep'][sim['AL']], 90) * 3.6)
    thr = 0.5 * vref
    print('--- %s  vref %.1f  thr %.1f  active %.1f' %
          (name, vref, thr, float(sim['AL'].sum(1).mean())))
    for K in (1, 2, 3, 4, 6):
        for p in (1.0, 0.5):
            r = runs_for(sim, K, p, 20, 30, thr)
            print('%-9s %-4s %-6d %-4.1f  %-16.2f %-16.2f' %
                  (name, 'all', K, p, np.mean(list(r['mean'].values())),
                   np.mean(list(r['median'].values()))))
    # per-mode detail at K=3, p=0.5
    r = runs_for(sim, 3, 0.5, 20, 30, thr)
    print('    K=3 p=0.5 detail  mean', {k: round(v, 2) for k, v in r['mean'].items()},
          ' median', {k: round(v, 2) for k, v in r['median'].items()})
