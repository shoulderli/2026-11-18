"""Cluster bootstrap confidence intervals.

Configurations inside a slice share seeds and onsets, so the effective sample
size is smaller than the configuration count.  We resample clusters
(dataset, onset) with replacement and recompute the feasibility rate.
"""
import collections
import json

import numpy as np

rng = np.random.default_rng(7)
B = 2000


def arith(x):
    if x['mode'] == 'C':
        return (x['clean_rate'] >= 0.5) and (x['atk_fs'] <= 0.1)
    return bool(x['succ'])


def boot(groups, key):
    """groups: dict cluster -> list of (arith, feas)."""
    keys = list(groups)
    out = []
    for _ in range(B):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        a = f = 0
        for i in pick:
            for ai, fi in groups[keys[i]]:
                a += ai
                f += fi
        if a:
            out.append(f / a)
    if not out:
        return (np.nan, np.nan)
    return tuple(np.percentile(out, [2.5, 97.5]))


print('=== main grid: feasibility rate among arithmetic-passing ===')
rows = [json.loads(l) for l in open('outputs/phase.jsonl')]
for mode, label in (('A', 'presence'), ('B', 'injection'), ('C', 'suppression')):
    g = collections.defaultdict(list)
    for x in rows:
        if x['mode'] != mode or not arith(x):
            continue
        g[(x['dataset'], x['t0'])].append((1, int(x['feas_seg'])))
    tot = sum(ai for v in g.values() for ai, _ in v)
    fea = sum(fi for v in g.values() for _, fi in v)
    lo, hi = boot(g, mode)
    print(' %-11s %3d/%3d = %.3f  95%% CI [%.3f, %.3f]  clusters %d'
          % (label, fea, tot, fea / tot if tot else 0, lo, hi, len(g)))

print()
print('=== n_min sweep: presence feasibility rate among all configs ===')
rows = []
for s in (0, 1):
    rows += [json.loads(l) for l in open('outputs/nmin_big_%d.jsonl' % s)]
for nm in (1, 2, 3, 4, 5):
    g = collections.defaultdict(list)
    for x in rows:
        if x['mode'] != 'A' or x['n_min'] != nm:
            continue
        g[(x['dataset'], x['t0'])].append((1, int(x['feas_seg'])))
    tot = sum(ai for v in g.values() for ai, _ in v)
    fea = sum(fi for v in g.values() for _, fi in v)
    lo, hi = boot(g, nm)
    print(' n_min=%d  %3d/%4d = %.4f  95%% CI [%.4f, %.4f]  clusters %d'
          % (nm, fea, tot, fea / tot if tot else 0, lo, hi, len(g)))

print()
print('=== n_min sweep: injection feasibility rate ===')
for nm in (1, 2, 3, 4, 5):
    g = collections.defaultdict(list)
    for x in rows:
        if x['mode'] != 'B' or x['n_min'] != nm:
            continue
        g[(x['dataset'], x['t0'])].append((1, int(x['feas_seg'])))
    tot = sum(ai for v in g.values() for ai, _ in v)
    fea = sum(fi for v in g.values() for _, fi in v)
    lo, hi = boot(g, nm)
    print(' n_min=%d  %3d/%4d = %.4f  95%% CI [%.4f, %.4f]'
          % (nm, fea, tot, fea / tot if tot else 0, lo, hi))
