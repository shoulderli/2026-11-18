"""Two extra sweeps for the expanded paper.

1. Minimum support:  n_min in {1..5}, threshold at 0.5*v_ref.
2. Decision threshold: thr in {0.4,0.5,0.6,0.7}*v_ref, n_min=3
   (this is Corollary 2: theta -> 0 at r = 1).

Usage: python run_sweeps.py <shard>   with shard in {0,1}; each handles two
dataset slices.
"""
import json
import sys

import numpy as np

sys.path.insert(0, 'src')
from overnight import run_one
from nextgen_load import load

DATASETS = [
    ('h2_train', 'highway_2', 'Train', (452150, 452300), (5100.0, 5400.0)),
    ('h2_test', 'highway_2', 'Test', (449900, 450000), (13500.0, 13800.0)),
    ('h7_train', 'highway_7', 'Train', (452150, 452300), (25200.0, 25500.0)),
    ('h7_test', 'highway_7', 'Test', (449900, 450000), (25200.0, 25500.0)),
]
shard = int(sys.argv[1])
mine = DATASETS[shard * 2:(shard + 1) * 2]

rows = []
for name, tag, split, band, t_win in mine:
    sim = load(split=split, band=band, t_win=t_win, tag=tag)
    if sim['n_senders'] == 0:
        continue
    vref = float(np.percentile(sim['V_rep'][sim['AL']], 90) * 3.6)
    print(name, 'vref', round(vref, 1), flush=True)
    base = dict(sim_override=sim, win=10)
    for mode in ('A', 'B', 'C'):
        for K in (2, 3, 4, 6):
            for t0 in (20, 100):
                for p in (1.0, 0.5):
                    for sd in (0, 1):
                        # --- minimum-support sweep ---
                        for nm in (1, 2, 3, 4, 5):
                            try:
                                r = run_one(mode, 0.02, p, K, 0.0, 5, (0, 4), sd,
                                            n_min=nm, thr_kmh=round(0.5 * vref, 1),
                                            t0=t0, a_falsify=(mode == 'A'),
                                            **base)
                            except Exception as e:
                                print('FAIL nm', mode, K, t0, p, nm, repr(e), flush=True)
                                continue
                            r.update(dataset=name, sweep='nmin', n_min=nm,
                                     thr=round(0.5 * vref, 1), K=K, t0=t0, p=p)
                            rows.append(r)
                        # --- decision-threshold sweep ---
                        for mult in (0.4, 0.6, 0.7):
                            try:
                                r = run_one(mode, 0.02, p, K, 0.0, 5, (0, 4), sd,
                                            n_min=3, thr_kmh=round(mult * vref, 1),
                                            t0=t0, a_falsify=(mode == 'A'),
                                            **base)
                            except Exception as e:
                                print('FAIL thr', mode, K, t0, p, mult, repr(e), flush=True)
                                continue
                            r.update(dataset=name, sweep='thr', n_min=3,
                                     thr=round(mult * vref, 1), thr_mult=mult,
                                     K=K, t0=t0, p=p)
                            rows.append(r)
    with open('outputs/sweeps_s%d.jsonl' % shard, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(name, 'rows so far', len(rows), flush=True)

print('shard', shard, 'rows', len(rows))
