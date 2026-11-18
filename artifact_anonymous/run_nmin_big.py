"""Large minimum-support sweep: 576 configurations per (mode, n_min) cell.

Usage: python run_nmin_big.py <shard>   with shard in {0,1}; shard 0 takes the
two sparse slices, shard 1 the two dense ones.
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
mine = DATASETS[0:2] if shard == 0 else DATASETS[2:4]

rows = []
for name, tag, split, band, t_win in mine:
    sim = load(split=split, band=band, t_win=t_win, tag=tag)
    if sim['n_senders'] == 0:
        continue
    vref = float(np.percentile(sim['V_rep'][sim['AL']], 90) * 3.6)
    thr = round(0.5 * vref, 1)
    print(name, 'thr', thr, flush=True)
    for mode in ('A', 'B', 'C'):
        for K in (1, 2, 3, 4, 6, 8):
            for t0 in (20, 60, 100, 150):
                for p in (1.0, 0.7, 0.5):
                    for sd in (0, 1):
                        for nm in (1, 2, 3, 4, 5):
                            try:
                                r = run_one(mode, 0.02, p, K, 0.0, 5, (0, 4), sd,
                                            sim_override=sim, win=10, t0=t0,
                                            n_min=nm, thr_kmh=thr,
                                            a_falsify=(mode == 'A'))
                            except Exception as e:
                                print('FAIL', mode, K, t0, p, nm, repr(e),
                                      flush=True)
                                continue
                            r.update(dataset=name, sweep='nmin_big', n_min=nm,
                                     thr=thr, K=K, t0=t0, p=p)
                            rows.append(r)
    print(name, 'rows so far', len(rows), flush=True)
    with open('outputs/nmin_big_%d.jsonl' % shard, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

print('shard', shard, 'rows', len(rows))
