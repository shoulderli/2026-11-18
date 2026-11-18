"""NextGen grid: highway_2 (2 AM) + highway_7 (7 AM), Train + Test.

Train and Test are spatially disjoint (different y bands).  highway_2 is
sparse (~1.5 active vehicles), highway_7 is ~10x denser.
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

rows = []
for name, tag, split, band, t_win in DATASETS:
    sim = load(split=split, band=band, t_win=t_win, tag=tag)
    if sim['n_senders'] == 0:
        print('empty', name)
        continue
    vref = float(np.percentile(sim['V_rep'][sim['AL']], 90) * 3.6)
    thr = round(0.5 * vref, 1)
    print(name, 'N', sim['n_senders'], 'vref_kmh', round(vref, 1), 'thr', thr,
          'active_mean', round(float(sim['AL'].sum(1).mean()), 1))
    for mode in ('B', 'A', 'C'):
        for K in (1, 2, 3, 4, 6):
            for win in (5, 10, 30):
                for t0 in (20, 50, 100, 150, 200, 250):
                    for p in (1.0, 0.7, 0.5):
                        for sd in (0, 1):
                            try:
                                r = run_one(mode, 0.02, p, K, 0.0, 5, (0, 4), sd,
                                            sim_override=sim, t0=t0, win=win,
                                            n_min=3, thr_kmh=thr)
                            except Exception as e:
                                print('FAIL', mode, K, win, t0, p, repr(e))
                                continue
                            r.update(dataset=name, v_ref=vref, thr=thr,
                                     win=win, p=p, t0=t0, n_min=3)
                            rows.append(r)
    for mode in ('B', 'A', 'C'):
        for K in (1, 2, 3, 4, 6):
            v = [r for r in rows if r['dataset'] == name
                 and r['mode'] == mode and r['K'] == K]
            if v:
                print(name, mode, 'K=%d' % K,
                      'feas %d/%d' % (sum(x['feas'] for x in v), len(v)),
                      'atk %.2f' % np.mean([x['atk_rate'] for x in v]),
                      'clean %.2f' % np.mean([x['clean_rate'] for x in v]))
    with open('outputs/nextgen_grid.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

with open('outputs/nextgen_grid.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
print('rows', len(rows))
