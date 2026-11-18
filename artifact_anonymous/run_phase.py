"""Unified phase sweep: A/B/C on all four NextGen slices with the corrected
attacker model (A falsifies position), the corrected verdict (C is fail-safe),
and the clean-state speed recorded for the phase diagram."""
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

out = 'outputs/phase.jsonl'
rows = []
for name, tag, split, band, t_win in DATASETS:
    sim = load(split=split, band=band, t_win=t_win, tag=tag)
    if sim['n_senders'] == 0:
        continue
    vref = float(np.percentile(sim['V_rep'][sim['AL']], 90) * 3.6)
    thr = round(0.5 * vref, 1)
    print(name, 'N', sim['n_senders'], 'vref', round(vref, 1), 'thr', thr,
          'active', round(float(sim['AL'].sum(1).mean()), 1))
    for mode in ('B', 'A', 'C'):
        for K in (1, 2, 3, 4, 6):
            for win in (5, 10, 30):
                for t0 in (20, 50, 100, 150, 200, 250):
                    for p in (1.0, 0.7, 0.5):
                        for sd in (0, 1):
                            try:
                                r = run_one(mode, 0.02, p, K, 0.0, 5, (0, 4), sd,
                                            sim_override=sim, t0=t0, win=win,
                                            n_min=3, thr_kmh=thr,
                                            a_falsify=(mode == 'A'))
                            except Exception as e:
                                print('FAIL', mode, K, win, t0, p, repr(e))
                                continue
                            r.update(dataset=name, v_ref=vref, thr=thr,
                                     win=win, p=p, t0=t0, n_min=3)
                            rows.append(r)
    for mode in ('B', 'A', 'C'):
        v = [x for x in rows if x['dataset'] == name and x['mode'] == mode]
        print('  %s succ %d feas %d feas_fs %d conf %d det %d' %
              (mode, sum(x['succ'] for x in v), sum(x['feas'] for x in v),
               sum(x['feas_fs'] for x in v), sum(1 for x in v if x['conf'] >= 0),
               sum(1 for x in v if x['det'])))
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

with open(out, 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
print('rows', len(rows))
