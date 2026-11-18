"""Mode A rerun with self-consistent position falsification (a_falsify=True).

The original run left the colluder's position reports untouched, so its own
moving positions collided with its stationary speed claim: the conflict
detector was comparing the attacker against itself.  Here the colluder claims
a stop at x0 and is removed from the honest-position cloud; conflict now means
an honest sender is within 4.5 m of the claimed stop point.
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

out = 'outputs/nextgen_a2.jsonl'
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
    for K in (1, 2, 3, 4, 6):
        for win in (5, 10, 30):
            for t0 in (20, 50, 100, 150, 200, 250):
                for p in (1.0, 0.7, 0.5):
                    for sd in (0, 1):
                        try:
                            r = run_one('A', 0.02, p, K, 0.0, 5, (0, 4), sd,
                                        sim_override=sim, t0=t0, win=win,
                                        n_min=3, thr_kmh=thr, a_falsify=True)
                        except Exception as e:
                            print('FAIL', K, win, t0, p, repr(e))
                            continue
                        r.update(dataset=name, v_ref=vref, thr=thr,
                                 win=win, p=p, t0=t0, n_min=3, a_falsify=True)
                        rows.append(r)
    v = [r for r in rows if r['dataset'] == name]
    print(name, 'succ %d/%d feas %d det %d conf>=0 %d'
          % (sum(x['succ'] for x in v), len(v), sum(x['feas'] for x in v),
             sum(1 for x in v if x['det']), sum(1 for x in v if x['conf'] >= 0)))
    with open(out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

with open(out, 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
print('rows', len(rows))
