"""Is conflict an artifact of how the phantoms are clustered?

Sweep the spacing between injected phantoms from 5 m to 200 m and record the
conflict rate and feasibility.  Spreading the claims across the segment gives
each of them an independent chance to land on a real vehicle, so if the
detector is not an artifact of the 10 m cluster the conflict rate should not
fall as the spacing grows.
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
SPACINGS = (5, 10, 25, 50, 100, 200)
rows = []
for name, tag, split, band, t_win in DATASETS:
    sim = load(split=split, band=band, t_win=t_win, tag=tag)
    if sim['n_senders'] == 0:
        continue
    vref = float(np.percentile(sim['V_rep'][sim['AL']], 90) * 3.6)
    thr = round(0.5 * vref, 1)
    for sp in SPACINGS:
        for K in (2, 3, 4, 6):
            for t0 in (20, 100):
                for p in (1.0, 0.5):
                    for sd in (0, 1):
                        r = run_one('B', 0.02, p, K, 0.0, 5, (0, 4), sd,
                                    sim_override=sim, t0=t0, win=10, n_min=3,
                                    thr_kmh=thr, phantom_spacing=sp)
                        r.update(dataset=name, spacing=sp, K=K, t0=t0, p=p,
                                 thr=thr)
                        rows.append(r)
    print(name, 'rows', len(rows), flush=True)
    with open('outputs/spacing.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
print('total', len(rows))
