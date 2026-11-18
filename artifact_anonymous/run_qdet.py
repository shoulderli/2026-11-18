"""How accurate must the passage counter be?

Sweep q_det from 0.999 down to 0.2 over many configurations and record the
detection rate among arithmetic-passing injections.  This turns the single
probe of the robustness paragraph into an operating curve.
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
]
QDET = (0.999, 0.99, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2)
rows = []
for name, tag, split, band, t_win in DATASETS:
    sim = load(split=split, band=band, t_win=t_win, tag=tag)
    if sim['n_senders'] == 0:
        continue
    vref = float(np.percentile(sim['V_rep'][sim['AL']], 90) * 3.6)
    thr = round(0.5 * vref, 1)
    for q in QDET:
        for K in (2, 3, 4):
            for p in (1.0, 0.5):
                for sd in (0, 1):
                    r = run_one('B', 0.02, p, K, 0.0, 5, (0, 4), sd,
                                sim_override=sim, t0=20, win=10, n_min=3,
                                thr_kmh=thr, q_det=q)
                    r.update(dataset=name, q_det=q, K=K, p=p, thr=thr)
                    rows.append(r)
    print(name, 'rows', len(rows), flush=True)
with open('outputs/qdet.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
print('total', len(rows))
