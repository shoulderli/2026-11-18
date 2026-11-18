"""SUMO grid with the corrected attacker model and segment-local conservation."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, 'src')
from overnight import run_one

OUT = 'outputs/sumo_v2.jsonl'


def load(path):
    d = np.load(path)
    return dict(X=d['X'], V=d['V'], AL=d['AL'], lane=d['lane'],
                ain=d['ain'], aout=d['aout'])


traces = [(0.40, 0.00, 0), (0.40, 0.05, 0), (0.40, 0.05, 1),
          (0.30, 0.05, 0), (0.50, 0.05, 0),
          (0.10, 0.00, 0), (0.15, 0.00, 0), (0.20, 0.00, 0)]
rows = []
for demand, ramp, seed in traces:
    path = 'outputs/sumo_d%03d_r%02d_s%d.npz' % (int(demand * 100), int(ramp * 100), seed)
    if not os.path.exists(path):
        print('missing', path)
        continue
    sim = load(path)
    print('trace', path, 'veh', int(sim['AL'].any(0).sum()), 'T', sim['X'].shape[0])
    for mode in ('B', 'A', 'C'):
        for K in (2, 3, 4, 6, 8):
            cellslist = (1, 3, 5) if mode != 'C' else (3,)
            for cells in cellslist:
                tgt = (0, cells - 1) if mode != 'C' else (9, 11)
                for rs in (0, 1):
                    try:
                        r = run_one(mode, demand, 0.5, K, ramp, cells, tgt, rs,
                                    sim_override=sim, a_falsify=(mode == 'A'))
                    except Exception as e:
                        print('FAIL', mode, K, cells, rs, repr(e))
                        continue
                    r['sumo'] = path
                    r['sumo_seed'] = seed
                    r['report_seed'] = rs
                    rows.append(r)
    print('  rows so far', len(rows))

with open(OUT, 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
print('rows', len(rows))
for m in ('A', 'B', 'C'):
    v = [r for r in rows if r['mode'] == m]
    print(m, 'arith %d' % sum(1 for r in v if r['succ']),
          'feas_seg %d/%d' % (sum(r['feas_seg'] for r in v), len(v)),
          'det_seg %d' % sum(1 for r in v if r['det_seg']),
          'conf %d' % sum(1 for r in v if r['conf'] >= 0))
