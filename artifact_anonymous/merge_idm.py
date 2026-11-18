"""Merge the IDM shards and produce the four-layer attribution table."""
import glob
import json

import numpy as np

rows = []
for f in sorted(glob.glob('outputs/idm_v2_s*.jsonl')):
    rows += [json.loads(l) for l in open(f)]
print('IDM rows', len(rows))

TH = {'A': lambda r: (r - 1) / max(r, 1e-3),
      'B': lambda r: r - 1,
      'C': lambda r: 1 - r}


def arith(x):
    if x['mode'] == 'C':
        return (x['clean_rate'] >= 0.5) and (x['atk_fs'] <= 0.1)
    return bool(x['succ'])


print('\nIDM attribution among arithmetic-passing configs')
print('%-3s %6s %6s %6s %6s' % ('md', 'arith', 'cons', 'conf', 'feas'))
for m in ('A', 'B', 'C'):
    r = [x for x in rows if x['mode'] == m and arith(x)]
    f = [x for x in r if x['feas_seg']]
    c = [x for x in r if not x['feas_seg'] and x['conf'] >= 0]
    k = [x for x in r if not x['feas_seg'] and x['conf'] < 0 and x['det_seg']]
    print('%-3s %6d %6d %6d %6d' % (m, len(r), len(k), len(c), len(f)))

print('\nIDM hard floor (feasible points under theta)')
for m in ('A', 'B', 'C'):
    pos = []
    for x in rows:
        if x['mode'] != m or not x['feas_seg']:
            continue
        nm = x.get('n_mass', 0.0)
        if nm <= 0:
            continue
        rr = (x.get('v_mass', 0.0) / nm) * 3.6 / x.get('thr', 40.0)
        ff = x['K'] / x['n_clean_mean'] if x['n_clean_mean'] > 0 else np.nan
        if np.isfinite(ff):
            pos.append((rr, ff))
    below = [p for p in pos if p[1] <= TH[m](p[0])]
    print(' %s feasible %3d  below %2d  r %.2f-%.2f' %
          (m, len(pos), len(below),
           min([p[0] for p in pos], default=0), max([p[0] for p in pos], default=0)))

with open('outputs/idm_v2.jsonl', 'w') as fh:
    for x in rows:
        fh.write(json.dumps(x) + '\n')
print('\nwrote outputs/idm_v2.jsonl')
