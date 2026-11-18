"""Validate the NextGen pipeline: Mode B/A/C on real CAM traces."""
import json
import sys

sys.path.insert(0, 'src')
from overnight import run_one
from nextgen_load import load

sim = load(split='Train', band=(452150, 452300), t_win=(5100.0, 5400.0))
print('N', sim['n_senders'], 'L', round(sim['L'], 1),
      'active', [int(sim['AL'][t].sum()) for t in (0, 150, 299)],
      'entries', int(sim['ain'].sum()), 'exits', int(sim['aout'].sum()))

rows = []
for mode in ('B', 'A', 'C'):
    for K in (1, 2, 3):
        for t0 in (50, 100, 150, 200):
            try:
                r = run_one(mode, 0.02, 1.0, K, 0.0, 5, (0, 4), 0,
                            sim_override=sim, t0=t0)
            except Exception as e:
                print('FAIL', mode, K, t0, repr(e))
                continue
            r['nextgen'] = 'h2_train'
            r['t0'] = t0
            rows.append(r)
            print(mode, 'K=%d t0=%d' % (K, t0),
                  'succ', r['succ'], 'atk', round(r['atk_rate'], 2),
                  'clean', round(r['clean_rate'], 2),
                  'conf', r['conf'], 'det', r['det'], 'feas', r['feas'])
json.dump(rows, open('outputs/nextgen_h2.json', 'w'), indent=1)
print('rows', len(rows))
