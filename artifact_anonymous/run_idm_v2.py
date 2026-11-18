"""Rerun the IDM grid with the corrected attacker model and verdicts.

Mode A now falsifies its own position (a_falsify=True) and Mode C is judged
fail-safe inside run_one (feas_fs).  Same job list as overnight.main().
"""
import json
import sys
import time
import traceback

sys.path.insert(0, 'src')
from overnight import run_one

out_path = 'outputs/overnight_v2.jsonl'
jobs = []
for seed in range(5):
    for mode in ('B', 'A', 'C'):
        for demand in (0.75, 0.30, 0.10):
            for pp in (1.0, 0.5):
                for K in (2, 4, 6, 8, 10):
                    if mode == 'C':
                        jobs.append((mode, demand, pp, K, 0.0, 3, (12, 13), seed, 1))
                    else:
                        for cells in (1, 5):
                            for ramp in ((0.0, 0.05, 0.10) if mode == 'B' else (0.0,)):
                                jobs.append((mode, demand, pp, K, ramp, cells,
                                             (0, cells - 1), seed, 1))
for seed in range(5):
    for demand in (0.10, 0.30):
        for K in (2, 3, 4, 6):
            for nm in (1, 2, 3, 4):
                jobs.append(('B', demand, 0.5, K, 0.0, 1, (0, 0), seed, nm))
                jobs.append(('C', demand, 0.5, K, 0.0, 3, (12, 13), seed, nm))

log = open('logs/idm_v2.log', 'a')
log.write('\n=== start %s | %d jobs ===\n' % (time.strftime('%F %T'), len(jobs)))
log.flush()
t0 = time.time()
rows = []
for i, j in enumerate(jobs):
    try:
        r = run_one(*j[:8], n_min=j[8], a_falsify=(j[0] == 'A'))
        rows.append(r)
        if i % 50 == 0:
            log.write('%d/%d %.0fs %s\n' % (i, len(jobs), time.time() - t0,
                                            {k: r[k] for k in ('mode', 'feas', 'feas_fs')}))
            log.flush()
    except Exception as e:
        log.write('FAIL %s %s\n' % (str(j), repr(e)))
        log.write(traceback.format_exc() + '\n')
        log.flush()
with open(out_path, 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
log.write('=== done %s %.0fs rows %d ===\n' % (time.strftime('%F %T'),
                                               time.time() - t0, len(rows)))
log.close()
print('rows', len(rows))
