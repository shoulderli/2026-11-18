"""Sharded IDM rerun: python run_idm_shard.py <shard> <nshards>."""
import json
import sys
import time
import traceback

sys.path.insert(0, 'src')
from overnight import run_one

shard = int(sys.argv[1])
nsh = int(sys.argv[2])

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

mine = jobs[shard::nsh]
log = open('logs/idm_v2_s%d.log' % shard, 'w')
log.write('shard %d/%d : %d jobs\n' % (shard, nsh, len(mine)))
log.flush()
t0 = time.time()
rows = []
for i, j in enumerate(mine):
    try:
        r = run_one(*j[:8], n_min=j[8], a_falsify=(j[0] == 'A'))
        rows.append(r)
        if i % 25 == 0:
            log.write('%d/%d %.0fs\n' % (i, len(mine), time.time() - t0))
            log.flush()
    except Exception as e:
        log.write('FAIL %s %s\n' % (str(j), repr(e)))
        log.write(traceback.format_exc() + '\n')
        log.flush()
with open('outputs/idm_v2_s%d.jsonl' % shard, 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
log.write('done %d rows %.0fs\n' % (len(rows), time.time() - t0))
log.close()
print('shard', shard, 'rows', len(rows))
