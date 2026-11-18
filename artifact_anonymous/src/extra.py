"""Extra overnight grid: more seeds, finer K, cells 1-5, q_det sweep."""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, 'src')
from overnight import run_one

OUT = 'outputs/extra.jsonl'


def main():
    jobs = []
    for seed in range(5, 15):
        for mode in ('B', 'C'):
            for demand in (0.75, 0.30, 0.10):
                for K in (1, 2, 3, 4, 5, 6, 8, 10):
                    if mode == 'C':
                        jobs.append((mode, demand, 0.5, K, 0.0, 3, (12, 13), seed, 1, 0.995))
                    else:
                        for cells in (1, 2, 3, 5):
                            jobs.append((mode, demand, 0.5, K, 0.0, cells,
                                         (0, cells - 1), seed, 1, 0.995))
    for q in (0.90, 0.95, 0.99):
        for mode in ('B', 'C'):
            for demand in (0.30, 0.10):
                for K in (2, 3, 4, 6):
                    if mode == 'C':
                        jobs.append((mode, demand, 0.5, K, 0.0, 3, (12, 13), 0, 1, q))
                    else:
                        jobs.append((mode, demand, 0.5, K, 0.0, 1, (0, 0), 0, 1, q))
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                d = json.loads(line)
                done.add((d['mode'], d['demand'], d['K'], d['cells'],
                          tuple(d['tgt']), d['seed'], d.get('q_det', 0.995)))
            except Exception:
                pass
    jobs = [j for j in jobs if (j[0], j[1], j[3], j[5], tuple(j[6]), j[7], j[9]) not in done]
    log = open('logs/extra.log', 'a')
    log.write('\n=== start %s | %d jobs ===\n' % (time.strftime('%F %T'), len(jobs)))
    log.flush()
    t0 = time.time()
    for i, j in enumerate(jobs):
        try:
            r = run_one(*j[:8], q_det=j[9], n_min=j[8])
            r['q_det'] = j[9]
            with open(OUT, 'a') as f:
                f.write(json.dumps(r) + '\n')
            if i % 20 == 0:
                log.write('%d/%d %.0fs %s\n' % (i, len(jobs), time.time() - t0,
                                                {k: r[k] for k in ('mode', 'demand', 'K', 'feas')}))
                log.flush()
        except Exception as e:
            log.write('FAIL %s %r\n' % (str(j), e))
            log.flush()
    log.write('=== done %s %.0fs ===\n' % (time.strftime('%F %T'), time.time() - t0))
    log.close()


if __name__ == '__main__':
    main()
