"""Third grid: decision-window length and penetration-rate error."""
import json
import os
import sys
import time

sys.path.insert(0, 'src')
from overnight import run_one

OUT = 'outputs/third.jsonl'


def main():
    jobs = []
    for win in (25, 50, 150):
        for seed in range(3):
            for mode in ('B', 'C'):
                for demand in (0.75, 0.30, 0.10):
                    for K in (2, 3, 4, 6, 8):
                        if mode == 'C':
                            jobs.append((mode, demand, 0.5, K, 0.0, 3, (12, 13),
                                         seed, 1, 0.995, win, 0.0))
                        else:
                            jobs.append((mode, demand, 0.5, K, 0.0, 1, (0, 0),
                                         seed, 1, 0.995, win, 0.0))
    for ph in (-0.2, -0.1, 0.1, 0.2):
        for seed in range(3):
            for mode in ('B', 'C'):
                for demand in (0.30, 0.10):
                    for K in (2, 3, 4, 6):
                        if mode == 'C':
                            jobs.append((mode, demand, 0.5, K, 0.0, 3, (12, 13),
                                         seed, 1, 0.995, 50, ph))
                        else:
                            jobs.append((mode, demand, 0.5, K, 0.0, 1, (0, 0),
                                         seed, 1, 0.995, 50, ph))
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                d = json.loads(line)
                done.add((d['mode'], d['demand'], d['K'], d['seed'], d['win'], d['ph']))
            except Exception:
                pass
    jobs = [j for j in jobs if (j[0], j[1], j[3], j[7], j[10], j[11]) not in done]
    log = open('logs/third.log', 'a')
    log.write('\n=== start %s | %d jobs ===\n' % (time.strftime('%F %T'), len(jobs)))
    log.flush()
    t0 = time.time()
    for i, j in enumerate(jobs):
        try:
            r = run_one(*j[:8], n_min=j[8], q_det=j[9], win=j[10], p_hat_rel=j[11])
            r['win'] = j[10]
            r['ph'] = j[11]
            with open(OUT, 'a') as f:
                f.write(json.dumps(r) + '\n')
            if i % 20 == 0:
                log.write('%d/%d %.0fs mode=%s d=%.2f K=%d win=%d ph=%.1f feas=%s\n'
                          % (i, len(jobs), time.time() - t0, r['mode'], r['demand'],
                             r['K'], r['win'], r['ph'], r['feas']))
                log.flush()
        except Exception as e:
            log.write('FAIL %s %r\n' % (str(j), e))
            log.flush()
    log.write('=== done %s %.0fs ===\n' % (time.strftime('%F %T'), time.time() - t0))
    log.close()


if __name__ == '__main__':
    main()
