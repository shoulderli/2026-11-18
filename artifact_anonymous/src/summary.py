"""Morning report from overnight.jsonl / extra.jsonl / third.jsonl."""
import collections
import json
import os

PATHS = ['outputs/overnight.jsonl', 'outputs/extra.jsonl', 'outputs/third.jsonl']
rows = []
for p in PATHS:
    if not os.path.exists(p):
        continue
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        r['q'] = r.get('q_det', 0.995)
        r['w'] = r.get('win', 50)
        r['ph'] = r.get('ph', 0.0)
        r['nm'] = r.get('n_min', 1)
        r['cells'] = r['cells']
        rows.append(r)

L = ['# Overnight report', '', 'rows: %d' % len(rows), '']


def grp(keyf, filt=None):
    g = collections.defaultdict(list)
    for r in rows:
        if filt and not filt(r):
            continue
        g[keyf(r)].append(r)
    return g


def stat(v):
    return (len(v), sum(x['feas'] for x in v), sum(x['succ'] for x in v),
            sum(x['atk_rate'] for x in v) / len(v),
            sum(x['clean_rate'] for x in v) / len(v),
            sum(x['tau'] for x in v) / len(v),
            sum(x['inc'] for x in v) / len(v),
            sum(1 for x in v if x['conf'] < 0) / len(v))


# 1. feasible configurations
L += ['## Feasible configurations (feas > 0)', '',
      '| mode | demand | p | K | ramp | cells | n_min | q_det | win | ph | feas/n | atk | clean |',
      '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
g = grp(lambda r: (r['mode'], r['demand'], r['p'], r['K'], r['ramp'],
                   r['cells'], r['nm'], r['q'], r['w'], r['ph']))
cnt = 0
for k, v in sorted(g.items(), key=lambda x: (-sum(y['feas'] for y in x[1]), str(x[0]))):
    n, f, s, atk, cl, tau, inc, nc = stat(v)
    if f == 0:
        continue
    L.append('| %s | %.2f | %.1f | %d | %.2f | %d | %d | %.3f | %d | %+.1f | %d/%d | %.2f | %.2f |'
             % (k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8], k[9], f, n, atk, cl))
    cnt += 1
    if cnt > 80:
        L.append('| ... | | | | | | | | | | | | |')
        break
L.append('')

# 2. feasibility pivot, base config (q=.995, w=50, ph=0, nm=1, ramp=0)
L += ['## Feasibility by mode / demand / K (base: ramp=0, q=.995, win=50)',
      '', '| mode | demand | K | cells=1 | cells=3 | cells=5 |', '|---|---|---|---|---|---|']
for mode in ('B', 'A', 'C'):
    for demand in (0.75, 0.30, 0.10):
        for K in (1, 2, 3, 4, 6, 8, 10):
            cells_row = []
            any_ = False
            for c in (1, 3, 5):
                v = [r for r in rows if r['mode'] == mode and abs(r['demand'] - demand) < 1e-9
                     and r['K'] == K and r['cells'] == c and r['ramp'] == 0.0
                     and r['q'] == 0.995 and r['w'] == 50 and r['ph'] == 0.0
                     and r['nm'] == 1]
                if v:
                    any_ = True
                    cells_row.append('%d/%d' % (sum(x['feas'] for x in v), len(v)))
                else:
                    cells_row.append('-')
            if any_:
                L.append('| %s | %.2f | %d | %s | %s | %s |' % (mode, demand, K, *cells_row))
L.append('')

# 3. n_min
L += ['## n_min defense', '', '| mode | demand | K | n_min | feas/n | atk |', '|---|---|---|---|---|---|']
for k, v in sorted(grp(lambda r: (r['mode'], r['demand'], r['K'], r['nm']),
                       lambda r: r['nm'] > 1).items()):
    n, f, s, atk, cl, tau, inc, nc = stat(v)
    L.append('| %s | %.2f | %d | %d | %d/%d | %.2f |' % (k[0], k[1], k[2], k[3], f, n, atk))
L.append('')

# 4. q_det
L += ['## boundary counter accuracy', '', '| mode | demand | K | q_det | feas/n | tau |', '|---|---|---|---|---|---|']
for k, v in sorted(grp(lambda r: (r['mode'], r['demand'], r['K'], r['q']),
                       lambda r: r['q'] != 0.995).items()):
    n, f, s, atk, cl, tau, inc, nc = stat(v)
    L.append('| %s | %.2f | %d | %.3f | %d/%d | %.1f |' % (k[0], k[1], k[2], k[3], f, n, tau))
L.append('')

# 5. window
L += ['## decision window', '', '| mode | demand | K | win(steps) | feas/n | atk |', '|---|---|---|---|---|---|']
for k, v in sorted(grp(lambda r: (r['mode'], r['demand'], r['K'], r['w'])).items()):
    n, f, s, atk, cl, tau, inc, nc = stat(v)
    L.append('| %s | %.2f | %d | %d | %d/%d | %.2f |' % (k[0], k[1], k[2], k[3], f, n, atk))
L.append('')

# 6. p_hat error
L += ['## penetration-rate error', '', '| mode | demand | K | p_hat err | feas/n | tau |', '|---|---|---|---|---|---|']
for k, v in sorted(grp(lambda r: (r['mode'], r['demand'], r['K'], r['ph']),
                       lambda r: r['ph'] != 0.0).items()):
    n, f, s, atk, cl, tau, inc, nc = stat(v)
    L.append('| %s | %.2f | %d | %+.1f | %d/%d | %.1f |' % (k[0], k[1], k[2], k[3], f, n, tau))
L.append('')

# 7. ramp
L += ['## unmetered ramp (Mode B)', '', '| demand | K | ramp | feas/n | tau | inc | atk |', '|---|---|---|---|---|---|---|']
for k, v in sorted(grp(lambda r: (r['demand'], r['K'], r['ramp']),
                       lambda r: r['mode'] == 'B' and r['ramp'] > 0).items()):
    n, f, s, atk, cl, tau, inc, nc = stat(v)
    L.append('| %.2f | %d | %.2f | %d/%d | %.1f | %.1f | %.2f |'
             % (k[0], k[1], k[2], f, n, tau, inc, atk))
L.append('')

open('outputs/summary.md', 'w').write('\n'.join(L))
print('wrote outputs/summary.md, %d lines' % len(L))
