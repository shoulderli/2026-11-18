"""Slice the NextGen grid: window length, penetration, and the Mode A corner."""
import collections
import json

rows = [json.loads(l) for l in open('outputs/nextgen_grid.jsonl')]


def agg(key, filt):
    g = collections.defaultdict(list)
    for r in rows:
        if filt(r):
            g[key(r)].append(r)
    return g


def fmt(v):
    n = len(v)
    return (n, sum(x['feas'] for x in v),
            sum(x['atk_rate'] for x in v) / n,
            sum(x['clean_rate'] for x in v) / n,
            sum(1 for x in v if x['det']) / n,
            sum(1 for x in v if x['conf'] >= 0) / n)


print('=== Mode B by dataset / K / penetration (all windows) ===')
for k, v in sorted(agg(lambda r: (r['dataset'], r['K'], r['p']),
                       lambda r: r['mode'] == 'B').items()):
    n, f, atk, cl, det, conf = fmt(v)
    if k[1] in (2, 3, 4, 6) and k[2] in (1.0, 0.5):
        print('%-9s K=%d p=%.1f  feas %3d/%3d  atk %.2f  clean %.2f  det %.2f  conf %.2f'
              % (k[0], k[1], k[2], f, n, atk, cl, det, conf))

print('')
print('=== Mode B decision-window effect (p=0.5, K>=3) ===')
for k, v in sorted(agg(lambda r: (r['dataset'], r['K'], r['win']),
                       lambda r: r['mode'] == 'B' and r['p'] == 0.5 and r['K'] >= 3).items()):
    n, f, atk, cl, det, conf = fmt(v)
    print('%-9s K=%d win=%2ds  feas %2d/%2d  atk %.2f' % (k[0], k[1], k[2], f, n, atk))

print('')
print('=== Mode A stealth (conf=-1 and det=False), by dataset / K ===')
for k, v in sorted(agg(lambda r: (r['dataset'], r['K']),
                       lambda r: r['mode'] == 'A').items()):
    n = len(v)
    stealth = sum(1 for x in v if x['conf'] < 0 and not x['det'])
    flips = sum(1 for x in v if x['atk_rate'] >= 0.9)
    print('%-9s K=%d  stealth %3d/%3d  flip(atk>=.9) %3d  feas %d'
          % (k[0], k[1], stealth, n, flips, sum(x['feas'] for x in v)))

print('')
print('=== Mode A feasible cases ===')
for r in rows:
    if r['mode'] == 'A' and r['feas']:
        print('%-9s K=%d win=%2ds t0=%3d p=%.1f sd=%d  atk %.2f clean %.2f'
              % (r['dataset'], r['K'], r['win'], r['t0'], r['p'], r['seed'],
                 r['atk_rate'], r['clean_rate']))

print('')
print('=== Mode C by dataset / K ===')
for k, v in sorted(agg(lambda r: (r['dataset'], r['K']),
                       lambda r: r['mode'] == 'C').items()):
    n, f, atk, cl, det, conf = fmt(v)
    print('%-9s K=%d  feas %2d/%3d  atk %.2f  clean %.2f  det %.2f'
          % (k[0], k[1], f, n, atk, cl, det))
