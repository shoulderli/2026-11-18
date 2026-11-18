import json, sys
import numpy as np
sys.path.insert(0, 'src')
from run_q7 import run

if __name__ == '__main__':
    out = []
    for cells in (1, 3, 5):
        for demand in (0.10, 0.30):
            for K in (2, 3):
                for seed in range(5):
                    out.append(run(demand, 0.5, K, win=50, seed=seed, cells=cells))
    json.dump(out, open('outputs/q8_cells.json', 'w'), indent=1)
    for cells in (1, 3, 5):
        for demand in (0.10, 0.30):
            for K in (2, 3):
                rr = [r for r in out if r['cells'] == cells and r['demand'] == demand
                      and r['K'] == K]
                f = sum(r['feas'] for r in rr)
                s = sum(r['succ'] for r in rr)
                a = np.mean([r['atk'] for r in rr])
                nc = np.mean([1 if r['conf'] < 0 else 0 for r in rr])
                print(f'cells={cells} d={demand:.2f} K={K} | feas {f}/5 succ {s}/5 '
                      f'atk_rate={a:.2f} no_conflict={nc:.1f}')
