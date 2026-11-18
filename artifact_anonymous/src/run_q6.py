import json, sys
sys.path.insert(0, 'src')
from run_q5 import run

if __name__ == '__main__':
    out = []
    for q in (0.995, 0.95, 0.90):
        for K in (2, 3, 4):
            out += [run(0.10, 0.5, K, q_det=q)]
    with open('outputs/q6_counter_accuracy.json', 'w') as f:
        json.dump(out, f, indent=1)
