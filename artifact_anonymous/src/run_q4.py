import json, sys
sys.path.insert(0, 'src')
from run_q3 import run

if __name__ == '__main__':
    out = []
    for demand in (0.30, 0.10):
        for share0 in (0.5, 0.2, 0.05):
            out += run(demand, 0.5, 0, share0=share0)
    with open('outputs/q4_lane_share.json', 'w') as f:
        json.dump(out, f, indent=1)
