import json, sys
import numpy as np
import torch
sys.path.insert(0, 'src')
from run_q3 import two_lanes
from metrics import report_layer, boundary_counts

DEV = 'cuda:0'

@torch.no_grad()
def run(demand, p, K, win=150, seed=0, p_hat_rel=0.0, q_det=0.995, cells=5):
    sim, cfg = two_lanes(demand, p, seed)
    cfg['q_det'] = q_det
    X, V, AL, rep, xr, vr = report_layer(sim, cfg, DEV)
    T, N = X.shape
    T0, T1 = 2000, 2000 + win
    lane = torch.tensor(sim['lane'], device=DEV)
    seg = torch.clamp((xr / 100.0).long(), 0, 19) // cells
    m = rep & (seg == 0)
    seg_pos0 = torch.where(m & (lane == 0), xr, torch.full_like(xr, 1e6))
    n_clean = m[T0:T1].sum(1).float()
    vsum = (vr * m)[T0:T1].sum(1)
    v_clean = vsum / n_clean.clamp(min=1)
    clean_cong = (n_clean >= 1) & (v_clean * 3.6 < 40)
    v_atk = vsum / (n_clean + K)
    atk_cong = v_atk * 3.6 < 40
    success = bool(atk_cong.all()) and (not bool(clean_cong.any()))
    p_eff = p * cfg['p_recv'] * (1 + p_hat_rel)
    nb = boundary_counts(sim, cfg)
    n_v2x = rep.sum(1).float()
    rb = (n_v2x / p_eff).cpu().numpy() - nb
    tau_c = float(np.percentile(np.abs(rb[cfg['t_analysis']:]), 99))
    det_c = K / p_eff > tau_c
    xp = 20.0 + np.arange(K) * 10.0
    xp_t = torch.tensor(xp, device=DEV, dtype=torch.float32)
    conf = -1
    for t in range(T0, T1):
        real = seg_pos0[t][seg_pos0[t] < 1e5]
        if real.numel() == 0:
            continue
        d = torch.cdist(real.unsqueeze(1), xp_t.unsqueeze(1)).min()
        if float(d) < cfg['dmin']:
            conf = t - T0
            break
    feas = success and (not det_c) and (conf < 0)
    return dict(demand=demand, K=K, win=win, seed=seed, ph=p_hat_rel, cells=cells,
                succ=success, atk=float(atk_cong.float().mean()),
                tau=tau_c, inc=K / p_eff, det=bool(det_c),
                conf=conf, feas=bool(feas))

if __name__ == '__main__':
    out = []
    for seed in range(5):
        for demand, K in ((0.03, 2), (0.10, 2), (0.10, 3), (0.30, 3)):
            for win in (50, 150):
                out.append(run(demand, 0.5, K, win=win, seed=seed))
    for ph in (-0.2, -0.1, 0.1, 0.2):
        for demand in (0.10, 0.30):
            for K in (2, 3, 4):
                out.append(run(demand, 0.5, K, win=50, p_hat_rel=ph))
    json.dump(out, open('outputs/q7_seed_phat.json', 'w'), indent=1)
    print('--- multi-seed (5 seeds), feasible count ---')
    for demand, K in ((0.03, 2), (0.10, 2), (0.10, 3), (0.30, 3)):
        for win in (50, 150):
            rr = [r for r in out if r['demand'] == demand and r['K'] == K
                  and r['win'] == win and r['ph'] == 0.0]
            f = sum(r['feas'] for r in rr)
            a = np.mean([r['atk'] for r in rr])
            print(f'd={demand:4.2f} K={K} win={win:3d} | feas {f}/{len(rr)} | '
                  f'atk_rate={a:.2f} | succ={sum(r["succ"] for r in rr)}')
    print('--- p_hat relative error, 10 s window ---')
    for ph in (-0.2, -0.1, 0.1, 0.2):
        for demand in (0.10, 0.30):
            for K in (2, 3, 4):
                rr = [r for r in out if r['ph'] == ph and r['demand'] == demand
                      and r['K'] == K][0]
                print(f'd={demand:4.2f} K={K} ph={ph:+.1f} | inc={rr["inc"]:5.1f} '
                      f'tau={rr["tau"]:5.1f} det={rr["det"]!s:5} succ={rr["succ"]!s:5} '
                      f'conf={rr["conf"]:4d} FEAS={rr["feas"]}')
