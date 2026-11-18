"""Q5: strict feasibility test.

Success = in a 30 s decision window the attacked run is CONGESTED for every
step while the clean run is never congested (including 'no data' steps, which
the honest RSU cannot call congested).  Stealth = conservation increment under
the benign 99% band AND no per-lane spatial conflict in the window.
"""
import json, sys
import numpy as np
import torch
sys.path.insert(0, 'src')
from run_q3 import two_lanes
from metrics import report_layer, boundary_counts

DEV = 'cuda:0'
T0, T1 = 2000, 2150


@torch.no_grad()
def run(demand, p, K, q_det=0.995):
    sim, cfg = two_lanes(demand, p, 0)
    cfg["q_det"] = q_det
    X, V, AL, rep, xr, vr = report_layer(sim, cfg, DEV)
    T, N = X.shape
    lane = torch.tensor(sim['lane'], device=DEV)
    seg = torch.clamp((xr / 100.0).long(), 0, 19) // 5
    m = rep & (seg == 0)
    m0 = m & (lane == 0)
    seg_pos0 = torch.where(m0, xr, torch.full_like(xr, 1e6))
    n_clean = m[T0:T1].sum(1).float()
    vsum = (vr * m)[T0:T1].sum(1)
    v_clean = vsum / n_clean.clamp(min=1)
    clean_cong = (n_clean >= 1) & (v_clean * 3.6 < 40)
    n_atk = n_clean + K
    v_atk = vsum / n_atk
    atk_cong = v_atk * 3.6 < 40
    success = bool(atk_cong.all()) and (not bool(clean_cong.any()))
    p_eff = p * cfg['p_recv']
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
    print(f'demand={demand:4.2f} p={p} K={K} | clean_cong={clean_cong.float().mean():.2f} '
          f'atk_cong={atk_cong.float().mean():.2f} succ={str(success):5} | '
          f'K/p={K/p_eff:5.1f} tau={tau_c:5.1f} cons={"DET" if det_c else "ok "} | '
          f'conf={conf:4d} | FEAS={feas}')
    return dict(demand=demand, p=p, K=K, success=bool(success),
                cons_det=bool(det_c), conflict_t=conf, feasible=bool(feas),
                tau_c=tau_c, clean_rate=float(clean_cong.float().mean()))


if __name__ == '__main__':
    out = []
    for demand in (0.30, 0.10, 0.03):
        for K in (2, 3, 4):
            out += [run(demand, 0.5, K)]
    with open('outputs/q5_feasible.json', 'w') as f:
        json.dump(out, f, indent=1)
