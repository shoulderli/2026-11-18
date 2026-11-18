"""Q2 sweep: phantom occupancy vs segmentation, penetration, K.

Detección de conservación = deterministic increment K/p_eff vs benign
99th percentile band (the attack is a step, so the delay is 0 or never).
Conflict = first step with a real report within dmin of a phantom slot.
"""
import json, sys
import numpy as np
import torch
sys.path.insert(0, "src")
from traffic import cfg_default, simulate
from metrics import report_layer, boundary_counts

DEV = "cuda:0"
W0, W1 = 2000, 2600


@torch.no_grad()
def run(p_pen, cells, Kmax):
    cfg = cfg_default(M=20, ell=100.0, p_pen=p_pen, p_recv=0.98,
                      sig_x=1.5, sig_v=0.3, q_det=0.995, anchor_err=0.0, dmin=4.5)
    sim = simulate(cfg)
    X, V, AL, rep, xr, vr = report_layer(sim, cfg, DEV)
    T, N = X.shape
    seg = torch.clamp((xr / 100.0).long(), 0, cfg["M"] - 1) // cells
    m = rep & (seg == 0)
    seg_pos = torch.where(m, xr, torch.full_like(xr, 1e6))
    n_rep = m.sum(1).float()
    p_eff = p_pen * cfg["p_recv"]
    t0 = cfg["t_analysis"]
    nb = boundary_counts(sim, cfg)
    n_v2x = rep.sum(1).float()
    rb = (n_v2x / p_eff).cpu().numpy() - nb
    tau_c = float(np.percentile(np.abs(rb[t0:]), 99))
    v_ben = (vr * m).sum(1) / n_rep.clamp(min=1)
    rows = []
    print(f"\n--- p={p_pen} cells/seg={cells} | benign n_seg={n_rep[t0:].mean():.1f} "
          f"v={v_ben[t0:].mean()*3.6:.1f} tau_c={tau_c:.2f}")
    for K in range(2, Kmax + 1, 2):
        xp = 20.0 + np.arange(K) * 10.0
        xp_t = torch.tensor(xp, device=DEV, dtype=torch.float32)
        n_tot = n_rep + K
        v_mean = (vr * m).sum(1) / n_tot.clamp(min=1)
        sb = torch.where(m, vr, torch.full_like(vr, 1e6)).sort(dim=1)[0]
        idx = (n_rep.long() + K - 1) // 2
        med = torch.where(idx < K, torch.zeros_like(v_mean),
                          sb.gather(1, (idx - K).clamp(min=0).unsqueeze(1)).squeeze(1))
        # first conflict inside window
        tc = -1
        for t in range(W0, W1):
            real = seg_pos[t][seg_pos[t] < 1e5]
            if real.numel() == 0:
                continue
            d = torch.cdist(real.unsqueeze(1), xp_t.unsqueeze(1)).min()
            if float(d) < cfg["dmin"]:
                tc = t - W0
                break
        wm = float(v_mean[W0:W1].mean()) * 3.6
        wd = float(med[W0:W1].mean()) * 3.6
        det_c = K / p_eff > tau_c
        feasible = (wm < 40) and (not det_c) and (tc < 0)
        print(f"K={K:2d} mean={wm:5.1f} med={wd:5.1f} flip={wm<40!s:5} "
              f"| K/p={K/p_eff:5.1f} vs tau={tau_c:5.1f} cons={'DET' if det_c else 'ok ':3} "
              f"| conflict_t={tc:4d} | feasible={feasible}")
        rows.append(dict(p=p_pen, cells=cells, K=K, mean=wm, median=wd,
                         flip=bool(wm < 40), k_over_p=K / p_eff, tau_c=tau_c,
                         cons_det=bool(det_c), conflict_t=tc, feasible=bool(feasible)))
    return rows


if __name__ == "__main__":
    out = []
    for p in (1.0, 0.5):
        for cells, kmax in ((5, 16), (3, 12), (1, 8)):
            out += run(p, cells, kmax)
    with open("outputs/q2_sweep.json", "w") as f:
        json.dump(out, f, indent=1)
