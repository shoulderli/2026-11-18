"""Q3: two lanes x density sweep x phantom warm-up.

Lane 0 = lane the phantom occupies (per-lane spatial conflict check).
Lane 1 = the other lane (still contributes to the segment state estimate).
Phantoms get a kinematic warm-up trajectory so the kinematic check is a
real test rather than an artefact of teleportation.
"""
import json, sys
import numpy as np
import torch
sys.path.insert(0, "src")
from traffic import cfg_default, simulate
from metrics import report_layer, boundary_counts

DEV = "cuda:0"
T0, T1 = 2000, 2600
WARM = 250          # 50 s approach at ~20 m/s
V_WARM = 20.0       # m/s


def trimmed_mean(vals, mask, alpha):
    v = torch.where(mask, vals, torch.full_like(vals, 1e6))
    v, _ = v.sort(dim=1)
    n = mask.sum(1)
    k = (alpha * n).long()
    idx = torch.arange(v.shape[1], device=v.device).unsqueeze(0)
    keep = (idx >= k.unsqueeze(1)) & (idx < (n - k).unsqueeze(1))
    return (v * keep).sum(1) / keep.sum(1).clamp(min=1)


def huber_mean(vals, mask, c=1.345):
    n = mask.sum(1).clamp(min=1)
    m = (vals * mask).sum(1) / n
    sd = torch.sqrt(((vals - m.unsqueeze(1)) ** 2 * mask).sum(1) / n).clamp(min=0.1)
    r = (vals - m.unsqueeze(1)) / sd.unsqueeze(1)
    w = torch.where(r.abs() <= c, 1.0, c / r.abs().clamp(min=1e-6))
    return (vals * w * mask).sum(1) / (w * mask).sum(1).clamp(min=1)


def two_lanes(demand, p_pen, seed, share0=0.5):
    s0 = simulate(cfg_default(seed=seed, demand=demand * share0, Npool=1200))
    s1 = simulate(cfg_default(seed=seed + 1, demand=demand * (1 - share0), Npool=1200))
    N0, N1 = s0["X"].shape[1], s1["X"].shape[1]
    X = np.concatenate([s0["X"], s1["X"]], 1)
    V = np.concatenate([s0["V"], s1["V"]], 1)
    AL = np.concatenate([s0["AL"], s1["AL"]], 1)
    lane = np.concatenate([np.zeros(N0, np.int64), np.ones(N1, np.int64)])
    ain = s0["ain"] + s1["ain"]
    aout = s0["aout"] + s1["aout"]
    cfg = cfg_default(seed=seed, demand=demand, p_pen=p_pen, p_recv=0.98,
                      sig_x=1.5, sig_v=0.3, q_det=0.995, anchor_err=0.0, dmin=4.5,
                      Npool=N0 + N1)
    return dict(X=X, V=V, AL=AL, ain=ain, aout=aout, lane=lane), cfg


@torch.no_grad()
def run(demand, p_pen, seed=0, share0=0.5):
    sim, cfg = two_lanes(demand, p_pen, seed, share0=share0)
    X, V, AL, rep, xr, vr = report_layer(sim, cfg, DEV)
    T, N = X.shape
    lane = torch.tensor(sim["lane"], device=DEV)
    seg = torch.clamp((xr / 100.0).long(), 0, 19) // 5
    m = rep & (seg == 0)
    m0 = m & (lane == 0)
    seg_pos0 = torch.where(m0, xr, torch.full_like(xr, 1e6))
    n_rep = m.sum(1).float()
    p_eff = p_pen * cfg["p_recv"]
    nb = boundary_counts(sim, cfg)
    n_v2x = rep.sum(1).float()
    rb = (n_v2x / p_eff).cpu().numpy() - nb
    tau_c = float(np.percentile(np.abs(rb[cfg["t_analysis"]:]), 99))
    tau_k = 10.5
    rows = []
    print(f"\n--- demand={demand} veh/s p={p_pen} seed={seed} | "
          f"n_seg={n_rep[cfg['t_analysis']:].mean():.1f} tau_c={tau_c:.2f}")
    for K in (2, 3, 4, 6, 8, 10):
        xp = 20.0 + np.arange(K) * 10.0
        xp_t = torch.tensor(xp, device=DEV, dtype=torch.float32)
        # warm-up trajectory (before T0) + stationary phantom (after)
        tt = torch.arange(T, device=DEV, dtype=torch.float32).unsqueeze(1)
        lead = torch.clamp((T0 - tt) * cfg["dt"] * V_WARM, min=0.0)
        xp_path = torch.clamp(xp_t.unsqueeze(0) - lead, min=0.0)   # [T,K]
        v_path = torch.where(tt < T0, torch.full_like(xp_path, V_WARM),
                             torch.zeros_like(xp_path))
        # segment stats with phantoms (only inside window)
        in_w = (torch.arange(T, device=DEV) >= T0) & (torch.arange(T, device=DEV) < T1)
        add = (K * in_w.float()).unsqueeze(1)
        n_tot = n_rep.unsqueeze(1) + add
        v_mean = (vr * m).sum(1, keepdim=True) / n_tot.clamp(min=1)
        z = torch.zeros(T, K, device=DEV)
        m_aug = torch.cat([m, torch.zeros(T, K, device=DEV, dtype=torch.bool)], 1)
        v_aug = torch.cat([vr, z], 1)
        # only turn phantoms on inside the window
        m_aug = torch.cat([m, in_w.unsqueeze(1).expand(T, K)], 1)
        v_mean = (v_aug * m_aug).sum(1) / m_aug.sum(1).clamp(min=1)
        sb = torch.where(m_aug, v_aug, torch.full_like(v_aug, 1e6)).sort(dim=1)[0]
        idx = ((m_aug.sum(1).long() - 1) // 2).clamp(min=0, max=sb.shape[1]-1)
        med = sb.gather(1, idx.unsqueeze(1)).squeeze(1)
        trim = trimmed_mean(v_aug, m_aug, 0.10)
        hub = huber_mean(v_aug, m_aug, 1.345)
        # per-lane spatial conflict (lane 0 only)
        tc = -1
        for t in range(T0, T1):
            real = seg_pos0[t][seg_pos0[t] < 1e5]
            if real.numel() == 0:
                continue
            d = torch.cdist(real.unsqueeze(1), xp_t.unsqueeze(1)).min()
            if float(d) < cfg["dmin"]:
                tc = t - T0
                break
        wm = float(v_mean[T0:T1].mean()) * 3.6
        wd = float(med[T0:T1].mean()) * 3.6
        wt = float(trim[T0:T1].mean()) * 3.6
        wh = float(hub[T0:T1].mean()) * 3.6
        det_c = K / p_eff > tau_c
        feasible = (wm < 40) and (not det_c) and (tc < 0)
        print(f"K={K:2d} mean={wm:5.1f} med={wd:5.1f} trim={wt:5.1f} hub={wh:5.1f} "
              f"flip={str(wm<40):5} | K/p={K/p_eff:5.1f} tau={tau_c:5.1f} "
              f"cons={'DET' if det_c else 'ok ':3} | conf={tc:4d} | feas={feasible}")
        rows.append(dict(demand=demand, p=p_pen, K=K, mean=wm, median=wd,
                         trim=wt, huber=wh, flip=bool(wm < 40), tau_c=tau_c,
                         cons_det=bool(det_c), conflict_t=tc, feasible=bool(feasible)))
    return rows


if __name__ == "__main__":
    out = []
    for demand in (0.75, 0.30, 0.10, 0.03):
        for p in (1.0, 0.5):
            out += run(demand, p)
    with open("outputs/q3_lanes_density.json", "w") as f:
        json.dump(out, f, indent=1)
