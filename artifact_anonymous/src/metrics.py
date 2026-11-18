"""CAM report layer + defender metrics.

Defender sees noisy reports and noisy boundary counts only.
Residual r(t) = N_hat_V2X(t) - N_BC(t), where N_BC integrates the
boundary counters from a trusted anchor at t_analysis.
"""
import numpy as np
import torch


def report_layer(sim, cfg, device):
    X = torch.as_tensor(sim["X"], device=device)
    V = torch.as_tensor(sim["V"], device=device)
    AL = torch.as_tensor(sim["AL"], device=device)
    T, N = X.shape
    g = torch.Generator()
    g.manual_seed(cfg["seed"] + 17)
    pen = (torch.rand(T, N, generator=g) < cfg["p_pen"]).to(device)
    recv = (torch.rand(T, N, generator=g) < cfg["p_recv"]).to(device)
    rep = AL & pen & recv
    xn = (torch.randn(T, N, generator=g) * cfg["sig_x"]).to(device)
    vn = (torch.randn(T, N, generator=g) * cfg["sig_v"]).to(device)
    return X, V, AL, rep, X + xn, V + vn


def boundary_counts(sim, cfg):
    """Observed cumulative entries/exits with per-vehicle detection prob q."""
    rng = np.random.default_rng(cfg["seed"] + 101)
    T = len(sim["ain"])
    ain_obs = rng.binomial(sim["ain"], cfg["q_det"])
    aout_obs = rng.binomial(sim["aout"], cfg["q_det"])
    t0 = cfg["t_analysis"]
    n0 = int(sim["AL"][t0].sum())
    anchor = n0 + cfg["anchor_err"] * rng.standard_normal()
    cum_in = np.cumsum(ain_obs)
    cum_out = np.cumsum(aout_obs)
    nbc = anchor + (cum_in - cum_in[t0]) - (cum_out - cum_out[t0])
    return nbc.astype(np.float32)


@torch.no_grad()
def compute(sim, cfg, device="cuda:0", target_seg=0, cells_per_seg=5):
    X, V, AL, rep, xr, vr = report_layer(sim, cfg, device)
    T, N = X.shape
    ell = cfg.get("ell", 100.0)
    dt = cfg["dt"]

    seg_rep = torch.clamp((xr / ell).long(), 0, cfg["M"] - 1) // cells_per_seg
    seg_true = torch.clamp((X / ell).long(), 0, cfg["M"] - 1) // cells_per_seg
    in_seg_rep = rep & (seg_rep == target_seg)
    in_seg_true = AL & (seg_true == target_seg)

    n_v2x = rep.sum(1).float()
    n_rep_seg = in_seg_rep.sum(1).float()
    v_rep_sum = (vr * in_seg_rep).sum(1)
    v_rep = v_rep_sum / n_rep_seg.clamp(min=1)
    n_true_seg = in_seg_true.sum(1).float()
    v_true = (V * in_seg_true).sum(1) / n_true_seg.clamp(min=1)

    # reported median in target segment
    big = torch.full_like(vr, 1e6)
    vals = torch.where(in_seg_rep, vr, big)
    sv, _ = vals.sort(dim=1)
    med_idx = (n_rep_seg.long() // 2).clamp(min=0, max=N - 1)
    v_med = sv.gather(1, med_idx.unsqueeze(1)).squeeze(1)
    v_med = torch.where(n_rep_seg > 0, v_med, torch.zeros_like(v_med))

    # kinematic residual on consecutive reports of the same vehicle
    e = xr[1:] - (xr[:-1] + vr[:-1] * dt)
    m = rep[1:] & rep[:-1]
    e_abs = e[m].abs()
    kin_p99 = torch.quantile(e_abs, 0.99).item() if e_abs.numel() > 100 else np.nan
    kin_p50 = torch.quantile(e_abs, 0.50).item() if e_abs.numel() > 100 else np.nan

    # spatial conflicts: sampled steps, same-lane 1D gap
    conf = 0
    nsamp = 0
    for t in range(cfg["t_analysis"], T, 20):
        xs = xr[t][in_seg_rep[t]]
        if xs.numel() > 1:
            xs, _ = xs.sort()
            gap = xs[1:] - xs[:-1]
            conf += int((gap < cfg["dmin"]).sum().item())
            nsamp += int(gap.numel())

    nb = boundary_counts(sim, cfg)
    p_eff = cfg["p_pen"] * cfg["p_recv"]
    r = (n_v2x / p_eff).cpu().numpy() - nb

    t0 = cfg["t_analysis"]
    out = dict(
        n_v2x=n_v2x.cpu().numpy(),
        nbc=nb,
        resid=r,
        n_true_seg=n_true_seg.cpu().numpy(),
        v_true=v_true.cpu().numpy() * 3.6,
        v_rep=v_rep.cpu().numpy() * 3.6,
        v_med=v_med.cpu().numpy() * 3.6,
        n_rep_seg=n_rep_seg.cpu().numpy(),
        kin_p50=kin_p50, kin_p99=kin_p99,
        conflict_rate=conf / max(nsamp, 1),
        t0=t0,
    )
    return out
