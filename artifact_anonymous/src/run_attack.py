"""Q2: phantom occupancy (Mode B) vs aggregators and detectors.

Attack: K stationary phantom vehicles at 10 m spacing inside the target
segment (FD-consistent jam spacing, so the fundamental-diagram check is
evaded by construction).  Reports are added, never remove real ones.
"""
import json, sys
import numpy as np
import torch
sys.path.insert(0, "src")
from traffic import cfg_default, simulate
from metrics import report_layer, boundary_counts

DEV = "cuda:0"
CELLS = 5
TARGET = 0


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


@torch.no_grad()
def main():
    results = []
    for p_pen in (1.0, 0.5):
        cfg = cfg_default(M=20, ell=100.0, p_pen=p_pen, p_recv=0.98,
                          sig_x=1.5, sig_v=0.3, q_det=0.995, anchor_err=0.0,
                          dmin=4.5, cells_per_seg=CELLS, target=TARGET)
        sim = simulate(cfg)
        X, V, AL, rep, xr, vr = report_layer(sim, cfg, DEV)
        T, N = X.shape
        seg = torch.clamp((xr / 100.0).long(), 0, cfg["M"] - 1) // CELLS
        m = rep & (seg == TARGET)
        seg_pos = torch.where(m, xr, torch.full_like(xr, 1e6))
        seg_v = torch.where(m, vr, torch.zeros_like(vr))
        n_rep = m.sum(1).float()
        p_eff = p_pen * cfg["p_recv"]

        t0 = cfg["t_analysis"]
        w0, w1 = 2000, 2600          # attack window
        nb = boundary_counts(sim, cfg)
        n_v2x = rep.sum(1).float()

        # benign thresholds
        rb = (n_v2x / p_eff).cpu().numpy() - nb
        tau_c = float(np.percentile(np.abs(rb[t0:]), 99))
        e = xr[1:] - (xr[:-1] + vr[:-1] * cfg["dt"])
        mm = rep[1:] & rep[:-1]
        tau_k = float(torch.quantile(e[mm].abs(), 0.99)) + 5.0
        print(f"\n=== p_pen={p_pen}  benign: n_seg={n_rep[t0:].mean():.1f} "
              f"tau_c={tau_c:.2f} tau_k={tau_k:.1f}m ===")

        for K in (2, 4, 6, 8, 10, 12, 16):
            xp = 50.0 + np.arange(K) * 10.0          # phantom positions
            xp_t = torch.tensor(xp, device=DEV, dtype=torch.float32)
            # count/speed including phantoms
            n_tot = n_rep + K
            v_sum = (seg_v * m).sum(1)
            v_mean = v_sum / n_tot.clamp(min=1)
            v_med_ben = torch.where(m, vr, torch.full_like(vr, 1e6)).sort(dim=1)[0]
            idx = ((n_rep.long() + K - 1) // 2)
            med = torch.where(idx < K, torch.zeros_like(v_mean, dtype=torch.float32),
                              v_med_ben.gather(1, (idx - K).clamp(min=0).unsqueeze(1)).squeeze(1))
            z = torch.zeros(T, K, device=DEV)
            m_aug = torch.cat([m, torch.ones(T, K, device=DEV, dtype=torch.bool)], 1)
            v_aug = torch.cat([vr, z], 1)
            v_trim = trimmed_mean(v_aug, m_aug, 0.10)
            v_hub = huber_mean(v_aug, m_aug, 1.345)
            # phantoms are zeros in the sample: recompute robust stats on augmented set
            # (approximate by injecting K zeros through the trimmed/huber formulas)
            # conflict: real reports vs phantom slots
            t_c = None
            for t in range(w0, w1):
                xr_t = seg_pos[t]
                real = xr_t[xr_t < 1e5]
                if real.numel() == 0:
                    continue
                d = torch.cdist(real.unsqueeze(1), xp_t.unsqueeze(1)).min()
                if float(d) < cfg["dmin"]:
                    t_c = t
                    break
            # conservation with phantoms
            r_atk = ((n_v2x + K) / p_eff).cpu().numpy() - nb
            det_c = np.abs(r_atk[w0:w1]) > tau_c
            dt_c = int(np.argmax(det_c)) if det_c.any() else -1
            wm = float(v_mean[w0:w1].mean()) * 3.6
            wd = float(med[w0:w1].mean()) * 3.6
            wt = float(v_trim[w0:w1].mean()) * 3.6
            wh = float(v_hub[w0:w1].mean()) * 3.6
            print(f"K={K:2d} | mean={wm:5.1f} flip={wm<40} | med={wd:5.1f} flip={wd<40} | "
                  f"trim={wt:5.1f} | huber={wh:5.1f} | "
                  f"conf_t={'--' if t_c is None else t_c-w0} | "
                  f"cons_det={'yes@'+str(dt_c) if dt_c>=0 else 'no':>8}")
            results.append(dict(p=p_pen, K=K, v_mean=wm, v_med=wd, v_trim=wt,
                                v_huber=wh, flip_mean=bool(wm < 40), flip_med=bool(wd < 40),
                                conflict_step=-1 if t_c is None else t_c - w0,
                                cons_det_step=dt_c, tau_c=tau_c))
    with open("outputs/attack_q2.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
