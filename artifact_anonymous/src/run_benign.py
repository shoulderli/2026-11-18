"""Q1: benign conservation residual vs penetration, loss, boundary accuracy."""
import json, sys, time
import numpy as np
sys.path.insert(0, "src")
from traffic import cfg_default, simulate
from metrics import compute

def one(seed, **kw):
    cfg = cfg_default(seed=seed, M=20, ell=100.0, p_pen=kw["p_pen"],
                      p_recv=kw["p_recv"], sig_x=1.5, sig_v=0.3,
                      q_det=kw["q_det"], anchor_err=kw.get("anchor_err", 0.0),
                      dmin=4.5)
    sim = simulate(cfg)
    out = compute(sim, cfg, device="cuda:0")
    return out, cfg

if __name__ == "__main__":
    import torch
    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
    grid = []
    for q in [1.0, 0.995, 0.99, 0.95]:
        for p in [1.0, 0.5]:
            grid.append(dict(q_det=q, p_pen=p, p_recv=0.98))
    rows = []
    t_start = time.time()
    for kw in grid:
        res = []
        for seed in range(3):
            out, cfg = one(seed, **kw)
            t0 = out["t0"]
            r = out["resid"][t0:]
            res.append(dict(
                mean=float(np.mean(r)), std=float(np.std(r)),
                p99=float(np.percentile(np.abs(r), 99)),
                drift=float((r[-1] - r[len(r)//10]) / (len(r) - len(r)//10)),
                v_true=float(np.mean(out["v_true"][t0:])),
                v_rep=float(np.mean(out["v_rep"][t0:])),
                v_med=float(np.mean(out["v_med"][t0:])),
                n_true=float(np.mean(out["n_true_seg"][t0:])),
                n_rep=float(np.mean(out["n_rep_seg"][t0:])),
                kin99=out["kin_p99"], conf=out["conflict_rate"],
            ))
        agg = {k: float(np.mean([x[k] for x in res])) for k in res[0]}
        agg.update(kw)
        rows.append(agg)
        print(f"q={kw['q_det']:.3f} p={kw['p_pen']:.1f} | "
              f"resid mean={agg['mean']:6.2f} std={agg['std']:5.2f} p99={agg['p99']:6.2f} "
              f"drift={agg['drift']:+.4f}/step | v_true={agg['v_true']:5.1f} "
              f"v_rep={agg['v_rep']:5.1f} v_med={agg['v_med']:5.1f} | "
              f"n_seg={agg['n_true']:4.1f}/{agg['n_rep']:4.1f} | "
              f"kin99={agg['kin99']:.1f}m conf={agg['conf']:.2e}")
    with open("outputs/benign_q1.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("elapsed", round(time.time() - t_start, 1), "s")
