"""Overnight driver: Mode B/A/C x ramp x density x K x segment x seeds.

Ground truth: two independent single-lane IDM streams.  Lane 0 may carry an
unmetered on-ramp source at x_ramp=900 m; the mainline boundary counter at
x=0 does NOT see it, so the defender's mass balance has a blind spot.

Defender: CAM reports + noisy boundary counts at x=0 and x=L.
Attack window: 10 s (50 steps) at t=2000 (400 s).

Success:
  Mode B/A: clean never congested, attacked always congested
  Mode C  : clean always congested, attacked never congested
Stealth: |conservation increment| under benign 99% band; no same-lane
spatial conflict in the window (B/A only).
"""
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

DEV = 'cuda:0'
T0, WIN = 2000, 50
THR_KMH = 40.0


def cfg_lane(**kw):
    c = dict(L=2000.0, dt=0.2, T=6000, v0=27.78, a=1.5, b=2.0, Th=1.2,
             s0=2.0, Lveh=4.5, xb0=1150.0, xb1=1350.0, vlim=8.0,
             demand=0.75, seed=0, Nmax=420, Npool=1100,
             ramp_flow=0.0, x_ramp=900.0)
    c.update(kw)
    return c


def sim_lane(cfg):
    rng = np.random.default_rng(cfg['seed'])
    dt, T, L = cfg['dt'], cfg['T'], cfg['L']
    Nmax, Npool = cfg['Nmax'], cfg['Npool']
    a, b, Th, s0, Lveh = cfg['a'], cfg['b'], cfg['Th'], cfg['s0'], cfg['Lveh']
    v0, vlim = cfg['v0'], cfg['vlim']
    x = np.zeros(Nmax)
    v = np.zeros(Nmax)
    pid = np.full(Nmax, -1, np.int32)
    n = 0
    next_id = 0
    X = np.zeros((T, Npool), np.float32)
    V = np.zeros((T, Npool), np.float32)
    AL = np.zeros((T, Npool), bool)
    ain = np.zeros(T, np.int32)
    aout = np.zeros(T, np.int32)
    aramp = np.zeros(T, np.int32)
    p_ins = cfg['demand'] * dt
    p_ramp = cfg['ramp_flow'] * dt
    xr = cfg['x_ramp']
    for t in range(T):
        if n > 0:
            gap = np.full(n, np.inf)
            gap[1:] = x[:n - 1] - x[1:n] - Lveh
            dv = np.zeros(n)
            dv[1:] = v[1:n] - v[:n - 1]
            zone = (x[:n] >= cfg['xb0']) & (x[:n] <= cfg['xb1'])
            v0l = np.where(zone, vlim, v0)
            ss = s0 + np.maximum(0.0, v[:n] * Th + v[:n] * dv / (2 * np.sqrt(a * b)))
            acc = a * (1.0 - (v[:n] / v0l) ** 4 - (ss / np.maximum(gap, 0.3)) ** 2)
            acc = np.clip(acc, -b, a)
            v[:n] = np.maximum(v[:n] + acc * dt, 0.0)
            x[:n] = x[:n] + v[:n] * dt
        k_out = 0
        while n > 0 and x[0] > L:
            AL[t, pid[0]] = False
            pid[:n - 1] = pid[1:n]
            x[:n - 1] = x[1:n]
            v[:n - 1] = v[1:n]
            n -= 1
            k_out += 1
        aout[t] = k_out
        k_in = 0
        if n < Nmax and next_id < Npool and rng.random() < p_ins:
            v_ins = v[n - 1] if n > 0 else v0
            if n == 0 or (x[n - 1] - 0.0) > (s0 + v_ins * Th):
                pid[n] = next_id
                x[n] = 0.0
                v[n] = v_ins
                n += 1
                next_id += 1
                k_in = 1
        ain[t] = k_in
        kr = 0
        if p_ramp > 0 and n < Nmax and next_id < Npool and rng.random() < p_ramp:
            i = int(np.searchsorted(-x[:n], -xr))
            ga = (x[i - 1] - xr - Lveh) if i > 0 else np.inf
            gb = (xr - x[i] - Lveh) if i < n else np.inf
            v_ins = v[i - 1] if i > 0 else v0
            if ga > (s0 + v_ins * Th) and gb > 2.0:
                x[i + 1:n + 1] = x[i:n]
                v[i + 1:n + 1] = v[i:n]
                pid[i + 1:n + 1] = pid[i:n]
                x[i] = xr
                v[i] = v_ins
                pid[i] = next_id
                n += 1
                next_id += 1
                kr = 1
        aramp[t] = kr
        ids = pid[:n]
        X[t, ids] = x[:n]
        V[t, ids] = v[:n]
        AL[t, ids] = True
    return dict(X=X, V=V, AL=AL, ain=ain, aout=aout, aramp=aramp)


def report_layer(X, V, AL, p_pen, p_recv, sig_x, sig_v, seed):
    Xt = torch.as_tensor(X, device=DEV)
    Vt = torch.as_tensor(V, device=DEV)
    ALt = torch.as_tensor(AL, device=DEV)
    T, N = Xt.shape
    g = torch.Generator()
    g.manual_seed(seed + 17)
    pen = (torch.rand(T, N, generator=g) < p_pen).to(DEV)
    recv = (torch.rand(T, N, generator=g) < p_recv).to(DEV)
    rep = ALt & pen & recv
    xr = Xt + (torch.randn(T, N, generator=g) * sig_x).to(DEV)
    vr = Vt + (torch.randn(T, N, generator=g) * sig_v).to(DEV)
    return Xt, Vt, ALt, rep, xr, vr


def boundary_residual(sim, n_v2x, p_eff, q_det, seed, t_analysis=1500):
    rng = np.random.default_rng(seed + 101)
    ai = rng.binomial(sim['ain'], q_det)
    ao = rng.binomial(sim['aout'], q_det)
    n0 = int(sim['AL'][t_analysis].sum())
    ci = np.cumsum(ai)
    co = np.cumsum(ao)
    nbc = n0 + (ci - ci[t_analysis]) - (co - co[t_analysis])
    r = (n_v2x / p_eff).cpu().numpy() - nbc
    return r


def two_lanes(demand, p, seed, ramp_flow):
    s0 = sim_lane(cfg_lane(seed=seed, demand=demand * 0.5, ramp_flow=ramp_flow))
    s1 = sim_lane(cfg_lane(seed=seed + 1, demand=demand * 0.5, ramp_flow=0.0))
    N0, N1 = s0['X'].shape[1], s1['X'].shape[1]
    X = np.concatenate([s0['X'], s1['X']], 1)
    V = np.concatenate([s0['V'], s1['V']], 1)
    AL = np.concatenate([s0['AL'], s1['AL']], 1)
    lane = np.concatenate([np.zeros(N0, np.int64), np.ones(N1, np.int64)])
    ain = s0['ain'] + s1['ain']
    aout = s0['aout'] + s1['aout']
    aramp = s0['aramp']
    return dict(X=X, V=V, AL=AL, ain=ain, aout=aout, lane=lane, aramp=aramp)


@torch.no_grad()
def run_one(mode, demand, p, K, ramp_flow, cells, tgt, seed,
            q_det=0.995, n_min=1, win=WIN, p_hat_rel=0.0, sim_override=None,
            t0=None, thr_kmh=THR_KMH, a_falsify=False, phantom_spacing=None):
    p_eff_base = p * 0.98 * (1.0 + p_hat_rel)
    t0 = T0 if t0 is None else t0
    sim = sim_override if sim_override is not None else two_lanes(demand, p, seed, ramp_flow)
    Xt, Vt, ALt, rep, xr, vr = report_layer(sim['X'], sim['V'], sim['AL'],
                                            p, 0.98, 1.5, 0.3, seed)
    if 'X_rep' in sim:
        xr = torch.as_tensor(sim['X_rep'], device=DEV)
        vr = torch.as_tensor(sim['V_rep'], device=DEV)
    T, N = Xt.shape
    lane = torch.tensor(sim['lane'], device=DEV)
    cell = torch.clamp((xr / 100.0).long(), 0, 19)
    tgt_m = (cell >= tgt[0]) & (cell <= tgt[1])
    m = rep & tgt_m
    seg_pos0 = torch.where(m & (lane == 0), xr, torch.full_like(xr, 1e6))
    a, b = t0, t0 + win
    n_clean = m[a:b].sum(1).float()
    vsum = (vr * m)[a:b].sum(1)
    v_clean = vsum / n_clean.clamp(min=1)
    clean_cong = (n_clean >= n_min) & (v_clean * 3.6 < thr_kmh)

    chosen = None
    if mode in ('A', 'C'):
        alive0 = ALt[t0] & (cell[t0] >= tgt[0]) & (cell[t0] <= tgt[1])
        idx = torch.nonzero(alive0).squeeze(1)
        if int(idx.numel()) < K:
            K = int(idx.numel())
        if K == 0:
            return dict(mode=mode, demand=demand, p=p, K=0, ramp=ramp_flow,
                        cells=cells, tgt=list(tgt), seed=seed, n_min=n_min,
                        succ=False, clean_rate=0.0, atk_rate=0.0, no_data=1.0,
                        tau=0.0, inc=0.0, det=False, conf=-1, feas=False,
                        n_atk_mean=0.0, n_clean_mean=0.0, v_clean_mean=0.0,
                        atk_fs=1.0, v_mass=0.0, n_mass=0.0, v_seg_med=0.0,
                        tau_clean=0.0, det_clean=False, tau_seg=0.0,
                        det_seg=False, feas_fs=False, feas_seg=False)
        order = torch.argsort(Vt[t0, idx])
        chosen = idx[order[-K:]] if mode == 'A' else idx[order[:K]]

    if mode == 'B':
        n_atk = n_clean + K
        v_atk = vsum / n_atk
        seg_len = (tgt[1] - tgt[0] + 1) * 100.0
        spacing = min(10.0, max(6.0, (seg_len - 10.0) / max(K, 1)))
        if phantom_spacing is not None:
            spacing = float(phantom_spacing)
        center = tgt[0] * 100.0 + seg_len / 2.0
        xp = center + (np.arange(K) - (K - 1) / 2.0) * spacing
        xp_t = torch.tensor(xp, device=DEV, dtype=torch.float32)
        n_v2x = rep.sum(1).float() + K
        inc = K / p_eff_base
    elif mode == 'A':
        ov = torch.zeros(T, N, device=DEV, dtype=torch.bool)
        ov[t0:t0 + win, chosen] = True
        m2 = m & ~ov
        n_atk = m2[a:b].sum(1).float() + K
        v_atk = ((vr * m2)[a:b].sum(1)) / n_atk.clamp(min=1)
        xp_t = Xt[t0, chosen].float()
        if a_falsify:
            # The colluder also falsifies its position: it claims to sit at the
            # stop point, so it is absent from the honest-position cloud.
            seg_pos0 = torch.where(m & (lane == 0) & ~ov,
                                   xr, torch.full_like(xr, 1e6))
        n_v2x = rep.sum(1).float()
        inc = 0.0
    else:
        sil = torch.zeros(T, N, device=DEV, dtype=torch.bool)
        sil[t0:t0 + win, chosen] = True
        rep2 = rep & ~sil
        m2 = rep2 & tgt_m
        n_atk = m2[a:b].sum(1).float()
        v_atk = ((vr * m2)[a:b].sum(1)) / n_atk.clamp(min=1)
        xp_t = None
        n_v2x = rep.sum(1).float() - K
        inc = K / p_eff_base

    atk_cong = (n_atk >= n_min) & (v_atk * 3.6 < thr_kmh)
    clean_rate = float(clean_cong.float().mean())
    atk_rate = float(atk_cong.float().mean())
    no_data = float((n_atk < n_min).float().mean())
    if mode == 'C':
        success = (clean_rate >= 0.5) and (atk_rate <= 0.1)
    else:
        success = (atk_rate >= 0.9) and (clean_rate <= 0.1)

    t_an = min(1500, max(0, T // 4))
    r = boundary_residual(sim, n_v2x, p_eff_base, q_det, seed, t_analysis=t_an)
    tau = float(np.percentile(np.abs(r[t_an:]), 99))
    det = inc > tau
    # The 99th percentile above is computed over a series that CONTAINS the
    # attack, so a sustained injection sets its own threshold.  Recompute the
    # threshold on the benign steps only (attack window excluded).
    keep = np.ones(T, bool)
    keep[a:b] = False
    keep[:t_an] = False
    r_base = r[keep]
    tau_clean = float(np.percentile(np.abs(r_base), 99)) if r_base.size else 0.0
    det_clean = bool(inc > tau_clean)
    # Segment-local conservation: the counter must watch the same segment the
    # decision uses.  Count entries/exits at the segment boundary from truth.
    lo, hi = tgt[0] * 100.0, (tgt[1] + 1) * 100.0
    inseg = ((Xt >= lo) & (Xt < hi) & ALt).cpu().numpy()
    ent = np.zeros((T, N), bool)
    ext = np.zeros((T, N), bool)
    ent[1:] = inseg[1:] & ~inseg[:-1]
    ext[1:] = ~inseg[1:] & inseg[:-1]
    rng2 = np.random.default_rng(seed + 202)
    ec = rng2.binomial(ent.sum(1).astype(np.int64), q_det)
    xc = rng2.binomial(ext.sum(1).astype(np.int64), q_det)
    ci = np.cumsum(ec)
    co = np.cumsum(xc)
    n0s = int(inseg[t_an].sum())
    cnt = n0s + (ci - ci[t_an]) - (co - co[t_an])
    n_atk_full = m.sum(1).float()
    n_atk_full[a:b] = n_atk
    r_seg = n_atk_full.cpu().numpy() / p_eff_base - cnt
    keep2 = np.ones(T, bool)
    keep2[a:b] = False
    keep2[:t_an] = False
    rb2 = r_seg[keep2]
    tau_seg = float(np.percentile(np.abs(rb2), 99)) if rb2.size else 0.0
    det_seg = bool(inc > tau_seg)

    conf = -1
    if xp_t is not None:
        for t in range(a, b):
            real = seg_pos0[t][seg_pos0[t] < 1e5]
            if real.numel() == 0:
                continue
            d = torch.cdist(real.unsqueeze(1), xp_t.unsqueeze(1)).min()
            if float(d) < 4.5:
                conf = t - a
                break
    feas = success and (not det) and (conf < 0)
    # Fail-safe verdict: an outage must never be read as the attacker's
    # preferred state.  For Mode C ("hide the jam") that means counting
    # no-data as congested; for A/B it is a no-op because they add reports.
    ok_steps = n_clean >= n_min
    v_seg_med = float(v_clean[ok_steps].median()) if bool(ok_steps.any()) else 0.0
    atk_fs = atk_rate + no_data
    if mode == 'C':
        succ_fs = (clean_rate >= 0.5) and (atk_fs <= 0.1)
    else:
        succ_fs = success
    feas_fs = succ_fs and (not det) and (conf < 0)
    return dict(mode=mode, demand=demand, p=p, K=int(K), ramp=ramp_flow,
                cells=cells, tgt=list(tgt), seed=seed, n_min=n_min,
                succ=success, clean_rate=clean_rate, atk_rate=atk_rate,
                no_data=no_data,
                tau=tau, inc=inc, det=bool(det), conf=conf, feas=bool(feas),
                n_atk_mean=float(n_atk.mean()), n_clean_mean=float(n_clean.mean()),
                v_clean_mean=float(v_clean.mean()), atk_fs=float(atk_fs),
                v_mass=float(vsum.sum()), n_mass=float(n_clean.sum()),
                v_seg_med=v_seg_med, tau_clean=tau_clean,
                det_clean=det_clean, tau_seg=tau_seg, det_seg=det_seg,
                feas_fs=bool(feas_fs), feas_seg=bool(feas_fs and not det_seg))


def main():
    out_path = 'outputs/overnight.jsonl'
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                d = json.loads(line)
                done.add((d['mode'], d['demand'], d['p'], d['K'], d['ramp'],
                          d['cells'], tuple(d['tgt']), d['seed'], d['n_min']))
            except Exception:
                pass
    jobs = []
    for seed in range(5):
        for mode in ('B', 'A', 'C'):
            for demand in (0.75, 0.30, 0.10):
                for pp in (1.0, 0.5):
                    for K in (2, 4, 6, 8, 10):
                        if mode == 'C':
                            jobs.append((mode, demand, pp, K, 0.0, 3, (12, 13), seed, 1))
                        else:
                            for cells in (1, 5):
                                for ramp in ((0.0, 0.05, 0.10) if mode == 'B' else (0.0,)):
                                    jobs.append((mode, demand, pp, K, ramp, cells,
                                                 (0, cells - 1), seed, 1))
    # n_min sweep on the Mode B / Mode C corners
    for seed in range(5):
        for demand in (0.10, 0.30):
            for K in (2, 3, 4, 6):
                for nm in (1, 2, 3, 4):
                    jobs.append(('B', demand, 0.5, K, 0.0, 1, (0, 0), seed, nm))
                    jobs.append(('C', demand, 0.5, K, 0.0, 3, (12, 13), seed, nm))
    jobs = [j for j in jobs if j not in done]
    log = open('logs/overnight.log', 'a')
    log.write('\n=== start %s | %d jobs ===\n' % (time.strftime('%F %T'), len(jobs)))
    log.flush()
    t_start = time.time()
    for i, j in enumerate(jobs):
        try:
            r = run_one(*j[:8], n_min=j[8])
            with open(out_path, 'a') as f:
                f.write(json.dumps(r) + '\n')
            if i % 20 == 0:
                log.write('%d/%d %.0fs %s\n' % (i, len(jobs), time.time() - t_start, r))
                log.flush()
        except Exception as e:
            log.write('FAIL %s %s\n' % (str(j), repr(e)))
            log.write(traceback.format_exc() + '\n')
            log.flush()
    log.write('=== done %s %.0fs ===\n' % (time.strftime('%F %T'), time.time() - t_start))
    log.close()


if __name__ == '__main__':
    main()
