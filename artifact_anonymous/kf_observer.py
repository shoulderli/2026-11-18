"""Model-based observer: Kalman filter on the segment mean speed.

State: x_t = mean speed of the decision segment.
Prior:  x_{t+1} = phi x_t + (1-phi) v_ref + w,  w ~ N(0, q)
Meas.:  z_t = x_t + e,  e ~ N(0, sigma_v^2 / n_t)

The attack perturbs z_t exactly as in the mean decision; the filter attenuates
it by the gain K_t.  We report the share of steps on which the *filtered*
decision flips, and the K at which that share crosses one half.
"""
import sys

import numpy as np

sys.path.insert(0, 'src')
from nextgen_load import load

DATASETS = [
    ('h2_train', 'highway_2', 'Train', (452150, 452300), (5100.0, 5400.0)),
    ('h2_test', 'highway_2', 'Test', (449900, 450000), (13500.0, 13800.0)),
    ('h7_train', 'highway_7', 'Train', (452150, 452300), (25200.0, 25500.0)),
]

PHI, Q, SIG = 0.9, 0.5, 0.3


def step_vals(sim, p, seed=0):
    rng = np.random.default_rng(seed)
    X, V, AL = sim['X'], sim['V'], sim['AL']
    T, N = V.shape
    rep = AL & (rng.random((T, N)) < p * 0.98)
    cell = np.clip((X / 100.0).astype(int), 0, 19)
    seg = rep & (cell >= 0) & (cell <= 4)
    out = []
    for t in range(T):
        v = V[t][seg[t]]
        out.append(v)
    return out


def kf_track(vals, vref, tau):
    x = np.mean([v.mean() for v in vals[:30] if len(v)]) if any(len(v) for v in vals[:30]) else vref
    P = 1.0
    dec = []
    for v in vals:
        x = PHI * x + (1 - PHI) * vref
        P = PHI * PHI * P + Q
        if len(v) == 0:
            dec.append(x)
            continue
        R = SIG * SIG / len(v)
        Kg = P / (P + R)
        x = x + Kg * (v.mean() - x)
        P = (1 - Kg) * P
        dec.append(x)
    return np.array(dec)


def run(sim, K, p, t0, win, tau, vref):
    vals = step_vals(sim, p)
    T = len(vals)
    clean_mean = np.array([v.mean() if len(v) else np.nan for v in vals])
    clean_kf = kf_track(vals, vref, tau)
    res = {}
    for mode in ('B', 'A', 'C'):
        atk = []
        for t, v in enumerate(vals):
            n = len(v)
            if t < t0 or t >= t0 + win or n == 0:
                atk.append(v)
                continue
            if mode == 'B':
                atk.append(np.concatenate([v, np.zeros(K)]))
            elif mode == 'A':
                if n <= K:
                    atk.append(v)
                    continue
                a = v.copy()
                a[np.argsort(v)[-K:]] = 0.0
                atk.append(a)
            else:
                atk.append(np.sort(v)[K:] if n > K else np.array([]))
        atk_mean = np.array([v.mean() if len(v) else np.nan for v in atk])
        atk_kf = kf_track(atk, vref, tau)
        for name, c, a in (('mean', clean_mean, atk_mean),
                           ('kf', clean_kf, atk_kf)):
            m = slice(t0, t0 + win)
            if mode == 'C':
                n_step = np.sum(~np.isnan(c[m]))
                flip = np.nansum((c[m] < tau) & (a[m] >= tau))
            else:
                n_step = np.sum(~np.isnan(c[m]))
                flip = np.nansum((c[m] >= tau) & (a[m] < tau))
            res.setdefault(name, {})[mode] = flip / max(n_step, 1)
    return res


print('%-9s %-4s %-4s %-4s  %-22s %-22s' %
      ('slice', 'K', 'p', 'w', 'mean flip B/A/C', 'KF flip B/A/C'))
for name, tag, split, band, t_win in DATASETS:
    sim = load(split=split, band=band, t_win=t_win, tag=tag)
    vref = float(np.percentile(sim['V_rep'][sim['AL']], 90))
    tau = 0.5 * vref
    for K in (1, 2, 3, 4, 6):
        for p in (1.0, 0.5):
            r = run(sim, K, p, 20, 30, tau, vref)
            m, k = r['mean'], r['kf']
            print('%-9s %-4d %-4.1f %-4d  %.2f/%.2f/%.2f          %.2f/%.2f/%.2f' %
                  (name, K, p, 30, m['B'], m['A'], m['C'], k['B'], k['A'], k['C']))
