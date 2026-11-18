"""IDM car-following ground truth (single lane, no overtaking).

A speed-limit zone acts as the bottleneck and produces a backward-moving
queue.  Vehicle identities are stable pool ids; the active array is ordered
most-downstream-first so exits are a left shift and entries are appends.
"""
import numpy as np


def cfg_default(**kw):
    c = dict(
        L=2000.0, dt=0.2, T=6000, v0=27.78,
        a=1.5, b=2.0, Th=1.2, s0=2.0, Lveh=4.5,
        xb0=1150.0, xb1=1350.0, vlim=8.0,
        demand=0.75,
        t_analysis=1500,
        seed=0, Nmax=420, Npool=1000,
    )
    c.update(kw)
    return c


def simulate(cfg):
    rng = np.random.default_rng(cfg["seed"])
    dt, T, L = cfg["dt"], cfg["T"], cfg["L"]
    Nmax, Npool = cfg["Nmax"], cfg["Npool"]
    a, b, Th, s0, Lveh = cfg["a"], cfg["b"], cfg["Th"], cfg["s0"], cfg["Lveh"]
    v0, vlim = cfg["v0"], cfg["vlim"]

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

    p_ins = cfg["demand"] * dt

    for t in range(T):
        if n > 0:
            gap = np.full(n, np.inf)
            gap[1:] = x[: n - 1] - x[1:n] - Lveh
            dv = np.zeros(n)
            dv[1:] = v[1:n] - v[: n - 1]
            in_zone = (x[:n] >= cfg["xb0"]) & (x[:n] <= cfg["xb1"])
            v0loc = np.where(in_zone, vlim, v0)
            sstar = s0 + np.maximum(0.0, v[:n] * Th + v[:n] * dv / (2 * np.sqrt(a * b)))
            acc = a * (1.0 - (v[:n] / v0loc) ** 4 - (sstar / np.maximum(gap, 0.3)) ** 2)
            acc = np.clip(acc, -b, a)
            v[:n] = np.maximum(v[:n] + acc * dt, 0.0)
            x[:n] = x[:n] + v[:n] * dt

        # exits (front = index 0)
        k_out = 0
        while n > 0 and x[0] > L:
            AL[t, pid[0]] = False
            pid[: n - 1] = pid[1:n]
            x[: n - 1] = x[1:n]
            v[: n - 1] = v[1:n]
            n -= 1
            k_out += 1
        aout[t] = k_out

        # entries
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

        ids = pid[:n]
        X[t, ids] = x[:n]
        V[t, ids] = v[:n]
        AL[t, ids] = True

    return dict(X=X, V=V, AL=AL, ain=ain, aout=aout, cfg=cfg, n_entries=next_id)


if __name__ == "__main__":
    cfg = cfg_default()
    sim = simulate(cfg)
    X, V, AL = sim["X"], sim["V"], sim["AL"]
    print("entries:", sim["n_entries"], "exits:", int(sim["aout"].sum()))
    bins = np.arange(0, cfg["L"] + 100.0, 100.0)
    for t in (1400, 3000, 5999):
        alive = AL[t]
        pos, spd = X[t, alive], V[t, alive] * 3.6
        h, _ = np.histogram(pos, bins=bins)
        sc = [spd[(pos >= bins[j]) & (pos < bins[j + 1])].mean()
              if ((pos >= bins[j]) & (pos < bins[j + 1])).any() else np.nan
              for j in range(len(bins) - 1)]
        print(f"t={t} n={alive.sum()}")
        print("  dens:", np.array2string(h, max_line_width=200))
        print("  km/h:", np.array2string(np.round(np.array(sc), 0), max_line_width=200))
