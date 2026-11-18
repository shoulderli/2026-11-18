"""Load VeReMi NextGen highway_2 into the sim format used by overnight.py.

- dedupe (sender_id, sendTime) across receiver logs
- select a spatial band (one road) and a time window
- reported values = sender.pos/spd (real V2X noise)
- truth = reported - pos_noise/spd_noise
- boundary crossings from the truth at x=0 and x=L
"""
import io
import json
import os
import zipfile

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def load(split='Train', band=(452150, 452300), t_win=(5100.0, 5400.0),
         dt=1.0, gap_max=2.5, tag='highway_2'):
    zp = os.path.join(DATA, 'InTAS_%s.zip' % tag)
    z = zipfile.ZipFile(zp)
    inner = z.read('InTAS_%s/%s/InTAS_%s.zip' % (tag, split, tag))
    zi = zipfile.ZipFile(io.BytesIO(inner))
    rows = {}
    for n in zi.namelist():
        if not n.endswith('.json'):
            continue
        for m in json.loads(zi.read(n)):
            k = (m['sender_id'], m['sendTime'])
            if k in rows:
                continue
            p = m['sender']['pos'].split(',')
            pn = m['sender']['pos_noise'].split(',')
            y = float(p[1]) - float(pn[1])
            if not (band[0] <= y <= band[1]):
                continue
            rows[k] = m
    senders = sorted(set(k[0] for k in rows))
    sidx = {s: i for i, s in enumerate(senders)}
    N = len(senders)
    T = int(round((t_win[1] - t_win[0]) / dt))
    bins = t_win[0] + np.arange(T) * dt
    per = {s: [] for s in senders}
    for (s, ts), m in rows.items():
        t = int(ts) / 1e9
        if t_win[0] - gap_max <= t < t_win[1]:
            per[s].append((t, m))
    Xr = np.full((T, N), np.nan, np.float32)
    Xt = np.full((T, N), np.nan, np.float32)
    Vr = np.full((T, N), np.nan, np.float32)
    Vt = np.full((T, N), np.nan, np.float32)
    for s in senders:
        arr = sorted(per[s], key=lambda x: x[0])
        if not arr:
            continue
        ts = np.array([a[0] for a in arr])
        xs = np.array([float(a[1]['sender']['pos'].split(',')[0]) for a in arr])
        xn = np.array([float(a[1]['sender']['pos_noise'].split(',')[0]) for a in arr])
        vs = np.array([float(a[1]['sender']['spd']) for a in arr])
        vn = np.array([float(a[1]['sender']['spd_noise']) for a in arr])
        idx = np.searchsorted(ts, bins, side='right') - 1
        ok = idx >= 0
        gap = np.where(ok, bins - ts[np.maximum(idx, 0)], 1e9)
        ok &= gap <= gap_max
        j = sidx[s]
        if ok.any():
            Xr[ok, j] = xs[idx[ok]]
            Xt[ok, j] = xs[idx[ok]] - xn[idx[ok]]
            Vr[ok, j] = vs[idx[ok]]
            Vt[ok, j] = vs[idx[ok]] - vn[idx[ok]]
    xmin = float(np.nanmin(Xt))
    xmax = float(np.nanmax(Xt))
    Xr = np.nan_to_num(Xr - xmin)
    Xt = np.nan_to_num(Xt - xmin)
    L = xmax - xmin
    AL = ~np.isnan(Vt)
    Vr = np.nan_to_num(Vr)
    Vt = np.nan_to_num(Vt)
    ain = np.zeros(T, np.int32)
    aout = np.zeros(T, np.int32)
    for j in range(N):
        x = Xt[:, j]
        a = AL[:, j]
        for t in range(1, T):
            if a[t] and a[t - 1]:
                if x[t - 1] < 0.0 <= x[t]:
                    ain[t] += 1
                if x[t - 1] < L <= x[t]:
                    aout[t] += 1
    return dict(X=Xt, V=Vt, X_rep=Xr, V_rep=Vr, AL=AL,
                lane=np.zeros(N, np.int64), ain=ain, aout=aout,
                L=L, n_senders=N)


if __name__ == '__main__':
    sim = load()
    print('N', sim['n_senders'], 'T', sim['X'].shape[0], 'L_m', round(sim['L'], 1))
    print('active at 0/150/299:', [int(sim['AL'][t].sum()) for t in (0, 150, 299)])
    print('entries', int(sim['ain'].sum()), 'exits', int(sim['aout'].sum()))
    print('speed m/s mean', float(sim['V'][sim['AL']].mean()))
    print('x range', float(sim['X'][sim['AL']].min()), float(sim['X'][sim['AL']].max()))
