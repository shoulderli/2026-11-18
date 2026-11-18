"""Generate a controlled-access highway scenario in SUMO and record ground truth.

Network: 2-lane mainline 0-1200 m, lane drop + 15 m/s limit 1200-2000 m,
on-ramp merging at 1200 m.  The ramp is NOT counted by the mainline
boundary at x=0 (the unmetered source term).

Run with the sumo_env python (has traci/sumolib).
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMO_BIN = os.path.join(HERE, 'venv_sumo', 'bin', 'sumo')
NETCONVERT = os.path.join(HERE, 'venv_sumo', 'bin', 'netconvert')
SUM_DIR = os.path.join(HERE, 'sumo')
os.makedirs(SUM_DIR, exist_ok=True)

NOD = """<nodes>
  <node id="n0" x="0" y="0" type="priority"/>
  <node id="n1" x="900" y="0" type="priority"/>
  <node id="n2" x="1200" y="0" type="priority"/>
  <node id="n3" x="2000" y="0" type="priority"/>
  <node id="r0" x="900" y="-60" type="priority"/>
</nodes>
"""
EDG = """<edges>
  <edge id="main0" from="n0" to="n1" numLanes="2" speed="27.78"/>
  <edge id="main1" from="n1" to="n2" numLanes="2" speed="27.78"/>
  <edge id="main2" from="n2" to="n3" numLanes="1" speed="15.0"/>
  <edge id="ramp" from="r0" to="n2" numLanes="1" speed="15.0"/>
</edges>
"""
CON = """<connections>
  <connection from="main0" to="main1" fromLane="0" toLane="0"/>
  <connection from="main0" to="main1" fromLane="1" toLane="1"/>
  <connection from="main1" to="main2" fromLane="0" toLane="0"/>
  <connection from="main1" to="main2" fromLane="1" toLane="0"/>
  <connection from="ramp" to="main2" fromLane="0" toLane="0"/>
</connections>
"""
ROUTE_MAIN = """<routes>
  <vType id="car" vClass="passenger" length="4.5" minGap="2.5"
         accel="1.5" decel="2.0" sigma="0.5" tau="1.2" maxSpeed="27.78"/>
  <flow id="main" type="car" begin="0" end="{T}" vehsPerHour="{main_vph}"
        departLane="random" departSpeed="max">
    <route edges="main0 main1 main2"/>
  </flow>
"""
RAMP_FLOW = """  <flow id="ramp" type="car" begin="0" end="{T}" vehsPerHour="{ramp_vph}"
        departSpeed="max">
    <route edges="ramp main2"/>
  </flow>
"""
ROUTE_TAIL = """</routes>
"""
CFG = """<configuration>
  <input><net-file value="highway.net.xml"/><route-files value="route.rou.xml"/></input>
  <time><begin value="0"/><end value="{T}"/><step-length value="{dt}"/></time>
  <processing><time-to-teleport value="-1"/></processing>
  <random_number><seed value="{seed}"/></random_number>
  <report><no-step-log value="true"/><no-warnings value="true"/></report>
</configuration>
"""


def write_static():
    with open(os.path.join(SUM_DIR, 'highway.nod.xml'), 'w') as f:
        f.write(NOD)
    with open(os.path.join(SUM_DIR, 'highway.edg.xml'), 'w') as f:
        f.write(EDG)
    with open(os.path.join(SUM_DIR, 'highway.con.xml'), 'w') as f:
        f.write(CON)
    subprocess.run([NETCONVERT,
                    '-n', os.path.join(SUM_DIR, 'highway.nod.xml'),
                    '-e', os.path.join(SUM_DIR, 'highway.edg.xml'),
                    '-x', os.path.join(SUM_DIR, 'highway.con.xml'),
                    '-o', os.path.join(SUM_DIR, 'highway.net.xml')],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run(demand, ramp, seed, T=1200.0, dt=0.2, Npool=1200, out=None):
    import traci
    write_static()
    rp = os.path.join(SUM_DIR, 'route.rou.xml')
    cp = os.path.join(SUM_DIR, 'highway.sumocfg')
    with open(rp, 'w') as f:
        f.write(ROUTE_MAIN.format(T=T, main_vph=int(demand * 3600)))
        if ramp > 0:
            f.write(RAMP_FLOW.format(T=T, ramp_vph=int(ramp * 3600)))
        f.write(ROUTE_TAIL)
    with open(cp, 'w') as f:
        f.write(CFG.format(T=T, dt=dt, seed=seed))
    steps = int(round(T / dt))
    X = np.zeros((steps, Npool), np.float32)
    V = np.zeros((steps, Npool), np.float32)
    AL = np.zeros((steps, Npool), bool)
    lane = np.zeros(Npool, np.int64)
    ain = np.zeros(steps, np.int32)
    aout = np.zeros(steps, np.int32)
    aramp = np.zeros(steps, np.int32)
    ids = {}
    next_id = 0
    prev = set()
    traci.start([SUMO_BIN, '-c', cp])
    for t in range(steps):
        traci.simulationStep()
        cur = set(traci.vehicle.getIDList())
        new = cur - prev
        gone = prev - cur
        aout[t] = len(gone)
        for vid in new:
            idx = next_id
            next_id += 1
            ids[vid] = idx
            lid = traci.vehicle.getLaneID(vid)
            if lid.startswith('ramp'):
                aramp[t] += 1
                lane[idx] = 2
            else:
                ain[t] += 1
                lane[idx] = int(lid.split('_')[-1])
        for vid in cur:
            idx = ids[vid]
            x, y = traci.vehicle.getPosition(vid)
            X[t, idx] = x
            V[t, idx] = traci.vehicle.getSpeed(vid)
            AL[t, idx] = True
        prev = cur
    traci.close()
    if next_id > Npool:
        raise RuntimeError('Npool too small: %d' % next_id)
    out = out or os.path.join(HERE, 'outputs', 'sumo_d%03d_r%02d_s%d.npz'
                              % (int(demand * 100), int(ramp * 100), seed))
    np.savez_compressed(out, X=X, V=V, AL=AL, lane=lane,
                        ain=ain, aout=aout, aramp=aramp,
                        demand=demand, ramp=ramp, seed=seed)
    print('saved', out, 'vehicles', next_id, 'steps', steps)
    return out


if __name__ == '__main__':
    d = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    r = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    s = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    run(d, r, s)
