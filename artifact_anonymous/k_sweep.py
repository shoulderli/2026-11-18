import json, sys
import numpy as np
sys.path.insert(0,'src')
from overnight import run_one
from nextgen_load import load
DS=[('h2_train','highway_2','Train',(452150,452300),(5100.,5400.)),
    ('h2_test','highway_2','Test',(449900,450000),(13500.,13800.))]
rows=[]
for name,tag,split,band,tw in DS:
    sim=load(split=split,band=band,t_win=tw,tag=tag)
    vref=float(np.percentile(sim['V_rep'][sim['AL']],90)*3.6)
    for K in (8,10,12,16,20):
        for t0 in (20,100):
            for p in (1.0,0.5):
                for sd in (0,1):
                    r=run_one('B',0.02,p,K,0.0,5,(0,4),sd,sim_override=sim,t0=t0,win=10,n_min=3,thr_kmh=round(0.5*vref,1))
                    r.update(dataset=name,K=K,t0=t0,p=p,thr=round(0.5*vref,1))
                    rows.append(r)
    print(name,'rows',len(rows),flush=True)
with open('outputs/k_sweep.jsonl','w') as f:
    for r in rows: f.write(json.dumps(r)+'\n')
for m in ('B',):
    for ds in ('h2_train','h2_test'):
        v=[x for x in rows if x['dataset']==ds]
        print(ds,'arith',sum(1 for x in v if x['succ']),'feas',sum(x['feas_seg'] for x in v),
              'det',sum(1 for x in v if x['det_seg']),'conf',sum(1 for x in v if x['conf']>=0))
