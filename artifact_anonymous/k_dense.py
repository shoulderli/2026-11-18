"""Large-K closure on the dense slices."""
import json, sys
import numpy as np
sys.path.insert(0,'src')
from overnight import run_one
from nextgen_load import load
DS=[('h7_train','highway_7','Train',(452150,452300),(25200.,25500.)),
    ('h7_test','highway_7','Test',(449900,450000),(25200.,25500.))]
rows=[]
for name,tag,split,band,tw in DS:
    sim=load(split=split,band=band,t_win=tw,tag=tag)
    if sim['n_senders']==0: continue
    vref=float(np.percentile(sim['V_rep'][sim['AL']],90)*3.6); thr=round(0.5*vref,1)
    for K in (8,10,12,16,20):
        for t0 in (20,100):
            for p in (1.0,0.5):
                for sd in (0,1):
                    r=run_one('B',0.02,p,K,0.0,5,(0,4),sd,sim_override=sim,t0=t0,win=10,n_min=3,thr_kmh=thr)
                    r.update(dataset=name,K=K,t0=t0,p=p,thr=thr)
                    rows.append(r)
    print(name,'rows',len(rows),flush=True)
with open('outputs/k_dense.jsonl','w') as f:
    for r in rows: f.write(json.dumps(r)+'\n')
print('total',len(rows))
