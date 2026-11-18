import json, sys
import numpy as np
sys.path.insert(0,'src')
from nextgen_load import load
sim=load(split='Train',band=(452150,452300),t_win=(5100.,5400.),tag='highway_2')
T,N=sim['V'].shape
rng=np.random.default_rng(0)
rep=sim['AL']&(rng.random((T,N))<0.5*0.98)
cell=np.clip((sim['X']/100.).astype(int),0,19)
seg=rep&(cell>=0)&(cell<=4)
t0,win,K=140,15,3
clean=[];atk=[]
for t in range(t0-3,t0+win+6):
    v=sim['V'][t][seg[t]]
    if len(v)==0:
        clean.append(None);atk.append(None);continue
    clean.append(float(v.mean()*3.6))
    a=np.concatenate([v,np.zeros(K)])
    atk.append(float(a.mean()*3.6))
vref=float(np.percentile(sim['V_rep'][sim['AL']],90)*3.6)
json.dump(dict(t0=t0,win=win,clean=clean,atk=atk,tau=0.5*vref,t_start=t0-3),
          open('outputs/series.json','w'))
print('tau',0.5*vref,'n',len(clean))
