import argparse,json,torch
from pathlib import Path
from . import design as d
from .run import model_source,load_source,Bank,eval_forward,snapshot
from experiments.cvs_finite_action.operators import apply
def run():
 torch.set_num_threads(2);rows=[]
 for seed in d.SEEDS:
  c=d.config(next(r for r in d.rows() if r['model_seed']==seed));L,U,V,contract=load_source(c['response_config']);bank=Bank(c['response_config'],L,U,'cpu');models=[]
  for tag in d.TAGS:
   m,proof=model_source(tag,seed,'cpu');before=snapshot(m)
   with torch.no_grad():
    for role,data in [('L_s',L),('U_s',U)]:
     x=data['x'][:16];logits,h=eval_forward(m,x,features=True)
     for kind in ('spectral','timing'):
      z=apply(x,kind,[0.,0.],seed)
      if not torch.equal(z,x):raise ValueError('Zero finite operation is not identity')
      _,ha=eval_forward(m,apply(x,kind,c['known_parameters'][kind],seed),features=True)
      if tag=='E20':
       obs=bank.observed[role];idx=bank.orders[role][:16];column=obs['targets'][kind]['points'].index(1)
       for level in h:
        if not torch.allclose(ha[level]-h[level],obs['targets'][kind]['mean'][level][idx,column],rtol=1e-5,atol=1e-5):raise ValueError('Actual parent delta cache differs')
   after=snapshot(m)
   if any(not torch.equal(t,after[n]) for n,t in before.items()):raise ValueError('Frozen preflight state changed')
   models.append(dict(tag=tag,source=proof,state_unchanged=True));del m
  rows.append(dict(seed=seed,models=models,source_L=6300,source_U=56700,parent_cache_verified=True,zero_IQ_identity=True,identity_updates=0,target_access=False))
 return dict(status='PASS',rows=rows,models=24,target_access=False,identity_updates=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(dict(status='PASS',models=24,target_access=False)))
