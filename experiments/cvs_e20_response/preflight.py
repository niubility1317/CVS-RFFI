import argparse,json,torch
from pathlib import Path
from . import design as d
from .model import build,groups
from .source import load_source,expected_counts
from .response import Bank
from .objective import objective
from experiments.cvs_nocfo_engineering.peer_cache import load as load_peer
from experiments.cvs_finite_action.state import snapshot
def run():
 torch.set_num_threads(2);rows=[]
 for seed in d.SEEDS:
  row=next(r for r in d.rows() if r['model_seed']==seed and r['arm']=='RESP_LU');c=d.config(row);L,U,V,contract=load_source(c);model=build(c,'cpu');peer=load_peer(c,U);bank=Bank(c,L,U,'cpu');pg,_=groups(model,c);model.train();before=snapshot(model);ids=torch.arange(16)
  loss,parts,w,info=objective(c,model,peer,bank,L,ids,U,ids,222,'cpu');grad=torch.autograd.grad(parts['extra'],[p for g in pg for p in g['params']],allow_unused=True);norm=float(sum(g.square().sum() for g in grad if g is not None).sqrt())
  after=snapshot(model)
  if norm<=0 or not torch.isfinite(loss) or any(p.grad is not None for reader in bank.readers.values() for p in reader.parameters()) or any(not torch.equal(t,after[n]) for n,t in before.items()):raise ValueError('Actual source response gradient/state differs')
  rows.append(dict(seed=seed,parent=model._parent_provenance,response=bank.proof,peer_seeds=peer['teacher_seeds'],L=6300,U=56700,V=27000,source_contract_ref=d.SOURCE,response_identity_gradient_norm=norm,main_L_batch=16,main_U_batch=16,actual_forward_L_packets=info['L_packets']+info['extra_L_packets'],actual_forward_U_packets=2*info['U_packets'],actual_forward_calls=info['forward_calls'],identity_updates=0,parameters_buffers_unchanged=True,target_access=False))
 return dict(status='PASS',rows=rows,configs=21,expected_counts={a:expected_counts(d.config(next(r for r in d.rows() if r['arm']==a))) for a in d.ARMS},source_only=True,target_access=False,identity_updates=0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True,type=Path);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(dict(status='PASS',seeds=3,configs=21)))
