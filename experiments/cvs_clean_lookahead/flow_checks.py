import json
import torch
from . import design as d
from .data import selection
from .worker import batch_response,qualify
from .virtual import discardable_step
from experiments.cvs_F_nontrue20 import design as base,model as api

def run():
 torch.set_num_threads(2);groups=[(t,r,day) for t in range(6) for r in (1,3,4,6,8) for day in (1,2,3) for j in range(70)]
 L=dict(ids=[str(i) for i in range(6300)],y=torch.tensor([k[0] for k in groups]),rx=torch.tensor([k[1] for k in groups]),day=torch.tensor([k[2] for k in groups]));a,b=selection(L,dict(role_ids={'L_s':L['ids']}))
 assert len(a)==len(b)==180 and not set(a)&set(b)
 for ix in (a,b):
  for start in range(0,180,30):assert set(zip(L['y'][ix[start:start+30]].tolist(),L['rx'][ix[start:start+30]].tolist()))=={(t,r) for t in range(6) for r in (1,3,4,6,8)}
 negative=0
 for key,value in [('target_access',True),('step_unique',181),('models',['B1']),('batch',128),('persistent_original_updates',1),('ordinary_augmentation',True)]:
  c=d.config();c[key]=value
  try:d.validate(c)
  except ValueError:negative+=1
  else:raise AssertionError('Unregistered protocol change accepted')
 torch.manual_seed(20261019627);c=base.config(next(r for r in base.rows() if r['arm']=='B1'));m=api.build(c,'cpu').eval();snap={k:v.detach().clone() for k,v in m.state_dict().items()}
 generator=torch.Generator().manual_seed(20261019628);train={v:torch.randn(2,2,256,generator=generator) for v in d.VIEWS};response={v:torch.randn(2,2,256,generator=generator) for v in d.VIEWS};y=torch.tensor([0,4]);ry=torch.tensor([1,3]);counts=dict(NN=0,network_backward_packets=0,virtual_updates=0);rng=torch.get_rng_state().clone()
 def forward(model,x):counts['NN']+=len(x);return model.forward_identity_only(x)['tx_logits']
 def step(model,g):counts['virtual_updates']+=1;return discardable_step(model,g)
 result,geometry,witness=batch_response(m,train,response,y,ry,forward,step,counts)
 assert counts==dict(NN=34,network_backward_packets=10,virtual_updates=3)
 assert set(result)=={'BASE','RAW_STEP','JOINT_STEP','FIRST_ORDER_LOOKAHEAD'} and all(set(v)==set(d.VIEWS) for v in result.values())
 assert torch.equal(rng,torch.get_rng_state()) and all(p.grad is None for p in m.parameters()) and all(torch.equal(v,m.state_dict()[k]) for k,v in snap.items())
 toy=[dict(arm=arm,response={mode:{v:{'CE':1. if mode=='FIRST_ORDER_LOOKAHEAD' else 1.1} for v in d.VIEWS} for mode in ('JOINT_STEP','FIRST_ORDER_LOOKAHEAD')}) for arm in ('B0','B1') for _ in range(6)]
 assert qualify(toy)['rationale_pass'];toy[0]['response']['FIRST_ORDER_LOOKAHEAD']['raw']['CE']=2.;assert not qualify(toy)['rationale_pass']
 return dict(status='PASS',source_selection_180_plus_180_disjoint=True,each_batch_all_TX_RX=True,protocol_negatives=negative,actual_shared_worker_batch_path=True,**counts,original_state_grad_RNG_unchanged=True,source_or_target_IQ=0,CP=0,new_trained_models=0,qualification_pass_and_fail=True)
if __name__=='__main__':print(json.dumps(run()))
