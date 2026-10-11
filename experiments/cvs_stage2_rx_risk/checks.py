import argparse,copy,json
import torch
from . import design as d
from .receiver import ReceiverActions,decision_state,frozen_forward
from .objective import objective
from .model import OriginalIdentity,CurrentStudentTarget
from experiments.cvs_rx_risk_learning.learning import Risk
from experiments.cvs_stage2_rx_actions.checks import synthetic

def synthetic_aux():
 return dict(model=Risk(206).state_dict(),normalizer=dict(mean=torch.zeros(206),std=torch.ones(206),delta_std=torch.ones(8)))

def check():
 from experiments.cvs_stage2_decouple.design import make_args
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_equivariant_identity.precision import numerical_context
 from .source import component_gradients
 from .model import groups
 torch.manual_seed(221);torch.set_num_threads(2);L=synthetic();U={k:v for k,v in L.items() if k!='y'};idx=torch.arange(32)*70
 pc=d.previous.previous.original.parent_config(d.SEEDS[0]);a=make_args(pc,'cpu');a.num_domains,a.input_len=5,256
 with numerical_context(d.FULL_FP32_POLICY),installed(pc) as native:base=native.build_baseline_model(a,torch.device('cpu'))
 base._stage2_baseline_args=vars(a);records=[]
 for row in [r for r in d.rows() if r['model_seed']==d.SEEDS[0]]:
  c=d.config(row);d.validate(c);model=OriginalIdentity(copy.deepcopy(base));pg,_=groups(model,c);model.train();heads=ReceiverActions(c,L,synthetic_aux())
  teacher=CurrentStudentTarget(model) if c['unlabeled'] else None;modes=[m.training for m in model.modules()]
  value,parts,weighted,details=objective(c,model,heads,L,idx,U,idx,teacher,None,1776,'cpu')
  assert torch.isfinite(value) and modes==[m.training for m in model.modules()]
  audit=component_gradients(weighted,pg,model);value.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
  assert all(torch.isfinite(p.grad).all() for p in list(model.parameters())+list(heads.parameters()) if p.grad is not None)
  if c['rx_action']['enabled']:
   assert details['RX_L_packets']==details['RX_U_packets']==32
   assert torch.equal(heads.candidates(U,idx,False),heads.candidates(dict(U,y=torch.full_like(L['y'],999)),idx,False))
  else:assert details['RX_L_packets']==details['RX_U_packets']==0 and not list(heads.parameters())
  if c['risk_learning']['enabled']:
   with frozen_forward(model):state,_=decision_state(model,L['x'][idx])
   state,zero=heads.normalized(state,torch.zeros(32,8,8));assert torch.equal(heads.critic(state,zero),torch.zeros(32,8))
   assert details['L_action']['critic_selection_active'] and details['L_action']['selection_total_variation']>=0
   if c['risk_learning']['online']:
    assert details['critic_packets']==32 and details['critic']['candidate_identity_forwards']==288 and details['critic']['identity_target_detached']
    assert all(v==0 for v in audit['weighted_dense_gradient_norms']['critic'].values())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in heads.parameters())
   else:assert details['critic_packets']==0 and all(not p.requires_grad and p.grad is None for p in heads.parameters())
  warm=objective(c,model,heads,L,idx,U,None,None,None,888,'cpu')[3];assert warm['RX_L_packets']==warm['RX_U_packets']==warm['critic_packets']==0
  records.append(dict(arm=row['arm'],online_auxiliary=c['risk_learning']['online'],auxiliary_parameters=sum(p.numel() for p in heads.parameters()),trainable_auxiliary_parameters=sum(p.numel() for p in heads.parameters() if p.requires_grad),finite_forward_backward=True))
 assert len(d.rows())==18 and all(d.config(r)['updates']==8880 and not d.config(r)['teacher'] for r in d.rows())
 # Controls execute the unchanged objective with matched RNG and identity state.
 from experiments.cvs_stage2_rx_actions import objective as old_objective
 from experiments.cvs_stage2_rx_actions.receiver import ReceiverActions as OldHeads
 for arm in ('CE','SSL','PHYS'):
  row=next(r for r in d.rows() if r['arm']==arm);c=d.config(row);oldc=d.previous.config(next(r for r in d.previous.rows() if r['arm']==arm and r['model_seed']==row['model_seed']))
  m1=OriginalIdentity(copy.deepcopy(base)).train();m2=OriginalIdentity(copy.deepcopy(base)).train();h1=ReceiverActions(c,L);h2=OldHeads(oldc,L)
  torch.manual_seed(991);v1=objective(c,m1,h1,L,idx,U,idx,CurrentStudentTarget(m1) if c['unlabeled'] else None,None,1776,'cpu')[0]
  torch.manual_seed(991);v2=old_objective.objective(oldc,m2,h2,L,idx,U,idx,CurrentStudentTarget(m2) if c['unlabeled'] else None,None,1776,'cpu')[0]
  assert torch.equal(v1,v2),(arm,float(v1),float(v2))
 return dict(status='PASS',rows=18,arms=records,matched_CE_SSL_PHYS_objectives=True,U_TX_not_used=True,auxiliary_identity_gradient_zero=True,delayed_activation=True,exact_zero_action=True,target_access=False)

def preflight():
 from .model import build
 from .source import load_source
 result=check();parents=[]
 for seed in d.SEEDS:
  for arm in ('RFIX','RANK','MARGIN'):
   c=d.config(next(r for r in d.rows() if r['arm']==arm and r['model_seed']==seed));L,U,V,contract,p=load_source(c);model=build(c,'cpu').eval();heads=ReceiverActions(c,L);idx=torch.arange(2)
   with torch.no_grad():
    changed,_,info=heads.choose(model,U['x'][idx],U,idx,False,1776,torch.Generator().manual_seed(seed));assert model(changed).shape==(2,6)
   if c['risk_learning']['online']:
    loss,metrics=heads.critic_objective(model,L['x'][idx],L['y'][idx],L,idx,1776,torch.Generator());loss.backward();assert torch.isfinite(loss)
    assert not any(p.grad is not None for p in model.parameters())
   parents.append(dict(seed=seed,arm=arm,parent=p,auxiliary=heads.provenance,real_U_action_forward=True))
 result['parents']=parents;return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();v=preflight() if a.remote_preflight else check();d.write(a.output,v);print(json.dumps(v))
