import argparse,copy,json
import torch
from . import design as d
from .receiver import ReceiverActions
from .objective import objective
from .model import OriginalIdentity,CurrentStudentTarget
from experiments.cvs_report_ssl.model import groups

def synthetic():
 y=torch.arange(6).repeat_interleave(3*5*70);day=torch.arange(3).repeat_interleave(5*70).repeat(6);rx=torch.tensor([1,3,4,6,8]).repeat_interleave(70).repeat(18)
 return dict(x=torch.randn(6300,2,256),y=y,day=day,rx=rx)

def check():
 from experiments.cvs_stage2_decouple.design import make_args
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_equivariant_identity.precision import numerical_context
 from .source import component_gradients
 torch.manual_seed(221);torch.set_num_threads(2);L=synthetic();U={k:v for k,v in L.items() if k!='y'};idx=torch.arange(32)*70
 pc=d.previous.original.parent_config(d.SEEDS[0]);a=make_args(pc,'cpu');a.num_domains,a.input_len=5,256
 with numerical_context(d.FULL_FP32_POLICY),installed(pc) as native:base=native.build_baseline_model(a,torch.device('cpu'))
 base._stage2_baseline_args=vars(a);records=[]
 for row in [r for r in d.rows() if r['model_seed']==d.SEEDS[0]]:
  c=d.config(row);d.validate(c);model=OriginalIdentity(copy.deepcopy(base));pg,_=groups(model,c);model.train();heads=ReceiverActions(c,L)
  teacher=CurrentStudentTarget(model) if c['unlabeled'] else None;modes=[m.training for m in model.modules()]
  # Both delayed activation and learned selection must execute in this check.
  loss,parts,weighted,details=objective(c,model,heads,L,idx,U,idx,teacher,None,1776,'cpu')
  assert torch.isfinite(loss) and modes==[m.training for m in model.modules()]
  audit=component_gradients(weighted,pg,model);loss.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
  assert all(torch.isfinite(p.grad).all() for p in list(model.parameters())+list(heads.parameters()) if p.grad is not None)
  if c['rx_action']['enabled']:
   assert details['RX_L_packets']==32 and details['RX_U_packets']==32
   # U has no y, and changing an irrelevant dummy y cannot change its actions.
   before=heads.candidates(U,idx,False);poison=dict(U,y=torch.full_like(L['y'],999));assert torch.equal(before,heads.candidates(poison,idx,False))
  else:assert details['RX_L_packets']==details['RX_U_packets']==0 and not list(heads.parameters())
  if heads.critic is not None:
   assert details['critic_packets']==32 and details['critic']['identity_target_detached']
   assert all(value==0 for value in audit['weighted_dense_gradient_norms']['critic'].values())
   assert any(p.grad is not None and p.grad.norm()>0 for p in heads.parameters())
   assert details['L_action']['critic_selection_active']
   zero=heads.risks(L['x'][idx],torch.zeros(32,8));assert torch.equal(zero,torch.zeros_like(zero))
  records.append(dict(arm=c['arm'],finite_loss=True,counts={k:details[k] for k in ('RX_L_packets','RX_U_packets','critic_packets')},auxiliary_parameters=sum(p.numel() for p in heads.parameters())))
 assert len(d.rows())==15 and all(d.config(r)['updates']==8880 and not d.config(r)['teacher'] for r in d.rows())
 return dict(status='PASS',arms=records,U_TX_not_used=True,critic_target_detached=True,critic_loss_no_identity_gradient=True,zero_action_critic_identity=True,all_identity_gradients_finite=True,target_access=False)

def preflight():
 from .model import build
 from .source import load_source
 result=check();parents=[]
 for seed in d.SEEDS:
  c=d.config(next(r for r in d.rows() if r['arm']=='CRIT' and r['model_seed']==seed));L,U,V,contract,p=load_source(c);model=build(c,'cpu').eval();heads=ReceiverActions(c,L)
  with torch.no_grad():
   idx=torch.arange(8);x=L['x'][idx];changed,_,_=heads.choose(x,L,idx,True,1776,torch.Generator().manual_seed(seed));assert model(changed).shape==(8,6)
   changed,_,_=heads.choose(U['x'][idx],U,idx,False,1776,torch.Generator().manual_seed(seed));assert model(changed).shape==(8,6)
  parents.append(dict(seed=seed,parent=p,L=len(L['ids']),U=len(U['ids']),V=len(V['ids']),real_L_U_action_forward=True))
 result['parents']=parents;return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--remote-preflight',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();v=preflight() if a.remote_preflight else check();d.write(a.output,v);print(json.dumps(v))
