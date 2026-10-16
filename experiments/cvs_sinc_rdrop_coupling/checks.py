"""Data-free actual-network and forbidden-output checks for source diagnosis."""
import json,sys,tempfile
from pathlib import Path
import torch
from .diagnose import moments,gather,verify_state,main
from experiments.cvs_rdrop_sinc_units import design as d,model as api
from experiments.cvs_sigtime_rdrop_legacy.model import terms
from experiments.cvs_equivariant_identity.precision import numerical_context
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches

def run(device='cpu'):
 torch.set_num_threads(2);observed=[]
 with numerical_context(d.FULL_FP32_POLICY):
  for arm in ('A','E'):
   c=d.config(next(r for r in d.rows() if r['arm']==arm and r['model_seed']==2026092701));m=api.build(c,device).train();named=[(n,p) for n,p in m.named_parameters() if n.startswith('id_backbone.encoder.') and p.requires_grad];saved={n:v.clone() for n,v in m.state_dict().items()}
   torch.manual_seed(117);x=torch.randn(28,2,256,device=device);y=torch.arange(28,device=device)%6
   with fresh_parameter_caches(m):
    first=m(x,y_tx=y,grl_lambda=1.,return_aux=True,domain_labels=None,update_crra_support=False)['tx_logits'];second=m.forward_identity_only(x)['tx_logits'];_,ce,kl,_,_=terms(first,second,y,1.)
    stats,raw=gather(m,(ce,kl),named,25000. if arm=='E' else 1.)
   verify_state(m,saved)
   if set(stats)!=set(('Sinc','time_middle','SIG_TIME','remaining_feature','default_W')) or len(raw['CE_physical_Hz_gradient'])!=48 or not all(s['CE_norm']>0 and s['KL_norm']>0 for s in stats.values()):raise ValueError('Missing real joint gradient path')
   observed.append(dict(arm=arm,physical_Sinc_CE_norm=stats['Sinc']['CE_norm'],physical_Sinc_KL_norm=stats['Sinc']['KL_norm'],model_parameters_preserved=True))
 if not moments(torch.tensor([1.,0.]),torch.tensor([-2.,0.]))['conflict']:raise ValueError('Conflict sign incorrect')
 if moments(torch.tensor([1.,0.]),torch.tensor([2.,0.]))['conflict']:raise ValueError('Compatible sign incorrect')
 old=sys.argv
 try:
  sys.argv=['diagnose','--output','forbidden_target_or_existing_root']
  try:main(Path('.'))
  except ValueError as e:
   if 'immutable source-only root' not in str(e):raise
  else:raise ValueError('Unregistered output accepted')
 finally:sys.argv=old
 return dict(status='PASS',device=device,actual_networks=observed,negative_cases=3,source_IQ_reads=0,checkpoint_loads=0,head_fit_updates=0,optimizer_updates=0)
if __name__=='__main__':print(json.dumps(run(sys.argv[1] if len(sys.argv)>1 else 'cpu')))
