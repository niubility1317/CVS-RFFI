from pathlib import Path
import torch
from . import design as d
from experiments.cvs_quality_film import design as q
from experiments.cvs_stochastic_ensemble import design as s
from experiments.cvs_stochastic_ensemble.model import build_member,probabilities
from experiments.cvs_quality_film.model import build as build_zero
from experiments.cvs_phase1_stack.recover import verify_contract
def verify(c,L,device='cpu',load=False):
 d.validate(c);fit=[L['ids'][i] for i in (L['rx']!=c['held_rx']).nonzero().flatten().tolist()];result=[];models=[]
 for m in range(3):
  cfg=q.config(next(r for r in q.rows() if r['arm']=='CE30' and r['held_rx']==c['held_rx'])) if m==0 else s.config(next(r for r in s.rows() if r['member']==m and r['held_rx']==c['held_rx']))
  p=Path(cfg['output']);done=d.read(p/'completion.json');resolved=d.read(p/'resolved_config.json');verify_contract(d.read(p/'source_contract.json'),d.read(d.SOURCE))
  if done['config']!=cfg or done['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or done['identity_updates']!=6660 or len(done['epochs'])!=30 or done['target_access'] or done['U_opened'] or not done['parameters_buffers_unchanged'] or done['fit_L_count']!=5040 or done['held_V_count']!=5400 or resolved['checkpoint_loaded'] or resolved['ancestors'] or resolved['identity_input_policy']['explicit_CFO_identity_input']:raise ValueError('Teacher provenance differs')
  ck=torch.load(p/'identity.pth',map_location='cpu',weights_only=False)
  if ck['config']!=cfg or ck['epoch']!=30 or ck['updates']!=6660 or ck['ancestors'] or ck['fit_ids']!=fit or ck['held_rx']!=c['held_rx'] or ck['target_access']:raise ValueError('Teacher actualfit roles/ancestry differs')
  result.append(dict(member=m,config=cfg,checkpoint=(p/'identity.pth').as_posix(),fit_L_count=5040,held_rx=c['held_rx'],epoch=30,source_roles='EXACT_MATCH',target_access=False,initialization='scratch'))
  if load:
   model=build_zero(cfg,L['x'][L['rx']!=c['held_rx']],device) if m==0 else build_member(cfg,device)
   model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False);models.append(model)
 return result,models
@torch.no_grad()
def cache(c,L,U,device):
 proof,models=verify(c,L,device,True);saved=[{k:v.clone() for k,v in m.state_dict().items()} for m in models];values=[]
 for i in range(0,len(U['ids']),256):values.append(probabilities([m(U['x'][i:i+256].to(device)) for m in models]).cpu())
 if any(any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()) for m,before in zip(models,saved)):raise ValueError('Frozen source teacher changed')
 return dict(probabilities=torch.cat(values),ids=U['ids'],provenance=proof,truth_read=False,held_rx_consumed=False)
