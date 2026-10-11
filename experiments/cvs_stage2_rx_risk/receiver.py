from contextlib import contextmanager
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_stage2_rx_actions.receiver import ReceiverActions as Bank
from experiments.cvs_rx_risk_learning.learning import Risk,normalize,probabilities,loss as risk_loss
from experiments.cvs_report_ssl.physical import signal_state
from experiments.cvs_rx_intervention_audit.actions import apply
from experiments.cvs_rx_critic_audit.metrics import margin

@contextmanager
def frozen_forward(model):
 modes=[(m,m.training) for m in model.modules()]
 try:
  model.eval()
  with torch.no_grad():yield
 finally:
  for m,mode in modes:m.training=mode

def decision_state(model,x):
 out=model.forward_identity_only(x);z=out['tx_logits']
 return torch.cat((signal_state(x),F.normalize(out['id_feat_joint'],dim=1),z),1).detach(),z

def read_pretrained(c):
 plan=c['risk_learning'];p=Path(plan['checkpoint']);source=plan['source_config']
 done=d.read(p.parent/'completion.json');resolved=d.read(p.parent/'resolved_config.json')
 expected=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',identity_updates=0,auxiliary_updates=12000,target_access=False,L_count=6300,V_count=27000,candidates=8,arms=6,source_model_unchanged=True,bank_unchanged=True)
 if any(done.get(k)!=v for k,v in expected.items()) or resolved['config']!=source:raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: auxiliary source')
 if resolved['commit']!=plan['source_execution_commit']:raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: auxiliary execution version')
 if source['parent_build_config']['parent_contract']!=c['parent_contract'] or source['source_checkpoint']!=c['parent_checkpoint'] or source['model_seed']!=c['model_seed']:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: auxiliary parent')
 ck=torch.load(p,map_location='cpu',weights_only=False)
 if ck['config']!=source or ck['arm']!=plan['kind']+'_STATE':raise ValueError('Auxiliary checkpoint config differs')
 return ck,dict(checkpoint=p.as_posix(),source_run=source['run_id'],source_execution_commit=resolved['commit'],pretrained_steps=2000,L_count=6300,pretrained_identity_updates=0,V_fitted=False,target_access=False,online=plan['online'],offline_source_diagnostic_seconds_all_six_arms=done['elapsed_seconds'])

class ReceiverActions(Bank):
 def __init__(self,c,L,pretrained=None):
  bank_config=dict(c,rx_action=dict(c['rx_action'],critic=False));super().__init__(bank_config,L)
  self.plan=c['rx_action'];self.risk_plan=c['risk_learning'];self.provenance=None
  if not self.risk_plan['enabled']:return
  if pretrained is None:ck,self.provenance=read_pretrained(c)
  else:ck=pretrained;self.provenance=dict(scope='local synthetic test only')
  self.critic=Risk(206);self.critic.load_state_dict(ck['model'],strict=True)
  for key,shape in [('mean',(206,)),('std',(206,)),('delta_std',(8,))]:
   value=ck['normalizer'][key]
   if value.shape!=shape or not torch.isfinite(value).all() or (key!='mean' and not (value>0).all()):raise ValueError('Invalid L normalizer')
   self.register_buffer('risk_'+key,value.detach().clone())
  self.critic.requires_grad_(self.risk_plan['online'])
 def normalized(self,state,delta):return normalize(state,delta,dict(mean=self.risk_mean,std=self.risk_std,delta_std=self.risk_delta_std))
 @torch.no_grad()
 def choose(self,model,x,cache,indices,labeled,step,generator):
  candidates=self.candidates(cache,indices,labeled);probs=x.new_full((len(x),8),1/8);score_std=None
  active=self.critic is not None
  if active:
   with frozen_forward(model):state,_=decision_state(model,x)
   state,delta=self.normalized(state,candidates);scores=self.critic(state,delta);probs=probabilities(scores,True);score_std=float(scores.std(1,unbiased=False).mean())
  selected=torch.multinomial(probs.cpu(),1,generator=generator).flatten().to(x.device)
  delta=candidates[torch.arange(len(x),device=x.device),selected]
  changed,info=apply(x,delta,1.,self.plan['taps'],self.plan['max_amplitude_dB'])
  return changed,delta,dict(critic_selection_active=active,full_strength_count=int((selected%2==1).sum()),selection_entropy=float(-(probs*probs.log()).sum(1).mean()),selection_total_variation=float(.5*(probs-1/8).abs().sum(1).mean()),risk_score_std=score_std,clipped_bin_fraction=float(info['clipped_fraction'].mean()))
 def critic_objective(self,model,x,y,cache,indices,step,generator):
  if not self.risk_plan['online']:raise ValueError('Frozen auxiliary update forbidden')
  candidates=self.candidates(cache,indices,True)
  with frozen_forward(model):
   state,z=decision_state(model,x);clean_ce=F.cross_entropy(z,y,reduction='none');clean_margin=margin(z,y);ces=[];margins=[]
   for a in range(8):
    changed,_=apply(x,candidates[:,a],1.,self.plan['taps'],self.plan['max_amplitude_dB']);z=model(changed)
    ces.append(F.cross_entropy(z,y,reduction='none')-clean_ce);margins.append(clean_margin-margin(z,y))
   ce=torch.stack(ces,1);target=torch.stack(margins,1)
  state,delta=self.normalized(state,candidates);predicted=self.critic(state,delta);value=risk_loss(predicted,ce,target,self.risk_plan['kind'])
  i,j=torch.triu_indices(8,8,1,device=x.device);diff=target[:,i]-target[:,j];mask=diff.abs()>.01;pd=predicted.detach()[:,i]-predicted.detach()[:,j]
  correct=torch.where(pd.abs()<=1e-8,torch.full_like(pd,.5),(pd.sign()==diff.sign()).float())*mask
  return value,dict(batch=len(x),candidate_identity_forwards=len(x)*9,eligible_margin_pairs=int(mask.sum()),pair_concordance=float(correct.sum()/mask.sum()) if mask.any() else None,target_margin_drop_mean=float(target.mean()),prediction_mean=float(predicted.detach().mean()),identity_target_detached=not target.requires_grad and not state.requires_grad,supervision=self.risk_plan['kind'])
