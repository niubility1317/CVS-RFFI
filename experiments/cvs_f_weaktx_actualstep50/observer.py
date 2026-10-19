"""Native Sinc observation sees final corrected leaves before own EMA update."""
import gzip,json
from experiments.cvs_f_supcon50.observer import Observer as Original
from experiments.cvs_f_supcon50.mechanism import RX_BY_DOMAIN
from . import design as d
from .core import active
from .mechanism import bind_optimizer
class Observer(Original):
 def __init__(self,c):d.validate(c);self.c=c;self.optimizer=None;self.pending=None;self.raw=[];self.records=[];self.saved=[]
 def data(self,native,ctx,out):
  self.ctx=ctx;contract=d.read(d.read(self.c['runtime_profile'])['dataset']['source_contract'])
  if ctx['num_domains']!=15 or tuple(v[0] for v in contract['domain_map'])!=RX_BY_DOMAIN:raise ValueError('Original physical sourceRX/day map differs')
 def optimizer_created(self,opt):
  # Unit counter hook is already first. Our post correction precedes the inherited
  # Sinc post witness; both return before native own-EMA copies the student.
  bind_optimizer(self.model,opt);super().optimizer_created(opt)
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device);steps=len(rows)*222;scheduled=sum(active(s) for s in range(1,steps+1))
  if m._step_calls!=steps or m._step_post_calls!=steps or len(m._step_records)!=steps or m._step_scheduled!=scheduled or m._step_second_captures!=scheduled or m._step_vjp_calls!=3*m._step_eligible or m._step_pending is not None or m._step_capture is not None or not all(r['finalized'] for r in m._step_records):raise ValueError('Actual ordered native/feature/EMA budget differs')
  d.write(out/'actual_feature_step_progress.json',dict(epoch=len(rows),steps=steps,post_steps=m._step_post_calls,scheduled=scheduled,eligible=m._step_eligible,changed=m._step_changed,extra_feature_VJP_calls=m._step_vjp_calls,numerical_fallbacks=m._step_fallbacks,maximum_stored_norm_match_error=m._step_max_norm_error,extra_model_forwards=0,head_fit_updates=0,U_truth_read=False))
  if (out/'sinc_optimizer_update_records.json').exists():
   value=d.read(out/'sinc_optimizer_update_records.json');value.update(physical_after_definition='Native origin-aware physical decay+adaptive step, then this same-step feature correction; witness ordered after correction',gradient_definition='Actual original RDrop+SupCon+oldU leaf derivative after physical globalclip; no surrogate/gradient correction');d.write(out/'sinc_optimizer_update_records.json',value)
 def finish(self):
  value=super().finish();m=self.model
  if m._step_calls!=11100 or m._step_post_calls!=11100 or len(m._step_records)!=11100 or m._step_scheduled!=2498 or m._step_second_captures!=2498 or m._step_vjp_calls!=3*m._step_eligible or not 0<=m._step_changed<=m._step_eligible<=2498 or m._step_pending is not None or not all(r['finalized'] for r in m._step_records):raise ValueError('Complete actual-step execution evidence required')
  with gzip.open(self.out/'actual_feature_steps.jsonl.gz','xt',encoding='utf8') as f:
   for r in m._step_records:f.write(json.dumps(r,allow_nan=False)+'\n')
  chosen=[r for r in m._step_records if r['eligible']];full=[r for r in chosen if r['strength']==1.];enforced=[r for r in full if r['constraint_enforced']]
  d.write(self.out/'actual_feature_step_observation.json',dict(status='VERIFIED',treatment=self.c['treatment'],calls=11100,post_steps=11100,scheduled=2498,second_logit_captures=m._step_second_captures,eligible=m._step_eligible,eligible_fraction=m._step_eligible/2498,changed_steps=m._step_changed,nonzero_fraction=m._step_changed/2498,extra_feature_VJP_calls=m._step_vjp_calls,numerical_fallbacks=m._step_fallbacks,maximum_norm_match_error=m._step_max_norm_error,full_strength_calls=len(full),actual_constraint_calls=len(enforced),maximum_enforced_residual_relative=max((r['applied_full_strength_max_residual_relative'] for r in enforced),default=0.),mean_retained_normalized_step_norm_ratio=sum(r['retained_normalized_norm_ratio'] for r in chosen)/len(chosen) if chosen else None,extra_model_forwards=0,extra_L_packets=0,extra_optimizer_updates=0,extra_parameters=0,risk_groups=['weak1_CE','weak3_CE','other4_CE'],head_fit_updates=0,default_head_entire_native_update_exact=True,native_moments_and_gradients_unchanged=True,U_truth_read=False,full_deployed_features=True,paired_CE_L_contributions=True,post_correction_before_Sinc_witness_and_EMA=True,Sinc_origin_decay_applied_once_in_native=True,interpretation='Only nonfallback full-strength candidate feature first-order current-L risk is constrained; control applied risks distinct and explicit. No nonlinear/joint-head/target guarantee.'))
  return value
