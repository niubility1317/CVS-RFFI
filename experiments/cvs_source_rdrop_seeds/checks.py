import copy,json,tempfile
from pathlib import Path
import torch
from . import design as d
from experiments.cvs_source_rdrop.checks import checks as first_checks
from experiments.cvs_source_rdrop.audit import measure_batch,recount

def checks():
 torch.set_num_threads(2);original=first_checks()
 q=dict(status='PASS',fixed_rules=d.QUALIFICATION,source_only=True,target_metrics_used=False,no_automatic_training_release=True)
 d.validate_qualification(q);bads=[]
 for key,value in [('status','FAIL'),('fixed_rules',{}),('source_only',False),('target_metrics_used',True),('no_automatic_training_release',False)]:
  invalid=copy.deepcopy(q);invalid[key]=value;bads.append(lambda v=invalid:d.validate_qualification(v))
 for key,value in [('role','U_s'),('coefficient',.1),('head_fit_updates',1),('model_seed',2026092701),('strong_rng_seed',1)]:
  invalid=d.config(d.rows()[0]);invalid[key]=value;bads.append(lambda v=invalid:d.validate(v))
 invalid=d.config(d.rows()[0]);invalid['source_config']['model_seed']=2026092703;bads.append(lambda:d.validate(invalid))
 bads.append(lambda:d.source_config(dict(row_id='UNREGISTERED',arm='SOURCE_RDROP',model_seed=2026092704)))
 bads.append(lambda:recount(Path('not_opened'),design=d,row=d.first.rows()[0]))
 for call in bads:
  try:call()
  except ValueError:pass
  else:raise AssertionError('Changed/unregistered source context accepted')
 from experiments.cvs_receiver_dro_seeds import model as api
 inputs=torch.randn(12,2,256,generator=torch.Generator().manual_seed(15519));labels=torch.arange(12)%6
 for row in d.rows():
  c=d.config(row);d.validate(c);model=api.build(c['source_config'],'cpu').eval()
  before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
  out=measure_batch(model,inputs,labels,c['strong_rng_seed'])
  if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()) or not torch.equal(rng,torch.get_rng_state()) or any(p.grad is not None for p in model.parameters()):raise ValueError('Native source adapter changed model/RNG/.grad')
  if any(not torch.isfinite(v).all() or v.norm()==0 for name in ('PAIRED_CE','SYMMETRIC_KL') for v in out['vectors'][name]):raise ValueError('Native derivative context missing')
  if out['features1'].shape!=(12,160) or torch.equal(out['features1'],out['features2']):raise ValueError('Native stochastic joint160 context differs')
 from .publish import configure,INSPECT,NEW
 configure();compile(INSPECT,'actual_remaining_inspector','exec')
 if NEW not in INSPECT or 'saved_vectors=40,source_only=True' in INSPECT:raise ValueError('Actual remaining completion inspector differs')
 return dict(status='PASS',old_default_interface=original['status'],remaining_native_contexts=2,negative_cases=len(bads),
  fixed_cohort_rng_and_formula=True,model_parameters_buffers_RNG_unchanged=True,actual_t1_t2_SIG_W_gradients=True,
  expected_new_source_models=2,expected_new_forward_packets=5400,expected_saved_vectors=80,
  real_source_IQ_or_checkpoint_read=False,head_fit_updates=0,optimizer_updates=0,first_model_remeasured=False)
if __name__=='__main__':print(json.dumps(checks()))
