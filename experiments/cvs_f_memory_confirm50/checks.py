"""Focused extension wrapper checks; reuse unchanged mechanism's prior CPU/CUDA evidence."""
import copy,tempfile
from pathlib import Path
from . import design as d
from experiments.cvs_f_crossbatch50 import design as old

def configuration_checks():
 base=Path(tempfile.gettempdir())/d.RUN
 for row in d.rows():
  c=d.config(row,str(base),'PROFILE','METADATA');d.validate(c)
  original=old.config(row,str(base.parent/old.RUN),'PROFILE','METADATA')
  allowed={'schema','run_id','launch_owner','output_root','runtime_base','source_selection','confirmation'}
  assert {k:v for k,v in c.items() if k not in allowed}=={k:v for k,v in original.items() if k not in allowed}
 assert [r['model_seed'] for r in d.rows()]==[2026092702,2026092703]
 assert d.fixed_authorization()['fixed_new_rows']==d.rows() and not d.fixed_authorization()['source_gate_applied']
 return True

def run(profile,metadata,output):
 configuration_checks();neg=0;base=Path(output)/d.RUN;c=d.config(d.rows()[0],str(base),profile,metadata)
 for key,value in [('epochs',20),('test_views',['practical_high']),('model_seed',2026092701),('separate_head_training',True),('contrastive_temperature',.1),('contrastive_keys_detached',False),('memory_previous_steps',4),('contrastive_class_reweight',True),('contrastive_mode','SYM4')]:
  bad=copy.deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:neg+=1
  else:raise AssertionError('Changed fixed recipe accepted '+key)
 for seeds in [(2026092701,),d.ALL_SEEDS]:
  try:d.rows(seeds)
  except ValueError:neg+=1
  else:raise AssertionError('Duplicate seed01 accepted')
 try:d.config(d.rows()[0],str(base.parent/old.RUN),profile,metadata)
 except ValueError:neg+=1
 else:raise AssertionError('Old immutable root accepted')
 from .score import preflight
 try:preflight(c,base)
 except FileNotFoundError:neg+=1
 else:raise AssertionError('Missing prediction passed truth preflight')
 import torch
 from . import model as api
 from experiments.cvs_f_crossbatch50 import model as prior_api
 from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
 from experiments.cvs_equivariant_identity.precision import numerical_context
 from experiments.cvs_f_crossrx50_local.provenance import source_proof,validate_checkpoint
 torch.set_num_threads(2)
 with numerical_context(d.FULL_FP32_POLICY):
  torch.manual_seed(9743);m=api.build(c,'cpu').eval()
  oc=old.config(d.rows()[0],str(base.parent/old.RUN),profile,metadata)
  torch.manual_seed(9743);ref=prior_api.build(oc,'cpu').eval()
  assert list(m.state_dict())==list(ref.state_dict()) and all(torch.equal(v,ref.state_dict()[k]) for k,v in m.state_dict().items())
  assert m._crossbatch_mode==ref._crossbatch_mode=='MEM4'
  with torch.no_grad(),fresh_parameter_caches(m),fresh_parameter_caches(ref):
   x=torch.zeros(2,2,256);a=m.forward_identity_only(x);b=ref.forward_identity_only(x)
   assert torch.equal(a['tx_logits'],b['tx_logits']) and torch.equal(a['id_feat_joint'],b['id_feat_joint']) and torch.equal(api.export(m)(x),prior_api.export(ref)(x))
  del ref
  root=Path(profile).parent.parent/'runs'/c['reference_local_run']/('BASE-s'+str(c['model_seed']))/'source'
  rc=d.read(root/'local_source_complete.json')['config'];source_proof(rc)
  cp=torch.load(root/'final_ssdg.pth',map_location='cpu',weights_only=False);validate_checkpoint(cp,rc);m.load_state_dict(cp['model'],strict=True)
  with torch.no_grad(),fresh_parameter_caches(m):out=m.forward_identity_only(torch.zeros(2,2,256))
  assert out['tx_logits'].shape==(2,6) and torch.isfinite(out['tx_logits']).all() and torch.isfinite(out['id_feat_joint']).all()
 return dict(status='PASS',protocol_negative_cases=neg,both_remaining_configs_scientifically_identical=True,prior_MEM4_model_state_and_zero_input_logits_features_export_exact=True,real_checkpoint_no_query_smoke=True,checkpoint_seed=2026092702,loaded_checkpoint_discarded_not_inherited=True,IQ_reads=0,target_reads=0,real_training_updates=0,head_fit_updates=0,prior_mechanism_CPU_CUDA_checks_reused=True)
