"""Focused data/parent/default-head, four-cell and source-interaction negatives."""
from copy import deepcopy
from . import design as d,protocol as p
from .report import source_checks

def rejects(call):
 try:call()
 except (ValueError,KeyError,FileNotFoundError):return
 raise AssertionError('Illegal input accepted')

def run():
 count=0;configs=[d.config(r) for r in d.rows()]
 assert len(configs)==4 and len({c['output_root'] for c in configs})==4 and len({c['parent_source'] for c in configs})==1
 for c in configs:
  assert d.validate(c)==c
  for key,value in [('model_seed',2026092702),('parent_source','unverified'),('share_propagation',not c['share_propagation']),('As_contrast',not c['As_contrast']),('checkpoint_sources',[])]:
   bad=deepcopy(c);bad[key]=value;rejects(lambda:d.validate(bad));count+=1
 c=configs[0];roles={'L_s':['L'],'U_s':['U'],'V':['V']}
 done=dict(config=c,epoch=10,optimizer_steps=2220,target_access=False);frozen=dict(config=c,selection='final_only',target_access=False);initial=dict(scratch_only=False,target_contact=False,checkpoint_sources=c['checkpoint_sources'],source_roles='EXACT_MATCH');contract=dict(role_ids=roles)
 p.final_metadata(c,done,frozen,initial,contract,roles)
 for obj,key,value in [('done','target_access',True),('done','head_fit_updates',1),('done','optimizer_steps',2221),('frozen','selection','best_target'),('initial','scratch_only',True),('initial','source_roles','MISMATCH'),('contract','role_ids',{})]:
  dd,ff,ii,cc=map(deepcopy,(done,frozen,initial,contract));target={'done':dd,'frozen':ff,'initial':ii,'contract':cc}[obj];target[key]=value;rejects(lambda:p.final_metadata(c,dd,ff,ii,cc,roles));count+=1
 cp=dict(epoch=10,a1_ema_successful_updates=2220,run_id=c['run_id'],candidate_id=c['row_id'],checkpoint_selection='final_only',ema_model={},args=dict(baseline_ckpt=c['parent_source']+'/final_ssdg.pth',teacher_ckpt=None,pseudo_temporal_mode='batch_neighbor',pseudo_temporal_gate=True));p.checkpoint(c,cp)
 for key,value in [('epoch',9),('candidate_id','other'),('ema_model',None),('checkpoint_selection','target_best')]:
  bad=deepcopy(cp);bad[key]=value;rejects(lambda:p.checkpoint(c,bad));count+=1
 vals={a:dict(fullV_clean=.98,midlow=.6,low=.5) for a in ('NOAS30','AS30','C00','C10','C01','C11')};vals['C11']['midlow']=.605
 effect,checks=source_checks(vals);assert all(checks.values())
 v=deepcopy(vals);v['C11']['fullV_clean']=.9789;assert not all(source_checks(v)[1].values())
 v=deepcopy(vals);v['C10']['midlow']=.604;assert not source_checks(v)[1]['midlow_vs_max_all_controls_at_least_0p5pp']
 v=deepcopy(vals);v['C01']['midlow']=.603;assert not source_checks(v)[1]['midlow_interaction_at_least_0p5pp']
 v=deepcopy(vals);v['C11']['fullV_clean']=.979;assert source_checks(v)[1]['clean_within_NoAs_0p1pp']
 from . import evaluate
 oldobs=evaluate.observations;evaluate.observations=lambda:(_ for _ in ()).throw(FileNotFoundError('No completed predictions;truth closed'))
 try:rejects(lambda:evaluate.preflight(configs[0]['row_id']))
 finally:evaluate.observations=oldobs
 return dict(status='PASS',protocol_groups=6,negative_cases=count,source_criteria_cases=5,missing_prediction_truth_closed=True,NN=0,IQ=0,CP=0,training_updates=0)
