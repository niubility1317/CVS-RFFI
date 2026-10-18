"""Frozen P43 students on the already saved eleven-condition source panel."""
import importlib,json,sys,time,traceback
from pathlib import Path
from . import design as d,protocol as p

def student(project,arm,device,torch):
 from experiments.cvs_noise_shape_factorial43 import design as old,model as api
 if Path(old.PROJECT)!=Path(project):raise ValueError('Bootstrap runtime/project differs from original P43')
 row=next(r for r in old.rows() if r['arm']==arm);c=old.config(row,30);source=d.source(project,arm)
 p.check_source_metadata(project,arm,c,d.read(source/'completion.json'),d.read(source/'source_frozen.json'),d.read(source/'initialization.json'),d.read(source/'source_contract.json'),d.read(old.SOURCE))
 # own_model performs the original own20 and fixed-final provenance checks.
 # Count its two declared checkpoint deserializations; no IQ before smoke.
 seen=[];original=torch.load;allowed={source/'final_ssdg.pth',source.parent/'stage20/final_ssdg.pth'}
 def counted(path,*args,**kwargs):
  if Path(path) not in allowed:raise ValueError('Only the two original own P43 checkpoint inputs before smoke')
  seen.append(str(path));return original(path,*args,**kwargs)
 torch.load=counted
 try:model=api.own_model(c,device)
 finally:torch.load=original
 if sorted(seen)!=sorted(str(path) for path in allowed):raise ValueError('Original own20/final checkpoint reads differ')
 head=model.id_backbone.encoder.core.id_backbone.cls_head
 p.check_head(head.weight.shape,float(head.scale),sum(q.numel() for q in model.parameters()),len(model.state_dict()))
 if model.training or any(q.requires_grad or q.grad is not None for q in model.parameters()) or any(not bool(torch.isfinite(q).all()) for q in model.state_dict().values()):raise ValueError('Finite frozen joint default-head source state required')
 if model.id_backbone.encoder._full_train_packets or model.id_backbone.encoder._full_eval_packets:raise ValueError('No source/model forward before no-query smoke')
 provenance=dict(run_id=d.REFERENCE_RUN,row_id=c['row_id'],checkpoint=str(source/'final_ssdg.pth'),state_key='model',EMA_weights_applied=False,EMA_checkpoint_state_may_be_deserialized_but_not_applied=True,checkpoint_deserializations=len(seen),checkpoint_paths=seen,student_and_default_head_loaded_together=True,historical_epochs=30,historical_updates=6660,final_segment_epochs=10,final_segment_updates=2220,parent_epochs=20,parent_updates=4440,selection='final_only',model_seed=d.SEED,role_handles='EXACT_MATCH',target_access=False,head_fit_updates=0)
 return model,provenance

def helper_contract():
 from experiments.cvs_fullsource_residual_diag24 import design as old
 if (old.N,old.BATCH,old.GRAD_N,tuple(old.DRAWS),tuple(old.SCENES),tuple(old.CONDITIONS))!=(d.N,d.BATCH,d.GRAD_N,d.DRAWS,d.SCENES,d.CONDITIONS):raise ValueError('Unchanged P24 prediction/gradient helper view budget differs')

def actual_modules():
 names=('SSDG.train_ssdg','DataAugmentation','leo_practical.channel','leo_practical.residual','experiments.cvs_noise_shape_factorial43.model','experiments.cvs_noise_shape_factorial43.quality','experiments.cvs_fullsource_residual_diag24.design','experiments.cvs_fullsource_residual_diag24.engine','experiments.cvs_fullsource_residual_diag24.model','experiments.cvs_fullsource_residual_diag24.observe')
 result={}
 for name in names:
  module=importlib.import_module(name);result[name]=str(module.__file__)
 print(json.dumps(dict(stage='ACTUAL_MODULE_FILES',modules=result),ensure_ascii=False),flush=True)
 return result

def read_cached_observations(project,panel,roles,np):
 root=d.paths(project)['panel'];meta=d.read(root/'observations/completion.json')
 p.check_panel(roles,panel['ids'],panel['y'].tolist(),panel['rx'].tolist(),panel['day'].tolist(),panel['x'].shape)
 iq=np.load(root/'observations/iq.npy',mmap_mode='r',allow_pickle=False);p.check_observations(meta,iq.shape,iq.dtype,bool(iq.flags.c_contiguous))
 with np.load(root/'observations/physics.npz',allow_pickle=False) as packed:physics={key:packed[key].copy() for key in packed.files}
 p.check_physics_headers({key:dict(shape=list(value.shape),kind=value.dtype.kind) for key,value in physics.items()})
 for key in ('state','lock'):
  value=physics[key]
  if not np.isfinite(value).all() or np.any((value<0)|(value>2)) or np.any(value!=value.astype(int)):raise ValueError('Cached state/lock index differs')
 # Original renderer used NaN for unavailable physical scalars.  Preserve this
 # missingness; it is neither a fabricated zero nor a sample-exclusion rule.
 statistics={key:dict(shape=list(value.shape),dtype=str(value.dtype),finite_count=int(np.isfinite(value).sum()),missing_nan_count=int(np.isnan(value).sum()),positive_infinity_count=int(np.isposinf(value).sum()),negative_infinity_count=int(np.isneginf(value).sum())) for key,value in physics.items()}
 engine=d.read(root/'observations/actual_engine.json')
 if set(engine.get('actual_configs',{}))!=set(d.SCENES) or engine.get('formal_test_changed') is not False:raise ValueError('Saved source engine metadata/scenes differ')
 return iq,dict(observation_root=str(root/'observations'),IQ_shape=list(iq.shape),IQ_dtype=str(iq.dtype),physics=statistics,physical_scalar_NaN_policy='original missing values preserved; no sample deletion',existing_source_cache_reused=True,new_IQ_generated=0,new_channel_draws=0,renderer_calls=0),engine

def unchanged(model,snapshot,torch):
 if model.training or any(module.training for module in model.modules()) or any(q.requires_grad or q.grad is not None for q in model.parameters()):raise ValueError('Readonly eval/autograd changed training/grad flags')
 if set(snapshot)!=set(model.state_dict()) or any(not torch.equal(value,model.state_dict()[key].detach().cpu()) for key,value in snapshot.items()):raise ValueError('Frozen source weights or buffers changed')
 head=model.id_backbone.encoder.core.id_backbone.cls_head
 p.check_head(head.weight.shape,float(head.scale),sum(q.numel() for q in model.parameters()),len(model.state_dict()))

def run(project,device='cuda:0'):
 project=Path(project)
 if not project.is_absolute():raise ValueError('Explicit absolute original project root required')
 out=d.paths(project)['output'];out.mkdir(exist_ok=False);start=time.perf_counter();stage='PROTOCOL_AND_RUNTIME'
 try:
  import numpy as np
  import torch
  from experiments.cvs_noise_shape_factorial43.quality import panel as original_panel
  from experiments.cvs_fullsource_residual_diag24 import observe,model as paths
  from experiments.cvs_equivariant_identity.precision import numerical_context
  config=p.check_config(d.config(project,device));helper_contract();torch.set_num_threads(2);stage='NOQUERY_SMOKE'
  d.write(out/'resolved_config.json',config);models={};snapshots={};provenance={}
  with numerical_context(d.FP32):
   # No source IQ cache or observations are opened before both real-CP smokes.
   for arm,label in zip(d.ARMS,d.LABELS):
    model,origin=student(project,arm,device,torch);snapshots[label]={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    with observe.capture(model) as captured,torch.no_grad():q,branches=observe.forward(model,torch.zeros(2,2,256,device=device),captured)
    if q['tx_logits'].shape!=(2,6) or not bool(torch.isfinite(q['tx_logits']).all()) or any(not bool(torch.isfinite(value).all()) for value in branches.values()):raise ValueError('Real-checkpoint default-head/five-branch no-query smoke failed')
    if model.id_backbone.encoder._full_eval_packets!=2 or model.id_backbone.encoder._full_train_packets:raise ValueError('Exactly two eval-only synthetic smoke packets per model')
    models[label]=model;provenance[label]=origin
   d.write(out/'checkpoint_noquery_smoke.json',dict(status='PASS',final_checkpoints=2,checkpoint_deserializations=sum(value['checkpoint_deserializations'] for value in provenance.values()),synthetic_NN=4,NN=4,source_IQ_before_smoke=0,query_reads=0,training_updates=0,head_fit_updates=0,provenance=provenance))
   print('SMOKE PASS P43 two fixed students/defaulthead/fivebranches;4 synthetic;0 query',flush=True)
   stage='EXISTING_SOURCE_INPUTS';modules=actual_modules();d.write(out/'actual_modules.json',modules)
   panel,roles=original_panel(torch);iq,input_summary,engine=read_cached_observations(project,panel,roles,np);d.write(out/'data_roles.json',roles);d.write(out/'input_summary.json',input_summary);d.write(out/'actual_engine.json',engine)
   print(json.dumps(dict(stage='EXISTING_SOURCE_ENGINE',actual_configs=engine['actual_configs'],renderer_calls=0),ensure_ascii=False),flush=True)
   results={}
   for label,model in models.items():
    stage='READOUT_'+label;dest=out/label;dest.mkdir();clean,satellite,branch=observe.predict(model,panel,iq,device,dest)
    p.check_prediction_shapes(clean.shape,satellite.shape,branch.shape,observe.NAMES,d.BRANCH_METRICS,bool(np.isfinite(clean).all() and np.isfinite(satellite).all() and np.isfinite(branch).all()))
    gradients=observe.gradients(model,panel,iq,device);p.check_gradients(gradients);unchanged(model,snapshots[label],torch);encoder=model.id_backbone.encoder;p.check_budget(encoder._full_eval_packets,encoder._full_train_packets)
    gates=paths.gates(model)
    # json.dumps rejects nonfinite existing gains without introducing thresholds.
    json.dumps(gates,allow_nan=False)
    result=dict(status='SOURCE_DIAGNOSED',label=label,arm='B00' if label=='NOAS30' else 'B01',row_id=provenance[label]['row_id'],run_id=d.RUN,checkpoint_state='model',EMA_weights_applied=False,default_head='original joint cosine6x160 scale30',NN=int(encoder._full_eval_packets),prediction_NN=d.PREDICTION_NN,gradient_forward_NN=d.GRADIENT_NN,synthetic_smoke_NN=d.SMOKE_NN,gradient_scope=config['gradient_scope'],training_updates=0,optimizer_updates=0,EMA_updates=0,head_fit_updates=0,source_unique_V=d.N,channel_draws=2,model_seed_count=1,source_only=True,U_iteration=False,U_truth_read=False,target_access=False,formal_scores_read=False,state_unchanged=True,param_grad_unchanged=True,training_gradients_measured=False,gates=gates,gradients=gradients,provenance=provenance[label],new_IQ_generated=0,new_channel_draws=0,automatic_training=False,automatic_formal=False,source_promotion=False)
    d.write(dest/'result.json',result);results[label]=result;d.write(out/'progress.json',dict(stage='MODEL_COMPLETE',completed=list(results),NN=sum(value['NN'] for value in results.values()),training_updates=0,target_access=False));print(json.dumps(dict(stage='MODEL_COMPLETE',label=label,NN=result['NN'])),flush=True)
   for label,model in models.items():unchanged(model,snapshots[label],torch)
   actual=sum(value['NN'] for value in results.values())
   if actual!=d.TOTAL_NN:raise ValueError('Two-row fixed diagnostic NN budget differs')
  complete=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',run_id=d.RUN,model_seed=d.SEED,model_seed_count=1,labels=list(d.LABELS),checkpoint_count=2,checkpoint_state='model',EMA_weights_applied=False,source_unique_V=d.N,channel_draws=2,scenes=list(d.SCENES),conditions=list(d.CONDITIONS),NN=actual,prediction_NN=2*d.PREDICTION_NN,gradient_forward_NN=2*d.GRADIENT_NN,synthetic_smoke_NN=4,training_updates=0,optimizer_updates=0,EMA_updates=0,head_fit_updates=0,new_IQ_generated=0,new_channel_draws=0,renderer_calls=0,U_iteration=False,U_truth_read=False,target_access=False,formal_scores_read=False,formal_test_changed=False,source_promotion=False,automatic_training=False,automatic_formal=False,automatic_budget_extension=False,state_unchanged=True,gradient_scope=config['gradient_scope'],wall_seconds=time.perf_counter()-start,goal_complete=False)
  d.write(out/'completion.json',complete);print(json.dumps(complete,ensure_ascii=False),flush=True);return complete
 except BaseException as exc:
  d.write(out/'failure.json',dict(status='TECHNICAL_FAILURE',stage=stage,type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),training_updates=0,head_fit_updates=0,target_access=False,automatic_retry=False,existing_outputs_preserved=True));raise
