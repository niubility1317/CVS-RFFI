"""Read only fixed source V; own checkpoint model/ema_model stay frozen."""
import json,os,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .protocol import check_state_pair,check_roles,check_ids,check_panel,check_panel_membership,decision
from .metrics import summarize

def unchanged(model,saved):
 if any(not torch.equal(v,model.state_dict()[k]) for k,v in saved.items()) or any(p.grad is not None for p in model.parameters()) or model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Source evaluation changed state/grad/freeze')

def load_models(rt,device,cost):
 from experiments.cvs_practical_budget50 import design as old,model as api
 models={};configs={};cp_orig=torch.load
 for arm,base in zip(d.ARMS,('NOAS','AS')):
  row=next(r for r in old.rows() if r['arm']==arm);c=old.config(row)
  if Path(c['output_root'])!=d.parent_source(rt,arm) or c['model_seed']!=d.SEED:raise ValueError('Only fixed own FINAL50 sources')
  seen=[];current=Path(c['output_root'])/'final_ssdg.pth'
  def counted(path,*a,**kw):
   p=Path(path);d.assert_source_path(p,[Path(rt['project'])/'runs'/old.RUN,Path(rt['project'])/'runs'/old.REFERENCE_RUN])
   cost['checkpoint_reads']+=1;cp=cp_orig(path,*a,**kw)
   if p==current:seen.append(cp)
   return cp
  torch.load=counted
  try:student=api.own_model(c,'cpu')
  finally:torch.load=cp_orig
  if len(seen)!=1:raise ValueError('Existing own_model must load its exact FINAL50 once')
  cp=seen[0];template=student.state_dict()
  check_state_pair(cp['model'],cp['ema_model'],template,lambda v:bool(torch.isfinite(v).all()))
  if sum(p.numel() for p in student.parameters())!=900722:raise ValueError('Existing F graph changed')
  ema=api.build(c,'cpu');ema.load_state_dict(cp['ema_model'],strict=True)
  for state,m in [('STUDENT',student),('EMA',ema)]:models[base+'_'+state]=m.to(device).eval().requires_grad_(False)
  configs[arm]=c
 return models,configs

def forward(model,x,cost):
 with torch.no_grad():z=model.forward_identity_only(x)['tx_logits']
 if z.shape!=(len(x),6) or not bool(torch.isfinite(z).all()):raise ValueError('Finite default-six-class source logits required')
 cost['identity_NN']+=len(x)
 return np.asarray(z.detach().cpu().tolist(),np.float32)

def source_context(config,device):
 from experiments.cvs_phase1_stack.runtime import installed
 from experiments.cvs_phase1_stack.fast_execution import configure
 from experiments.cvs_practical_budget50 import design as old
 from cvsrffi.xuc_fusion.native import role_ids_from_native
 with installed(config) as native:ctx=native._build_ssdg_wisig_data(configure(old.make_args(config,str(device))),device)
 expected=old.read(old.SOURCE)['role_ids'];actual=role_ids_from_native(ctx)
 counts={r:len(ctx[k].dataset) for r,k in [('L_s','train_loader'),('U_s','unlabeled_loader'),('V','val_loader')]}
 check_roles(actual,expected,ctx['named_test_loaders'],counts)
 if not getattr(ctx['unlabeled_loader'].dataset,'hide',False):raise ValueError('Original hidden U dataset required, never iterated')
 return ctx,expected

def full_v(models,ctx,expected,device,cost,out):
 from cvsrffi.game_tracking.data import opaque_id
 loader=ctx['val_loader'];dataset=loader.dataset
 if loader.drop_last or getattr(dataset,'hide',False) or len(dataset)!=27000:raise ValueError('Full visible source V only')
 # Index-to-opaque mapping does not read any U example or hidden U label.
 if not hasattr(dataset,'selected') or len(dataset.selected)!=27000:raise ValueError('Original native V subset base indices required')
 by_physical={tuple(int(getattr(v,k)) for k in ('eq_i','rx_i','day_i','sig_i'))+(int(base),):opaque_id(v) for base,v in zip(dataset.selected,dataset.index)}
 if len(by_physical)!=27000:raise ValueError('Unique V physical identities required')
 result={key:[] for key in ('ids','physical_ids','TX','RX','day','eq',*d.LABELS)}
 for batch,(x,y,domain,meta) in enumerate(loader):
  values={k:meta[k].tolist() if hasattr(meta[k],'tolist') else list(meta[k]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')}
  physical=[tuple(int(values[k][i]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')) for i in range(len(x))]
  ids=[by_physical[pid] for pid in physical]
  if x.shape!=(len(y),2,256) or not bool(torch.isfinite(x).all()) or any(int(v) not in range(6) for v in y):raise ValueError('Original finite V256/six-class data required')
  x=x.to(device);cost['fullV_raw_IQ_reads']+=len(x)
  for label,m in models.items():result[label].append(forward(m,x,cost))
  for k,v in [('ids',ids),('physical_ids',physical),('TX',y.tolist()),('RX',values['rx_i']),('day',values['day_i']),('eq',values['eq_i'])]:result[k].extend(v)
  if batch%20==0:d.write(out/'progress.json',dict(stage='FULL_V',packets=len(result['ids']),identity_NN=cost['identity_NN'],U_iterator_calls=0,target_access=False));print('FULL_V '+str(len(result['ids'])),flush=True)
 check_ids(result['ids'],result['RX'],result['day'],result['eq'],expected['V'],27000)
 for k in result:result[k]=np.concatenate(result[k]) if k in d.LABELS else np.asarray(result[k])
 return result

def panel_v(models,rt,full,device,cost,out):
 from experiments.cvs_practical_budget50 import design as old
 from experiments.cvs_practical_budget50.quality import panel
 from experiments.cvs_practical_budget50.channel import tensor_iq
 data,roles=panel(torch)
 result={k:np.asarray(data[n].tolist() if hasattr(data[n],'tolist') else data[n]) for k,n in [('ids','ids'),('TX','y'),('RX','rx'),('day','day')]}
 check_panel_membership(result,full)
 cost['panel_clean_IQ_reads']+=1800
 for arm,base in zip(d.ARMS,('NOAS','AS')):
  path=d.assert_source_path(d.panel_predictions(rt,arm),[d.parent_source(rt,arm).parent/'source_qualification'])
  with np.load(path,allow_pickle=False) as f:oldpred={k:f[k].copy() for k in f.files}
  check_panel(oldpred,dict(result,clean=oldpred['clean'],satellite=oldpred['satellite']))
  complete=d.read(path.parent/'complete.json')
  if complete['target_access'] or complete.get('formal_scores_read') or complete['NN']!=12600 or complete.get('training_updates',0) or complete.get('head_fit_updates',0):raise ValueError('Original frozen source-only student panel required')
  result[base+'_STUDENT']=dict(clean=oldpred['clean'],satellite=oldpred['satellite'])
 obs_path=d.assert_source_path(old.PANEL/'observations/iq.npy',[old.PANEL/'observations'])
 obs=np.load(obs_path,mmap_mode='r',allow_pickle=False)
 if obs.shape[:4]!=(2,3,1,1800):raise ValueError('Original two-draw three-scene fixed source observations required')
 for base in ('NOAS','AS'):result[base+'_EMA']=dict(clean=np.empty((1800,6),np.float32),satellite=np.empty((2,3,1800,6),np.float32))
 for i in range(0,1800,90):
  x=data['x'][i:i+90].to(device)
  for base in ('NOAS','AS'):result[base+'_EMA']['clean'][i:i+90]=forward(models[base+'_EMA'],x,cost)
 for draw in range(2):
  for si,scene in enumerate(d.SCENES):
   for i in range(0,1800,90):
    observed=obs[draw,si,0,i:i+90];cost['panel_observation_IQ_reads']+=len(observed);x=tensor_iq(observed,device)
    for base in ('NOAS','AS'):result[base+'_EMA']['satellite'][draw,si,i:i+90]=forward(models[base+'_EMA'],x,cost)
   print('PANEL bothEMA '+str(draw)+' '+scene,flush=True)
   d.write(out/'progress.json',dict(stage='PANEL',readout='bothEMA',draw=draw,scene=scene,identity_NN=cost['identity_NN'],target_access=False))
 return result

def run():
 rt=d.runtime();out=d.output(rt)
 if out.exists():raise FileExistsError('Immutable own source output exists; reconcile original owner, never overwrite/relaunch')
 out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();torch.set_num_threads(2)
 device=torch.device('cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES','') else 'cpu')
 cost=dict(identity_NN=0,checkpoint_reads=0,fullV_raw_IQ_reads=0,panel_clean_IQ_reads=0,panel_observation_IQ_reads=0,
  optimizer_updates=0,head_fit_updates=0,EMA_updates=0,U_iterator_calls=0,U_truth_read=False,target_IQ_reads=0,target_truth_reads=0)
 d.write(out/'runtime.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,device=str(device),owner=d.OWNER,run_id=d.RUN,release_commit=(d.ROOT/'release_commit.txt').read_text().strip()))
 try:
  from experiments.cvs_practical_budget50 import design as old
  from experiments.cvs_equivariant_identity.precision import numerical_context
  with numerical_context(old.FULL_FP32_POLICY):
   models,configs=load_models(rt,device,cost);before={k:{n:v.clone() for n,v in m.state_dict().items()} for k,m in models.items()}
   for label,m in models.items():
    forward(m,torch.zeros(1,2,256,device=device),cost);unchanged(m,before[label])
   d.write(out/'smoke.json',dict(status='PASS',actual_NN=4,checkpoint_count=2,state_keys=['model','ema_model'],readouts=list(d.LABELS),state_entries_each=206,default_head=[6,160],IQ_reads=0,query_reads=0,optimizer_updates=0,all_frozen=True))
   print('REAL_OWN_FINAL50_STUDENT_EMA_NO_QUERY_SMOKE PASS4NN;continue source',flush=True)
   ctx,expected=source_context(configs['B00'],device)
   full=full_v(models,ctx,expected,device,cost,out)
   panel=panel_v(models,rt,full,device,cost,out)
   for label,m in models.items():
    unchanged(m,before[label])
    expectedNN=27001+(12600 if label.endswith('_EMA') else 0)
    if m.id_backbone.encoder._full_eval_packets!=expectedNN:raise ValueError('Actual native eval NN differs: '+label)
   if cost['identity_NN']!=d.EXPECTED_NN:raise ValueError('Complete fixed source inference budget differs')
   arrays={'fullV_'+k:v for k,v in full.items()}
   arrays.update({'panel_'+k:v for k,v in panel.items() if k not in d.LABELS})
   for label in d.LABELS:
    arrays['panel_'+label+'_clean']=panel[label]['clean'];arrays['panel_'+label+'_satellite']=panel[label]['satellite']
   np.savez_compressed(out/'predictions.npz',**arrays)
   summary=summarize(full,panel,out);qualification=decision(summary['fullV'],summary['panel'])
   original={}
   for arm,base in zip(d.ARMS,('NOAS','AS')):
    native=old.read(d.parent_source(rt,arm)/'completion.json')['final_source_metrics'];fresh=summary['fullV'][base+'_STUDENT']
    if native['source_val_count']!=27000:raise ValueError('Original native fullV count differs')
    original[base]=dict(native_accuracy=native['source_val_accuracy'],fresh_accuracy=fresh['accuracy'],accuracy_delta_pp=100*(fresh['accuracy']-native['source_val_accuracy']),exact_hit_count_match=round(native['source_val_accuracy']*27000)==fresh['hits'])
   cost.update(wall_seconds=time.perf_counter()-start,new_model_epochs=0,reused_parent_model_epochs=100,reused_student_panel_NN=25200,historical_training_recharged=False,fullV_student_NN=54000,fullV_EMA_NN=54000,new_EMA_panel_NN=25200,smoke_NN=4,parameter_buffer_changes=0)
   summary.update(status='SOURCE_ANALYZED',run_id=d.RUN,model_seed=d.SEED,model_seed_count=1,readouts=list(d.LABELS),qualification=qualification,native_student_fullV_recount=original,cost=cost,source_fullV_count=27000,panel_unique_V=1800,channel_draws=2,source_scenes=list(d.SCENES),source_high_urban_available=False,source_panel_fc_hz=2450000000,training_formal_fc_hz=2462000000,target_access=False,formal_scores_read=False,automatic_formal=False,goal_complete=False)
   d.write(out/'summary.json',summary)
   for label,arm,state in [('NOAS_STUDENT','B00','STUDENT'),('NOAS_EMA','B00','EMA'),('AS_STUDENT','B01','STUDENT'),('AS_EMA','B01','EMA')]:
    row=arm+'_'+state+'-s'+str(d.SEED)
    d.write(out/'readouts'/row/'summary.json',dict(status='SOURCE_ANALYZED',row_id=row,readout=label,source_root=str(out),shared_predictions='../../predictions.npz',fullV=summary['fullV'][label],panel=summary['panel'][label],state_key='model' if state=='STUDENT' else 'ema_model',new_NN=27001+(12600 if state=='EMA' else 0),training_updates=0,target_access=False))
   d.write(out/'costs.json',cost)
   d.write(out/'complete.json',dict(status='SOURCE_ANALYZED',run_id=d.RUN,source_qualified=qualification['passed'],source_only=True,model_seed_count=1,readout_count=4,identity_NN=cost['identity_NN'],all_parameters_frozen=True,parameter_buffer_changes=0,optimizer_updates=0,EMA_updates=0,U_truth_read=False,target_access=False,formal_scores_read=False,automatic_formal=False,scientific_failure_preserves_full_output=True))
   print('SOURCE_ANALYZED '+json.dumps(qualification),flush=True)
 except BaseException as error:
  cost['wall_seconds']=time.perf_counter()-start;d.write(out/'failure.json',dict(status='TECHNICAL_FAILURE',error=repr(error),traceback=traceback.format_exc(),cost=cost,preserve_outputs=True,no_automatic_retry=True,target_access=False));raise
