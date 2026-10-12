import argparse,csv,gzip,json,os,sys,time
from pathlib import Path
import torch
from . import design as d
from .model import build,groups
from .objective import objective
from .warmup import head_warmup
from experiments.cvs_report_ssl.sampling import DomainStream
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_stage2_decouple.common import clean
from experiments.cvs_stage2_rx_risk.source import validation,component_gradients,state_bytes
from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
from .calibration import begin_calibration,advance_calibration,require_calibration,scheduled_after


def publish_calibration_ready(out,resolved,calibration,completed_updates):
 if (calibration['status']!='FIXED_CALIBRATION_COMPLETE'
     or calibration['after_identity_updates']!=completed_updates):raise ValueError('Calibration-ready event has the wrong update boundary')
 final=dict(resolved,fishr_calibration=calibration)
 event=dict(schema='calibration_ready_v1',event='CALIBRATION_READY',after_identity_updates=completed_updates,
  calibration=calibration,resolved=final)
 path=Path(out)/'calibration_ready.json'
 with path.open('x',encoding='utf-8') as stream:json.dump(clean(event),stream,allow_nan=False);stream.write('\n')
 d.write(Path(out)/'resolved_config_final.json',clean(final));d.write(Path(out)/'resolved_config.json',clean(final))
 print('CALIBRATION_READY '+json.dumps(clean(event)),flush=True)
 return final,event

def load_source(c):
 from experiments.cvs_phase1_stack.recover import verify_contract
 d.validate(c);prepared=Path(c['preparation']);done=d.read(prepared/'preparation_complete.json')
 if done['target_access'] or done['identity_updates'] or done['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE':raise ValueError('Raw IQ preparation differs')
 packed=torch.load(prepared/'source_cache.pt',map_location='cpu',weights_only=False)
 verify_contract(packed['contract'],d.read(d.SOURCE));verify_contract(done['source_contract'],packed['contract'])
 keep=('x','y','rx','day','ids');L={k:packed['labeled'][k] for k in keep};V={k:packed['validation'][k] for k in keep};contract=packed['contract']
 for role,cache,n in [('L_s',L,6300),('V',V,27000)]:
  if len(cache['ids'])!=n or len(set(cache['ids']))!=n or sorted(cache['ids'])!=sorted(contract['role_ids'][role]):raise ValueError('Source physical IDs differ')
 if set(L['ids'])&set(V['ids']):raise ValueError('Overlapping source roles')
 U=None
 if c['unlabeled']:
  U=torch.load(c['unlabeled_cache'],map_location='cpu',weights_only=False);validate_unlabeled(U,contract['role_ids']['U_s'])
  if len(U['ids'])!=56700 or set(U['ids'])&(set(L['ids'])|set(V['ids'])):raise ValueError('U source role differs')
 return L,U,V,contract

def expected_counts(c):
 n=c['updates'];active=n-c['semi']['warmup_updates'];packets=active*c['batch_size']
 aux=(n//c['multilevel']['interval']-c['semi']['warmup_updates']//c['multilevel']['interval']) if c['multilevel']['enabled'] else 0
 return dict(identity_updates=n,L_packets=n*c['batch_size'],U_packets=packets if c['unlabeled'] else 0,
  weak_forwards=active if c['unlabeled'] else 0,physical_L=0,physical_U=0,feature_U=0,
  margin_U=0,EMA_updates=0,satellite_calls=0,auxiliary_steps=aux,
  auxiliary_L=aux*c['multilevel']['batch_size'],auxiliary_forwards=aux*2,
  fishr_steps=aux if c['multilevel']['fishr_mode']!='none' else 0,
  mixstyle_steps=active if c['multilevel']['mixstyle_l'] else 0,
  style_L_packets=packets if c['multilevel']['mixstyle_l'] else 0,
  style_search_forwards=active*5 if c['multilevel']['style_selection'] in ('random','hard') else 0,
  style_search_L=packets*5 if c['multilevel']['style_selection'] in ('random','hard') else 0,
  mixstyle_U_steps=active if c['multilevel']['mixstyle_u'] else 0)

def train(c,device='cuda:0'):
 d.validate(c);out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();torch.set_num_threads(2)
 L,U,V,contract=load_source(c);seed_all(c['model_seed']);cuda=str(device).startswith('cuda')
 from .peer_cache import load as load_peer
 from .provenance import validate as validate_teachers,expected_sources,student_initialization
 bank=load_peer(c,U) if c['teacher']['source']=='peer' else None
 teachers=validate_teachers(c,bank['provenance'] if bank is not None else {})
 sources=expected_sources(c)
 with numerical_context(d.FULL_FP32_POLICY):
  model=build(c,device)
  parent_provenance=dict(model._parent_provenance)
  if parent_provenance!=student_initialization(c):raise ValueError('Scratch student initialization metadata differs')
  provenance=dict(parent_provenance)
  provenance.update(scratch_only=True,scratch_scope='student_identity_only',parent_checkpoint=None,checkpoint_sources=sources,
   ancestors=sources,teacher_provenance=teachers,initialization_checkpoint_sources=[],
   parent_training_epochs=0,incremental_epochs=20,total_epochs=20,target_contact=False,source_roles='EXACT_MATCH',seed=c['model_seed'],
   reference_E20_loaded=False,source_IQ_cache_only=True,auxiliary_inheritance=False,old_features_or_auxiliary_consumed=False,
   parent_provenance=parent_provenance,source_probability_cache_used=bank is not None,
   whole_method_pretrained_teachers=bool(teachers),student_initialization='random native identity, no checkpoint loaded')
  d.write(out/'initialization.json',clean(provenance));d.write(out/'source_contract.json',clean(contract))
  warmup=head_warmup(c,model,L,device,out)
  calibration=begin_calibration(c,model,L,device,out)
  pg,names=groups(model,c)
  # A fresh main optimizer starts the random student; no optimizer or head state is inherited.
  optimizer=torch.optim.AdamW(pg,weight_decay=c['optimizer']['weight_decay'])
  seed_all(c['model_seed']);lstream=DomainStream(L['rx'],L['day'],c['model_seed']+101)
  ustream=DomainStream(U['rx'],U['day'],c['model_seed']+901337) if U is not None else None
  initial={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
  resolved=dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),
   hardware=torch.cuda.get_device_name(device) if cuda else 'CPU',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),backend_flags=actual_flags(),
   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_parameters=sum(p.numel() for p in model.parameters()),parameter_groups=names,
   parent_training_epochs=0,incremental_epochs=20,total_epochs=20,head_warmup=warmup,fishr_calibration=calibration,
   execution=dict(scratch=True,scratch_scope='student_identity_only',E20_weights_loaded=False,E200_weights_loaded=False,U_iterator=U is not None,U_truth_hidden=True,EMA=False,satellite_generator=False,
    auxiliary_neural_networks=c['multilevel']['statistic']=='probe',sensitivity_probes_trainable=False,physical_bank_fit_L_only=False,physical_actions=False,
    auxiliary_inheritance=False,main_optimizer_reset=True,multilevel=c['multilevel'],
    auxiliary_CE_to_identity=c['multilevel']['aux_ce_to_backbone'],Fishr_levels=c['multilevel']['fishr_levels'],Fishr_fixed_denominator=2,
    teacher_source=c['teacher']['source'],U_selection=c['teacher']['selection'],teacher_seeds=c['teacher']['seeds'],
    teacher_model_forward_per_step=False,live_weak_forward_role='student diagnostic' if bank is not None else 'self teacher' if U is not None else 'none'))
  for name,val in [('initial_resolved_config.json',resolved),('resolved_config.json',resolved),('initialization.json',provenance),('source_contract.json',contract)]:d.write(out/name,clean(val))
  print('RESOLVED '+json.dumps(clean(resolved)),flush=True)
  calibration_events=[]
  if calibration['status']=='FIXED_CALIBRATION_COMPLETE':
   resolved,event=publish_calibration_ready(out,resolved,calibration,0);calibration_events.append(event)
  counts={k:0 for k in expected_counts(c)};counts['mixed_U_packets']=0;counts['mixed_L_packets']=0;visited=set();epochs=[];last_validation=None
  main_started=time.perf_counter()
  if cuda:torch.cuda.reset_peak_memory_stats(device)
  with gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as log:
   for epoch in range(1,21):
    model.train();sums={};epstart=time.perf_counter();audit=None
    for group in pg:group['lr']=d.learning_rate(c,epoch)
    for local in range(1,223):
     step=(epoch-1)*222+local;start=time.perf_counter();li,_=lstream.batch(128);ui=None
     if U is not None and step>c['semi']['warmup_updates']:ui,_=ustream.batch(128);visited.update(ui.tolist())
     require_calibration(c,model,calibration,step)
     optimizer.zero_grad(set_to_none=True);loss,parts,weighted,detail=objective(c,model,bank,L,li,U,ui,step,device)
     if not torch.isfinite(loss):raise FloatingPointError('Nonfinite scratch20 loss')
     audit=component_gradients(weighted,pg,model) if step%444==0 else None
     if audit is not None and not c['multilevel']['aux_ce_to_backbone']:
      aux=audit['weighted_dense_gradient_norms']['aux_ce']
      if any(value!=0. for group,value in aux.items() if group!='multilevel_heads'):
       raise ValueError('Detached auxiliary CE unexpectedly updates identity parameters')
     loss.backward();params=[p for group in pg for p in group['params']]
     norm=torch.nn.utils.clip_grad_norm_(params,float('inf'),error_if_nonfinite=True);optimizer.step()
     counts['identity_updates']+=1;counts['L_packets']+=128
     for k in ('U_packets','weak_forwards','physical_L','physical_U','feature_U','margin_U','auxiliary_steps','auxiliary_L','auxiliary_forwards','fishr_steps','mixstyle_steps','mixstyle_U_steps','mixed_U_packets','style_L_packets','mixed_L_packets','style_search_forwards','style_search_L'):counts[k]+=detail[k]
     if cuda:torch.cuda.synchronize(device)
     scalars={k:float(v.detach()) for k,v in parts.items()};scalars.update({'weighted_'+k:float(v.detach()) for k,v in weighted.items()})
     scalars.update(total_loss=float(loss.detach()),gradient_norm=float(norm),lr=pg[0]['lr'],L_accuracy=detail['L_accuracy'],accepted_U=detail['accepted_U'],step_seconds=time.perf_counter()-start)
     for k,v in scalars.items():sums[k]=sums.get(k,0.)+v
     log.write(json.dumps(clean(dict(step=step,epoch=epoch,metrics=scalars,details=detail,gradient_audit=audit)),allow_nan=False)+'\n')
     # The completed update is counted/logged before calibration. No next L/U
     # stream batch is sampled until the unique readiness event is recorded.
     calibration,became_ready=advance_calibration(c,model,L,device,out,calibration,counts['identity_updates'])
     if became_ready:
      resolved,event=publish_calibration_ready(out,resolved,calibration,counts['identity_updates']);calibration_events.append(event)
    log.flush();v=validation(model,V,device) if epoch in c['validation']['epochs'] else None
    if v is not None:last_validation=v
    current=dict(model.named_parameters());movement={}
    for key,keys in names.items():
     keys=[n for n in keys if not n.endswith(('low_hz_','band_hz_'))]
     diff=sum((current[n].detach()-initial[n]).square().sum() for n in keys).sqrt();scale=sum(initial[n].square().sum() for n in keys).sqrt()
     movement[key]=dict(delta_l2=float(diff),relative_l2=float(diff/scale.clamp_min(1e-10)))
    row=clean(dict(epoch=epoch,total_epoch=epoch,optimizer_steps=step,**{k:v/222 for k,v in sums.items()},validation=v,counts=dict(counts),last_gradient_audit=audit,weight_movement=movement,distinct_U_visited=len(visited),U_full_passes=ustream.pass_index-1 if ustream else 0,epoch_seconds=time.perf_counter()-epstart))
    epochs.append(row);(out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in epochs),encoding='utf-8')
    flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in epochs]
    with (out/'epoch_metrics.csv').open('w',encoding='utf-8',newline='') as stream:writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print('EPOCH '+json.dumps(row),flush=True)
  expected=expected_counts(c)
  require_calibration(c,model,calibration,4440)
  if len(calibration_events)!=1 or calibration_events[0]['after_identity_updates']!=scheduled_after(c):raise ValueError('Incomplete or repeated calibration lifecycle')
  if any(counts[k]!=value for k,value in expected.items()) or (U is not None and len(visited)!=56700):raise ValueError('Incomplete identity or mechanism budget')
  if not 0<=counts['mixed_U_packets']<=counts['mixstyle_U_steps']*c['batch_size']*len(c['multilevel']['levels']):raise ValueError('Invalid measured mixed U packet-layer count')
  if not 0<=counts['mixed_L_packets']<=counts['style_L_packets']*len(c['multilevel']['style_levels']):raise ValueError('Invalid L MixStyle packet-layer count')
  final=out/'final_ssdg.pth'
  torch.save(dict(model=model.state_dict(),baseline_args=model._stage2_baseline_args,config=c,run_id=c['run_id'],candidate_id=c['row_id'],epoch=20,total_epochs=20,
   parent_training_epochs=0,incremental_epochs=20,optimizer_steps=4440,checkpoint_selection='final_only',parent_checkpoint=c['parent_checkpoint'],
   parent_provenance=parent_provenance,teacher_provenance=teachers,checkpoint_sources=sources,optimizer=optimizer.state_dict(),target_access=False,
   fixed_sensitivity_state=model.multilevel.export_state(),model_seed=c['model_seed'],scratch_only=True,scratch_scope='student_identity_only',reference_E20_loaded=False),final)
  torch.save(dict(L_sampler=lstream.state_dict(),U_sampler=ustream.state_dict() if ustream else None,teacher=None,
   multilevel=model.multilevel.state_dict() if c['multilevel']['enabled'] else None,head_warmup=warmup,fishr_calibration=calibration,fixed_sensitivity_state=model.multilevel.export_state(),
   physical_actions=False,auxiliary_inheritance=False,selected_for_test=False),out/'auxiliary_state.pt')
  main_elapsed=time.perf_counter()-main_started
  delayed_calibration_seconds=calibration['seconds'] if scheduled_after(c) else 0.
  resources=dict(identity_training_and_source_diagnostics_seconds=main_elapsed-delayed_calibration_seconds,
   identity_training_and_source_diagnostics_including_delayed_calibration_seconds=main_elapsed,
   calibration_inside_main_timer_seconds=delayed_calibration_seconds,calibration_ready_events=len(calibration_events),
   calibration_after_identity_updates=calibration['after_identity_updates'],
   incremental_total_seconds_including_warmup=time.perf_counter()-started,head_warmup=warmup,fishr_calibration=calibration,fixed_sensitivity_state=model.multilevel.export_state(),
   peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,
   trainable_parameters=resolved['trainable_parameters'],total_parameters=resolved['total_parameters'],auxiliary_trainable_parameters=sum(p.numel() for p in model.multilevel.parameters()) if c['multilevel']['enabled'] else 0,teacher_parameters=0,
   optimizer_state_tensor_bytes=state_bytes(optimizer.state_dict()),checkpoint_bytes=final.stat().st_size,auxiliary_saved_bytes=(out/'auxiliary_state.pt').stat().st_size,
   parent_training_epochs=0,incremental_epochs=20,total_epochs=20,parent_training_cost_reused=False,scratch_only=True,scratch_scope='student_identity_only',
   student_initialization_checkpoint_count=0,whole_method_pretrained_teachers=bool(teachers),
   frozen_teacher_model_count=len(teachers),teacher_training_epochs_reused=20*len(teachers),historical_model_epochs_including_teachers=20+20*len(teachers),
   teacher_probability_bank_bytes=state_bytes(bank) if bank is not None else 0,teacher_cache_resources_ref=str(Path(c['teacher']['cache_root'])/'completion.json') if bank is not None else None,
   teacher_scope='teacher_parameters=0 means no teacher models resident during student training; extra original E20 training and shared cache forward costs reported separately',
   mixed_U_packets_unit='U MixStyle disabled',mixed_L_packets_unit='actual changed L packet-layer occurrences; not distinct packets',**counts)
  done=dict(status='SOURCE_TRAINED',epoch=20,total_epochs=20,optimizer_steps=4440,logged_steps=4440,config=c,checkpoint=str(final),checkpoint_selection='final_only',
   target_access=False,target_evaluated=False,parent_checkpoint=c['parent_checkpoint'],checkpoint_sources=sources,teacher_provenance=teachers,final_source_metrics=last_validation,
   mechanism_execution=counts,initialization=provenance,resources=resources,distinct_U_visited=len(visited))
  d.write(out/'completion.json',clean(done));d.write(out.parent/'source_frozen.json',dict(status='SOURCE_FROZEN',run_id=c['run_id'],row_id=c['row_id'],config=c,
   checkpoint=str(final),epoch=20,total_epochs=20,parent_checkpoint=c['parent_checkpoint'],optimizer_steps=4440,target_access=False,target_scores_consumed=False,selection='final_only'))
  print('SOURCE_TRAINED '+json.dumps(clean(done)),flush=True)
 return done

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();train(d.read(a.config))
