import argparse,csv,gzip,json,os,sys,time
from pathlib import Path
import torch
from . import design as d
from .model import build,groups
from .objective import objective
from .provenance import initialization
from .response import Bank
from experiments.cvs_nocfo_engineering.peer_cache import load as load_peer
from experiments.cvs_nocfo_engineering.teacher import teacher_config
from experiments.cvs_nocfo_foundation.source import load_source as foundation_source
from experiments.cvs_report_ssl.sampling import DomainStream
from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_stage2_decouple.common import clean
from experiments.cvs_stage2_rx_risk.source import validation,component_gradients,state_bytes
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
def load_source(c):
 d.validate(c);L,unused,V,contract=foundation_source(teacher_config(c['model_seed']));assert unused is None
 L['role']='L_s';U=None
 if c['unlabeled']:
  U=torch.load(c['unlabeled_cache'],map_location='cpu',weights_only=False);validate_unlabeled(U,contract['role_ids']['U_s'])
  if len(U['ids'])!=56700 or set(U['ids'])&(set(L['ids'])|set(V['ids'])):raise ValueError('Legal U physical pool differs')
 return L,U,V,contract
def expected_counts(c):
 n=4440;b=128;rc=c['response'];L_extra=rc['extra_eval'] or 'L_s' in rc['roles'] or rc['IQ_action'];U_extra=rc['extra_eval'] or 'U_s' in rc['roles'] or rc['IQ_action']
 return dict(identity_updates=n,L_packets=n*b,U_packets=n*b if c['unlabeled'] else 0,forward_calls=n*(3 if c['unlabeled'] else 1)+n*int(L_extra),extra_forward_calls=n*int(L_extra),extra_L_packets=n*b if L_extra else 0,extra_U_packets=n*b if U_extra else 0,response_readout_calls=n*4*len(rc['roles']),response_targets_L=n*b if 'L_s' in rc['roles'] else 0,response_targets_U=n*b if 'U_s' in rc['roles'] else 0,teacher_forwards=0,EMA_updates=0,satellite_calls=0,auxiliary_optimizer_steps=0)
def train(c,device='cuda:0'):
 d.validate(c);out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();torch.set_num_threads(2);L,U,V,contract=load_source(c);seed_all(c['model_seed']);cuda=str(device).startswith('cuda')
 with numerical_context(d.FULL_FP32_POLICY):
  model=build(c,device);peer=load_peer(c,U) if c['unlabeled'] else None;response=Bank(c,L,U,device) if c['response']['enabled'] else None
  init=initialization(c,model._parent_provenance,peer['provenance'] if peer else {},response.proof if response else None)
  pg,names=groups(model,c);optimizer=torch.optim.AdamW(pg,weight_decay=.0001);seed_all(c['model_seed']);initial={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
  if optimizer.state:raise ValueError('Continuation optimizer must be fresh')
  lstream=DomainStream(L['rx'],L['day'],c['model_seed']+101);ustream=DomainStream(U['rx'],U['day'],c['model_seed']+901337) if U is not None else None
  frozen_before={key:{n:t.detach().clone() for n,t in reader.state_dict().items()} for key,reader in response.readers.items()} if response else None
  resolved=dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),hardware=torch.cuda.get_device_name(device) if cuda else 'CPU',CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),backend_flags=actual_flags(),initialization=init,parameter_groups=names,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_parameters=sum(p.numel() for p in model.parameters()),execution=dict(E20_weights_loaded=True,parent_training_epochs=20,total_epochs=40,optimizer_reset=True,U_iterator=U is not None,U_truth_hidden=True,EMA=False,satellite_generator=False,auxiliary_optimizer=False,response_readers=c['response']['tasks'],response_parameters_frozen=True,response_input_gradient=c['response']['enabled'],teacher_source=c['teacher']['source'],teacher_forwards=0,explicit_CFO_identity_input=False,identity_input_policy=model._identity_input_policy))
  for name,value in [('resolved_config.json',resolved),('initialization.json',init),('source_contract.json',contract)]:d.write(out/name,clean(value))
  print('RESOLVED '+json.dumps(clean(resolved)),flush=True);counts={k:0 for k in expected_counts(c)};visited=set();epochs=[];last_validation=None
  if cuda:torch.cuda.reset_peak_memory_stats(device)
  main_start=time.perf_counter()
  with gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as logs:
   for epoch in range(1,21):
    model.train();sums={};epstart=time.perf_counter();audit=None
    for group in pg:group['lr']=d.learning_rate(c,epoch)
    for local in range(1,223):
     step=(epoch-1)*222+local;ts=time.perf_counter();li,_=lstream.batch(128);ui=None
     if ustream is not None:ui,_=ustream.batch(128);visited.update(ui.tolist())
     optimizer.zero_grad(set_to_none=True);loss,parts,weighted,details=objective(c,model,peer,response,L,li,U,ui,step,device)
     if not torch.isfinite(loss):raise FloatingPointError('Nonfinite identity/response loss')
     audit=component_gradients(weighted,pg,model) if step%444==0 else None
     loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for group in pg for p in group['params']],c['optimizer']['gradient_clip'])
     if not torch.isfinite(norm):raise FloatingPointError('Nonfinite identity gradient')
     if response and any(p.grad is not None for reader in response.readers.values() for p in reader.parameters()):raise ValueError('Frozen D received optimizer gradient')
     optimizer.step();counts['identity_updates']+=1
     for key in counts:
      if key!='identity_updates':counts[key]+=details[key]
     if cuda:torch.cuda.synchronize(device)
     metrics={k:float(v.detach()) for k,v in parts.items()};metrics.update({'weighted_'+k:float(v.detach()) for k,v in weighted.items()});metrics.update(total_loss=float(loss.detach()),gradient_norm=float(norm),lr=pg[0]['lr'],L_accuracy=details['L_accuracy'],L_margin=details['L_margin'],step_seconds=time.perf_counter()-ts)
     metrics.update({f'response_{role}_{task}':value['loss'] for role,tasks in details['role_task_losses'].items() for task,value in tasks.items()})
     for key,value in metrics.items():sums[key]=sums.get(key,0.)+value
     logs.write(json.dumps(clean(dict(step=step,epoch=epoch,metrics=metrics,details=details,gradient_audit=audit)),allow_nan=False)+'\n')
    logs.flush();val=validation(model,V,device) if epoch in (5,10,15,20) else None
    if val is not None:val.pop('source_selection_score',None)
    if val is not None:last_validation=val
    current=dict(model.named_parameters());movement={};movement_groups=dict(names)
    for name in initial:movement_groups.setdefault('module:'+name.split('.')[2],[]).append(name)
    for key,keys in movement_groups.items():
     delta=sum((current[n].detach()-initial[n]).square().sum() for n in keys).sqrt();scale=sum(initial[n].square().sum() for n in keys).sqrt();movement[key]=dict(delta_l2=float(delta),relative_l2=float(delta/scale.clamp_min(1e-10)))
    row=clean(dict(epoch=epoch,total_epoch=epoch+20,optimizer_steps=step,**{k:v/222 for k,v in sums.items()},validation=val,counts=dict(counts),last_gradient_audit=audit,weight_movement=movement,distinct_U_visited=len(visited),U_full_passes=ustream.pass_index-1 if ustream else 0,epoch_seconds=time.perf_counter()-epstart))
    epochs.append(row);(out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in epochs),encoding='utf-8');flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in epochs]
    with (out/'epoch_metrics.csv').open('w',encoding='utf-8',newline='') as stream:writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print('EPOCH '+json.dumps(row),flush=True)
  if counts!=expected_counts(c) or (U is not None and len(visited)!=56700):raise ValueError('Incomplete fixed update/role/response budget')
  if response and any(not torch.equal(t,frozen_before[key][name]) for key,reader in response.readers.items() for name,t in reader.state_dict().items()):raise ValueError('Frozen response state changed')
  final=out/'final_ssdg.pth';payload=dict(init);payload.update(model=model.state_dict(),baseline_args=model._stage2_baseline_args,config=c,model_seed=c['model_seed'],epoch=20,total_epochs=40,parent_training_epochs=20,incremental_epochs=20,optimizer_steps=4440,lifetime_optimizer_steps=8880,run_id=d.RUN,candidate_id=c['row_id'],parent_checkpoint=c['parent_checkpoint'],checkpoint_selection='final_only',target_access=False,scratch_only=False,reference_E20_loaded=True,optimizer=optimizer.state_dict())
  torch.save(payload,final)
  try:
   import resource
   rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
  except ImportError:rss=None
  resources=dict(total_seconds=time.perf_counter()-started,main_seconds=time.perf_counter()-main_start,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0,peak_process_rss_bytes=rss,RSS_scope='Linux lifetime maximum' if rss is not None else 'N/A local platform',optimizer_state_bytes=state_bytes(optimizer.state_dict()),teacher_forward_calls=0,response_trainable_parameters=0,response_parameters_buffers_unchanged=True,identity_trainable_parameters=resolved['trainable_parameters'],shared_peer_cache_reused=c['unlabeled'],shared_response_cache_reused=c['response']['enabled'],source_response_cache_file_bytes=(Path(c['response']['source_root'])/'source_observations.pt').stat().st_size if response else 0,pretraining_cost='parentCE20;other2CE peers40 if enabled;finite source diagnostic costs separate')
  done=dict(status='SOURCE_TRAINED',config=c,epoch=20,total_epochs=40,optimizer_steps=4440,logged_steps=4440,parent_checkpoint=c['parent_checkpoint'],checkpoint=str(final),checkpoint_selection='final_only',target_access=False,target_evaluated=False,initialization=init,counts=counts,resources=resources,last_validation=last_validation,validation_packets=sum(e['validation']['V_count'] for e in epochs if e['validation'] is not None),validation_forward_calls=4*((len(V['x'])+255)//256),distinct_U_visited=len(visited))
  d.write(out/'completion.json',clean(done));d.write(d.BASE/c['row_id']/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,epoch=20,total_epochs=40,parent_checkpoint=c['parent_checkpoint'],optimizer_steps=4440,checkpoint=str(final),selection='final_only',target_access=False))
  print('SOURCE_COMPLETE '+json.dumps(clean(done)),flush=True)
 return done
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--row',required=True);a=p.parse_args();train(d.config(next(r for r in d.rows() if r['row_id']==a.row)))
