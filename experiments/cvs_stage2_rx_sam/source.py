"""E200 ->40 epochs: L readaptation, then source soft SSL and physical tasks."""
import argparse,csv,gzip,json,os,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .model import build,groups,make_target
from experiments.cvs_report_ssl.sampling import DomainStream
from experiments.cvs_report_ssl.objectives import soft_identity_kl,identity_weights
from experiments.cvs_report_ssl.auxiliary import PhysicalHeads,linear_objective,repetition_plan,repetition_objective
from experiments.cvs_report_ssl.conditional import ConditionalTeacher
from experiments.cvs_stage2_decouple.common import verify_parent,clean
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_stage2_decouple.deterministic_forward import deterministic_identity
from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled
from experiments.cvs_stage2_ssl.frequency_optim import FrequencyAdamW
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def component_gradients(weighted,pg,model):
 names={id(p):n for n,p in model.named_parameters()};result={}
 for loss_name,value in weighted.items():
  row={}
  for group in pg:
   selected=[p for p in group['params'] if not names[id(p)].endswith(('low_hz_','band_hz_'))]
   gradients=torch.autograd.grad(value,selected,retain_graph=True,allow_unused=True)
   row[group['name']]=float(sum((g.detach().square().sum() for g in gradients if g is not None),value.new_zeros(())).sqrt())
  result[loss_name]=row
 return dict(weighted_dense_gradient_norms=result,frequency_parameters_excluded_for_unit_comparability=True)

def state_bytes(value):
 if torch.is_tensor(value):return value.numel()*value.element_size()
 if isinstance(value,dict):return sum(state_bytes(v) for v in value.values())
 if isinstance(value,(tuple,list)):return sum(state_bytes(v) for v in value)
 return 0

def load_source(c):
 from experiments.cvs_phase1_stack.recover import verify_contract
 pc=c['parent_contract'];provenance=verify_parent(pc)
 prepared=Path(c['preparation']);done=d.read(prepared/'preparation_complete.json')
 if done['status']!='SOURCE_GEOMETRY_AUDIT_COMPLETE' or done['parent_contract']!=pc or done['target_access'] or done['identity_updates']!=0:raise ValueError('IQ preparation provenance differs')
 packed=torch.load(prepared/'source_cache.pt',map_location='cpu',weights_only=False)
 verify_contract(packed['contract'],d.read(d.SOURCE));verify_contract(done['source_contract'],packed['contract'])
 # Discard every old fitted state, parent feature, margin and auxiliary head.
 keep=('x','y','rx','day','ids');L={k:packed['labeled'][k] for k in keep};V={k:packed['validation'][k] for k in keep}
 for role,cache,n in [('L_s',L,6300),('V',V,27000)]:
  if len(cache['ids'])!=n or len(set(cache['ids']))!=n or sorted(cache['ids'])!=sorted(packed['contract']['role_ids'][role]):raise ValueError('Source physical IDs differ')
 if set(L['ids'])&set(V['ids']):raise ValueError('Overlapping legal roles')
 U=None
 if c['unlabeled']:
  U=torch.load(c['semi']['cache'],map_location='cpu',weights_only=False)
  validate_unlabeled(U,packed['contract']['role_ids']['U_s'])
  if len(U['ids'])!=56700 or set(U['ids'])&(set(L['ids'])|set(V['ids'])):raise ValueError('Unlabeled physical role differs')
 return L,U,V,packed['contract'],provenance

@torch.no_grad()
def validation(model,V,device):
 old=model.training;model.eval();correct=0;loss=0.;rxs={};days={}
 for start in range(0,len(V['x']),256):
  x=V['x'][start:start+256].to(device);y=V['y'][start:start+256].to(device);scores=model(x)
  yes=(scores.argmax(1)==y).cpu();correct+=int(yes.sum());loss+=float(F.cross_entropy(scores,y,reduction='sum'))
  for r in V['rx'][start:start+256].unique().tolist():
   mask=V['rx'][start:start+256]==r;a,b=rxs.get(str(r),(0,0));rxs[str(r)]=(a+int(yes[mask].sum()),b+int(mask.sum()))
  for r in V['day'][start:start+256].unique().tolist():
   mask=V['day'][start:start+256]==r;a,b=days.get(str(r),(0,0));days[str(r)]=(a+int(yes[mask].sum()),b+int(mask.sum()))
 model.train(old)
 byrx={k:a/b for k,(a,b) in rxs.items()};byday={k:a/b for k,(a,b) in days.items()}
 return dict(V_count=len(V['x']),V_accuracy=correct/len(V['x']),V_NLL=loss/len(V['x']),V_RX_accuracy=byrx,V_day_accuracy=byday,V_worst_RX=min(byrx.values()),source_selection_score=.5*correct/len(V['x'])+.5*min(byrx.values()))

@torch.no_grad()
def teacher_validation(teacher,V,device,conditional=None):
 probabilities={'target':[],'shared':[],'condition_L':[],'condition_U':[]};ys=[]
 for start in range(0,len(V['x']),256):
  x=V['x'][start:start+256].to(device);rx=V['rx'][start:start+256].to(device);day=V['day'][start:start+256].to(device)
  ema=teacher.probabilities(x);probabilities['target'].append(ema.cpu());ys.append(V['y'][start:start+256])
  if conditional:
   for mode in ('shared','condition_L','condition_U'):
    q=conditional.probabilities(x,rx,day,mode)
    probabilities[mode].append((.5*ema+.5*q).cpu())
 y=torch.cat(ys);result={}
 for name,parts in probabilities.items():
  if not parts:continue
  q=torch.cat(parts);conf=q.max(1).values;error=q.argmax(1)!=y;rows=[]
  for coverage in (.5,.9,.95,1.):
   count=int(len(q)*coverage);idx=torch.argsort(conf,descending=True,stable=True)[:count]
   rows.append(dict(coverage=coverage,count=count,error_rate=float(error[idx].float().mean()),NLL=float(-q[idx,y[idx]].clamp_min(1e-8).log().mean())))
  result[name]=rows
 return dict(source='independent V; never U truth',fixed_coverage=result,temperature_fitted_on_V=False)

@torch.no_grad()
def U_diagnostic(teacher,U,device):
 from experiments.cvs_report_ssl.physical import signal_state
 counts=torch.zeros(6,dtype=torch.long);accepted=torch.zeros(6,dtype=torch.long);strata={};entropy=0.;confidence=0.;total=0;state_sum=torch.zeros(40,dtype=torch.float64);state_square=torch.zeros(40,dtype=torch.float64)
 for start in range(0,len(U['x']),256):
  q=teacher.probabilities(U['x'][start:start+256].to(device)).cpu();conf,pred=q.max(1);mask=conf>=.95
  counts+=torch.bincount(pred,minlength=6);accepted+=torch.bincount(pred[mask],minlength=6)
  state=signal_state(U['x'][start:start+256].to(device)).cpu().double();state_sum+=state.sum(0);state_square+=state.square().sum(0)
  entropy+=float(-(q*q.clamp_min(1e-8).log()).sum());confidence+=float(conf.sum());total+=len(q)
  for i in range(len(q)):
   key=f"{int(U['rx'][start+i])}:{int(U['day'][start+i])}";v=strata.setdefault(key,dict(seen=0,accepted=0));v['seen']+=1;v['accepted']+=int(mask[i])
 return dict(count=total,class_prediction_counts=counts.tolist(),class_accepted_counts=accepted.tolist(),mean_confidence=confidence/total,mean_entropy=entropy/total,RX_day=strata,U_accuracy=None,reason='TX truth hidden',recomputed_state40_mean=(state_sum/total).tolist(),recomputed_state40_variance=(state_square/total-(state_sum/total).square()).clamp_min(0).tolist(),state_version='report_no_explicit_cfo_v1',state_last2='lag1/lag2 correlation magnitudes; no old real/imag correlation phase input',state40_scope='diagnostic only; new readouts consume current intermediate349, never old state40 caches',scope='post-L-readaptation scan; no threshold fitting')

from .objective import objective
from .sam import buffers, sam_backward
from .receiver import ReceiverActions

@torch.no_grad()
def physical_validation(c,model,heads,L,V,device):
 result={};model.eval()
 selection,_=DomainStream(V['rx'],V['day'],c['model_seed']+401337).batch(1024)
 linear_readout=None
 if heads.linear is not None:
  source_selection,_=DomainStream(L['rx'],L['day'],c['model_seed']+501337).batch(512)
  fit_h=[];fit_p=[]
  for start in range(0,512,128):
   x=L['x'][source_selection[start:start+128]].to(device);g=torch.Generator().manual_seed(c['model_seed']+501337+start)
   p=2*torch.rand(len(x),4,generator=g).to(device)-1
   from experiments.cvs_report_ssl.physical import known_linear
   h0=model.representation(x)[2];dh=model.representation(known_linear(x,p))[2]-h0
   fit_h.append(torch.cat((h0,dh,torch.ones(len(x),1,device=device)),1).double());fit_p.append(p.double())
  a=torch.cat(fit_h);b=torch.cat(fit_p)
  linear_readout=torch.linalg.solve(a.T@a/len(a)+.001*torch.eye(a.shape[1],device=device,dtype=a.dtype),a.T@b/len(a))
 # Independent V UID holdout, deterministic fixed actions; never train on V.
 for start in range(0,1024,128):
  x=V['x'][selection[start:start+128]].to(device);g=torch.Generator().manual_seed(c['model_seed']+401337+start)
  if heads.linear is not None:
   p=2*torch.rand(len(x),4,generator=g).to(device)-1
   _,_,info=linear_objective(model,heads,x,p,None)
   # Analytic inverse-free least squares from actual interior IQ changes.
   # Exclude both wrap-around positions; compare with the network readout.
   from experiments.cvs_report_ssl.physical import iq,known_linear
   z=iq(x);delta=iq(known_linear(x,p))[:,1:-1]-z[:,1:-1]
   left=z[:,:-2];right=z[:,2:]
   columns=torch.stack((left,1j*left,right,1j*right),2)*.015
   a=torch.cat((columns.real,columns.imag),1);b=torch.cat((delta.real,delta.imag),1)
   estimate=torch.linalg.lstsq(a,b[:,:,None]).solution[:,:,0]
   info['analytic_interior_tap_MSE']=float((estimate-p).square().mean())
   h0=model.representation(x)[2];dh=model.representation(known_linear(x,p))[2]-h0
   matrix=torch.cat((h0,dh,torch.ones(len(x),1,device=device)),1).double()
   info['L_fit_linear_readout_MSE']=float((matrix@linear_readout-p.double()).square().mean())
   info['zero_action_head_MSE']=float(heads.linear(torch.cat((h0,torch.zeros_like(h0)),1)).square().mean())
   masked=x.clone();masked[:,:,:16]=0;masked[:,:,240:]=0
   changed_masked=known_linear(x,p);changed_masked[:,:,:16]=0;changed_masked[:,:,240:]=0
   hm=model.representation(masked)[2];dm=model.representation(changed_masked)[2]-hm
   info['same_neural_head_boundary_masked_MSE']=float((heads.linear(torch.cat((hm,dm),1))-p).square().mean())
   info['boundary_control_scope']='same source-trained neural head, both views edge16 zeroed; sensitivity diagnostic, not isolated causal attribution'
   changed=known_linear(x,p);interior=(changed-x)[:,:,1:-1].square().mean();edge=(changed-x)[:,:,[0,255]].square().mean()
   info.update(interior_IQ_delta_power=float(interior),edge_IQ_delta_power=float(edge),circular_boundary_interpretation='edges excluded from analytic diagnostic; not a calibrated channel')
   result.setdefault('linear',[]).append(info)
  if heads.repetition is not None:
   _,info=repetition_objective(model,heads,repetition_plan(model,x,(start//128)%4))
   result.setdefault('repetition',[]).append(info)
 return dict(independent_UID_count=1024,role='V diagnostic only, no parameter fit',linear_baseline_fit_L_count=512 if linear_readout is not None else 0,linear_baseline_ridge=.001,linear_baseline_selected_for_identity=False,batches=result)

def train(c,device='cuda:0'):
 d.validate(c);L,U,V,contract,provenance=load_source(c);out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False)
 seed_all(c['model_seed']);torch.set_num_threads(2);started=time.perf_counter();cuda=str(device).startswith('cuda')
 lstream=DomainStream(L['rx'],L['day'],c['model_seed']+101)
 ustream=DomainStream(U['rx'],U['day'],c['model_seed']+901337) if U is not None else None
 with numerical_context(d.FULL_FP32_POLICY):
  model=build(c,device);pg,names=groups(model,c)
  heads=ReceiverActions(c,L).to(device)
  auxparams=[p for p in heads.parameters() if p.requires_grad];auxopt=torch.optim.AdamW(auxparams,lr=c['risk_learning']['online_lr'],weight_decay=1e-4) if auxparams else None
  optimizer=FrequencyAdamW(model.base,pg,c['optimizer']['frequency']['peak_lr'],c['optimizer']['weight_decay']) if c['enabled']['frequency'] else torch.optim.AdamW(pg,weight_decay=c['optimizer']['weight_decay'])
  # Added-module initialization never changes the initial ordinary CE RNG.
  seed_all(c['model_seed'])
  initial={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad};teacher=None;conditional=None
  resolved=dict(c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),hardware=torch.cuda.get_device_name(device) if cuda else 'CPU',cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),backend_flags=actual_flags(),parameter_groups=names,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_parameters=sum(p.numel() for p in model.parameters()),auxiliary_trainable_parameters=sum(p.numel() for p in auxparams),execution=dict(U_truth_hidden=True,old_aux_cache_read=False,satellite_generator=False,EMA_enabled=c['teacher'],target_provider=c['semi']['target_provider'],teacher_at_start=False,target_after_updates=888 if U is not None else None,unlabeled_iterator=U is not None,student_selected_for_test=True,physical_heads_from_scratch=False,auxiliary_initialization=heads.provenance,RX_action=c['rx_action'],RX_bank_fit_L_only=c['rx_action']['enabled']))
  for name,value in [('resolved_config.json',resolved),('initialization.json',provenance),('source_contract.json',contract)]:d.write(out/name,clean(value))
  print('RESOLVED '+json.dumps(clean(resolved)),flush=True)
  counts=dict(identity_gradient_passes=0,RX_L_packets=0,RX_U_packets=0,critic_packets=0,identity_updates=0,main_L_packets=0,U_identity_packets=0,teacher_updates=0,auxiliary_steps=0,accepted_U=0,linear_steps=0,repetition_steps=0,physical_L_packets=0,physical_U_packets=0,satellite_calls=0)
  visited=set();epochs=[];diagnostics=[];history={};U_strata={};flip_count=0;repeat_visits=0
  if cuda:torch.cuda.reset_peak_memory_stats(device)
  with gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8') as log:
   for epoch in range(1,41):
    model.train();sums={};epoch_start=time.perf_counter()
    for local in range(1,223):
     step=(epoch-1)*222+local;t=time.perf_counter();indices,_=lstream.batch(128)
     uindices=None
     if teacher is not None:
      uindices,_=ustream.batch(128);visited.update(uindices.tolist())
     for group in pg:group['lr']=d.learning_rate(c,step,group['peak'])
     if c['enabled']['frequency']:optimizer.set_frequency_lr(d.frequency_learning_rate(c,step))
     optimizer.zero_grad(set_to_none=True)
     if auxopt:auxopt.zero_grad(set_to_none=True)
     replay={} if c['sam']['enabled'] else None
     before_buffers=buffers(model) if replay is not None else None
     loss,parts,weighted,details=objective(c,model,heads,L,indices,U,uindices,teacher,conditional,step,device,replay=replay)
     if not torch.isfinite(loss):raise FloatingPointError('Nonfinite teacher-factorial SSL loss')
     gradient_audit=component_gradients(weighted,pg,model) if step%888==0 else None
     loss.backward();params=[p for g in pg for p in g['params']]
     sam_metrics=None
     if replay is not None:
      sam_metrics=sam_backward(model,optimizer,c['sam']['rho'],before_buffers,lambda: objective(c,model,heads,L,indices,U,uindices,teacher,conditional,step,device,replay=replay,reuse=True)[0])
      details['SAM']=sam_metrics
     counts['identity_gradient_passes']+=2 if replay is not None else 1
     norm=torch.nn.utils.clip_grad_norm_(params,float('inf'),error_if_nonfinite=True)
     dense=torch.stack([p.grad.norm() for p in params if p.grad is not None and p.numel()>64]).norm()
     aux_norm=torch.nn.utils.clip_grad_norm_(auxparams,float('inf'),error_if_nonfinite=True) if auxparams else None
     optimizer.step()
     if auxopt and details['critic_packets']>0:auxopt.step();counts['auxiliary_steps']+=1
     for key in ('RX_L_packets','RX_U_packets','critic_packets'):counts[key]+=details[key]
     if teacher is not None:teacher.update(model);counts['teacher_updates']+=int(c['teacher']);counts['U_identity_packets']+=128
     counts['identity_updates']+=1;counts['main_L_packets']+=128;counts['accepted_U']+=details['accepted_U']
     if uindices is not None:
      for i,pred,accept in zip(uindices.tolist(),details['U_predictions'],details['U_accepted_mask']):
       if i in history:repeat_visits+=1;flip_count+=int(history[i]!=pred)
       history[i]=pred
       key=f"{int(U['rx'][i])}:{int(U['day'][i])}";v=U_strata.setdefault(key,dict(seen=0,accepted=0));v['seen']+=1;v['accepted']+=int(accept)
     counts['linear_steps']+=int('linear' in details);counts['repetition_steps']+=int('repetition' in details)
     if 'linear' in details or 'repetition' in details:
      counts['physical_L_packets']+=32;counts['physical_U_packets']+=32 if uindices is not None else 0
     if cuda:torch.cuda.synchronize(device)
     scalars={k:float(v.detach()) for k,v in parts.items()};scalars.update({'weighted_'+k:float(v.detach()) for k,v in weighted.items()})
     scalars.update(total_loss=float(loss.detach()),gradient_norm=float(norm),dense_gradient_norm=float(dense),lr_encoder=pg[0]['lr'],lr_head=pg[1]['lr'],L_accuracy=details['L_accuracy'],U_accept_rate=details['accepted_U']/128.,step_seconds=time.perf_counter()-t)
     if sam_metrics is not None:
      scalars.update(SAM_perturbed_identity_loss=sam_metrics['perturbed_identity_loss'],SAM_perturbation_l2=sam_metrics['actual_perturbation_l2'],SAM_loss_increase=sam_metrics['perturbed_identity_loss']-float((loss-weighted['critic']).detach()))
     for k,v in scalars.items():sums[k]=sums.get(k,0.)+v
     log.write(json.dumps(clean(dict(step=step,epoch=epoch,metrics=scalars,details=details,gradient_audit=gradient_audit,auxiliary_gradient_norm=float(aux_norm) if aux_norm is not None else None,frequency=getattr(optimizer,'last_metrics',None))),allow_nan=False)+'\n')
     if step==888 and U is not None:
      teacher=make_target(c,model)
      d.write(out/'post_readapt_U_scan.json',U_diagnostic(teacher,U,device))
      print('TARGET_INITIALIZED_AFTER_L_READAPT '+json.dumps(dict(step=step,provider=c['semi']['target_provider'])),flush=True)
    log.flush();metrics=validation(model,V,device)
    movement={}
    for key,keys in names.items():
     current=dict(model.named_parameters());keys=[n for n in keys if not n.endswith(('low_hz_','band_hz_'))]
     delta=sum((current[n].detach()-initial[n]).square().sum() for n in keys).sqrt();scale=sum(initial[n].square().sum() for n in keys).sqrt()
     movement[key]=dict(delta_l2=float(delta),relative_l2=float(delta/scale.clamp_min(1e-10)),frequency_parameters_excluded=True)
    if teacher is not None and epoch in c['semi']['conditional_refresh_epochs']:
     if c['enabled']['condition']:
      conditional=ConditionalTeacher(teacher,L,U,device,c['semi']['U_mean_alpha_cap']);torch.save(conditional.state(),out/f'conditional_epoch{epoch}.pt')
     diag=teacher_validation(teacher,V,device,conditional);diagnostics.append(dict(epoch=epoch,**diag));d.write(out/'source_teacher_diagnostics.json',dict(rows=diagnostics))
    row=clean(dict(epoch=epoch,optimizer_steps=counts['identity_updates'],**{k:v/222 for k,v in sums.items()},**metrics,counts=dict(counts),weight_movement=movement,last_gradient_audit=gradient_audit,frequency_last_update=getattr(optimizer,'last_metrics',None),epoch_seconds=time.perf_counter()-epoch_start,distinct_U_visited=len(visited),U_full_passes=ustream.pass_index-1 if ustream else 0,U_prediction_flip_count=flip_count,U_repeat_visits=repeat_visits,U_flip_rate=flip_count/repeat_visits if repeat_visits else None,U_RX_day=copy_strata(U_strata)))
    epochs.append(row);(out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in epochs),encoding='utf-8')
    flat=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in epochs]
    with (out/'epoch_metrics.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print('EPOCH '+json.dumps(row),flush=True)
  if counts['identity_updates']!=8880 or (U is not None and (counts['teacher_updates']!=(7992 if c['teacher'] else 0) or counts['U_identity_packets']!=1022976 or len(visited)!=56700)):raise ValueError('Fixed stage2/SSL budget incomplete')
  if c['enabled']['linear'] and counts['linear_steps']!=2220:raise ValueError('Linear schedule incomplete')
  if c['enabled']['repetition'] and counts['repetition_steps']!=2220:raise ValueError('Repetition schedule incomplete')
  if teacher is not None:
   diagnostics.append(dict(epoch=40,**teacher_validation(teacher,V,device,conditional)))
   d.write(out/'source_teacher_diagnostics.json',dict(rows=diagnostics))
  physical_diag=dict(status='FIXED_L_IQ_BANK' if c['rx_action']['enabled'] else 'NOT_APPLICABLE',recipe=c['rx_action'],counts={k:counts[k] for k in ('RX_L_packets','RX_U_packets','critic_packets','auxiliary_steps')},V_used_for_bank=False,U_TX_used=False);d.write(out/'physical_holdout.json',clean(physical_diag))
  if c['rx_action']['enabled'] and (counts['RX_L_packets']!=1022976 or counts['RX_U_packets']!=1022976):raise ValueError('Incomplete RX action schedule')
  if c['risk_learning']['online'] and (counts['critic_packets']!=63936 or counts['auxiliary_steps']!=1998):raise ValueError('Incomplete auxiliary critic schedule')
  if not c['risk_learning']['online'] and (counts['critic_packets'] or counts['auxiliary_steps']):raise ValueError('Frozen or absent auxiliary unexpectedly updated')
  if counts['identity_gradient_passes']!=(17760 if c['sam']['enabled'] else 8880):raise ValueError('SAM gradient pass budget differs')
  final=out/'final_ssdg.pth'
  torch.save(dict(model=model.state_dict(),baseline_args=model._stage2_baseline_args,config=c,run_id=c['run_id'],candidate_id=c['row_id'],epoch=40,optimizer_steps=8880,checkpoint_selection=c['checkpoint_selection'],parent_checkpoint=c['parent_checkpoint'],frontend_version=c['frontend']['version'],optimizer=optimizer.state_dict(),target_access=False),final)
  torch.save(dict(heads=heads.state_dict(),optimizer=auxopt.state_dict() if auxopt else None,teacher=teacher.model.state_dict() if c['teacher'] else None,teacher_updates=teacher.updates if teacher else 0,L_sampler=lstream.state_dict(),U_sampler=ustream.state_dict() if ustream else None,selected_for_test=False,frontend_version=c['frontend']['version']),out/'auxiliary_state.pt')
  result=clean(dict(status='SOURCE_TRAINED',epoch=40,optimizer_steps=8880,logged_steps=8880,config=c,checkpoint=str(final),checkpoint_selection=c['checkpoint_selection'],target_access=False,target_evaluated=False,final_source_metrics=metrics,mechanism_execution=counts,parent_provenance=provenance,elapsed_seconds=time.perf_counter()-started,resources=dict(identity_training_and_source_diagnostics_seconds=time.perf_counter()-started,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if cuda else None,trainable_parameters=resolved['trainable_parameters'],total_parameters=resolved['total_parameters'],auxiliary_trainable_parameters=resolved['auxiliary_trainable_parameters'],teacher_parameters=sum(p.numel() for p in teacher.model.parameters()) if c['teacher'] else 0,conditional_snapshot_parameters=sum(p.numel() for p in conditional.model.parameters()) if conditional else 0,main_L_packets=counts['main_L_packets'],U_identity_packets=counts['U_identity_packets'],parent_training_epochs=200,incremental_epochs=40)))
  result['resources'].update(optimizer_state_tensor_bytes=state_bytes(optimizer.state_dict()),auxiliary_optimizer_state_tensor_bytes=state_bytes(auxopt.state_dict()) if auxopt else 0,checkpoint_bytes=final.stat().st_size,auxiliary_saved_bytes=(out/'auxiliary_state.pt').stat().st_size,conditional_snapshot_saved_bytes=sum(p.stat().st_size for p in out.glob('conditional_epoch*.pt')),frequency_master_and_moment_dtype='float64' if c['enabled']['frequency'] else None,physical_L_packets=counts['physical_L_packets'],physical_U_packets=counts['physical_U_packets'],RX_L_packets=counts['RX_L_packets'],RX_U_packets=counts['RX_U_packets'],critic_packets=counts['critic_packets'],risk_initialization=heads.provenance,critic_target_identity_packets=counts['critic_packets']*9,action_state_identity_packets=counts['RX_L_packets']+counts['RX_U_packets'] if c['rx_action']['critic'] else 0,teacher_identity_packets=counts['U_identity_packets'],teacher_probability_method_batch_calls=teacher.calls if teacher else 0,teacher_diagnostic_packets=56700+6*27000 if teacher else 0,conditional_fit_snapshot_count=5 if conditional else 0)
  d.write(out/'completion.json',result);d.write(out.parent/'source_frozen.json',dict(status='SOURCE_FROZEN',run_id=c['run_id'],row_id=c['row_id'],config=c,checkpoint=str(final),epoch=40,optimizer_steps=8880,target_access=False,target_scores_consumed=False,selection=c['checkpoint_selection']))
  print('SOURCE_TRAINED '+json.dumps(result),flush=True)
 return result

def copy_strata(values):return {k:dict(v) for k,v in values.items()}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();train(d.read(a.config),a.device)
