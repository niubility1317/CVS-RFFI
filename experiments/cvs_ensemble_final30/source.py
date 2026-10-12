import gzip,json,math,os,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_nocfo_foundation.source import load_source as native_source
from .model import build_member,groups
from experiments.cvs_report_ssl.sampling import DomainStream
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled,VisitPassBank
from experiments.cvs_finite_margin.objective import deterministic
from experiments.cvs_stage2_rx_risk.source import validation
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def load(c):
 d.validate_member(c);L,closed,V,contract=native_source(c['source_loader_config']);assert closed is None;U=None
 if c['source_U']:
  U=torch.load(c['U_cache'],map_location='cpu',weights_only=False);validate_unlabeled(U,contract['role_ids']['U_s'])
  if len(U['ids'])!=56700 or set(U['ids'])&(set(L['ids'])|set(V['ids'])):raise ValueError('U source role differs')
 return L,U,V,contract

def objective(model,x,y,ux,ids,passes,bank,c,step):
 logits=model(x);ce=F.cross_entropy(logits,y);kl=logits.sum()*0;detail=dict(L_accuracy=float((logits.argmax(1)==y).float().mean()),U_packets=0,accepted_U=0,ramp=0.,teacher_forward_calls=0)
 if ux is not None:
  if not c['source_U'] or bank is None or step<=1110:raise ValueError('Unexpected U branch')
  with deterministic(model),torch.no_grad():q=model(ux).softmax(1).detach()
  confidence,pseudo=q.max(1);mask,history=bank.update(ids,pseudo,confidence,passes)
  gen=torch.Generator(device=ux.device).manual_seed(c['model_seed']+step*9901)
  strong=ux+torch.randn(ux.shape,generator=gen,device=ux.device,dtype=ux.dtype)*.015
  devices=[ux.device.index or 0] if ux.is_cuda else []
  with torch.random.fork_rng(devices=devices):
   torch.manual_seed(c['model_seed']+step*1979);strong_logits=model(strong)
  kl=(F.kl_div(strong_logits.log_softmax(1),q,reduction='none').sum(1)*mask).mean()
  ramp=min(1.,max(0.,(step-1110)/1110))
  detail.update(U_packets=len(ux),accepted_U=int(mask.sum()),ramp=ramp,teacher_forward_calls=1,history=history,U_confidence=float(confidence.mean()),U_entropy=float(-(q*q.clamp_min(1e-12).log()).sum(1).mean()))
 else:ramp=0.
 return ce+.25*ramp*kl,dict(CE=float(ce.detach()),KL=float(kl.detach())),detail

def train_member(c,device='cuda:0'):
 d.validate_member(c);torch.set_num_threads(2);out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();L,U,V,contract=load(c)
 seed_all(c['member_init_seed'])
 with numerical_context(d.FULL_FP32_POLICY):
  model=build_member(c,device);pg,names=groups(model,c);params=[p for g in pg for p in g['params']];opt=torch.optim.AdamW(pg,weight_decay=.0001)
  lstream=DomainStream(L['rx'],L['day'],c['model_seed']+101);ustream=DomainStream(U['rx'],U['day'],c['model_seed']+901337) if U else None;bank=VisitPassBank(2,.95) if U else None
  resolved=dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),backend_flags=actual_flags(),identity_input_policy=model._identity_input_policy,checkpoint_loaded=False,ancestors=[],actual_initialization_seed=c['member_init_seed'],trainable_parameters=sum(p.numel() for p in params),U_iterator=U is not None,teacher='current_student' if U else 'none',teacher_pretrained=False,EMA=False,satellite_generator=False,L_count=6300,U_count=56700 if U else 0,V_count=27000)
  d.write(out/'resolved_config.json',resolved);d.write(out/'source_contract.json',contract);print('RESOLVED '+json.dumps(resolved),flush=True);epochs=[];counts=dict(identity_updates=0,L_packets=0,U_packets=0,accepted_U=0,teacher_forwards=0);visited=set();seed_all(c['member_init_seed'])
  with gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as log:
   for epoch in range(1,31):
    model.train();sums={};lr=c['min_lr']+(c['peak_lr']-c['min_lr'])*(1+math.cos(math.pi*(epoch-1)/29))/2
    for group in pg:group['lr']=lr
    for local in range(222):
     step=(epoch-1)*222+local+1;idx,_=lstream.batch(128);x=L['x'][idx].to(device);y=L['y'][idx].to(device);ux=None;uids=None;passes=None
     if U is not None and step>1110:
      ui,passes=ustream.batch(128);visited.update(ui.tolist());ux=U['x'][ui].to(device);uids=[U['ids'][i] for i in ui.tolist()]
     opt.zero_grad(set_to_none=True);loss,m,detail=objective(model,x,y,ux,uids,passes,bank,c,step)
     if not torch.isfinite(loss):raise FloatingPointError('Scratch30 objective')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True);opt.step();counts['identity_updates']+=1;counts['L_packets']+=128;counts['U_packets']+=detail['U_packets'];counts['accepted_U']+=detail['accepted_U'];counts['teacher_forwards']+=detail['teacher_forward_calls']
     values=dict(**m,total_loss=float(loss.detach()),gradient_norm=float(norm),**{k:v for k,v in detail.items() if k not in ('history',)})
     for key,value in values.items():sums[key]=sums.get(key,0.)+value
     log.write(json.dumps(dict(step=step,epoch=epoch,metrics=values,history=detail.get('history')))+'\n')
    log.flush();v=validation(model,V,device) if epoch in (10,20,30) else None
    if v:v.pop('source_selection_score',None)
    r=dict(epoch=epoch,updates=counts['identity_updates'],lr=lr,**{k:v/222 for k,v in sums.items()},counts=dict(counts),distinct_U=len(visited),validation=v);epochs.append(r);(out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in epochs),encoding='utf-8');print('EPOCH '+json.dumps(r),flush=True)
  expected_U=5550*128 if U else 0
  if counts['identity_updates']!=6660 or counts['L_packets']!=852480 or counts['U_packets']!=expected_U or counts['teacher_forwards']!=(5550 if U else 0) or len(visited)!=(56700 if U else 0):raise ValueError('Scratch30 fixed role/update budget differs')
  ck=dict(model=model.state_dict(),baseline_args=model._stage2_baseline_args,config=c,epoch=30,updates=6660,ancestors=[],initialization='scratch',target_access=False,checkpoint_selection='fixed_final_epoch30');torch.save(ck,out/'identity.pth')
  done=dict(status='SOURCE_TRAINED',config=c,epoch=30,updates=6660,logged_steps=6660,counts=counts,distinct_U=len(visited),target_access=False,U_truth_read=False,ancestors=[],elapsed_seconds=time.perf_counter()-start,parameters=resolved['trainable_parameters'])
  d.write(out/'completion.json',done);d.write(out/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,checkpoint=str(out/'identity.pth'),epoch=30,updates=6660,selection='fixed_final_epoch30',target_access=False));print('SOURCE_TRAINED '+json.dumps(done),flush=True)
 return done

def train(rowid,device='cuda:0'):
 from .predict import source_proof,baseline_proof
 row=next(r for r in d.rows() if r['row_id']==rowid);c=d.config(row)
 baseline_proof(c)
 parent=Path(c['output']);parent.mkdir(parents=True,exist_ok=False)
 completions=[]
 for member in (1,2):completions.append(train_member(d.member_config(row,member),device))
 bundle=dict(config=c,members=[dict(member=0,config=d.baseline_config(row['model_seed']),checkpoint=d.baseline_config(row['model_seed'])['output']+'/identity.pth'),
  *[dict(member=i,config=d.member_config(row,i),checkpoint=d.member_config(row,i)['output']+'/identity.pth') for i in (1,2)]],
  formula=c['ensemble']['formula'],epochs_per_member=30,new_updates=13320,reused_updates=6660,total_updates=19980,target_access=False,ensemble_fit_updates=0)
 torch.save(bundle,parent/'identity.pth');d.write(parent/'source_contract.json',d.read(d.SOURCE))
 done=dict(status='SOURCE_TRAINED',config=c,members=3,new_members=2,epochs_per_member=30,new_updates=13320,reused_updates=6660,total_updates=19980,
  ensemble_fit_updates=0,target_access=False,U_truth_read=False,allmembersfrozen=True,trainable_parameters=531075)
 d.write(parent/'completion.json',done);d.write(parent/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,checkpoint=str(parent/'identity.pth'),selection=c['checkpoint_selection'],target_access=False))
 source_proof(c);print('ENSEMBLE_SOURCE_TRAINED '+json.dumps(done),flush=True);return done
