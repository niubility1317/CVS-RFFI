import gzip,json,math,os,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_nocfo_foundation.source import load_source as native_source
from experiments.cvs_nocfo_foundation.model import build,groups
from experiments.cvs_report_ssl.sampling import DomainStream
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled,VisitPassBank
from experiments.cvs_finite_margin.objective import deterministic
from experiments.cvs_stage2_rx_risk.source import validation
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

from .objective import objective,layer_info

def load(c):
 d.validate(c);L,closed,V,contract=native_source(c['source_loader_config'])
 if closed is not None or c['source_U']:raise ValueError('Unexpected unlabeled branch')
 return L,None,V,contract

def train(rowid,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==rowid));d.validate(c);torch.set_num_threads(2);out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();L,U,V,contract=load(c)
 seed_all(c['model_seed'])
 with numerical_context(d.FULL_FP32_POLICY):
  model=build(c['source_loader_config'],device);pg,names=groups(model,c['source_loader_config']);params=[p for g in pg for p in g['params']];opt=torch.optim.AdamW(pg,weight_decay=.0001)
  lstream=DomainStream(L['rx'],L['day'],c['model_seed']+101);ustream=DomainStream(U['rx'],U['day'],c['model_seed']+901337) if U else None;bank=VisitPassBank(2,.95) if U else None
  resolved=dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),backend_flags=actual_flags(),identity_input_policy=model._identity_input_policy,checkpoint_loaded=False,ancestors=[],trainable_parameters=sum(p.numel() for p in params),U_iterator=U is not None,teacher='current_student' if U else 'none',teacher_pretrained=False,actual_layers=layer_info(model,L['x'][:4].to(device)),challenge=c['challenge'],EMA=False,satellite_generator=False,L_count=6300,U_count=56700 if U else 0,V_count=27000)
  d.write(out/'resolved_config.json',resolved);d.write(out/'source_contract.json',contract);print('RESOLVED '+json.dumps(resolved),flush=True);epochs=[];counts=dict(identity_updates=0,L_packets=0,U_packets=0,accepted_U=0,teacher_forwards=0,challenge_forwards=0,rank_grad_calls=0);visited=set();seed_all(c['model_seed'])
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
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True);opt.step();counts['identity_updates']+=1;counts['L_packets']+=128;counts['U_packets']+=detail['U_packets'];counts['accepted_U']+=detail['accepted_U'];counts['teacher_forwards']+=detail['teacher_forward_calls'];counts['challenge_forwards']+=detail['challenge_forwards'];counts['rank_grad_calls']+=detail['rank_grad_calls']
     values=dict(**m,total_loss=float(loss.detach()),gradient_norm=float(norm),**{k:v for k,v in detail.items() if k not in ('history',)})
     for key,value in values.items():sums[key]=sums.get(key,0.)+value
     log.write(json.dumps(dict(step=step,epoch=epoch,metrics=values,history=detail.get('history')))+'\n')
    log.flush();v=validation(model,V,device) if epoch in (10,20,30) else None
    if v:v.pop('source_selection_score',None)
    r=dict(epoch=epoch,updates=counts['identity_updates'],lr=lr,**{k:v/222 for k,v in sums.items()},counts=dict(counts),distinct_U=len(visited),validation=v);epochs.append(r);(out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in epochs),encoding='utf-8');print('EPOCH '+json.dumps(r),flush=True)
  expected_challenge=5550 if c['challenge']['enabled'] else 0
  if counts['challenge_forwards']!=expected_challenge or counts['rank_grad_calls']!=expected_challenge:raise ValueError('Challenge schedule differs')
  expected_U=5550*128 if U else 0
  if counts['identity_updates']!=6660 or counts['L_packets']!=852480 or counts['U_packets']!=expected_U or counts['teacher_forwards']!=(5550 if U else 0) or len(visited)!=(56700 if U else 0):raise ValueError('Scratch30 fixed role/update budget differs')
  ck=dict(model=model.state_dict(),baseline_args=model._stage2_baseline_args,config=c,epoch=30,updates=6660,ancestors=[],initialization='scratch',target_access=False,checkpoint_selection='fixed_final_epoch30');torch.save(ck,out/'identity.pth')
  done=dict(status='SOURCE_TRAINED',config=c,epoch=30,updates=6660,logged_steps=6660,counts=counts,distinct_U=len(visited),target_access=False,U_truth_read=False,ancestors=[],elapsed_seconds=time.perf_counter()-start,parameters=resolved['trainable_parameters'])
  d.write(out/'completion.json',done);d.write(out/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,checkpoint=str(out/'identity.pth'),epoch=30,updates=6660,selection='fixed_final_epoch30',target_access=False));print('SOURCE_TRAINED '+json.dumps(done),flush=True)
 return done
