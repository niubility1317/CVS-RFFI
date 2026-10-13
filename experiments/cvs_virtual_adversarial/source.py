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

def load(c):
 d.validate(c);L,closed,V,contract=native_source(c['source_loader_config']);assert closed is None;U=None
 if c['source_U']:
  U=torch.load(c['U_cache'],map_location='cpu',weights_only=False);validate_unlabeled(U,contract['role_ids']['U_s'])
  if len(U['ids'])!=56700 or set(U['ids'])&(set(L['ids'])|set(V['ids'])):raise ValueError('U source role differs')
 return L,U,V,contract

from .objective import objective,layer_info
from .budget import final_artifacts

def train(rowid,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==rowid));d.validate(c);torch.set_num_threads(2);out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();L,U,V,contract=load(c)
 seed_all(c['model_seed'])
 with numerical_context(d.FULL_FP32_POLICY):
  model=build(c['source_loader_config'],device);pg,names=groups(model,c['source_loader_config']);params=[p for g in pg for p in g['params']];opt=torch.optim.AdamW(pg,weight_decay=.0001)
  lstream=DomainStream(L['rx'],L['day'],c['model_seed']+101);ustream=DomainStream(U['rx'],U['day'],c['model_seed']+901337) if U else None;bank=None
  resolved=dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),backend_flags=actual_flags(),identity_input_policy=model._identity_input_policy,checkpoint_loaded=False,ancestors=[],trainable_parameters=sum(p.numel() for p in params),U_iterator=U is not None,teacher='current_student' if U else 'none',teacher_pretrained=False,actual_layers=layer_info(model,L['x'][:4].to(device)),vat=c['vat'],EMA=False,satellite_generator=False,L_count=6300,U_count=56700 if U else 0,V_count=27000)
  d.write(out/'resolved_config.json',resolved);d.write(out/'source_contract.json',contract);print('RESOLVED '+json.dumps(resolved),flush=True);epochs=[];counts=dict(identity_updates=0,L_packets=0,U_packets=0,accepted_U=0,teacher_forwards=0,search_forwards=0,search_grad_calls=0,consistency_forwards=0,zero_direction_fallbacks=0);visited=set();seed_all(c['model_seed'])
  with gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as log:
   for epoch in range(1,d.EPOCHS+1):
    model.train();sums={};lr=c['min_lr']+(c['peak_lr']-c['min_lr'])*(1+math.cos(math.pi*(epoch-1)/(d.EPOCHS-1)))/2
    for group in pg:group['lr']=lr
    for local in range(222):
     step=(epoch-1)*222+local+1;idx,_=lstream.batch(128);x=L['x'][idx].to(device);y=L['y'][idx].to(device);ux=None;uids=None;passes=None
     if U is not None and step>1110:
      ui,passes=ustream.batch(128);visited.update(ui.tolist());ux=U['x'][ui].to(device);uids=[U['ids'][i] for i in ui.tolist()]
     opt.zero_grad(set_to_none=True);loss,m,detail=objective(model,x,y,ux,uids,passes,bank,c,step)
     if not torch.isfinite(loss):raise FloatingPointError('Source objective')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True);opt.step();counts['identity_updates']+=1;counts['L_packets']+=128;counts['U_packets']+=detail['U_packets'];counts['accepted_U']+=detail['accepted_U'];counts['teacher_forwards']+=detail['teacher_forward_calls'];counts['search_forwards']+=detail['search_forwards'];counts['search_grad_calls']+=detail['search_grad_calls'];counts['consistency_forwards']+=detail['consistency_forwards'];counts['zero_direction_fallbacks']+=detail['zero_direction_fallbacks']
     values=dict(**m,total_loss=float(loss.detach()),gradient_norm=float(norm),**{k:v for k,v in detail.items() if k not in ('history',)})
     for key,value in values.items():sums[key]=sums.get(key,0.)+value
     log.write(json.dumps(dict(step=step,epoch=epoch,metrics=values,history=detail.get('history')))+'\n')
    log.flush();v=validation(model,V,device) if epoch in (10,30,d.EPOCHS) else None
    if v:v.pop('source_selection_score',None)
    r=dict(epoch=epoch,updates=counts['identity_updates'],lr=lr,**{k:v/222 for k,v in sums.items()},counts=dict(counts),distinct_U=len(visited),validation=v);epochs.append(r);(out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in epochs),encoding='utf-8');print('EPOCH '+json.dumps(r),flush=True)
  ck,done,frozen=final_artifacts(c,model,counts,len(visited),time.perf_counter()-start,resolved['trainable_parameters'])
  torch.save(ck,out/'identity.pth');d.write(out/'completion.json',done);d.write(out/'source_frozen.json',frozen);print('SOURCE_TRAINED '+json.dumps(done),flush=True)
 return done
