import json,math,os,sys,time,gzip
from pathlib import Path
import torch
from . import design as d
from .objective import objective
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_nocfo_foundation.model import build,groups
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_report_ssl.sampling import DomainStream
from experiments.cvs_wave_geometry.run import score
from experiments.cvs_finite_action.operators import apply
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def subset(data,mask):
 return {k:([v[i] for i in mask.nonzero().flatten().tolist()] if k=='ids' else v[mask]) for k,v in data.items() if k in ('x','y','rx','day','ids')}

@torch.no_grad()
def predict(model,data,device,actions):
 out=[]
 for kind,a in [(None,None),*actions]:
  chunks=[]
  for b in range(0,len(data['x']),256):
   x=data['x'][b:b+256].to(device)
   if kind:x=apply(x,kind,a,0)
   chunks.append(model(x).cpu())
  out.append(torch.cat(chunks))
 return out

def run(rowid,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==rowid));d.validate(c);torch.set_num_threads(2)
 out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();L,U,V,contract=load_source(c['source_loader_config'])
 if U is not None:raise ValueError('U must remain closed in source falsification')
 fit=subset(L,L['rx']!=c['held_rx']);held=subset(L,L['rx']==c['held_rx']);vv=subset(V,V['rx']==c['held_rx'])
 if len(fit['ids'])!=5040 or len(held['ids'])!=1260 or len(vv['ids'])!=5400 or c['held_rx'] in fit['rx'].tolist():raise ValueError('True sourceRX holdout differs')
 seed_all(c['model_seed'])
 with numerical_context(d.FULL_FP32_POLICY):
  model=build(c['source_loader_config'],device);pg,names=groups(model,c['source_loader_config']);params=[p for g in pg for p in g['params']]
  opt=torch.optim.AdamW(pg,lr=c['peak_lr'],weight_decay=c['weight_decay']);stream=DomainStream(fit['rx'],fit['day'],c['model_seed']+101)
  resolved=dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),training_rx=fit['rx'].unique().tolist(),L_fit=len(fit['ids']),L_held=len(held['ids']),V_held=len(vv['ids']),U_opened=False,teacher=False,checkpoint_loaded=False,ancestors=[],backend_flags=actual_flags(),parameters=sum(p.numel() for p in params),identity_input_policy=model._identity_input_policy,heldRX_excluded_before_any_training=True)
  d.write(out/'resolved_config.json',resolved);d.write(out/'source_contract.json',contract);print('RESOLVED '+json.dumps(resolved),flush=True)
  seed_all(c['model_seed']);epochs=[];updates=0
  with gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as log:
   for epoch in range(1,21):
    model.train();sums={};lr=c['min_lr']+(c['peak_lr']-c['min_lr'])*(1+math.cos(math.pi*(epoch-1)/19))/2
    for g in pg:g['lr']=lr
    for j in range(222):
     idx,_=stream.batch(128);x=fit['x'][idx].to(device);y=fit['y'][idx].to(device);opt.zero_grad(set_to_none=True)
     loss,m=objective(model,x,y,c)
     if not torch.isfinite(loss):raise FloatingPointError('Actual finite margin loss')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True);opt.step();updates+=1
     for key,value in m.items():sums[key]=sums.get(key,0.)+value
     log.write(json.dumps(dict(step=updates,epoch=epoch,metrics=m,gradient_norm=float(norm)))+'\n')
    log.flush();r=dict(epoch=epoch,updates=updates,lr=lr,**{k:v/222 for k,v in sums.items()});epochs.append(r)
    (out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in epochs),encoding='utf-8');print('EPOCH '+json.dumps(r),flush=True)
  if updates!=4440:raise ValueError('Incomplete fixedE20 native identity training')
  frozen=dict(config=c,model=model.state_dict(),baseline_args=model._stage2_baseline_args,epoch=20,updates=updates,initialization='scratch',ancestors=[],training_rx=resolved['training_rx'],target_access=False,selected_by='fixed_final_epoch')
  torch.save(frozen,out/'identity.pth');d.write(out/'source_frozen.json',dict(config=c,epoch=20,updates=4440,checkpoint=str(out/'identity.pth'),source_only=True,target_access=False))
  model.eval();lp=predict(model,held,device,[])[0];vp=predict(model,vv,device,c['actions'])
  torch.save(dict(L_logits=lp,V_logits=vp,L_ids=held['ids'],V_ids=vv['ids'],truth_read=False,target_access=False),out/'source_predictions.pt')
  metrics=dict(L_held=score(lp,held['y'],held['rx'],held['day']),V_views=[score(z,vv['y'],vv['rx'],vv['day']) for z in vp])
  done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,metrics=metrics,epochs=epochs,identity_updates=4440,main_L_packets=4440*128,extra_L_packets=4440*160,extra_mode_updates_normalization=False,target_access=False,U_opened=False,checkpoint_sources=[],elapsed_seconds=time.perf_counter()-start)
  d.write(out/'completion.json',done);print('SOURCE_DIAGNOSTIC_COMPLETE '+json.dumps(dict(row_id=rowid,accuracy=metrics['V_views'][0]['accuracy'])),flush=True)
 return done
