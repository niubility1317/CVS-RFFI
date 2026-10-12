import gzip,json,math,os,sys,time
from pathlib import Path
import torch
from . import design as d
from .method import objective,ema_update
from .teachers import cache
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_nocfo_foundation.model import build,groups
from experiments.cvs_finite_margin.run import subset,predict
from experiments.cvs_report_ssl.sampling import DomainStream
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled,VisitPassBank
from experiments.cvs_wave_geometry.run import score
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
def load(c):
 d.validate(c);L,closed,V,contract=load_source(c['source_loader_config'])
 if closed is not None:raise ValueError('Original loader must stay labeled-only')
 U=torch.load(c['U_cache'],map_location='cpu',weights_only=False);validate_unlabeled(U,contract['role_ids']['U_s'])
 if len(U['ids'])!=56700 or set(U['ids'])&(set(L['ids'])|set(V['ids'])):raise ValueError('Source U roles differ')
 fit=subset(L,L['rx']!=c['held_rx']);held=subset(V,V['rx']==c['held_rx']);u=subset(U,U['rx']!=c['held_rx'])
 if (len(fit['ids']),len(u['ids']),len(held['ids']))!=(5040,45360,5400) or c['held_rx'] in fit['rx'].tolist() or c['held_rx'] in u['rx'].tolist():raise ValueError('WholeRX role isolation differs')
 return L,fit,u,held,contract
def run(rowid,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==rowid));d.validate(c);torch.set_num_threads(2);out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();L,fit,U,held,contract=load(c)
 with numerical_context(d.FULL_FP32_POLICY):
  anchor=cache(c,L,U,device) if c['semi']['anchor'] else None
  if anchor is not None:torch.save(anchor,out/'U_anchor.pt')
  seed_all(c['model_seed']);model=build(c['source_loader_config'],device);pg,names=groups(model,c['source_loader_config']);params=[p for g in pg for p in g['params']];opt=torch.optim.AdamW(pg,weight_decay=.0001)
  ema=build(c['source_loader_config'],device);ema.load_state_dict(model.state_dict(),strict=True);ema.eval().requires_grad_(False)
  ls=DomainStream(fit['rx'],fit['day'],c['model_seed']+101);us=DomainStream(U['rx'],U['day'],c['model_seed']+901337);bank=VisitPassBank(2,.90)
  resolved=dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),fit_RX=fit['rx'].unique().tolist(),fit_L=5040,fit_U=45360,held_V=5400,checkpoint_loaded=False,ancestors=anchor['provenance'] if anchor else [],U_opened=True,U_truth_read=False,target_access=False,trainable_parameters=sum(p.numel() for p in params),parameter_groups=names,EMA=True,teacher='fixed source ensemble handoff to currentstudent EMA' if anchor else 'currentstudent EMA',backend_flags=actual_flags(),identity_input_policy=model._identity_input_policy)
  d.write(out/'resolved_config.json',resolved);d.write(out/'source_contract.json',contract);print('RESOLVED '+json.dumps(resolved),flush=True);epochs=[];counts=dict(identity_updates=0,EMA_updates=0,L_packets=0,U_packets=0,accepted_U=0,teacher_forwards=0);visited=set();seed_all(c['model_seed'])
  with gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as log:
   for epoch in range(1,31):
    model.train();sums={};lr=.000001+(.0002-.000001)*(1+math.cos(math.pi*(epoch-1)/29))/2
    for g in pg:g['lr']=lr
    for local in range(222):
     step=(epoch-1)*222+local+1;li,_=ls.batch(128);x=fit['x'][li].to(device);y=fit['y'][li].to(device);ux=aa=ids=passes=None
     if step>1110:
      ui,passes=us.batch(128);visited.update(ui.tolist());ux=U['x'][ui].to(device);ids=[U['ids'][i] for i in ui.tolist()];aa=anchor['probabilities'][ui].to(device) if anchor else None
     opt.zero_grad(set_to_none=True);loss,m,detail=objective(model,x,y,ux,ema,aa,ids,passes,bank,step,c['model_seed'])
     if not torch.isfinite(loss):raise FloatingPointError('EMA source objective')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True);opt.step();decay=ema_update(ema,model,step)
     counts['identity_updates']+=1;counts['EMA_updates']+=1;counts['L_packets']+=128
     for k in ('U_packets','accepted_U','teacher_forwards'):counts[k]+=detail[k]
     values=dict(**m,total_loss=float(loss.detach()),gradient_norm=float(norm),EMA_decay=decay,**{k:v for k,v in detail.items() if k!='history'})
     for k,v in values.items():sums[k]=sums.get(k,0.)+v
     log.write(json.dumps(dict(step=step,epoch=epoch,metrics=values,history=detail.get('history')))+'\n')
    log.flush();r=dict(epoch=epoch,updates=step,lr=lr,**{k:v/222 for k,v in sums.items()},counts=dict(counts),distinct_U=len(visited));epochs.append(r);(out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in epochs),encoding='utf-8');print('EPOCH '+json.dumps(r),flush=True)
  if counts['identity_updates']!=6660 or counts['EMA_updates']!=6660 or counts['L_packets']!=852480 or counts['U_packets']!=710400 or counts['teacher_forwards']!=5550 or len(visited)!=45360:raise ValueError('EMA source fixed budget differs')
  torch.save(dict(model=model.state_dict(),config=c,epoch=30,updates=6660,ancestors=resolved['ancestors'],fit_ids=fit['ids'],fit_U_ids=U['ids'],held_rx=c['held_rx'],target_access=False,U_truth_read=False,initialization='scratch'),out/'identity.pth');model.eval().requires_grad_(False);before={k:t.clone() for k,t in model.state_dict().items()};z=predict(model,held,device,[])[0]
  if any(not torch.equal(t,model.state_dict()[k]) for k,t in before.items()):raise ValueError('Frozen heldRX prediction changed model')
  torch.save(dict(logits=z,ids=held['ids'],target_access=False),out/'source_predictions.pt');metrics=score(z,held['y'],held['rx'],held['day'])
  done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,metrics=metrics,identity_updates=6660,epochs=epochs,held_V_count=5400,fit_L_count=5040,fit_U_count=45360,target_access=False,U_opened=True,U_truth_read=False,parameters_buffers_unchanged=True,counts=counts,distinct_U=45360,ancestors=resolved['ancestors'],parameters=resolved['trainable_parameters'],elapsed_seconds=time.perf_counter()-start)
  d.write(out/'completion.json',done);print('SOURCE_DIAGNOSTIC_COMPLETE '+json.dumps(dict(row_id=rowid,accuracy=metrics['accuracy'])),flush=True)
 return done
