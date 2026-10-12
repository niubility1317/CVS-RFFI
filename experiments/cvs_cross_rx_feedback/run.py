import gzip,json,math,os,sys,time
from pathlib import Path
import torch
from . import design as d
from .model import build,groups
from .method import objective,parameter_selection
from .sampling import RotatingStreams
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_finite_margin.run import subset,predict
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled,VisitPassBank
from experiments.cvs_wave_geometry.run import score
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def load(c):
 d.validate(c);L,closed,V,contract=load_source(c['source_loader_config'])
 if closed is not None:raise ValueError('Native loader U closed')
 fit=subset(L,L['rx']!=c['held_rx']);held=subset(V,V['rx']==c['held_rx']);u=None
 if c['source_U']:
  U=torch.load(c['U_cache'],map_location='cpu',weights_only=False);validate_unlabeled(U,contract['role_ids']['U_s'])
  if len(U['ids'])!=56700 or set(U['ids'])&(set(L['ids'])|set(V['ids'])):raise ValueError('U source role differs')
  u=subset(U,U['rx']!=c['held_rx'])
 if len(fit['ids'])!=5040 or len(held['ids'])!=5400 or u is not None and len(u['ids'])!=45360:raise ValueError('Whole RX exclusion differs')
 return fit,u,held,contract

def run(rowid,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==rowid));out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter();fit,U,held,contract=load(c);seed_all(c['model_seed'])
 with numerical_context(d.FULL_FP32_POLICY):
  model=build(c,device);pg,names=groups(model,c);params=[p for g in pg for p in g['params']];opt=torch.optim.AdamW(pg,weight_decay=.0001);feedback_names,fp=parameter_selection(model,pg)
  ls=RotatingStreams(fit,c['model_seed']+101,labeled=True);us=RotatingStreams(U,c['model_seed']+901337) if U else None;bank=VisitPassBank(2,.95) if U else None;rxs=sorted(ls.indices);visited=set();epochs=[];counts=dict(identity_updates=0,L_packets=0,U_packets=0,accepted_U=0,teacher_forwards=0,feedback_forwards=0,feedback_gradients=0,U_gradient_groups=0)
  resolved=dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),fit_RX=rxs,fit_L=5040,fit_U=45360 if U else 0,held_V=5400,checkpoint_loaded=False,ancestors=[],U_opened=U is not None,U_truth_read=False,target_access=False,trainable_parameters=sum(p.numel() for p in params),parameter_groups=names,feedback_parameter_names=feedback_names,classifier=c['classifier'],EMA=False,backend_flags=actual_flags(),identity_input_policy=model._identity_input_policy)
  d.write(out/'resolved_config.json',resolved);d.write(out/'source_contract.json',contract);print('RESOLVED '+json.dumps(resolved),flush=True);seed_all(c['model_seed'])
  with gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as log:
   for epoch in range(1,31):
    model.train();sums={};lr=.000001+(.0002-.000001)*(1+math.cos(math.pi*(epoch-1)/29))/2
    for g in pg:g['lr']=lr
    for local in range(222):
     step=(epoch-1)*222+local+1;mr=rxs[(step-1)%4];inner=[r for r in rxs if r!=mr];parts=[ls.batch(r,32)[0] for r in inner+[mr]];li=torch.cat(parts);mi=parts[-1];x=fit['x'][li].to(device);y=fit['y'][li].to(device);mx=fit['x'][mi].to(device);my=fit['y'][mi].to(device);ux=ids=passes=None
     if U is not None and step>1110:
      batches=[us.batch(r,n) for r,n in zip(inner,(43,43,42))];ui=torch.cat([a for a,b in batches]);passes=torch.cat([b for a,b in batches]);visited.update(ui.tolist());ux=U['x'][ui].to(device);ids=[U['ids'][i] for i in ui.tolist()]
     opt.zero_grad(set_to_none=True);loss,m,detail=objective(model,x,y,mx,my,ux,ids,passes,bank,opt,fp,c,step)
     if U is not None and step>1110:
      accepted=torch.tensor(detail.pop('accepted_mask'));detail['RX_day']={f'{r}:{day}':dict(seen=int(((U['rx'][ui]==r)&(U['day'][ui]==day)).sum()),accepted=int((accepted&(U['rx'][ui]==r)&(U['day'][ui]==day)).sum())) for r in inner for day in (1,2,3)}
     if not torch.isfinite(loss):raise FloatingPointError('CrossRX feedback loss')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True);opt.step();counts['identity_updates']+=1;counts['L_packets']+=128
     for k in ('U_packets','accepted_U','teacher_forwards','feedback_forwards','feedback_gradients','U_gradient_groups'):counts[k]+=detail[k]
     values=dict(**m,total_loss=float(loss.detach()),gradient_norm=float(norm),**{k:v for k,v in detail.items() if isinstance(v,(float,int))})
     for k,v in values.items():sums[k]=sums.get(k,0.)+v
     log.write(json.dumps(dict(step=step,epoch=epoch,metrics=values,feedback_RX=mr,inner_RX=inner,detail={k:v for k,v in detail.items() if not isinstance(v,(float,int))}))+'\n')
    log.flush();r=dict(epoch=epoch,updates=step,lr=lr,**{k:v/222 for k,v in sums.items()},counts=dict(counts),distinct_U=len(visited));epochs.append(r);(out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in epochs),encoding='utf-8');print('EPOCH '+json.dumps(r),flush=True)
  expected=710400 if U else 0
  if counts['identity_updates']!=6660 or counts['L_packets']!=852480 or counts['U_packets']!=expected or len(visited)!=(45360 if U else 0):raise ValueError('CrossRX fixed source budget differs')
  torch.save(dict(model=model.state_dict(),config=c,epoch=30,updates=6660,ancestors=[],fit_ids=fit['ids'],fit_U_ids=U['ids'] if U else [],held_rx=c['held_rx'],target_access=False,U_truth_read=False,initialization='scratch'),out/'identity.pth');model.eval().requires_grad_(False);before={k:v.clone() for k,v in model.state_dict().items()};z=predict(model,held,device,[])[0]
  if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):raise ValueError('Frozen state mutated')
  torch.save(dict(logits=z,ids=held['ids'],target_access=False),out/'source_predictions.pt');metrics=score(z,held['y'],held['rx'],held['day']);done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,metrics=metrics,identity_updates=6660,epochs=epochs,fit_L_count=5040,fit_U_count=45360 if U else 0,held_V_count=5400,target_access=False,U_opened=U is not None,U_truth_read=False,ancestors=[],parameters_buffers_unchanged=True,counts=counts,distinct_U=len(visited),parameters=resolved['trainable_parameters'],elapsed_seconds=time.perf_counter()-start);d.write(out/'completion.json',done);print('SOURCE_DIAGNOSTIC_COMPLETE '+json.dumps(dict(row_id=rowid,accuracy=metrics['accuracy'])),flush=True)
 return done
