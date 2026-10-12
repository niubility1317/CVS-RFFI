import gzip,json,math,os,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .model import build,groups
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_finite_margin.run import subset,predict
from experiments.cvs_report_ssl.sampling import DomainStream
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_wave_geometry.run import score
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
def run(rowid,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==rowid));d.validate(c);torch.set_num_threads(2);out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();L,U,V,contract=load_source(c['source_loader_config'])
 if U is not None:raise ValueError('FullwaveCNN source U must remain closed')
 fit=subset(L,L['rx']!=c['held_rx']);held=subset(V,V['rx']==c['held_rx'])
 if len(fit['ids'])!=5040 or len(held['ids'])!=5400 or c['held_rx'] in fit['rx'].tolist():raise ValueError('Whole RX holdout differs')
 seed_all(c['model_seed'])
 with numerical_context(d.FULL_FP32_POLICY):
  model=build(c,fit['x'],device);pg,names=groups(model,c);params=[p for g in pg for p in g['params']];opt=torch.optim.AdamW(pg,weight_decay=.0001);stream=DomainStream(fit['rx'],fit['day'],c['model_seed']+101)
  resolved=dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),fit_RX=fit['rx'].unique().tolist(),fit_L=5040,held_V=5400,checkpoint_loaded=False,ancestors=[],U_opened=False,target_access=False,trainable_parameters=sum(p.numel() for p in params),parameter_groups=names,actual_layers=model.layer_info,quality_stats_fit_L_only=False,initial_transform='identity',backend_flags=actual_flags(),identity_input_policy=model._identity_input_policy)
  d.write(out/'resolved_config.json',resolved);d.write(out/'source_contract.json',contract);print('RESOLVED '+json.dumps(resolved),flush=True);epochs=[];updates=0;seed_all(c['model_seed'])
  with gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as log:
   for epoch in range(1,31):
    model.train();sums=dict(CE=0.,accuracy=0.);lr=.000001+(.0002-.000001)*(1+math.cos(math.pi*(epoch-1)/29))/2
    for g in pg:g['lr']=lr
    for i in range(222):
     idx,_=stream.batch(128);x=fit['x'][idx].to(device);y=fit['y'][idx].to(device);opt.zero_grad(set_to_none=True);logits=model(x);loss=F.cross_entropy(logits,y)
     if not torch.isfinite(loss):raise FloatingPointError('Bounded fullwaveCNN CE')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True);opt.step();updates+=1;sums['CE']+=float(loss);sums['accuracy']+=float((logits.argmax(1)==y).float().mean());log.write(json.dumps(dict(step=updates,epoch=epoch,CE=float(loss),gradient_norm=float(norm)))+'\n')
    log.flush();r=dict(epoch=epoch,updates=updates,lr=lr,**{k:v/222 for k,v in sums.items()});epochs.append(r);(out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in epochs),encoding='utf-8');print('EPOCH '+json.dumps(r),flush=True)
  if updates!=6660 or model._current_q is not None or model.augmented_packets!=c['mechanism']['augmentation_count']:raise ValueError('FullwaveCNN budget/forward context differs')
  torch.save(dict(model=model.state_dict(),config=c,epoch=30,updates=6660,ancestors=[],fit_ids=fit['ids'],fit_RX=fit['rx'].unique().tolist(),held_rx=c['held_rx'],target_access=False),out/'identity.pth');model.eval().requires_grad_(False);before={k:t.clone() for k,t in model.state_dict().items()};z=predict(model,held,device,[])[0]
  if any(not torch.equal(t,model.state_dict()[k]) for k,t in before.items()) or model._current_q is not None:raise ValueError('HeldRX evaluation changed frozen state')
  torch.save(dict(logits=z,ids=held['ids'],target_access=False),out/'source_predictions.pt');metrics=score(z,held['y'],held['rx'],held['day'])
  done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,metrics=metrics,identity_updates=6660,epochs=epochs,held_V_count=5400,fit_L_count=5040,target_access=False,U_opened=False,parameters_buffers_unchanged=True,actual_layers=model.layer_info,parameters=resolved['trainable_parameters'],augmented_L_packets=model.augmented_packets,elapsed_seconds=time.perf_counter()-start)
  d.write(out/'completion.json',done);print('SOURCE_DIAGNOSTIC_COMPLETE '+json.dumps(dict(row_id=rowid,accuracy=metrics['accuracy'])),flush=True)
 return done
