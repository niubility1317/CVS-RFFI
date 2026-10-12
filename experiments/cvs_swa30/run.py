import gzip,json,math,os,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_nocfo_foundation.model import build,groups
from .average import MeanParameters
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_finite_margin.run import subset,predict
from experiments.cvs_report_ssl.sampling import DomainStream
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_wave_geometry.run import score
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags
def run(rowid,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==rowid));d.validate(c);torch.set_num_threads(2);out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();L,U,V,contract=load_source(c['source_loader_config'])
 if U is not None:raise ValueError('Quality-FiLM source U must remain closed')
 fit=subset(L,L['rx']!=c['held_rx']);held=subset(V,V['rx']==c['held_rx'])
 if len(fit['ids'])!=5040 or len(held['ids'])!=5400 or c['held_rx'] in fit['rx'].tolist():raise ValueError('Whole RX holdout differs')
 seed_all(c['model_seed'])
 with numerical_context(d.FULL_FP32_POLICY):
  model=build(c['source_loader_config'],device);pg,names=groups(model,c['source_loader_config']);avg=MeanParameters(model,pg);params=[p for g in pg for p in g['params']];opt=torch.optim.AdamW(pg,weight_decay=.0001);stream=DomainStream(fit['rx'],fit['day'],c['model_seed']+101)
  resolved=dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),fit_RX=fit['rx'].unique().tolist(),fit_L=5040,held_V=5400,checkpoint_loaded=False,ancestors=[],U_opened=False,target_access=False,trainable_parameters=sum(p.numel() for p in params),parameter_groups=names,classifier=c['classifier'],SWA_parameter_names=avg.names,BN=False,backend_flags=actual_flags(),identity_input_policy=model._identity_input_policy)
  d.write(out/'resolved_config.json',resolved);d.write(out/'source_contract.json',contract);print('RESOLVED '+json.dumps(resolved),flush=True);epochs=[];updates=0;seed_all(c['model_seed'])
  with gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as log:
   for epoch in range(1,31):
    model.train();sums=dict(CE=0.,accuracy=0.);lr=d.learning_rate(epoch)
    for g in pg:g['lr']=lr
    for i in range(222):
     idx,_=stream.batch(128);x=fit['x'][idx].to(device);y=fit['y'][idx].to(device);opt.zero_grad(set_to_none=True);logits=model(x);loss=F.cross_entropy(logits,y)
     if not torch.isfinite(loss):raise FloatingPointError('Bounded quality-FiLM CE')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True);opt.step();updates+=1;sums['CE']+=float(loss);sums['accuracy']+=float((logits.argmax(1)==y).float().mean());log.write(json.dumps(dict(step=updates,epoch=epoch,CE=float(loss),gradient_norm=float(norm)))+'\n')
    
    if epoch>=21:avg.update(model)
    log.flush();r=dict(epoch=epoch,updates=updates,lr=lr,**{k:v/222 for k,v in sums.items()});epochs.append(r);(out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in epochs),encoding='utf-8');print('EPOCH '+json.dumps(r),flush=True)
  if updates!=6660:raise ValueError('Quality-FiLM budget/forward context differs')
  final={k:v.detach().clone() for k,v in model.state_dict().items()};averaged=avg.state(model)
  torch.save(dict(config=c,epochs=list(range(21,31)),parameter_names=avg.names,snapshots=avg.snapshots),out/'SWA_snapshots.pth')
  for name,state in (('FINAL',final),('SWA',averaged)):
   torch.save(dict(model=state,config=c,readout=name,epoch=30,updates=6660,ancestors=[],fit_ids=fit['ids'],held_rx=c['held_rx'],target_access=False,swa_snapshots=10 if name=='SWA' else 0,swa_parameter_names=avg.names if name=='SWA' else []),out/(name+'.pth'))
  model.eval().requires_grad_(False);zz={};readouts={}
  for name,state in (('FINAL',final),('SWA',averaged)):
   model.load_state_dict(state,strict=True);before={k:t.clone() for k,t in model.state_dict().items()};z=predict(model,held,device,[])[0]
   if any(not torch.equal(t,model.state_dict()[k]) for k,t in before.items()):raise ValueError('SWA heldRX evaluation changed frozen state')
   torch.save(dict(logits=z,ids=held['ids'],target_access=False),out/(name+'_source_predictions.pt'));zz[name]=z
  for name,z in zz.items():readouts[name]=score(z,held['y'],held['rx'],held['day'])
  done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,readouts=readouts,identity_updates=6660,epochs=epochs,held_V_count=5400,fit_L_count=5040,target_access=False,U_opened=False,parameters_buffers_unchanged=True,swa_snapshots=avg.count,swa_parameter_names=avg.names,parameters=resolved['trainable_parameters'],elapsed_seconds=time.perf_counter()-start)
  d.write(out/'completion.json',done);print('SOURCE_DIAGNOSTIC_COMPLETE '+json.dumps(dict(row_id=rowid,readouts={k:v['accuracy'] for k,v in readouts.items()})),flush=True)
 return done
