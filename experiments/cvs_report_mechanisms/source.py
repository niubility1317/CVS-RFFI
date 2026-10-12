import gzip,json,math,os,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .model import build,groups
from .frontends import digital_action,features,DIMS,bounded_tangent_distance
from .feedback import objective as feedback_objective,parameter_selection
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_cross_rx_feedback.sampling import RotatingStreams
from experiments.cvs_report_ssl.sampling import DomainStream
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled,VisitPassBank
from experiments.cvs_swa30.average import MeanParameters
from experiments.cvs_swa30.design import learning_rate as swa_lr
from experiments.cvs_finite_margin.objective import deterministic
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def load(c):
 d.validate(c);L,U,V,contract=load_source(c['source_loader_config'])
 if U is not None or len(L['ids'])!=6300 or len(V['ids'])!=27000:raise ValueError('Native fullsource role mismatch')
 if c['source_U']:
  U=torch.load(c['U_cache'],map_location='cpu',weights_only=False);validate_unlabeled(U,contract['role_ids']['U_s'])
  if len(U['ids'])!=56700 or set(U['ids'])&(set(L['ids'])|set(V['ids'])):raise ValueError('Unlabeled source role mismatch')
 return L,U,contract

def tangent(model,x,step):
 kind=('smooth_gain','delay')[(step-1)%2]
 with deterministic(model):
  plus=F.normalize(model.forward_identity_only(digital_action(x,kind,1.))['id_feat_joint'],dim=1,eps=1e-4)
  minus=F.normalize(model.forward_identity_only(digital_action(x,kind,-1.))['id_feat_joint'],dim=1,eps=1e-4)
  loss=((plus-minus)/2).square().sum(1).mean()
 return loss,kind

def feature_census(model,L,device):
 if model.arm not in DIMS:return None
 values=[]
 with torch.no_grad():
  for begin in range(0,6300,256):values.append(features(model.arm,L['x'][begin:begin+256].to(device)).cpu())
 z=torch.cat(values);return dict(count=6300,feature_dimension=z.shape[1],std=z.std(0).tolist(),mean=z.mean(0).tolist(),zero_packet_fraction=float((z.abs().sum(1)==0).float().mean()),constant_coordinate_fraction=float((z.std(0)<1e-7).float().mean()),selection_gate=False,scope='sourceL only content/degeneration diagnostic;noVperformance/noquery')

def tangent_census(model,L,device):
 if model.arm!='TANGENT':return None
 hh=[];jj=[]
 with deterministic(model),torch.no_grad():
  for begin in range(0,6300,256):
   x=L['x'][begin:begin+256].to(device);hh.append(F.normalize(model.forward_identity_only(x)['id_feat_joint'],dim=1,eps=1e-4).cpu());derivatives=[]
   for kind in ('smooth_gain','delay'):
    plus=F.normalize(model.forward_identity_only(digital_action(x,kind,1.))['id_feat_joint'],dim=1,eps=1e-4);minus=F.normalize(model.forward_identity_only(digital_action(x,kind,-1.))['id_feat_joint'],dim=1,eps=1e-4);derivatives.append(((plus-minus)/2).cpu())
   jj.append(torch.stack(derivatives,2))
 h=torch.cat(hh).double();j=torch.cat(jj).double();cells={};pairs=[]
 for tx in range(6):
  for rx in (1,3,4,6,8):
   ix=(L['y']==tx)&(L['rx']==rx);cells[tx,rx]=(h[ix].mean(0),j[ix].mean(0))
 keys=list(cells)
 for a,ka in enumerate(keys):
  for kb in keys[a+1:]:
   if ka[0]!=kb[0] and ka[1]!=kb[1]:continue
   delta=cells[ka][0]-cells[kb][0];jac=(cells[ka][1]+cells[kb][1])/2;value,coef=bounded_tangent_distance(delta,jac);pairs.append(dict(cell_a=ka,cell_b=kb,same_TX=ka[0]==kb[0],raw_squared_distance=float(delta.square().sum()),bounded_squared_residual=float(value),eta=coef.tolist()))
 return dict(source_L=6300,pairs=pairs,coefficient_box=[-1,1],physical_proxy_bounds=['smooth spectral .1dB','circular fractional .02sample'],classifier_uses_distance=False,scope='fixed sourcecenters workingmodel two-action boxLS diagnostic;not querymatching/not automatic sampledependent projection invariance')

def train(rowid,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==rowid));out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter();L,U,contract=load(c);seed_all(c['model_seed'])
 with numerical_context(d.FULL_FP32_POLICY):
  model=build(c,device);pg,names=groups(model,c);params=[p for g in pg for p in g['params']];opt=torch.optim.AdamW(pg,weight_decay=.0001);fpnames,fp=parameter_selection(model,pg);matched=c['arm'] in ('CE_MATCHED','UNIFORM_SSL','META_SSL');ls=RotatingStreams(L,c['model_seed']+101,True) if matched else DomainStream(L['rx'],L['day'],c['model_seed']+101);us=RotatingStreams(U,c['model_seed']+901337) if U else None;bank=VisitPassBank(2,.95) if U else None;rxs=sorted(L['rx'].unique().tolist());avg=MeanParameters(model,pg) if c['arm']=='SWA_TRAJECTORY' else None;visited=set();epochs=[];counts=dict(identity_updates=0,L_packets=0,U_packets=0,accepted_U=0,teacher_forwards=0,tangent_forwards=0,C4_native_forwards=0)
  resolved=dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),backend_flags=actual_flags(),identity_input_policy=model._identity_input_policy,checkpoint_loaded=False,ancestors=[],trainable_parameters=sum(p.numel() for p in params),parameter_groups=names,feedback_parameter_names=fpnames if U else [],U_opened=U is not None,U_truth_read=False,V_forward=False,source_performance_gate=False,target_access=False,classifier=c['classifier'],SWA_parameter_names=avg.names if avg else [],BN=False,source_L=6300,source_U=56700 if U else 0)
  d.write(out/'resolved_config.json',resolved);d.write(out/'source_contract.json',contract);print('RESOLVED '+json.dumps(resolved),flush=True);d.write(out/'feature_diagnostics.json',feature_census(model,L,device));seed_all(c['model_seed'])
  with gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as log:
   for epoch in range(1,31):
    model.train();sums={};lr=swa_lr(epoch) if avg else .000001+(.0002-.000001)*(1+math.cos(math.pi*(epoch-1)/29))/2
    for g in pg:g['lr']=lr
    for local in range(222):
     step=(epoch-1)*222+local+1;ux=ids=passes=None;mr=None
     if matched:
      mr=rxs[(step-1)%5];inner=[r for r in rxs if r!=mr];parts=[ls.batch(r,n)[0] for r,n in zip(rxs,(26,26,26,25,25))];li=torch.cat(parts);mi=parts[rxs.index(mr)]
     else:li,_=ls.batch(128)
     x=L['x'][li].to(device);y=L['y'][li].to(device)
     if U is not None and step>1110:
      batches=[us.batch(r,32) for r in inner];ui=torch.cat([a for a,b in batches]);passes=torch.cat([b for a,b in batches]);visited.update(ui.tolist());ux=U['x'][ui].to(device);ids=[U['ids'][i] for i in ui.tolist()]
     opt.zero_grad(set_to_none=True)
     if matched:loss,m,detail=feedback_objective(model,x,y,L['x'][mi].to(device),L['y'][mi].to(device),ux,ids,passes,bank,opt,fp,c,step)
     else:
      logits=model(x);loss=F.cross_entropy(logits,y);m=dict(CE=float(loss.detach()),L_accuracy=float((logits.argmax(1)==y).float().mean()));detail=dict(U_packets=0,accepted_U=0,teacher_forwards=0)
      if c['arm']=='TANGENT':aux,kind=tangent(model,x,step);loss=loss+.05*aux;m.update(tangent=float(aux.detach()));detail['tangent_action']=kind;counts['tangent_forwards']+=2
      if c['arm']=='C4_ORBIT':counts['C4_native_forwards']+=4
     if not torch.isfinite(loss):raise FloatingPointError('Reportmechanism objective')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True);opt.step();counts['identity_updates']+=1;counts['L_packets']+=128
     for k in ('U_packets','accepted_U','teacher_forwards'):counts[k]+=detail[k]
     values=dict(**m,total_loss=float(loss.detach()),gradient_norm=float(norm),**{k:v for k,v in detail.items() if isinstance(v,(float,int))})
     for k,v in values.items():sums[k]=sums.get(k,0.)+v
     log.write(json.dumps(dict(step=step,epoch=epoch,metrics=values,feedback_RX=mr,detail={k:v for k,v in detail.items() if not isinstance(v,(float,int))}))+'\n')
    if avg and epoch>=21:avg.update(model)
    log.flush();r=dict(epoch=epoch,updates=step,lr=lr,**{k:v/222 for k,v in sums.items()},counts=dict(counts),distinct_U=len(visited));epochs.append(r);(out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in epochs),encoding='utf-8');print('EPOCH '+json.dumps(r),flush=True)
  if counts['identity_updates']!=6660 or counts['L_packets']!=852480 or counts['U_packets']!=(710400 if U else 0) or len(visited)!=(56700 if U else 0):raise ValueError('Fixed30 fullsource budget differs')
  d.write(out/'tangent_diagnostics.json',tangent_census(model,L,device));final={k:v.detach().clone() for k,v in model.state_dict().items()};states={'FINAL':final}
  if avg:states['SWA']=avg.state(model);torch.save(dict(config=c,epochs=list(range(21,31)),parameter_names=avg.names,snapshots=avg.snapshots),out/'SWA_snapshots.pth')
  for name,state in states.items():torch.save(dict(model=state,config=c,readout=name,epoch=30,updates=6660,ancestors=[],fit_ids=L['ids'],fit_U_ids=U['ids'] if U else [],target_access=False,U_truth_read=False,initialization='scratch'),out/(name+'.pth'))
  done=dict(status='SOURCE_TRAINED',config=c,epochs=epochs,epoch=30,updates=6660,logged_steps=6660,counts=counts,distinct_U=len(visited),ancestors=[],target_access=False,U_truth_read=False,V_forward=False,source_performance_gate=False,readouts=list(states),elapsed_seconds=time.perf_counter()-start)
  d.write(out/'completion.json',done);d.write(out/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,readouts=list(states),epoch=30,updates=6660,selection='fixed_final_epoch30/SWA21..30',target_access=False));print('SOURCE_TRAINED '+rowid,flush=True)
 return done
