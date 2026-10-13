import gzip,json,math,os,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .model import build,groups
from .contrastive import capture,penalty
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_cross_rx_feedback.sampling import RotatingStreams
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled,VisitPassBank
from experiments.cvs_finite_margin.objective import deterministic
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def dropout_seed(c,step):return c['member_init_seed']+step*1979
def load(c):
 d.validate_member(c);L,U,V,contract=load_source(c['source_loader_config'])
 if U is not None or len(L['ids'])!=6300 or len(V['ids'])!=27000:raise ValueError('Fullsource role contract')
 U=torch.load(c['U_cache'],map_location='cpu',weights_only=False);validate_unlabeled(U,contract['role_ids']['U_s'])
 if len(U['ids'])!=56700 or set(U['ids'])&(set(L['ids'])|set(V['ids'])):raise ValueError('Source hiddenU role contract')
 return L,U,contract
def objective(model,x,y,rx,ux,ids,passes,bank,c,step,epoch):
 weight=d.coefficient(epoch)
 if weight:
  with capture(model) as maps:logits=model(x)
  contrastive,details=penalty(maps,y,rx,c)
 else:logits=model(x);contrastive=logits.new_zeros(());details=[]
 ce=F.cross_entropy(logits,y);kl=logits.new_zeros(());ramp=0.;accepted=0
 if ux is not None:
  if step<=1110 or len(ux)!=128 or bank is None:raise ValueError('Illegal sourceU timing/batch')
  with deterministic(model),torch.no_grad():q=model(ux).softmax(1).detach()
  confidence,pseudo=q.max(1);mask,history=bank.update(ids,pseudo,confidence,passes);accepted=int(mask.sum());gen=torch.Generator(device=ux.device).manual_seed(c['model_seed']+step*9901);strong=ux+torch.randn(ux.shape,generator=gen,device=ux.device)*.015
  with torch.random.fork_rng(devices=[ux.device.index or 0] if ux.is_cuda else []):
   torch.manual_seed(dropout_seed(c,step));uz=model(strong)
  micro=(F.kl_div(uz.log_softmax(1),q,reduction='none').sum(1)*mask).reshape(4,32).mean(1);kl=micro.mean();ramp=min(1.,max(0.,(step-1110)/1110))
 else:history=None
 loss=ce+.25*ramp*kl+weight*contrastive;metrics=dict(CE=float(ce.detach()),L_accuracy=float((logits.argmax(1)==y).float().mean()),contrastive=float(contrastive.detach()),weighted_contrastive=float(weight*contrastive.detach()),contrastive_weight=weight,KL=float(kl.detach()),weighted_KL=float(.25*ramp*kl.detach()),ramp=ramp,U_packets=128 if ux is not None else 0,accepted_U=accepted,teacher_forwards=int(ux is not None),total_loss=float(loss.detach()))
 return loss,metrics,dict(actual_layers=details,history=history)
def train(rowid,device='cuda:0'):
 c=d.member_config(next(r for r in d.member_rows() if r['member_row_id']==rowid));out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter();L,U,contract=load(c);seed_all(c['model_seed'])
 with numerical_context(d.FULL_FP32_POLICY):
  model=build(c,device);pg,names=groups(model,c);params=[p for g in pg for p in g['params']];opt=torch.optim.AdamW(pg,weight_decay=.0001);ls=RotatingStreams(L,c['model_seed']+101,True);us=RotatingStreams(U,c['model_seed']+901337);bank=VisitPassBank(2,.95);visited=set();epochs=[];counts=dict(identity_updates=0,L_packets=0,U_packets=0,accepted_U=0,teacher_forwards=0,contrastive_updates=0)
  resolved=dict(config=c,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,commit=(d.ROOT/'release_commit.txt').read_text().strip(),backend_flags=actual_flags(),identity_input_policy=model._identity_input_policy,checkpoint_loaded=False,actual_initialization_seed=c['member_init_seed'],ancestors=[],trainable_parameters=sum(p.numel() for p in params),parameter_groups=names,U_opened=True,U_truth_read=False,V_forward=False,source_performance_gate=False,target_access=False,classifier=c['classifier'],actual_layers=list(d.LEVELS),contrastive_runtime_state=False,EMA=False,teacher='currentstudent',control_loss_equivalence='sameUNIFORM_SSL objective/microreduction/data/dropout;unnecessarymetagradients notexecuted,not equalcomputeclaim')
  d.write(out/'resolved_config.json',resolved);d.write(out/'source_contract.json',contract);print('RESOLVED '+json.dumps(resolved),flush=True);seed_all(c['member_init_seed'])
  with gzip.open(out/'step_metrics.jsonl.gz','xt',encoding='utf-8',compresslevel=3) as log:
   for epoch in range(1,31):
    model.train();sums={};lr=.000001+(.0002-.000001)*(1+math.cos(math.pi*(epoch-1)/29))/2
    for g in pg:g['lr']=lr
    for local in range(222):
     step=(epoch-1)*222+local+1;excluded=d.RX[(step-1)%5];li=torch.cat([ls.batch(r,n)[0] for r,n in zip(d.RX,(26,26,26,25,25))]);ux=ids=passes=None
     if step>1110:
      batches=[us.batch(r,32) for r in d.RX if r!=excluded];ui=torch.cat([a for a,b in batches]);passes=torch.cat([b for a,b in batches]);visited.update(ui.tolist());ux=U['x'][ui].to(device);ids=[U['ids'][i] for i in ui.tolist()]
     opt.zero_grad(set_to_none=True);loss,metrics,detail=objective(model,L['x'][li].to(device),L['y'][li].to(device),L['rx'][li].to(device),ux,ids,passes,bank,c,step,epoch)
     if not torch.isfinite(loss):raise FloatingPointError('CrossRX contrastive total loss')
     loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True);opt.step();counts['identity_updates']+=1;counts['L_packets']+=128
     for k in ('U_packets','accepted_U','teacher_forwards'):counts[k]+=metrics[k]
     counts['contrastive_updates']+=int(epoch>5);metrics['gradient_norm']=float(norm)
     for k,v in metrics.items():sums[k]=sums.get(k,0.)+v
     log.write(json.dumps(dict(step=step,epoch=epoch,metrics=metrics,detail=detail))+'\n')
    log.flush();r=dict(epoch=epoch,updates=step,lr=lr,**{k:v/222 for k,v in sums.items()},counts=dict(counts),distinct_U=len(visited));epochs.append(r);(out/'epoch_metrics.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in epochs),encoding='utf-8');print('EPOCH '+json.dumps(r),flush=True)
  if counts['identity_updates']!=6660 or counts['L_packets']!=852480 or counts['U_packets']!=710400 or counts['contrastive_updates']!=5550 or len(visited)!=56700:raise ValueError('Conditionalcontrastive30budget')
  torch.save(dict(model=model.state_dict(),config=c,readout='FINAL',epoch=30,updates=6660,ancestors=[],fit_ids=L['ids'],fit_U_ids=U['ids'],target_access=False,U_truth_read=False,initialization='scratch',actual_initialization_seed=c['member_init_seed']),out/'FINAL.pth');done=dict(status='SOURCE_TRAINED',config=c,epoch=30,updates=6660,logged_steps=6660,epochs=epochs,counts=counts,distinct_U=len(visited),ancestors=[],target_access=False,U_truth_read=False,V_forward=False,source_performance_gate=False,readouts=['FINAL'],elapsed_seconds=time.perf_counter()-start);d.write(out/'completion.json',done);d.write(out/'source_frozen.json',dict(status='SOURCE_FROZEN',config=c,readouts=['FINAL'],epoch=30,updates=6660,selection='fixed_final_epoch30',target_access=False));print('SOURCE_TRAINED '+rowid,flush=True)
 return done
