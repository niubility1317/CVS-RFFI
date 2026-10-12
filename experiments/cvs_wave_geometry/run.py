import json,math,os,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .geometry import Head,features,projection,diagnostics
from experiments.cvs_nocfo_foundation.source import load_source
from experiments.cvs_stage2_decouple.source import seed_all
from experiments.cvs_equivariant_identity.precision import numerical_context,actual_flags

def score(logits,y,rx,day):
 pred=logits.argmax(1);cm=torch.bincount(y*6+pred,minlength=36).reshape(6,6)
 denom=cm.sum(0)+cm.sum(1);f1=torch.where(denom>0,2*cm.diag()/denom,0.).mean()
 mask=torch.eye(6,dtype=torch.bool)[y];other=logits.masked_fill(mask,-torch.inf).max(1).values
 margins=logits.gather(1,y[:,None]).squeeze(1)-other
 pairs=[]
 for a in range(6):
  for b in range(a+1,6):
   choose=(y==a)|(y==b);s=torch.where(y[choose]==a,logits[choose,a]-logits[choose,b],logits[choose,b]-logits[choose,a])
   pairs.append(dict(classes=[a,b],count=int(choose.sum()),binary_error=float((s<=0).float().mean()),mean_signed_margin=float(s.mean()),confusion_a_to_b=int(cm[a,b]),confusion_b_to_a=int(cm[b,a])))
 return dict(count=len(y),accuracy=float((pred==y).float().mean()),macro_f1=float(f1),NLL=float(F.cross_entropy(logits,y)),mean_margin=float(margins.mean()),confusion=cm.tolist(),class_pairs=pairs,by_day={str(k):dict(count=int((day==k).sum()),accuracy=float((pred[day==k]==y[day==k]).float().mean())) for k in day.unique()},by_rx={str(k):dict(count=int((rx==k).sum()),accuracy=float((pred[rx==k]==y[rx==k]).float().mean())) for k in rx.unique()})

@torch.no_grad()
def encode(x,kind,k,device):
 return torch.cat([features(x[b:b+256].to(device),kind,k).cpu() for b in range(0,len(x),256)])

@torch.no_grad()
def predict(head,z,device):return torch.cat([head(z[b:b+256].to(device)).cpu() for b in range(0,len(z),256)])

def run(rowid,device='cuda:0'):
 c=d.config(next(r for r in d.rows() if r['row_id']==rowid));d.validate(c)
 out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 start=time.perf_counter();L,U,V,contract=load_source(c['source_loader_config'])
 if U is not None:raise ValueError('U must remain closed')
 fit=(L['rx']!=c['held_rx']);test=(L['rx']==c['held_rx']);vt=(V['rx']==c['held_rx'])
 if int(fit.sum())!=5040 or int(test.sum())!=1260 or int(vt.sum())!=5400:raise ValueError('Whole sourceRX partition differs')
 if set(L['ids'][i] for i in fit.nonzero().flatten().tolist())&set(L['ids'][i] for i in test.nonzero().flatten().tolist()):raise ValueError('Holdout physical overlap')
 steps=math.ceil(int(fit.sum())/c['batch_size']);k=projection(c['projection_width'],c['projection_seed']).to(device)
 resolved=dict(config=c,commit=(d.ROOT/'release_commit.txt').read_text().strip(),pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,backend_flags=actual_flags(),training_rx=sorted(L['rx'][fit].unique().tolist()),L_fit=int(fit.sum()),L_held=int(test.sum()),V_held=int(vt.sum()),U_opened=False,checkpoint_loaded=False,target_access=False,explicit_CFO_feature=False,rank_diagnostics=dict(L=diagnostics(L['x'],k.cpu()),V=diagnostics(V['x'][vt],k.cpu())))
 d.write(out/'resolved_config.json',resolved);d.write(out/'source_contract.json',contract)
 print('RESOLVED '+json.dumps(resolved),flush=True);results=[]
 with numerical_context(d.FULL_FP32_POLICY):
  for arm in d.ARMS:
   seed_all(c['model_seed']);head=Head().to(device);opt=torch.optim.AdamW(head.parameters(),lr=c['peak_lr'],weight_decay=c['weight_decay'])
   zfit=encode(L['x'][fit],arm,k,device);ztest=encode(L['x'][test],arm,k,device);zv=encode(V['x'][vt],arm,k,device);yfit=L['y'][fit]
   ap=out/arm;ap.mkdir();updates=0;history=[]
   g=torch.Generator().manual_seed(c['model_seed']+101)
   with (ap/'epoch_metrics.jsonl').open('x',encoding='utf-8') as log:
    for epoch in range(1,21):
     head.train();order=torch.randperm(len(zfit),generator=g);total=0.;correct=0
     lr=c['min_lr']+(c['peak_lr']-c['min_lr'])*(1+math.cos(math.pi*(epoch-1)/19))/2
     for group in opt.param_groups:group['lr']=lr
     for begin in range(0,len(order),128):
      idx=order[begin:begin+128];y=yfit[idx].to(device);opt.zero_grad(set_to_none=True)
      logits=head(zfit[idx].to(device));loss=F.cross_entropy(logits,y)
      if not torch.isfinite(loss):raise FloatingPointError('Source head loss')
      loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),10.,error_if_nonfinite=True);opt.step();updates+=1
      total+=float(loss)*len(y);correct+=int((logits.argmax(1)==y).sum())
     epochrow=dict(epoch=epoch,updates=updates,lr=lr,train_CE=total/len(zfit),train_accuracy=correct/len(zfit));history.append(epochrow);log.write(json.dumps(epochrow)+'\n');log.flush()
     print('EPOCH '+json.dumps(dict(row_id=rowid,arm=arm,**epochrow)),flush=True)
   if updates!=20*steps:raise ValueError('Head update budget differs')
   head.eval();lt=predict(head,ztest,device);vv=predict(head,zv,device)
   # Freeze source predictions before computing source validation metrics.
   torch.save(dict(L_logits=lt,V_logits=vv,L_ids=[L['ids'][i] for i in test.nonzero().flatten().tolist()],V_ids=[V['ids'][i] for i in vt.nonzero().flatten().tolist()],truth_read=False,target_access=False),ap/'predictions.pt')
   torch.save(dict(model=head.state_dict(),config=c,arm=arm,epoch=20,updates=updates,ancestors=[],training_rx=resolved['training_rx'],held_rx=c['held_rx'],target_access=False),ap/'head.pth')
   record=dict(arm=arm,held_rx=c['held_rx'],model_seed=c['model_seed'],parameters=sum(p.numel() for p in head.parameters()),epochs=history,updates=updates,L_held=score(lt,L['y'][test],L['rx'][test],L['day'][test]),V_held=score(vv,V['y'][vt],V['rx'][vt],V['day'][vt]))
   d.write(ap/'metrics.json',record);results.append(record);del zfit,ztest,zv,head,opt
 done=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',config=c,records=results,target_access=False,U_opened=False,neural_backbone_updates=0,head_updates=4*20*steps,elapsed_seconds=time.perf_counter()-start)
 d.write(out/'completion.json',done);print('SOURCE_DIAGNOSTIC_COMPLETE '+json.dumps(dict(row_id=rowid,head_updates=done['head_updates'])),flush=True)
 return done
