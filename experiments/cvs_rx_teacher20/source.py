from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from . import design as d,model,subset
from experiments.cvs_feature_sequence50.source import train as shared_train
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
class Observer:
 def __init__(self,c):self.c=c
 def data(self,native,ctx,out):
  self.allowed,self.held,records=subset.apply(native,ctx,self.c['arm'])
  from cvsrffi.xuc_fusion.native import role_ids_from_native
  self.roles=role_ids_from_native(ctx)
  d.write(Path(out)/'teacher_subset.json',dict(config=self.c,train_records=records,fit_V_RX_counts={str(rx):sum(int(v.rx_i)==rx for v in ctx['val_loader'].dataset.index) for rx in d.FIT},U_iterator_calls=0,held_V_IQ_read_before_freeze=False,checkpoint_sources=[],ancestors=[]))
 def source_contract(self,expected,ctx):
  if {k:len(v) for k,v in self.roles.items()}!={'L_s':5040,'U_s':0,'V':21600} or any(not set(v)<=set(expected['role_ids'][k]) for k,v in self.roles.items()):raise ValueError('Source exposure escapes original legal roles')
  return dict(expected,role_ids=self.roles,source_receivers=self.c['teacher_exposure']['train_RX'],teacher_exposure=self.c['teacher_exposure'],native_role_comparison='EXACT_MATCH_CARRIER_THEN_TEACHER_SUBSET',carrier_contract_ref=d.SOURCE)
 def initialize(self,m,c,out):self.model=m;self.out=Path(out);m._teacher_allowed_L=self.allowed
 @contextmanager
 def native_scope(self,native):yield
 def epoch(self,m,ctx,rows,out,device):
  if m._feature_l_steps!=len(rows)*222 or m._teacher_calls!=len(rows)*222:raise ValueError('Teacher actual CE update/context budget differs')
  d.write(self.out/'teacher_progress.json',dict(epoch=len(rows),verified_L_steps=m._teacher_calls,unique_L_packets=len(m._teacher_seen),U_iterator_calls=0,EMA=False,target_access=False,head_fit_updates=0))
 def finish(self):
  done=d.read(self.out/'completion.json');freeze=d.read(self.out/'source_frozen.json')
  if done['epoch']!=20 or done['optimizer_steps']!=4440 or freeze['selection']!='final_only' or done['checkpoint_sources'] or done['head_fit_updates']:raise ValueError('Fixed own teacher must freeze before held source V forward')
  m=self.model.eval();device=next(m.parameters()).device;pred=[];ys=[];ids=[];logp=[];errors=ce=0
  loader=torch.utils.data.DataLoader(self.held,batch_size=128,shuffle=False,num_workers=0)
  with torch.no_grad():
   for x,y,domain,meta in loader:
    with fresh_parameter_caches(m):logits=m.forward_identity_only(x.to(device))['tx_logits'].double()
    labels=y.to(device)
    if logits.shape!=(len(y),6) or not torch.isfinite(logits).all() or not bool((meta['rx_i']==1).all()):raise ValueError('Finite frozen default-head RX1 source V required')
    p=logits.argmax(1);errors+=int((p!=labels).sum());ce+=float(F.cross_entropy(logits,labels,reduction='sum'))
    pred+=p.cpu().tolist();ys+=y.tolist();logp+=logits.log_softmax(1).cpu().tolist()
    ids += [list(map(int,v)) for v in zip(*[meta[k].tolist() for k in ('eq_i','rx_i','day_i','sig_i','base_index')])]
  if len(ids)!=5400 or len({tuple(v) for v in ids})!=5400:raise ValueError('Complete held source V required')
  result=dict(status='SOURCE_TEACHER_COMPLETE',config=self.c,epoch=20,optimizer_steps=4440,source_health=done['final_training_L_CE']<=.5,count=5400,errors=errors,accuracy=1-errors/5400,mean_CE=ce/5400,physical_ids=ids,labels=ys,predictions=pred,log_probabilities=logp,U_truth_read=False,target_access=False,head_fit_updates=0)
  d.write(self.out/'held_RX1_V.json',result);return result
def fit_validation(model,loader,device):
 model.eval();hits=total=0;counts={}
 with torch.no_grad():
  for x,y,domain,meta in loader:
   with fresh_parameter_caches(model):ok=model(x.to(device)).argmax(1).cpu()==y
   hits+=int(ok.sum());total+=len(y)
   for rx in meta['rx_i'].unique().tolist():
    mask=meta['rx_i']==rx;n=counts.setdefault(str(rx),[0,0]);n[0]+=int(ok[mask].sum());n[1]+=int(mask.sum())
 rates={k:h/n for k,(h,n) in counts.items()}
 if total!=21600 or set(rates)!=set(map(str,d.FIT)):raise ValueError('Fixed fit-only V differs')
 return dict(source_val_count=total,source_val_accuracy=hits/total,source_val_rx_accuracy=rates,source_val_worst_rx=min(rates.values()))
def train(c):
 d.validate(c);view=SimpleNamespace(**{k:getattr(d,k) for k in ('validate','make_args','read','write','SOURCE','FULL_FP32_POLICY','ROOT')},UPDATES=4440,source_validation=fit_validation)
 observer=Observer(c);shared_train(c,design=view,model_api=model,observer=observer);return observer.finish()
