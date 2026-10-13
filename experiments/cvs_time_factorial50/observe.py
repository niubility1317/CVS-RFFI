"""Read-only source training observations; masks/results returned unchanged."""
from contextlib import contextmanager
import torch
from . import design as d
from .model import classifier
from experiments.cvs_feature_restitution.features import layout
class Observer:
 def initialize(self,model,c,out):
  self.model=model;self.c=c;self.initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()};self.groups={};self.pending=None;self.handle=None;self.stage_z=None
 def collect(self,module,args,kwargs,result):
  if self.pending is None or not module.training or kwargs.get('y_tx') is not None:return
  p=self.pending;pred=result['tx_logits'].detach().argmax(1).cpu();self.pending=None
  if pred.shape!=p['pseudo'].shape:raise ValueError('U observer teacher/student sample alignment differs')
  agree=pred==p['pseudo'];accepted=p['base'] & agree
  for rx,day in sorted(set(zip(p['rx'].tolist(),p['day'].tolist()))):
   ix=(p['rx']==rx)&(p['day']==day);g=self.groups.setdefault(f'{rx}/{day}',dict(total=0,accepted=0,agreement=0,predicted_class=[0]*6,accepted_class=[0]*6))
   g['total']+=int(ix.sum());g['accepted']+=int((ix&accepted).sum());g['agreement']+=int((ix&agree).sum())
   g['predicted_class']=[a+b for a,b in zip(g['predicted_class'],torch.bincount(p['pseudo'][ix],minlength=6).tolist())]
   g['accepted_class']=[a+b for a,b in zip(g['accepted_class'],torch.bincount(p['pseudo'][ix&accepted],minlength=6).tolist())]
 @contextmanager
 def native_scope(self,n):
  threshold,temporal=n._threshold_mask,n._temporal_mask_tensor
  def conf(*a,**kw):
   result=threshold(*a,**kw);self.conf=result.detach().cpu();return result
  def time(pseudo,confidence,extra,args,device):
   result=temporal(pseudo,confidence,extra,args,device)
   if args.pseudo_domain_gate or not args.pseudo_strong_agreement:raise ValueError('Read-only U observer requires registered no-domain/strong-agreement gates')
   meta=n._meta_from_extra(extra)
   if any(k in meta for k in ('tx','tx_i','true_tx_i','tx_label','y_tx')):raise ValueError('Hidden U metadata exposed TX')
   if self.pending is not None:raise ValueError('Unconsumed U observation')
   self.pending=dict(pseudo=pseudo.detach().cpu(),base=self.conf&result.detach().cpu(),rx=meta['rx_i'].detach().cpu(),day=meta['day_i'].detach().cpu())
   # EMA was already deepcopied; attach only now to the original student.
   if self.handle is None:self.handle=self.model.register_forward_hook(self.collect,with_kwargs=True)
   return result
  n._threshold_mask=conf;n._temporal_mask_tensor=time
  try:yield
  finally:
   n._threshold_mask=threshold;n._temporal_mask_tensor=temporal
   if self.handle is not None:self.handle.remove();self.handle=None
 def epoch(self,model,ctx,rows,out,device):
  epoch=len(rows)
  if self.pending is not None:raise ValueError('U observer failed to consume strong prediction')
  record=dict(epoch=epoch,U_truth_read=False,groups=self.groups,gate_changed=False,head_fit_updates=0);self.groups={}
  if epoch>=34 and sum(g['total'] for g in record['groups'].values())!=56700:raise ValueError('U observer full epoch count differs')
  with (out/'source_mechanism_epochs.jsonl').open('a',encoding='utf-8') as f:
   import json
   f.write(json.dumps(record)+'\n')
  if epoch not in (32,33,34,40,50):return
  was=model.training;core,prefix,_=layout(model);device_ids=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
  with torch.random.fork_rng(devices=device_ids),torch.no_grad():
   from experiments.cvs_feature_sequence50.source import source_validation
   snapshot=dict(epoch=epoch,source=source_validation(model,ctx['val_loader'],device),parameter_updates={})
   for group in ('sinc','t1','t2','t3','f_proj','pa_b1','pa_b2','pa_b3','cls_head'):
    pairs=[(p.detach().cpu(),self.initial[n]) for n,p in model.named_parameters() if n.startswith(prefix+'.'+group+'.')]
    snapshot['parameter_updates'][group]=dict(changed_values=sum(int((p!=q).sum()) for p,q in pairs),l2_change=sum(float((p-q).square().sum()) for p,q in pairs)**.5)
   x=torch.stack([ctx['train_loader'].dataset[i][0] for i in range(32)]).to(device);z=model.forward_identity_only(x)['id_feat_joint'].cpu()
   if epoch==33:self.stage_z=z.clone()
   snapshot['feature_change_vs_epoch33_L32']=float((z-self.stage_z).norm(dim=1).mean()) if self.stage_z is not None else None
   d.write(out/f'source_stage_epoch{epoch}.json',snapshot)
  model.train(was)
