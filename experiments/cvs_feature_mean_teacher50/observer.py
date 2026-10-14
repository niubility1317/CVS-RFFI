import gzip,json
from contextlib import contextmanager
import torch
from . import design as d
from .objective import consistency,coefficient
from experiments.cvs_sigtime_pair50.observer import Observer as Parent,derivative
from experiments.cvs_source_layer_order.probe import gradient_stats

class Observer(Parent):
 def __init__(self,c):
  d.validate(c)
  # Parent counters and source sampler are reused, with this candidate's design.
  self.c=c;self.steps=0;self.U_steps=0;self.U_packets=0;self.U_accepted=0;self.records=[];self.raw=[];self.saved=[];self.sampling=[];self.pending=None;self.extra_records=[];self.extra_raw=[];self.extra_saved=[];self.feature_packets=0
 def initialize(self,model,c,out):
  d.validate(c)
  if c!=self.c or model._source_neighbor_sampling['mode']!='RANDOM':raise ValueError('Fixed native RANDOM U path differs')
  self.model=model;self.out=out;self.params=[dict(model.named_parameters())[n] for n in d.PARAMETERS];self.head=model.id_backbone.encoder.core.id_backbone.cls_head.weight
  if not all(p.requires_grad for p in self.params+[self.head]):raise ValueError('Default feature/head joint training disabled')
  self.stream=gzip.open(out/'feature_mean_teacher_steps.jsonl.gz','xt',encoding='utf8',compresslevel=3)
 @contextmanager
 def native_scope(self,n):
  if n._unlabeled_feature_observer_hook is not None:raise ValueError('Another U feature owner active')
  with super().native_scope(n):
   n._unlabeled_feature_observer_hook=self.capture;n._source_feature_auxiliary_hook=self.aux
   try:yield
   finally:n._unlabeled_feature_observer_hook=None;n._source_feature_auxiliary_hook=None;self.stream.close()
 def capture(self,*,model,args,epoch,batch_idx,teacher_features,student_features):
  if model is not self.model or epoch<34 or self.pending is not None or teacher_features.requires_grad or teacher_features.shape!=student_features.shape:raise ValueError('Same-packet actual U feature graph differs')
  self.pending=dict(epoch=epoch,batch=batch_idx,loss=consistency(student_features,teacher_features),count=len(student_features));self.feature_packets+=len(student_features)
 def observe(self,**kw):
  self.main_CE=kw['loss_l']
  # Parent runtime's fixed step/gradient constants match this identical native recipe.
  return super().observe(**kw)
 def aux(self,model,step):
  if model is not self.model or step!=self.steps:raise ValueError('Actual native feature step differs')
  epoch=(step-1)//222+1;batch=(step-1)%222+1;active=epoch>=34
  if active:
   if self.pending is None or (self.pending['epoch'],self.pending['batch'])!=(epoch,batch):raise ValueError('Missing same-step U features')
   loss=self.pending['loss'];count=self.pending['count']
  else:
   if self.pending is not None:raise ValueError('U feature signal before pseudo stage')
   loss=self.main_CE.detach()*0;count=0
  weight=coefficient(epoch,self.c['feature_mean_teacher']['mode']);effective=weight*loss
  if active and torch.autograd.grad(loss,self.head,allow_unused=True,retain_graph=True)[0] is not None:raise ValueError('Feature objective depends on classW')
  if not torch.isfinite(effective):raise FloatingPointError('Nonfinite feature consistency')
  if epoch in d.DIAGNOSTIC_EPOCHS:
   vectors=dict(main_L_CE=derivative(self.main_CE,self.params),feature=derivative(loss,self.params),effective=derivative(effective,self.params))
   self.extra_raw.append(dict(step=step,epoch=epoch,vectors=vectors));self.extra_records.append(dict(step=step,epoch=epoch,layers=[dict(parameter=p,**{key:gradient_stats(vectors['main_L_CE'][k],vectors[key][k]) for key in ('feature','effective')}) for k,p in enumerate(d.PARAMETERS)]))
  self.stream.write(json.dumps(dict(step=step,epoch=epoch,batch=batch,U_packets=count,feature_loss=float(loss.detach()),coefficient=weight,effective_loss=float(effective.detach())),allow_nan=False)+chr(10));self.pending=None;self.main_CE=None
  return effective,dict(feature_mean_teacher_loss=float(loss.detach()),feature_mean_teacher_coefficient=weight,feature_mean_teacher_U_packets=count)
 def epoch(self,model,ctx,rows,out,device):
  super().epoch(model,ctx,rows,out,device);epoch=int(rows[-1]['epoch']);self.stream.flush()
  if self.feature_packets!=max(0,epoch-33)*56700:raise ValueError('AllU feature exposure differs')
  if epoch in d.DIAGNOSTIC_EPOCHS:
   if len(self.extra_raw)!=222:raise ValueError('Feature gradient period incomplete')
   path=out/f'feature_mean_teacher_gradients_e{epoch}.pt';torch.save(self.extra_raw,path);saved=torch.load(path,map_location='cpu',weights_only=False);records=[r for r in self.extra_records if r['epoch']==epoch]
   for a,b in zip(saved,records):
    for k,p in enumerate(d.PARAMETERS):
     for key in ('feature','effective'):
      if gradient_stats(a['vectors']['main_L_CE'][k],a['vectors'][key][k])!=b['layers'][k][key]:raise ValueError('Saved actual feature gradient recount differs')
   self.extra_raw=[];self.extra_saved.append(epoch);d.write(out/'feature_mean_teacher_gradient_records.json',dict(status='VERIFIED',epochs=self.extra_saved,batch_records=self.extra_records,saved_gradient_recount='VERIFIED',no_classW_gradient=True,extra_model_forwards=0,head_fit_updates=0,U_truth_read=False))
 def finish(self):
  parent=super().finish()
  if self.pending is not None or self.feature_packets!=963900 or self.extra_saved!=[33,34,50] or len(self.extra_records)!=666:raise ValueError('Actual feature consistency budget incomplete')
  v=dict(status='VERIFIED',steps=11100,U_feature_packets=self.feature_packets,U_feature_steps=3774,gradient_epochs=self.extra_saved,gradient_batches=666,layer_gradient_records=1998,saved_vectors=5994,extra_model_forwards=0,head_fit_updates=0,auxiliary_classW_dependency=False,default_head_jointly_trained=True,U_truth_read=False,hard_pseudo_loss_returned_unchanged=True,mode=self.c['feature_mean_teacher']['mode'],saved_gradient_recount='VERIFIED')
  d.write(self.out/'feature_mean_teacher_training_observation.json',v);return v
