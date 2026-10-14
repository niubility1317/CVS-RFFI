"""Call-local supervised hooks; EMA/U/eval unchanged, no added trainable state."""
from contextlib import contextmanager
from pathlib import Path
import torch
from . import design as default_design
d=default_design
from experiments.cvs_phase1_stack.runtime import installed as native_installed
from experiments.cvs_phase1_overlay.model import Phase1IdentityAdapter
from experiments.cvs_source_identity_readout.input_policy import apply
from experiments.cvs_feature_multiscale50.model import source_domain_count
from experiments.cvs_e20_budget_fishr.multilevel import weak_mixstyle_u

def metadata(y,extra):
 if y.ndim!=1 or y.dtype!=torch.long or len(extra)!=2 or not isinstance(extra[1],dict):raise ValueError('Actual L metadata boundary differs')
 raw=extra[1];values={}
 for k in ('rx_i','day_i','eq_i','sig_i','base_index'):
  t=raw.get(k)
  if not torch.is_tensor(t) or t.ndim!=1 or len(t)!=len(y) or t.dtype!=torch.long:raise ValueError('Actual physical L metadata missing: '+k)
  values[k]=t.detach().cpu()
 if not set(y.detach().cpu().tolist())<=set(range(6)) or not set(values['rx_i'].tolist())<={1,3,4,6,8} or not set(values['day_i'].tolist())<={1,2,3}:raise ValueError('Forbidden source labels/RX/day')
 ids=[tuple(int(values[k][i]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')) for i in range(len(y))]
 return dict(y=y.detach(),rx=values['rx_i'],day=values['day_i'],ids=ids)

def intervene(value,meta,seed):
 mixed,detail=weak_mixstyle_u(value,meta['y'],meta['rx'],meta['day'],meta['ids'],seed,p=.5,alpha=.1,eta=1.)
 if detail['enabled']:
  y=meta['y'].detach().cpu().tolist();rx=meta['rx'].tolist();day=meta['day'].tolist()
  for i,j in enumerate(detail['donors']):
   if i!=j and not (y[i]==y[j] and day[i]==day[j] and rx[i]!=rx[j] and meta['ids'][i]!=meta['ids'][j]):raise ValueError('Identity-protected actual donor failed')
 return mixed,detail

def install(model,c,*,design=None):
 d=design or default_design
 d.validate(c);model._identity_input_policy=apply(model);model._source_mixstyle=dict(c['identity_mixstyle'],head='unchanged cosine6x160',separate_head_fit=False)
 b=model.id_backbone
 if b.mixstyle_on:raise ValueError('Original MixStyle unexpectedly enabled')
 b._source_style_contract=dict(c['identity_mixstyle']);b._source_style_seed=c['model_seed'];b._source_style_calls=0;b._source_style_history=[]
 return model

@contextmanager
def installed(c,training=False,*,design=None):
 d=design or default_design
 d.validate(c)
 with native_installed(c) as n:
  previous_build=n.build_baseline_model;previous_move=n.move_batch;previous_forward=Phase1IdentityAdapter.forward;context={}
  if n._source_feature_auxiliary_hook is not None:raise ValueError('Unregistered auxiliary hook')
  def build(args,device):return install(previous_build(args,device),c,design=d).to(device)
  def move(batch,device):
   x,y,extra=previous_move(batch,device)
   # U remains -1 and no U metadata enters the intervention. V may be fetched
   # by native validation but eval/y=None prevents hooks and it cannot train.
   context.clear()
   if bool((y>=0).all()):context.update(metadata(y,extra))
   return x,y,extra
  def forward(self,x,y=None,return_aux=False,domain_labels=None):
   contract=getattr(self,'_source_style_contract',None)
   active=self.training and contract and contract['enabled'] and y is not None
   if not active:return previous_forward(self,x,y,return_aux,domain_labels)
   if not context or len(y)!=len(context['y']) or not torch.equal(y,context['y']):raise ValueError('Supervised feature hook missing exact L context')
   if not bool((y>=0).all()) or len(x)!=len(y):raise ValueError('U/expanded input intervention forbidden')
   self._source_style_calls+=1;call=self._source_style_calls;record=dict(call=call,count=len(y),layers={});hooks=[];core=self.encoder.core.id_backbone
   def hook(name):
    def run(module,args,out):
     mixed,detail=intervene(out,context,self._source_style_seed+95001+call*701+d.STYLE_LAYERS.index(name)*1709)
     # Retain aggregate counts only; no raw source identity or donor list output.
     record['layers'][name]={k:v for k,v in detail.items() if k!='donors'}
     return mixed
    return run
   try:
    for name in d.STYLE_LAYERS:hooks.append(getattr(core,name).register_forward_hook(hook(name)))
    result=previous_forward(self,x,y,return_aux,domain_labels)
   finally:
    for h in hooks:h.remove()
   if set(record['layers'])!=set(d.STYLE_LAYERS):raise ValueError('Actual shallow-middle hook bypassed')
   self._source_style_history.append(record)
   return result
  n.build_baseline_model=build
  if training:n.move_batch=move;Phase1IdentityAdapter.forward=forward
  try:
   if training:
    from experiments.cvs_time_factorial.monitor import installed_monitor
    with installed_monitor(n,c,d.write):yield n
   else:yield n
  finally:n.build_baseline_model=previous_build;n.move_batch=previous_move;Phase1IdentityAdapter.forward=previous_forward

def build(c,device='cpu',*,design=None):
 d=design or default_design
 a=d.make_args(c,str(device));a.num_domains=source_domain_count(a);a.input_len=256
 with installed(c,design=d) as n:return n.build_baseline_model(a,torch.device(device))
