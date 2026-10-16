"""Original E20 loop, exact source fold and default head;independent extra observation only."""
from contextlib import contextmanager
from pathlib import Path
import gzip,json,sys
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .iq import Runtime,AugmentorProxy
from experiments.cvs_additive_as20.source import Observer as ParentObserver
from experiments.cvs_feature_sequence50.source import train as shared_train
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_u_schedule50.counterfactual import _preserve_rng

def serial(v):
 if isinstance(v,dict):return {str(k):serial(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [serial(x) for x in v]
 if hasattr(v,'tolist'):return serial(v.tolist())
 if isinstance(v,complex):return [v.real,v.imag]
 if isinstance(v,float) and not __import__('math').isfinite(v):return None
 return v

@contextmanager
def capture_physics(channel):
 cls=channel.residual.ResidualChannel;old=cls.process;records=[]
 def call(instance,x):
  y,m=old(instance,x);records.append(serial(m));return y,m
 cls.process=call
 try:yield records
 finally:cls.process=old

class Observer(ParentObserver):
 def __init__(self,c):super().__init__(c);self.runtime=Runtime(c['arm'])
 def data(self,native,ctx,out):
  super().data(native,ctx,out);self.refresh_stream=gzip.open(Path(out)/'refresh_steps.jsonl.gz','xt',encoding='utf8',compresslevel=3);self.runtime.stream=self.refresh_stream
 def initialize(self,m,c,out):
  super().initialize(m,c,out);m._refresh_runtime=self.runtime;d.write(Path(out)/'refresh_actual_modules.json',dict(native=sys.modules['SSDG.train_ssdg'].__file__,augmentation=sys.modules['DataAugmentation'].__file__,refresh=__import__('experiments.cvs_f_viewrefresh20.iq',fromlist=['x']).__file__,all_scalar_parameters_native=True,PA_DAC_class_signature_unchanged=True,U_grouping=False,network_and_default_head_unchanged=True))
 @contextmanager
 def native_scope(self,native):
  old=native.make_augmentor;old_move=native.move_batch
  def moved(*args,**kwargs):
   value=old_move(*args,**kwargs);x,y,extra=value
   if torch.is_tensor(y) and y.numel()==len(x) and bool((y>=0).all()):self.runtime.set_L(extra[1])
   else:self.runtime.set_U()
   return value
  def factory(*args,**kwargs):return AugmentorProxy(old(*args,**kwargs),self.runtime)
  native.make_augmentor=factory;native.move_batch=moved
  try:
   with super().native_scope(native):yield
  finally:
   native.make_augmentor=old;native.move_batch=old_move
   if hasattr(self,'refresh_stream'):self.refresh_stream.close();self.runtime.stream=None
 def epoch(self,m,ctx,rows,out,device):
  super().epoch(m,ctx,rows,out,device);steps=len(rows)*222
  if self.runtime.active or self.runtime.pending or self.runtime.calls!=steps or self.runtime.consumed!=steps or self.runtime.fresh_packets!=steps*128 or self.teacher._refresh_runtime is not None or self.reader._refresh_runtime is not None:raise ValueError('Exactly one labeled ordinary view/two stages per real step required')
  self.refresh_stream.flush();d.write(Path(out)/'refresh_progress.json',dict(epoch=len(rows),steps=steps,**self.runtime.summary(),source_only=True,target_access=False))
 def evaluate(self,state):
  self.reader.load_state_dict(state,strict=True);self.reader.eval();device=next(self.reader.parameters()).device;pred=[];truth=[];margin=[];ce=0.
  with _preserve_rng(),torch.no_grad(),fresh_parameter_caches(self.reader):
   for start in range(0,1260,128):
    x=self.held['x'][start:start+128].to(device);y=self.held['y'][start:start+128].to(device);z=self.reader.forward_identity_only(x)['tx_logits'].double()
    if z.shape!=(len(y),6) or not bool(torch.isfinite(z).all()):raise ValueError('Finite held default logits required')
    other=z.masked_fill(F.one_hot(y,6).bool(),-torch.inf);margin+=((z.gather(1,y[:,None]).flatten()-other.max(1).values)/30).cpu().tolist();ce+=float(F.cross_entropy(z,y,reduction='sum'));pred+=z.argmax(1).cpu().tolist();truth+=y.cpu().tolist()
  return dict(count=1260,accuracy=sum(a==b for a,b in zip(pred,truth))/1260,mean_CE=ce/1260,macro_F1=d.macro_f1(truth,pred),mean_margin=sum(margin)*30/1260,margin_over30=margin,predictions=pred,labels=truth,physical_ids=self.held['ids'],head_fit_updates=0)
 def scene(self,scene):
  labels=[];pred=[];margin=[];physics=[];ce=0.;device=next(self.reader.parameters()).device
  with _preserve_rng(),torch.no_grad(),fresh_parameter_caches(self.reader):
   for start in range(0,1260,128):
    raw=self.held['x'][start:start+128].to(device);y=self.held['y'][start:start+128].to(device);ids=self.held['ids'][start:start+128];keys=('eq_i','rx_i','day_i','sig_i','base_index');meta={k:torch.tensor([v[i] for v in ids],dtype=torch.long) for i,k in enumerate(keys)}
    with capture_physics(self.reader._joint_channel) as full:x,observed=self.reader._joint_channel.apply(raw,meta,namespace='source_eval/RX1/final20',scene=scene)
    if len(full)!=len(y) or len(observed)!=len(y):raise ValueError('One original physical source channel per packet')
    physics.extend({**a,**b} for a,b in zip(full,serial(observed)));z=self.reader.forward_identity_only(x)['tx_logits'].double()
    if z.shape!=(len(y),6) or not bool(torch.isfinite(z).all()):raise ValueError('Finite complete source readout required')
    other=z.masked_fill(F.one_hot(y,6).bool(),-torch.inf);margin+=((z.gather(1,y[:,None]).flatten()-other.max(1).values)/30).cpu().tolist();ce+=float(F.cross_entropy(z,y,reduction='sum'));pred+=z.argmax(1).cpu().tolist();labels+=y.cpu().tolist()
  result=dict(count=1260,accuracy=sum(a==b for a,b in zip(pred,labels))/1260,mean_CE=ce/1260,macro_F1=d.macro_f1(labels,pred),margin_over30=margin,physics=physics,predictions=pred,labels=labels,physical_ids=self.held['ids'],head_fit_updates=0);d.write(self.out/'source_readouts'/(scene+'.json'),result);return result
 def _final_record(self,done,payload,views):
  result=super()._final_record(done,payload,views);r=self.runtime
  if r.active or r.pending or r.calls!=4440 or r.consumed!=4440 or r.fresh_packets!=568320 or set(r.coverage)!={(y,rx,day) for y in range(6) for rx in d.FIT for day in (1,2,3)}:raise ValueError('Complete unchanged L training coverage required')
  d.write(self.out/'refresh_audit.json',dict(summary=r.summary(),coverage=[dict(TX=k[0],RX=k[1],day=k[2],count=v) for k,v in sorted(r.coverage.items())],model_parameters_added=0,original_U_unchanged=True,source_only=True,target_access=False))
  result.update(Ag='existing_IQ_refresh_candidate_not_validated' if self.c['arm']=='REFRESH' else False,As=False,view_refresh=r.summary());result['views']={k:{a:b for a,b in v.items() if a not in ('physical_ids','labels','predictions','physics','margin_over30')} for k,v in views.items()};d.write(self.out/'satmix_result.json',result);return result

def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
