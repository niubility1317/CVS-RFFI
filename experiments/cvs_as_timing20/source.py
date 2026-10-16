"""Original source contracts/U/EMA/final20;all seven held-source readouts automatic."""
from contextlib import contextmanager
from pathlib import Path
import json,sys
import torch
from torch.nn import functional as F
from . import design as d,model as api
from experiments.cvs_additive_as20.source import Observer as ParentObserver
from experiments.cvs_crossrx_satmix20.source import Observer as OriginalObserver
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
 def initialize(self,m,c,out):
  super().initialize(m,c,out);d.write(Path(out)/'timing_actual_modules.json',dict(native=sys.modules['SSDG.train_ssdg'].__file__,augmentation=sys.modules['DataAugmentation'].__file__,timing=__import__('experiments.cvs_as_timing20.model',fromlist=['x']).__file__,zero_BatchNorm=True,no_new_model_parameters=True,U_unchanged=True,architecture_fixed=True))
 def epoch(self,m,ctx,rows,out,device):
  epoch=len(rows);steps=epoch*222;expected=d.packets(self.c['arm'],epoch)
  if m._feature_l_steps!=steps or m._joint_calls!=steps or m._joint_L_packets!=steps*1024 or m._additive_calls!=steps or m._additive_L_packets!=steps*256 or m._supcon_calls!=steps or m._supcon_applied!=max(0,epoch-5)*222 or m._supcon_pending is not None or m._supcon_handles or self.optimizer._sinc_physical_clip_calls!=steps or self.optimizer._sinc_unit_steps!=steps or m._timing_calls!=steps or m._timing_generated!=steps*128 or m._joint_As_packets!=expected or sum(m._joint_coverage.values())!=expected or self.teacher._timing_calls or self.reader._timing_calls:raise ValueError('Actual fixed20/8L/timing/U/teacher/source coverage differs')
  d.write(out/'timing_progress.json',dict(epoch=epoch,steps=steps,L_packets=m._joint_L_packets,extra_L_packets=m._additive_L_packets,As_generated=m._timing_generated,As_consumed=expected,active=d.active(self.c['arm'],epoch),supcon_applied=m._supcon_applied,target_access=False))
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
  result=OriginalObserver._final_record(self,done,payload,views);m=self.model
  if m._additive_calls!=4440 or m._additive_L_packets!=1136640 or self.teacher._additive_calls or self.teacher._joint_As_packets:raise ValueError('Original extra L/teacher budget differs')
  result.update(additive_steps=m._additive_calls,extra_L_forward_packets=m._additive_L_packets,total_pair_weight=1.25,teacher_additive_calls=0,final_native_logged_L_CE=result['final_L_CE'],final_base_L_objective=sum(v['base_total_L'] for v in m._joint_details[-222:])/222,final_extra_pair_objective=sum(v['extra_pair_loss'] for v in m._joint_details[-222:])/222)
  if m._timing_calls!=4440 or m._timing_generated!=568320 or sum(m._timing_generated_coverage.values())!=568320 or self.teacher._timing_calls or self.reader._timing_calls:raise ValueError('Complete generated and consumed source-only budget required')
  d.write(self.out/'timing_audit.json',dict(arm=self.c['arm'],timing=self.c['As_timing'],steps=m._timing_calls,generated=m._timing_generated,consumed=m._joint_As_packets,generated_coverage=[dict(TX=k[0],RX=k[1],day=k[2],scene=k[3],state=k[4],lock=k[5],count=v) for k,v in sorted(m._timing_generated_coverage.items())],new_parameters=0,extra_U=False,head_fit_updates=0,target_access=False))
  result.update(status='SOURCE_AS_TIMING_COMPLETE',As=self.c['arm']!='RAW',Ag=False,As_generated_packets=m._timing_generated,timing_steps=m._timing_calls,As_active_epochs=self.c['As_timing']['active_epochs']);result['views']={k:{a:b for a,b in v.items() if a not in ('physical_ids','labels','predictions','physics','margin_over30')} for k,v in views.items()};d.write(self.out/'satmix_result.json',result);return result

def train(c):
 observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
