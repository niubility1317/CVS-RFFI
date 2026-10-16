"""Cache first pre-PA IQ and sample one independent native pre-PA observation."""
from contextlib import contextmanager
from collections import Counter
import random,time,types,json
import torch
SEED=20261017961
KEYS={'anti':('p_dc_offset','p_bandedge_taper'),'channel':('p_time_shift','p_amp_scale','p_phase_rot','p_cfo','p_phase_noise','p_multipath','p_awgn')}
class TorchProxy:
 def __init__(self):self.draws=[]
 def __getattr__(self,k):return getattr(torch,k)
 def rand(self,*args,**kwargs):
  value=torch.rand(*args,**kwargs)
  if args!=((),) or kwargs:raise ValueError('Native direct gate signature changed')
  self.draws.append(float(value));return value

def observed(method,x,name,base):
 proxy=TorchProxy();fn=method.__func__;copy=types.FunctionType(fn.__code__,dict(fn.__globals__,torch=proxy),fn.__name__,fn.__defaults__,fn.__closure__);copy.__kwdefaults__=fn.__kwdefaults__;y=copy(method.__self__,x);keys=[k for k in KEYS[name] if getattr(base,k)>0]
 if len(keys)!=len(proxy.draws):raise ValueError('Native gate order differs')
 return y,{k:dict(probability=float(getattr(base,k)),triggered=u<float(getattr(base,k))) for k,u in zip(keys,proxy.draws)}
@contextmanager
def private_rng(seed,x):
 py=random.getstate();cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state(x.device) if x.is_cuda else None
 try:
  with torch.random.fork_rng(devices=[x.device.index] if x.is_cuda else [],enabled=True):
   torch.manual_seed(seed);random.seed(seed);yield
 finally:
  random.setstate(py)
  if not torch.equal(cpu,torch.get_rng_state()) or random.getstate()!=py or (gpu is not None and not torch.equal(gpu,torch.cuda.get_rng_state(x.device))):raise ValueError('Fresh IQ changed native RNG stream')
class Runtime:
 def __init__(self,arm,stream=None):
  if arm not in ('RAW','REPLAY','REFRESH'):raise ValueError('Only fixed three view modes')
  self.arm=arm;self.stream=stream;self.role=None;self.meta=None;self.active=False;self.pending=False;self.calls=0;self.consumed=0;self.fresh_packets=0;self.seconds=0.;self.coverage=Counter();self.nonidentical_steps=0;self.first_gate_counts=Counter();self.fresh_gate_counts=Counter();self.gate_disagreement=Counter()
 def __deepcopy__(self,memo):return None
 def set_L(self,meta):self.role='L';self.meta=meta
 def set_U(self):self.role='U';self.meta=None
 def begin(self,x,y):
  if self.active or self.pending or self.meta is None or tuple(x.shape)!=(128,2,256) or y.shape!=(128,) or y.dtype!=torch.long or not bool(((y>=0)&(y<6)).all()) or not bool(torch.isfinite(x).all()):raise ValueError('Exact visible L128 once per step required')
  self.ids=list(zip(*(self.meta[k].detach().cpu().tolist() for k in ('eq_i','rx_i','day_i','sig_i','base_index'))))
  if len(self.ids)!=128 or any(v[0]!=1 or v[1] not in (3,4,6,8) or v[2] not in (1,2,3) for v in self.ids):raise ValueError('Only fit source L roles permitted')
  self.labels=y.detach().clone();self.raw=x.detach().clone();self.first=None;self.stages=[];self.calls+=1;self.active=True
  for tx,pid in zip(y.detach().cpu().tolist(),self.ids):self.coverage[int(tx),int(pid[1]),int(pid[2])]+=1
 def stage(self,method,name,base,x):
  if not self.active:return method(x)
  y,gates=observed(method,x,name,base);self.stages.append(dict(stage=name,gates=gates))
  if name=='channel':self.first=y.detach().clone()
  return y
 def finish_normal(self,base,anti,channel,normal):
  if not self.active or len(self.stages)!=2 or self.first is None or normal.shape!=self.raw.shape or not bool(torch.isfinite(normal).all()) or base.p_dac or base.enable_class_signature:raise ValueError('Exact original normal/pre-PA path required')
  started=time.perf_counter()
  with private_rng(SEED+self.calls,self.raw):
   z,a=observed(anti,self.raw,'anti',base);z,b=observed(channel,z,'channel',base)
  if not bool(torch.isfinite(z).all()):raise ValueError('Finite fresh native pre-PA IQ required')
  self.fresh=z.detach();self.normal=normal.detach();self.fresh_stages=[dict(stage='anti',gates=a),dict(stage='channel',gates=b)];self.fresh_packets+=128;self.seconds+=time.perf_counter()-started;self.nonidentical_steps+=not torch.equal(self.first,self.fresh)
  for one,two in zip(self.stages,self.fresh_stages):
   if one['stage']!=two['stage'] or set(one['gates'])!=set(two['gates']):raise ValueError('Fresh stage support differs')
   for k,v in one['gates'].items():
    if v['probability']!=two['gates'][k]['probability']:raise ValueError('Native scalar probability changed')
    self.first_gate_counts[k]+=int(v['triggered'])*128;self.fresh_gate_counts[k]+=int(two['gates'][k]['triggered'])*128;self.gate_disagreement[k]+=int(v['triggered']!=two['gates'][k]['triggered'])*128
  self.active=False;self.pending=True
 def take(self,raw,y,meta,step):
  ids=list(zip(*(meta[k].detach().cpu().tolist() for k in ('eq_i','rx_i','day_i','sig_i','base_index'))))
  if self.active or not self.pending or step!=self.calls or step!=self.consumed+1 or not torch.equal(raw,self.raw) or not torch.equal(y,self.labels) or ids!=self.ids:raise ValueError('Fourth IQ pair must consume its same current real L once')
  chosen={'RAW':self.raw,'REPLAY':self.first,'REFRESH':self.fresh}[self.arm]
  def mse(a,b):return float((a-b).square().mean())
  rec=dict(step=step,epoch=(step-1)//222+1,arm=self.arm,ids=ids,labels=y.detach().cpu().tolist(),seed=SEED+step,first_stages=self.stages,fresh_stages=self.fresh_stages,first_prepa_raw_MSE=mse(self.first,self.raw),fresh_prepa_raw_MSE=mse(self.fresh,self.raw),fresh_first_prepa_MSE=mse(self.fresh,self.first),normal_prepa_MSE=mse(self.normal,self.first),selected_raw_MSE=mse(chosen,self.raw),selected_first_prepa_MSE=mse(chosen,self.first),raw_power=float(self.raw.double().square().mean()),first_prepa_power=float(self.first.double().square().mean()),fresh_prepa_power=float(self.fresh.double().square().mean()),selected_power=float(chosen.double().square().mean()),all_L_supervised=True,new_PA_DAC_applications=0,original_normal_view_unchanged=True)
  if self.stream:self.stream.write(json.dumps(rec,allow_nan=False)+'\n')
  self.pending=False;self.consumed+=1;self.raw=None;self.labels=None;self.first=None;self.fresh=None;self.normal=None;self.stages=[];self.fresh_stages=[];return chosen
 def summary(self):return dict(calls=self.calls,consumed=self.consumed,fresh_packets=self.fresh_packets,coverage_cells=len(self.coverage),construct_audit_seconds=self.seconds,nonidentical_steps=self.nonidentical_steps,first_operator_packets=dict(self.first_gate_counts),fresh_operator_packets=dict(self.fresh_gate_counts),view_gate_disagreement_packets=dict(self.gate_disagreement),extra_NN_per_L_packet=2,U_fresh_calls=0,new_PA_DAC_applications=0,seed=SEED)
class AugmentorProxy:
 def __init__(self,base,runtime):
  object.__setattr__(self,'base',base);object.__setattr__(self,'runtime',runtime);object.__setattr__(self,'anti',base._apply_antis_shortcut);object.__setattr__(self,'channel',base._apply_general_channel)
  for name,attr,method in (('anti','_apply_antis_shortcut',self.anti),('channel','_apply_general_channel',self.channel)):setattr(base,attr,lambda x,name=name,method=method:runtime.stage(method,name,base,x))
 def __getattr__(self,k):return getattr(self.base,k)
 def __setattr__(self,k,v):setattr(self.base,k,v)
 def __call__(self,x,labels=None,**kwargs):
  normal=self.runtime.role=='L' and labels is not None and bool((labels>=0).all()) and not any(kwargs.get(k,False) for k in ('dac_only','pa_only','dac_pa'))
  if not normal:return self.base(x,labels=labels,**kwargs)
  self.runtime.begin(x,labels)
  try:
   out=self.base(x,labels=labels,**kwargs)
   if not torch.is_tensor(out):raise ValueError('Actual native L must return one IQ tensor')
   self.runtime.finish_normal(self.base,self.anti,self.channel,out);return out
  except BaseException:self.runtime.active=False;raise
