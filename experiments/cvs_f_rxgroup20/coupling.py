"""Change only labeled ordinary-view RX correlation, not marginal operations."""
from contextlib import contextmanager
import random,time,types
import torch

KEYS={'anti':('p_dc_offset','p_bandedge_taper'),'channel':('p_time_shift','p_amp_scale','p_phase_rot','p_cfo','p_phase_noise','p_multipath','p_awgn')}
SEED=20261017831

class TorchProxy:
 def __init__(self):self.draws=[]
 def __getattr__(self,key):return getattr(torch,key)
 def rand(self,*args,**kwargs):
  value=torch.rand(*args,**kwargs)
  if args!=((),) or kwargs:raise ValueError('Unexpected native top-level gate draw')
  self.draws.append(float(value));return value

def observed(method,x):
 # The exact native function bytecode/globals are reused; only its direct
 # torch.rand(()) gate draws are observed. Helpers retain their own globals.
 proxy=TorchProxy();fn=method.__func__;scope=dict(fn.__globals__,torch=proxy)
 copied=types.FunctionType(fn.__code__,scope,fn.__name__,fn.__defaults__,fn.__closure__);copied.__kwdefaults__=fn.__kwdefaults__
 return copied(method.__self__,x),proxy.draws

@contextmanager
def private_rng(seed,x):
 py=random.getstate();cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state(x.device) if x.is_cuda else None
 try:
  with torch.random.fork_rng(devices=[x.device.index] if x.is_cuda else [],enabled=True):
   torch.manual_seed(seed);random.seed(seed)
   yield
 finally:
  random.setstate(py)
  if not torch.equal(cpu,torch.get_rng_state()) or random.getstate()!=py or (gpu is not None and not torch.equal(gpu,torch.cuda.get_rng_state(x.device))):raise ValueError('Private group draws changed native streams')

class Runtime:
 def __init__(self,arm,stream=None):
  if arm not in ('BATCH','RXGROUP'):raise ValueError('Only fixed two coupling arms')
  self.arm=arm;self.stream=stream;self.meta=None;self.role=None;self.active=False;self.calls=0;self.stage_calls=0;self.packets=0;self.seconds=0.;self.coverage={};self.pair_totals={};self.nonzero_group_stages=0;self.groups_total=0
 def set_L(self,meta):self.meta=meta;self.role='L'
 def set_U(self):self.meta=None;self.role='U'
 def begin(self,x,labels):
  if self.active or self.meta is None or tuple(x.shape)!=(128,2,256) or labels.shape!=(128,) or labels.dtype!=torch.long or not bool(((labels>=0)&(labels<6)).all()):raise ValueError('Exact real L128 metadata required')
  rx=self.meta['rx_i'].detach().cpu().long();day=self.meta['day_i'].detach().cpu().long();eq=self.meta['eq_i'].detach().cpu().long()
  if rx.shape!=(128,) or set(rx.tolist())-set((3,4,6,8)) or set(day.tolist())-set((1,2,3)) or set(eq.tolist())!={1}:raise ValueError('Only fit-source L,never held/U/V')
  keys=('eq_i','rx_i','day_i','sig_i','base_index');self.ids=list(zip(*(self.meta[k].detach().cpu().tolist() for k in keys)))
  if any(len(v)!=5 for v in self.ids) or not bool(torch.isfinite(x).all()):raise ValueError('Finite current L IDs required')
  self.rx=rx;self.labels=labels.detach().cpu().long();self.calls+=1;self.packets+=128;self.active=True;self.current=[];self.raw=x.detach().clone();self.epoch=(self.calls-1)//222+1
  for y,r,dd in zip(self.labels.tolist(),rx.tolist(),day.tolist()):self.coverage[(y,r,dd)]=self.coverage.get((y,r,dd),0)+1
 def stage(self,base,method,name,x):
  if not self.active:return method(x)
  started=time.perf_counter();before=x.clone();whole,draws=observed(method,x);keys=[k for k in KEYS[name] if getattr(base,k)>0]
  if len(draws)!=len(keys):raise ValueError('Native gate audit differs from active operator order')
  whole_gates={k:u<float(getattr(base,k)) for k,u in zip(keys,draws)};grouped=torch.empty_like(x);groups=[]
  for rx in sorted(self.rx.unique().tolist()):
   ix=(self.rx==rx).nonzero(as_tuple=False).flatten().to(x.device);seed=SEED+self.calls*64+(0 if name=='anti' else 32)+rx
   with private_rng(seed,x):part,gu=observed(method,x.index_select(0,ix))
   if len(gu)!=len(keys):raise ValueError('Grouped gate audit differs')
   grouped.index_copy_(0,ix,part);groups.append(dict(RX=int(rx),count=len(ix),seed=seed,gates={k:u<float(getattr(base,k)) for k,u in zip(keys,gu)}))
  if not torch.equal(before,x) or not bool(torch.isfinite(whole).all()) or not bool(torch.isfinite(grouped).all()):raise ValueError('Finite transformed view and unchanged input required')
  positive=(self.labels[:,None]==self.labels[None,:])&(self.rx[:,None]!=self.rx[None,:]);pairs=int(positive.sum());mismatch={}
  for k in keys:
   gate=torch.tensor([next(g['gates'][k] for g in groups if g['RX']==r) for r in self.rx.tolist()]);different=gate[:,None]!=gate[None,:];n=int((positive&different).sum());mismatch[k]=dict(positive_pairs=pairs,would_group_disagree=n,actual_disagree=n if self.arm=='RXGROUP' else 0,probability=float(getattr(base,k)))
   acc=self.pair_totals.setdefault(k,[0,0,0]);acc[0]+=pairs;acc[1]+=n;acc[2]+=n if self.arm=='RXGROUP' else 0
  mse=float((grouped-whole).square().mean());self.nonzero_group_stages+=mse>0.;self.stage_calls+=1;self.groups_total+=len(groups);elapsed=time.perf_counter()-started;self.seconds+=elapsed
  self.current.append(dict(stage=name,whole_gates=whole_gates,groups=groups,crossRX_sameTX_gate_disagreement=mismatch,group_vs_whole_stage_MSE=mse,seconds=elapsed))
  return whole if self.arm=='BATCH' else grouped
 def finish(self,result):
  if not self.active or len(self.current)!=2 or not torch.is_tensor(result) or result.shape!=self.raw.shape or not bool(torch.isfinite(result).all()):raise ValueError('Exactly two normal-view stages required')
  record=dict(step=self.calls,epoch=self.epoch,arm=self.arm,ids=self.ids,stages=self.current,actual_ordinary_raw_MSE=float((result-self.raw).square().mean()),all_L_supervised=True,PA_DAC_class_signature_rules_unchanged=True)
  if self.stream:self.stream.write(__import__('json').dumps(record,allow_nan=False)+'\n')
  self.active=False;self.raw=None
 def summary(self):
  return dict(calls=self.calls,packets=self.packets,stages=self.stage_calls,group_calls=self.groups_total,nonzero_group_stages=self.nonzero_group_stages,construct_audit_seconds=self.seconds,coverage_cells=len(self.coverage),pair_gate_counts={k:dict(positive_pairs=v[0],would_group_disagree=v[1],actual_disagree=v[2]) for k,v in self.pair_totals.items()},new_NN=0,U_grouped_calls=0,grouping='actual physical RX only;never TX/day',private_seed=SEED,formal_test_changed=False)

class AugmentorProxy:
 def __init__(self,base,runtime):
  object.__setattr__(self,'base',base);object.__setattr__(self,'runtime',runtime)
  for name,attr in (('anti','_apply_antis_shortcut'),('channel','_apply_general_channel')):
   method=getattr(base,attr)
   setattr(base,attr,lambda x,method=method,name=name:runtime.stage(base,method,name,x))
 def __getattr__(self,key):return getattr(self.base,key)
 def __setattr__(self,key,value):setattr(self.base,key,value)
 def __call__(self,x,labels=None,**kwargs):
  normal=self.runtime.role=='L' and labels is not None and bool((labels>=0).all()) and not any(kwargs.get(k,False) for k in ('dac_only','pa_only','dac_pa'))
  if not normal:return self.base(x,labels=labels,**kwargs)
  self.runtime.begin(x,labels)
  try:
   result=self.base(x,labels=labels,**kwargs);self.runtime.finish(result);return result
  except BaseException:self.runtime.active=False;raise
