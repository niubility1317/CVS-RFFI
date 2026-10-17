"""Matched L-only six-update buffering; original model/loss/U path untouched."""
from contextlib import contextmanager
from collections import defaultdict,deque
import gzip,json,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader,RandomSampler
from .reorder import reorder,metrics
L_SAMPLER_SEED=202610175201
L_LOADER_SEED=202610175202

def private_L_loader(loader,model_seed):
 actual=loader;seen=set()
 while hasattr(actual,'loader'):
  if id(actual) in seen:raise ValueError('Cyclic L wrapper')
  seen.add(id(actual));actual=actual.loader
 if not isinstance(actual,DataLoader) or not isinstance(actual.sampler,RandomSampler) or actual.batch_size!=128 or actual.num_workers!=0 or not actual.drop_last:raise ValueError('Original finite L128 RandomSampler/worker0/drop_last required')
 if bool(getattr(actual.dataset,'hide',False)) or len(actual.dataset)!=5040:raise ValueError('Visible original fit L5040 only')
 offset=int(model_seed)-2026092701
 actual.sampler.generator=torch.Generator().manual_seed(L_SAMPLER_SEED+offset)
 actual.generator=torch.Generator().manual_seed(L_LOADER_SEED+offset)
 return dict(L_sampler_seed=L_SAMPLER_SEED+offset,L_loader_seed=L_LOADER_SEED+offset,model_seed=model_seed,private_L_generators=True,original_U_loader_unmodified=True)

def identities(batch):
 if not isinstance(batch,(list,tuple)) or len(batch)!=4:raise ValueError('Original x/y/domain/meta L batch required')
 x,y,domain,meta=batch
 if x.shape!=(128,2,256) or y.shape!=(128,) or domain.shape!=(128,) or x.requires_grad or y.dtype!=torch.long:raise ValueError('Original unmodified CPU L128 required')
 if x.device.type!='cpu' or y.device.type!='cpu' or not torch.isfinite(x).all():raise ValueError('Finite host L IQ only')
 keys=('eq_i','rx_i','day_i','sig_i','base_index');columns=[meta[k].tolist() for k in keys]
 return [(int(t),*(int(c[i]) for c in columns),int(t)) for i,t in enumerate(y.tolist())]

def concatenate(values):
 v=values[0]
 if torch.is_tensor(v):
  if any(not torch.is_tensor(x) or x.ndim<1 or len(x)!=128 or x.shape[1:]!=v.shape[1:] or x.dtype!=v.dtype or x.device.type!='cpu' for x in values):raise ValueError('Unexpected batched tensor')
  return torch.cat(values,0)
 if isinstance(v,dict):
  if any(not isinstance(x,dict) or tuple(x)!=tuple(v) for x in values):raise ValueError('Metadata keys differ')
  return {k:concatenate([x[k] for x in values]) for k in v}
 if isinstance(v,(list,tuple)) and len(v)==128 and all(isinstance(a,(str,int,float,bool,type(None))) for a in v):
  if any(type(x)is not type(v) or len(x)!=128 for x in values):raise ValueError('Unexpected metadata sequences')
  flat=[a for x in values for a in x];return tuple(flat) if isinstance(v,tuple) else flat
 raise ValueError('Unsupported original metadata shape;do not discard fields')

def take(v,indices):
 if torch.is_tensor(v):return v.index_select(0,torch.tensor(indices,dtype=torch.long))
 if isinstance(v,dict):return {k:take(x,indices) for k,x in v.items()}
 values=[v[i] for i in indices];return tuple(values) if isinstance(v,tuple) else values

class Order:
 def __init__(self,arm,out):
  if arm not in ('O0','O1'):raise ValueError('One matched order pair')
  self.arm=arm;self.epochs=self.steps=self.blocks=self.moved=0;self.seconds=0.;self.original_stats=[];self.actual_stats=[]
  self.stream=gzip.open(Path(out)/'L_batch_order_steps.jsonl.gz','xt',encoding='utf8',compresslevel=3)
 def transform(self,batches):
  started=time.perf_counter();ids=[identities(b) for b in batches]
  balanced=reorder(ids,self.blocks) # computed in both rows,including matched no-reorder control
  after=ids if self.arm=='O0' else balanced
  original=[p for row in ids for p in row];queues=defaultdict(deque)
  for i,p in enumerate(original):queues[p].append(i)
  order=[[queues[p].popleft() for p in row] for row in after]
  if sorted(i for row in order for i in row)!=list(range(768)):raise ValueError('An original L position lost/duplicated')
  combined=tuple(concatenate([b[k] for b in batches]) for k in range(4))
  result=[tuple(take(v,idx) for v in combined) for idx in order]
  self.blocks+=1
  for slot,(before,expected,observed,indices) in enumerate(zip(ids,after,result,order)):
   if identities(observed)!=expected:raise ValueError('IQ/labels/domains/metadata gather disagrees')
   self.original_stats.append(metrics(before));self.actual_stats.append(metrics(expected));self.moved+=sum(i!=slot*128+j for j,i in enumerate(indices))
  self.seconds+=time.perf_counter()-started
  return result,ids,after
 def wrap(self,original):
  def paired(*args,**kwargs):
   self.epochs+=1
   if self.epochs>20:raise ValueError('No extra training epochs')
   iterator=iter(original(*args,**kwargs))
   for _ in range(37):
    source=[]
    for __ in range(6):
     try:pair=next(iterator)
     except StopIteration as e:raise ValueError('Original epoch below fixed222 L steps') from e
     if not isinstance(pair,(tuple,list)) or len(pair)!=2 or pair[1] is not None:raise ValueError('Original MUSE/U pair must not enter L buffer')
     source.append(pair[0])
    batches,before,after=self.transform(source)
    for slot,batch in enumerate(batches):
     self.steps+=1
     self.stream.write(json.dumps(dict(step=self.steps,epoch=self.epochs,block=self.blocks,slot=slot,arm=self.arm,original_ids=[list(p[1:6]) for p in before[slot]],original_labels=[p[0] for p in before[slot]],actual_ids=[list(p[1:6]) for p in after[slot]],actual_labels=[p[0] for p in after[slot]],IQ_changed=False,U_used=False))+'\n')
     yield batch,None
   try:next(iterator)
   except StopIteration:pass
   else:raise ValueError('Original epoch above fixed222 L steps')
  return paired
 def summary(self):
  return dict(arm=self.arm,epochs=self.epochs,steps=self.steps,blocks=self.blocks,moved_positions=self.moved,buffer_CPU_seconds=self.seconds,maximum_buffer_L_packets=768,original_stats=self.original_stats,actual_stats=self.actual_stats,new_IQ_transform=False,new_NN_forward_packets=0,U_buffered=False,hidden_U_truth_read=False)
 def close(self):self.stream.close()

@contextmanager
def scope(native,controller):
 original=native._muse_epoch_pairs;native._muse_epoch_pairs=controller.wrap(original)
 try:yield
 finally:native._muse_epoch_pairs=original;controller.close()
