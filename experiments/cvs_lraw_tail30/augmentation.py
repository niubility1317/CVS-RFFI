"""L-only raw return after a complete original augmentation draw;no global monkeypatch."""
import torch

def draw_and_discard(augmentor,x,labels):
 if x.shape!=(128,2,256) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()) or labels.shape!=(128,) or bool(((labels<0)|(labels>5)).any()):raise ValueError('Finite visible L128 IQ256 and original labels0..5 required')
 raw=x.detach().clone()
 discarded=augmentor(raw.clone(),labels=labels,no_pa=False)
 if discarded.shape!=raw.shape or discarded.dtype!=raw.dtype or not bool(torch.isfinite(discarded).all()):raise ValueError('Original discarded ordinary generation must remain finite and same shape')
 return raw,dict(discardedordinary_MSE=float((discarded-raw).square().mean()),discardedordinary_changed_packets=int((discarded!=raw).flatten(1).any(1).sum()),returned_raw_MSE=float((raw-x).square().mean()),full_native_draws=1,count=128,input_clone=True,original_input_unchanged=True,full_native_aug_computation=True)

class Wrapper:
 def __init__(self,native,records):
  object.__setattr__(self,'_native',native);object.__setattr__(self,'_records',records)
 def __getattr__(self,key):return getattr(self._native,key)
 def __setattr__(self,key,value):setattr(self._native,key,value)
 def __call__(self,x,*args,**kwargs):
  if args or set(kwargs)!={'labels','no_pa'} or kwargs['no_pa'] is not False or kwargs['labels'] is None:raise ValueError('Only actual normal visible L labels/no_paFalse call is wrapped;U paths remain native')
  raw,record=draw_and_discard(self._native,x,kwargs['labels']);self._records.append(dict(step=len(self._records)+1,**record))
  return raw

def ordinary_factory(original,records):
 def create(*args,**kwargs):
  create.calls+=1
  if create.calls!=1:raise ValueError('Exactly one native ordinary L augmentor is supported')
  return Wrapper(original(*args,**kwargs),records)
 create.calls=0
 return create
