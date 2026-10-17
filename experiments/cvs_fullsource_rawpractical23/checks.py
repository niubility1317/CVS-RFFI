"""Synthetic data-flow checks only; no model/real IQ/channel/checkpoint."""
import copy,io,json,tempfile,types
from pathlib import Path
import torch
from . import design as d
from .observer import Observer
from concat_sat_channel_aug import ConcatSatChannelAugment
from cvsrffi import practical_adapter as pa
def run():
 c=d.config(d.rows()[0]);d.validate(c);negative=0
 for key,value in [('held_RX',1),('epochs',50),('As',False),('source_roles',dict(L_s=5040,U_s=45360,V=21600))]:
  bad=copy.deepcopy(c);bad[key]=value
  try:d.validate(bad)
  except ValueError:negative+=1
  else:raise AssertionError('Changed role/As/budget accepted')
 x=torch.randn(128,2,256,generator=torch.Generator().manual_seed(2301));saved=x.clone();ordinary=x*.9+.03
 y=torch.arange(128)%6;meta={k:torch.arange(128) for k in ('eq_i','sig_i','base_index')};meta.update(rx_i=torch.tensor([d.RX[(i//3)%5] for i in range(128)]),day_i=torch.arange(128)%3+1)
 o=Observer(c);o.sequence=io.StringIO();o.physics=io.StringIO();seen=[]
 previous=pa._last_meta
 def apply(x,scene,args,*,gen=None,return_meta=False):
  seen.append(x.clone());pa._last_meta=dict(config={'synthetic_only':True},records=[dict(state_end='LOS',receiver_lock_status='locked') for _ in range(128)])
  return x*.5,{}
 n=types.SimpleNamespace(ConcatSatChannelAugment=ConcatSatChannelAugment,make_augmentor=lambda cfg:None,deepcopy=copy.deepcopy,_detach_log_mapping=lambda v:v)
 try:
  with tempfile.TemporaryDirectory() as td:
   o.out=Path(td)
   with o.native_scope(n):
    aug=n.ConcatSatChannelAugment(scenarios=d.TRAIN_SCENES,p=1.,seed=202610172201,apply_fn=apply)
    o.record('L_s',(x,y,None,meta));result=aug.expand(ordinary,y,torch.zeros(128,dtype=torch.long),args=types.SimpleNamespace(),epoch=1,batch_idx=1)
    assert len(seen)==1 and torch.equal(seen[0],x) and torch.equal(x,saved)
    assert torch.equal(result.x[:128],ordinary) and torch.equal(result.x[128:],x*.5)
    assert torch.equal(result.y[:128],y) and torch.equal(result.y[128:],y)
    assert o.raw_L is None and o.current is None and o.route_calls==o.concat_calls==1
    o.record('U_s',(x,torch.full_like(y,-1),None,meta));assert o.raw_L is None
    try:aug.transform(ordinary,args=types.SimpleNamespace(),epoch=1,batch_idx=2)
    except ValueError:negative+=1
    else:raise AssertionError('Stale/U cache accepted')
    try:o.record('U_s',(x,y,None,meta))
    except ValueError:negative+=1
    else:raise AssertionError('Visible U truth accepted')
    try:o.record('L_s',(x[:127],y,None,meta))
    except ValueError:negative+=1
    else:raise AssertionError('Mismatched physical L accepted')
 finally:pa._last_meta=previous
 assert n.ConcatSatChannelAugment is ConcatSatChannelAugment
 a=d.make_args(c,'cpu')
 assert (a.epochs,a.label_epochs,a.pseudo_epochs)==(20,13,7) and not hasattr(a,'source_fold_held_rx')
 return dict(status='PASS',protocol_negatives=negative,synthetic_packets=128,original_raw_exact=True,clean_ordinary_exact=True,labels_identical=True,cache_consumed_once=True,NN=0,real_IQ=0,CP=0,physical_channel_calls=0,optimizer_updates=0)
if __name__=='__main__':print(json.dumps(run()))
