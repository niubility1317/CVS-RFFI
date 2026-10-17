"""Frozen source qualification;all full-source roles retained,never target scores."""
import os,time
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .timing import transform,SOURCE_SEED
PANEL=Path(d.PROJECT)/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
def _guard(m):
 h=m.id_backbone.encoder.core.id_backbone.cls_head
 if sum(p.numel() for p in m.parameters())!=900722 or len(m.state_dict())!=206 or h.weight.shape!=(6,160) or float(h.scale)!=30.:raise ValueError('Actual fixed F/defaultcos differs')
 before={k:v.clone() for k,v in m.state_dict().items()}
 with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
 if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Real CP no-query smoke failed')
 return before
def _unchanged(m,before,NN):
 if m.id_backbone.encoder._full_eval_packets!=NN or m.id_backbone.encoder._full_train_packets or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()) or any(p.grad is not None for p in m.parameters()):raise ValueError('Frozen source state/NN differs')
def _timing(m,data):
 values=np.empty((2,1800,6),np.float32);offsets=np.empty((2,1800),np.int64)
 with torch.no_grad():
  for draw in range(2):
   for i in range(0,1800,90):
    x,info=transform(data['x'][i:i+90],data['ids'][i:i+90],'sourceV/timing/draw'+str(draw),seed=SOURCE_SEED);z=m.forward_identity_only(x)['tx_logits']
    if z.shape!=(len(x),6) or not torch.isfinite(z).all():raise ValueError('Finite timing-source logits required')
    values[draw,i:i+90]=z.tolist();offsets[draw,i:i+90]=info['offset_samples']
 return values,offsets
def reference():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Reference is CPU-only source readout')
 # Separate fresh process:do not import the new model module before the old own_model.
 from experiments.cvs_cycleorder_factorial32 import model as old_api
 from experiments.cvs_fullsource_cycleorder30.probe import panel
 from experiments.cvs_equivariant_identity.precision import numerical_context
 out=d.BASE/'reference_timing';out.mkdir(exist_ok=False);tic=time.perf_counter();torch.set_num_threads(2);c=d.control_config('B00')
 with numerical_context(d.FULL_FP32_POLICY):
  m=old_api.own_model(c,'cpu');before=_guard(m);d.write(out/'smoke.json',dict(status='PASS',NN=2,source_IQ_before_smoke=0,target_access=False));data,roles=panel(torch);d.write(out/'data_roles.json',roles);z,offsets=_timing(m,data);_unchanged(m,before,3602)
  y=np.asarray(data['y'].tolist(),np.int64);np.savez_compressed(out/'predictions.npz',ids=np.asarray(data['ids']),TX=y,RX=data['rx'].tolist(),day=data['day'].tolist(),timing=z,offset_samples=offsets)
  q=dict(status='REFERENCE_TIMING_SOURCE_COMPLETE',config=c,source_unique_V=1800,NN=3602,timing_accuracies=(z.argmax(-1)==y).mean(-1).tolist(),state_unchanged=True,training_updates=0,head_fit_updates=0,target_access=False,formal_scores_read=False,wall_seconds=time.perf_counter()-tic);d.write(out/'complete.json',q);return q
def run(row):
 from . import model as api
 from .channel import tensor_iq
 from experiments.cvs_fullsource_cycleorder30.probe import panel
 from experiments.cvs_equivariant_identity.precision import numerical_context
 c=d.config(next(r for r in d.rows() if r['row_id']==row));out=d.BASE/row/'source_qualification';out.mkdir(exist_ok=False);tic=time.perf_counter();torch.set_num_threads(2)
 with numerical_context(d.FULL_FP32_POLICY):
  m=api.own_model(c,'cpu');before=_guard(m);d.write(out/'smoke.json',dict(status='PASS',NN=2,source_IQ_before_smoke=0,target_access=False));data,roles=panel(torch);d.write(out/'data_roles.json',roles);ids=data['ids'];x=data['x'];y=np.asarray(data['y'].tolist(),np.int64)
  obs=d.read(PANEL/'observations/completion.json')
  if obs['status']!='SOURCE_OBSERVATIONS_COMPLETE' or obs['target_access'] or obs['draws']!=2 or obs['scenes']!=list(d.TRAIN_SCENES) or obs['conditions'][0]!='full':raise ValueError('Same original source full residual observations required')
  a=np.load(PANEL/'observations/iq.npy',mmap_mode='r',allow_pickle=False)
  if a.shape!=(2,3,11,1800,2,256):raise ValueError('Fixed source observation matrix required')
  clean=np.empty((1800,6),np.float32);sat=np.empty((2,3,1800,6),np.float32)
  with torch.no_grad():
   for i in range(0,1800,90):clean[i:i+90]=m.forward_identity_only(x[i:i+90])['tx_logits'].tolist()
   for draw in range(2):
    for si in range(3):
     for i in range(0,1800,90):sat[draw,si,i:i+90]=m.forward_identity_only(tensor_iq(a[draw,si,0,i:i+90],'cpu'))['tx_logits'].tolist()
  timing,offsets=_timing(m,data);_unchanged(m,before,16202)
  if not np.isfinite(clean).all() or not np.isfinite(sat).all():raise ValueError('Finite source default logits required')
  np.savez_compressed(out/'predictions.npz',ids=np.asarray(ids),TX=y,RX=data['rx'].tolist(),day=data['day'].tolist(),clean=clean,satellite=sat,timing=timing,offset_samples=offsets)
  q=dict(status='SOURCE_QUALIFIED_TIMING_READOUT',config=c,source_accuracies=(sat.argmax(-1)==y).mean(-1).tolist(),timing_accuracies=(timing.argmax(-1)==y).mean(-1).tolist(),clean_accuracy=float((clean.argmax(1)==y).mean()),NN=16202,source_unique_V=1800,channel_draws=2,head_fit_updates=0,training_updates=0,target_access=False,formal_scores_read=False,state_unchanged=True,wall_seconds=time.perf_counter()-tic);d.write(out/'complete.json',q);return q
