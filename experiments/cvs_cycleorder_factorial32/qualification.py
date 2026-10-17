"""One frozen FINAL30 source readout before its automatic formal prediction."""
import time
import numpy as np
import torch
from pathlib import Path
from . import design as d,model as api
from .channel import tensor_iq
PANEL=Path(d.PROJECT)/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
def run(row):
 from experiments.cvs_fullsource_cycleorder30.probe import panel
 c=d.config(next(r for r in d.rows() if r['row_id']==row));out=d.BASE/row/'source_qualification';out.mkdir(exist_ok=False);tic=time.perf_counter();torch.set_num_threads(2)
 from experiments.cvs_equivariant_identity.precision import numerical_context
 with numerical_context(d.FULL_FP32_POLICY):
  m=api.own_model(c,'cpu');before={k:v.clone() for k,v in m.state_dict().items()}
  with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
  if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Actual FINAL30 defaulthead no-query smoke failed')
  d.write(out/'smoke.json',dict(status='PASS',NN=2,IQ_before_smoke=0,target_access=False))
  data,roles=panel(torch);d.write(out/'data_roles.json',roles);ids=data['ids'];x=data['x'];y=np.asarray(data['y'].tolist(),np.int64)
  obs=d.read(PANEL/'observations/completion.json')
  if obs['status']!='SOURCE_OBSERVATIONS_COMPLETE' or obs['target_access'] or obs['draws']!=2 or obs['scenes']!=list(d.TRAIN_SCENES) or obs['conditions'][0]!='full':raise ValueError('Same existing full source observations required')
  array=np.load(PANEL/'observations/iq.npy',mmap_mode='r',allow_pickle=False)
  if array.shape!=(2,3,11,1800,2,256):raise ValueError('Full fixed source matrix required')
  clean=np.empty((1800,6),np.float32);sat=np.empty((2,3,1800,6),np.float32)
  with torch.no_grad():
   for i in range(0,1800,90):clean[i:i+90]=m.forward_identity_only(x[i:i+90])['tx_logits'].tolist()
   for draw in range(2):
    for si in range(3):
     for i in range(0,1800,90):sat[draw,si,i:i+90]=m.forward_identity_only(tensor_iq(array[draw,si,0,i:i+90],'cpu'))['tx_logits'].tolist()
  if not np.isfinite(clean).all() or not np.isfinite(sat).all() or m.id_backbone.encoder._full_eval_packets!=12602 or m.id_backbone.encoder._full_train_packets or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Source logits/state/NN differs')
  np.savez_compressed(out/'predictions.npz',ids=np.asarray(ids),TX=y,RX=data['rx'].tolist(),day=data['day'].tolist(),clean=clean,satellite=sat)
  result=dict(status='SOURCE_QUALIFIED_READOUT',config=c,source_accuracies=(sat.argmax(-1)==y).mean(-1).tolist(),clean_accuracy=float((clean.argmax(1)==y).mean()),NN=12602,source_unique_V=1800,channel_draws=2,head_fit_updates=0,training_updates=0,target_access=False,formal_scores_read=False,state_unchanged=True,wall_seconds=time.perf_counter()-tic)
  d.write(out/'complete.json',result);return result
