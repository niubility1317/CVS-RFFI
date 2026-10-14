from contextlib import contextmanager
import torch
from torch.nn import functional as F
from experiments.cvs_source_layer_order.objective import relation_indices,losses,metadata

@contextmanager
def capture(model):
 pool=model.id_backbone.encoder.sequence_pool
 if len(pool.views)!=1 or pool.views[0].mode!='SIG_TIME' or not torch.equal(pool.views[0].descriptor_lift.cpu(),torch.eye(28)):raise ValueError('Actual deployed single SIG_TIME view differs')
 maps={};handles=[]
 try:
  def t2(module,inputs,out):
   if 't2' in maps or out.ndim!=3 or not torch.isfinite(out).all():raise ValueError('Actual t2 map differs')
   maps['t2']=out
  def descriptor(module,inputs,out):
   if 'SIG_DESCRIPTOR' in maps or out.shape[1:]!=(112,) or not torch.isfinite(out).all():raise ValueError('Actual deployed descriptor differs')
   maps['SIG_DESCRIPTOR']=out
  handles.append(model.id_backbone.encoder.core.id_backbone.t2.register_forward_hook(t2));handles.append(pool.views[0].register_forward_hook(descriptor));yield maps
  if tuple(maps)!=('t2','SIG_DESCRIPTOR'):raise ValueError('Actual deployed forward capture incomplete')
 finally:
  for h in handles:h.remove()

def descriptors(s):
 if s.ndim!=2 or s.shape[1]!=112 or not torch.isfinite(s).all():raise ValueError('Actual SIG112 required')
 windows=s.reshape(-1,4,28);clock=windows[:,:,6]
 if not torch.allclose(clock,torch.ones_like(clock),atol=2e-6,rtol=0):raise ValueError('Declared fixed time step is not constant1')
 keep=torch.arange(28,device=s.device)!=6;dynamic=windows[:,:,keep].reshape(-1,108)
 return dynamic,clock

def views(maps,out):
 s=maps['SIG_DESCRIPTOR'];dynamic,clock=descriptors(s)
 return {'T2_WINDOW':F.normalize(F.adaptive_avg_pool1d(maps['t2'],4).flatten(1),dim=1,eps=1e-4),'SIG_RAW':F.normalize(s,dim=1,eps=1e-4),'SIG_DYNAMIC':F.normalize(dynamic,dim=1,eps=1e-4),'JOINT160':F.normalize(out['id_feat_joint'],dim=1,eps=1e-4)}
