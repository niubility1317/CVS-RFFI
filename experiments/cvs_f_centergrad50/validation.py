"""One frozen sourceV pass: existing default head plus descriptive raw joint160 angles."""
import numpy as np
import torch
from torch.nn import functional as F
from experiments.cvs_f_source_geometry50.geometry import describe
from . import design as d

@torch.no_grad()
def validate(model,loader,device):
 model.eval();cm=np.zeros((6,6),dtype=np.int64);rxcm={r:np.zeros((6,6),dtype=np.int64) for r in (1,3,4,6,8)};daycm={r:np.zeros((6,6),dtype=np.int64) for r in (1,2,3)};sums=np.zeros((6,160),dtype=np.float64);ce_sum=0.
 features=[];labels=[];receivers=[];days=[]
 versions=[p._version for p in model.parameters()];buffers={n:v.clone() for n,v in model.named_buffers()}
 for x,y,domain,meta in loader:
  out=model.forward_identity_only(x.to(device));logits=out['tx_logits'];pred=logits.argmax(1).cpu().numpy();truth=y.numpy();z=F.normalize(out['id_feat_joint'],dim=1).cpu().numpy()
  if not torch.isfinite(logits).all() or not np.isfinite(z).all():raise FloatingPointError('Nonfinite frozen source outputs')
  features.append(out['id_feat_joint'].detach().cpu().numpy().copy());labels.append(truth.copy());receivers.append(meta['rx_i'].numpy().copy());days.append(meta['day_i'].numpy().copy())
  np.add.at(cm,(truth,pred),1);np.add.at(sums,truth,z);ce_sum+=float(F.cross_entropy(logits,y.to(device),reduction='sum'))
  for r,mat in rxcm.items():
   mask=meta['rx_i'].numpy()==r;np.add.at(mat,(truth[mask],pred[mask]),1)
  for day,mat in daycm.items():
   mask=meta['day_i'].numpy()==day;np.add.at(mat,(truth[mask],pred[mask]),1)
 if cm.sum()!=27000 or not np.all(cm.sum(1)==4500) or any(not np.all(v.sum(1)==900) for v in rxcm.values()):raise ValueError('Source V full role coverage differs')
 if versions!=[p._version for p in model.parameters()] or any(not torch.equal(v,dict(model.named_buffers())[n]) for n,v in buffers.items()):raise ValueError('Validation updated persistent model state')
 centers=sums/4500;norms=np.linalg.norm(centers,axis=1);unit=centers/np.maximum(norms[:,None],1e-12);cos=unit@unit.T
 geometry=describe(np.concatenate(features),np.concatenate(labels),np.concatenate(receivers),np.concatenate(days),d.CLASSES)
 return dict(geometry=geometry,count=27000,accuracy=float(cm.trace()/27000),CE=ce_sum/27000,recall=(cm.diagonal()/4500).tolist(),worst_RX=min(float(v.trace()/v.sum()) for v in rxcm.values()),confusion=cm.tolist(),receiver_confusion={str(r):v.tolist() for r,v in rxcm.items()},day_confusion={str(r):v.tolist() for r,v in daycm.items()},mean_within_TX_pair_cosine=((4500*norms**2-1)/4499).tolist(),class_centroid_cosine=cos.tolist(),feature_diagnostic_fitting=False)
