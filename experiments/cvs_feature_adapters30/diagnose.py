"""No-fit readout: actual training As, same-L new channel, and held-out source V."""
import numpy as np
import torch
from . import design as d
from .data import load,diagnostic_indices
from experiments.cvs_F_nontrue20.channel import tensor_iq
from experiments.cvs_F_nontrue20.quality import panel

def readout(m,out):
 out.mkdir(exist_ok=False);device=next(m.parameters()).device;before={k:v.clone() for k,v in m.state_dict().items()};start=m.id_backbone.encoder._full_eval_packets;bank=d.BASE/'observation_bank';L=torch.load(bank/'labeled.pt',map_location='cpu',weights_only=False);di=diagnostic_indices(L);V=load('V');vp,roles=panel(torch);y=np.asarray(L['y'][di].tolist(),np.int64);yv=np.asarray(vp['y'].tolist(),np.int64);yvfull=np.asarray(V['y'].tolist(),np.int64)
 def predict(x):
  values=[]
  for i in range(0,len(x),180):
   v=x[i:i+180].to(device) if torch.is_tensor(x) else tensor_iq(x[i:i+180],device)
   z=m.forward_identity_only(v)['tx_logits']
   if not torch.isfinite(z).all():raise ValueError('Nonfinite source logits')
   values.extend(z.tolist())
  return np.asarray(values,np.float32)
 m.eval()
 with torch.no_grad():
  clean=predict(L['x'][di]);actual=predict(np.load(bank/'actual_first_training_As.npy',mmap_mode='r',allow_pickle=False));fresh=np.load(bank/'same_L_new_channel.npy',mmap_mode='r',allow_pickle=False);new=np.asarray([[predict(fresh[a,b]) for b in range(2)] for a in range(2)])
  vc=predict(vp['x']);obs=np.load(d.PANEL/'observations/iq.npy',mmap_mode='r',allow_pickle=False)
  if obs.shape!=(2,3,11,1800,2,256):raise ValueError('Fixed V physical observation panel differs')
  vs=np.asarray([[predict(obs[a,b,0]) for b in range(3)] for a in range(2)]);vf=predict(V['x'])
 if any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Source readout mutated parameters')
 NN=m.id_backbone.encoder._full_eval_packets-start
 if NN!=50400:raise ValueError('Full source readout forward cost differs')
 np.savez_compressed(out/'predictions.npz',L_ids=np.asarray([L['ids'][i] for i in di]),L_TX=y,L_RX=L['rx'][di].tolist(),L_day=L['day'][di].tolist(),L_clean=clean,L_actual_training_As=actual,L_new=new,V_ids=np.asarray(vp['ids']),V_TX=yv,V_clean=vc,V_satellite=vs,V_full_ids=np.asarray(V['ids']),V_full_TX=yvfull,V_full_clean=vf)
 result=dict(status='SOURCE_READOUT_COMPLETE',L_train_clean_accuracy=float((clean.argmax(1)==y).mean()),L_actual_training_As_accuracy=float((actual.argmax(1)==y).mean()),L_new_midlow_accuracies=(new.argmax(-1)==y).mean(-1).tolist(),V_clean_accuracy=float((vc.argmax(1)==yv).mean()),source_accuracies=(vs.argmax(-1)==yv).mean(-1).tolist(),full_V_clean_accuracy=float((vf.argmax(1)==yvfull).mean()),NN=NN,fit_updates=0,head_fit_updates=0,target_access=False,U_used=False,formal_scores_read=False)
 d.write(out/'complete.json',result);return result

def run(row):
 from .model import own_model
 c=d.config(next(r for r in d.rows() if r['row_id']==row));m=own_model(c,'cuda:0');return readout(m,d.BASE/row/'source_qualification')
