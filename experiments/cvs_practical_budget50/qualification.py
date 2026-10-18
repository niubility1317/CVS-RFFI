"""Same frozen legal source V panel after cumulative FINAL50."""
from pathlib import Path
import numpy as np
import torch
from . import design as d
from .quality import panel
from .channel import tensor_iq

def run(row):
 from .model import own_model
 c=d.config(next(r for r in d.rows() if r['row_id']==row));out=d.BASE/row/'source_qualification';out.mkdir(exist_ok=False)
 m=own_model(c,'cuda:0');before={k:v.clone() for k,v in m.state_dict().items()};data,roles=panel(torch)
 obs=np.load(d.PANEL/'observations/iq.npy',mmap_mode='r',allow_pickle=False)
 clean=np.empty((1800,6),np.float32);sat=np.empty((2,3,1800,6),np.float32)
 with torch.no_grad():
  for i in range(0,1800,90):clean[i:i+90]=m.forward_identity_only(data['x'][i:i+90].to('cuda:0'))['tx_logits'].tolist()
  for draw in range(2):
   for si in range(3):
    for i in range(0,1800,90):sat[draw,si,i:i+90]=m.forward_identity_only(tensor_iq(obs[draw,si,0,i:i+90],'cuda:0'))['tx_logits'].tolist()
 if any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()) or m.id_backbone.encoder._full_eval_packets!=12600:raise ValueError('Frozen source readout state/NN differs')
 y=np.asarray(data['y'].tolist(),np.int64);np.savez_compressed(out/'predictions.npz',ids=np.asarray(data['ids']),TX=y,RX=data['rx'].tolist(),day=data['day'].tolist(),clean=clean,satellite=sat)
 d.write(out/'complete.json',dict(status='SOURCE_READOUT_COMPLETE',row_id=row,config=c,clean_accuracy=float((clean.argmax(1)==y).mean()),source_accuracies=(sat.argmax(-1)==y).mean(-1).tolist(),NN=12600,training_updates=0,head_fit_updates=0,target_access=False,formal_scores_read=False))
