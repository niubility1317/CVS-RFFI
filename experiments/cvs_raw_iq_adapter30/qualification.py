"""Frozen source-only readout on the already shared full-physics V panel."""
import numpy as np
from . import design as d


def same_panel(data,reference):
 for key,value in (('ids',data['ids']),('TX',data['y'].tolist()),('RX',data['rx'].tolist()),('day',data['day'].tolist())):
  if key not in reference or not np.array_equal(np.asarray(value),reference[key]):
   raise ValueError('Existing local-adapter source panel IDs/labels/roles differ: '+key)
 if reference['clean'].shape!=(1800,6) or reference['satellite'].shape!=(2,3,1800,6):
  raise ValueError('Existing local-adapter full source readout shape differs')


def run(row):
 import torch
 from .model import own_model
 from experiments.cvs_F_nontrue20.quality import panel
 from experiments.cvs_F_nontrue20.channel import tensor_iq
 from experiments.cvs_dual_teacher_adapter30 import design as old_design
 if (d.OLD_CONTROL_RUN!=old_design.RUN or d.PANEL!=old_design.PANEL or d.SOURCE_CACHE!=old_design.SOURCE_CACHE):
  raise ValueError('Reuse the existing local-adapter fixed source cache and received observations')
 c=d.config(next(r for r in d.rows() if r['row_id']==row));out=d.BASE/row/'source_qualification';out.mkdir(exist_ok=False)
 m=own_model(c,'cuda:0');before={k:v.clone() for k,v in m.state_dict().items()};data,roles=panel(torch);obs=np.load(d.PANEL/'observations/iq.npy',mmap_mode='r',allow_pickle=False)
 if obs.shape!=(2,3,11,1800,2,256):raise ValueError('Fixed source V1800 physical observations differ')
 old=d.OLD_CONTROL_BASE/('S3-s'+str(d.SEED))/'source_qualification'
 old_done=d.read(old/'complete.json')
 if (old_done['status']!='SOURCE_READOUT_COMPLETE' or old_done['NN']!=12600
     or old_done['target_access'] or old_done['formal_scores_read']
     or old_done['config']['run_id']!=d.OLD_CONTROL_RUN
     or old_done['config']['row_id']!='S3-s'+str(d.SEED)
     or old_done['config']['data']!=c['data']):raise ValueError('Existing frozen source reference required')
 with np.load(old/'predictions.npz',allow_pickle=False) as reference:same_panel(data,reference)
 if m.training or any(p.requires_grad for p in m.parameters()):raise ValueError('Source readout requires the frozen default-head model')
 clean=np.empty((1800,6),np.float32);sat=np.empty((2,3,1800,6),np.float32)
 with torch.no_grad():
  for i in range(0,1800,90):clean[i:i+90]=m.forward_identity_only(data['x'][i:i+90].to('cuda:0'))['tx_logits'].tolist()
  for draw in range(2):
   for si in range(3):
    for i in range(0,1800,90):sat[draw,si,i:i+90]=m.forward_identity_only(tensor_iq(obs[draw,si,0,i:i+90],'cuda:0'))['tx_logits'].tolist()
 if any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()) or m.id_backbone.encoder._full_eval_packets!=12600:raise ValueError('Frozen source readout state/count differs')
 y=np.asarray(data['y'].tolist(),np.int64);np.savez_compressed(out/'predictions.npz',ids=np.asarray(data['ids']),TX=y,RX=data['rx'].tolist(),day=data['day'].tolist(),clean=clean,satellite=sat)
 correct=(sat.argmax(-1)==y).sum(-1)
 d.write(out/'complete.json',dict(status='SOURCE_READOUT_COMPLETE',row_id=row,config=c,clean_accuracy=float((clean.argmax(1)==y).mean()),
  clean_panel_n=1800,clean_panel_correct=int((clean.argmax(1)==y).sum()),source_accuracies=(correct/1800).tolist(),source_correct=correct.tolist(),
  source_panel_n=1800,draws=2,scenes=['practical_low_suburban','practical_mid_urban','practical_low_urban'],observation_variant_index=0,
  source_observation_array=str(d.PANEL/'observations/iq.npy'),same_panel_as_old_local_S3=True,
  old_reference=dict(run_id=d.OLD_CONTROL_RUN,row_id='S3-s'+str(d.SEED),predictions=str(old/'predictions.npz'),new_NN=0),
  NN=12600,training_updates=0,head_fit_updates=0,target_access=False,formal_scores_read=False,
  fullVclean_source='source/completion.json final_source_metrics; separate from this V1800 clean panel'))
