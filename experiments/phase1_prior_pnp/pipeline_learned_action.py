"""Fixed independent learned channel and observation-conditioned recovery."""
import numpy as np
import torch
from pipeline_unroll import load_fixed as load_unroll,s,core,action_matrix,posterior
from learned_channel import ChannelRegressor
from channel_operator import corrected_action
from observation_unroll import restore_observation_conditioned

def load_fixed(root,device):
 blind,unroll,stats,gate,eye=load_unroll(root,device);run=root/'runs/20261102-phase1-learned-channel-source-s2026092701-r03';result=core.read(run/'result.json');state=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False);old=torch.load(root/'runs/20261102-phase1-observation-unroll-s2026092701-r01/checkpoint.pt',map_location='cpu',weights_only=False)
 if state.get('commit')!='0831bcec99be5d2e3d20aa093a674a5eed6eb68d' or result.get('status')!='SOURCE_TRAINING_COMPLETE' or not np.array_equal(state['fit_ids'],old['fit_ids']) or not np.array_equal(state['source_view_assignment'],old['source_view_assignment']):raise ValueError('Exact independent source training contract required')
 channel=ChannelRegressor().to(device);channel.load_state_dict(state['model'],strict=True);channel.eval().requires_grad_(False);return blind,unroll,stats,gate,eye,channel

@torch.no_grad()
def evaluate_wave(x,models,threshold=.8):
 blind,unroll,stats,gate,eye,channel=models;active=gate(x).sigmoid()>threshold;parent=x.clone();candidate=x.clone()
 if active.any():
  signal=x[active];h,lv=blind(signal);A=action_matrix(h);states,lp=posterior(signal.reshape(len(signal),512),A,eye[None]*lv.exp()[:,None,None],*stats);z=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(signal);forward=torch.bmm(A,z.reshape(len(z),512,1)).reshape_as(z);gain=channel(signal,forward,lv);refined=corrected_action(A,gain);parent[active]=z;candidate[active]=restore_observation_conditioned(unroll,signal,z,refined,lv)
 if not torch.isfinite(parent).all() or not torch.isfinite(candidate).all():raise ValueError('Finite source-trained channel recovery required')
 return parent,candidate,active
