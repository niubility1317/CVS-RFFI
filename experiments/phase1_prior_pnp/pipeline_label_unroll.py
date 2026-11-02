"""Independent blind inverse with source-trained observation-conditioned unrolled corrector."""
import sys
from pathlib import Path
import torch
from observation_unroll import ObservationCorrector,restore_observation_conditioned
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'phase1_joint_mixture'))
from source_pilot import s,core,BlindActionNet,action_matrix,posterior,SOURCE,PRIOR_RUN

def load_fixed(root,device):
 blindstate=torch.load(root/'runs'/SOURCE/'checkpoint_epoch10.pt',map_location='cpu',weights_only=False);p=torch.load(root/'runs'/PRIOR_RUN/'source_prior.pt',map_location='cpu',weights_only=False);run=root/'runs/20261102-phase1-label-unroll-s2026092701-r01';result=core.read(run/'result.json');state=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
 if blindstate.get('commit')!='94693a342bb2daa684c0f21415d7b949f9fb2107' or blindstate.get('epoch')!=10 or p.get('source_ids')!=8192 or state.get('commit')!='a0e150d97e32d4298aeb19e35d035fced11bb28d' or result.get('status')!='SOURCE_TRAINING_COMPLETE':raise ValueError('Exact independent source models required')
 blind=BlindActionNet().to(device);blind.load_state_dict(blindstate['model']);blind.eval().requires_grad_(False);prior=ObservationCorrector().to(device);prior.load_state_dict(state['model']);prior.eval().requires_grad_(False);stats=(p['means'].to(device),p['covariances'].to(device),p['priors'].to(device));fixed=torch.load(root/'runs'/s.GATE_RUN/'source/selected.pt',map_location='cpu',weights_only=False)
 if fixed.get('commit')!=s.GATE_COMMIT or fixed.get('selected',{}).get('kind')!='gated_direct_iq':raise ValueError('Exact source-only corruption gate required')
 gate=s.BlindGate().to(device);gate.load_state_dict(fixed['gate'],strict=True);gate.eval().requires_grad_(False)
 return blind,prior,stats,gate,torch.eye(512,device=device)

@torch.no_grad()
def evaluate_wave(x,models,threshold=.8):
 blind,prior,stats,gate,eye=models;active=gate(x).sigmoid()>threshold;parent=x.clone();candidate=x.clone()
 if active.any():
  signal=x[active];h,lv=blind(signal);A=action_matrix(h);states,lp=posterior(signal.reshape(len(signal),512),A,eye[None]*lv.exp()[:,None,None],*stats);z=(states*lp.exp().unsqueeze(-1)).sum(1).reshape_as(signal);parent[active]=z;candidate[active]=restore_observation_conditioned(prior,signal,z,A,lv)
 if not torch.isfinite(parent).all() or not torch.isfinite(candidate).all():raise ValueError('Finite physical and prior output required')
 return parent,candidate,active
