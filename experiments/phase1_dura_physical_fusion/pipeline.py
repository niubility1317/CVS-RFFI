"""Frozen physical posterior and DURA packed for single-record fusion."""
import torch
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_posterior.pipeline import PhysicalPosterior


def load_components(root,device,deps):
 blind_state=torch.load(root/'runs'/deps.SOURCE/'checkpoint_epoch10.pt',map_location='cpu',weights_only=False)
 prior=torch.load(root/'runs'/deps.PRIOR_RUN/'source_prior.pt',map_location='cpu',weights_only=False)
 gate_state=torch.load(root/'runs'/deps.s.GATE_RUN/'source/selected.pt',map_location='cpu',weights_only=False)
 if (blind_state.get('commit')!='94693a342bb2daa684c0f21415d7b949f9fb2107' or blind_state.get('epoch')!=10
     or prior.get('source_ids')!=8192 or gate_state.get('commit')!=deps.s.GATE_COMMIT
     or gate_state.get('selected',{}).get('kind')!='gated_direct_iq'):
  raise ValueError('The fixed source-only physical estimator, prior and gate are required')
 blind=deps.BlindActionNet();blind.load_state_dict(blind_state['model'],strict=True)
 gate=deps.s.BlindGate();gate.load_state_dict(gate_state['gate'],strict=True)
 physical=PhysicalPosterior(blind,prior['means'],prior['covariances'],prior['priors'],
                            action_matrix=deps.action_matrix,posterior=deps.posterior)
 physical=physical.to(device).eval().requires_grad_(False)
 gate=gate.to(device).eval().requires_grad_(False)
 dura=frozen_dura(root,device).eval().requires_grad_(False)
 return physical.blind,gate,dura,physical


@torch.no_grad()
def input_noise(x,blind,dura,physical,deps):
 anchor=dura(x);posterior,logvar=physical(x)
 if anchor.shape!=x.shape or posterior.shape!=x.shape or logvar.shape!=(len(x),):
  raise ValueError('One posterior, DURA and noise estimate per received record required')
 if not all(torch.isfinite(v).all() for v in (x,anchor,posterior,logvar)):
  raise FloatingPointError('Finite physical fusion evidence required')
 return torch.cat((anchor,posterior),dim=1),logvar


@torch.no_grad()
def predict_batch(x,model,blind,gate,dura,physical,reader,deps):
 scores=gate(x).reshape(-1);anchor=dura(x)
 if scores.shape!=(len(x),) or anchor.shape!=x.shape or not all(torch.isfinite(v).all() for v in (x,scores,anchor)):
  raise FloatingPointError('Finite source-only inference evidence required')
 active=scores.sigmoid()>.8;output=anchor.clone()
 if active.any():
  posterior,lv=physical(x[active]);packed=torch.cat((anchor[active],posterior),dim=1)
  output[active]=model(x[active],packed,lv)
 choices=[]
 for wave in (x,anchor,output):
  logits=reader.classify_features(reader.features(wave))
  if logits.shape!=(len(x),6) or not all(torch.isfinite(v).all() for v in (wave,logits)):
   raise FloatingPointError('Finite IQ and frozen reader logits required')
  choices.append(logits.argmax(1))
 return anchor,output,active,choices
