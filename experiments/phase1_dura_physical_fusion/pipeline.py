"""Frozen physical posterior and DURA packed for single-record fusion."""
import torch
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura


def load_components(root,device,deps):
 from experiments.phase1_dura_posterior.physical_source_smoke import load_source_models
 physical,gate=load_source_models(root,'mean')
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
