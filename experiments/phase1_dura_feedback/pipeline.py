"""Frozen same-window F features conditioning continuous DURA correction."""
import torch
from experiments.phase1_dura_feedback.model import pack_context
from experiments.phase1_dura_volterra.pipeline import load_components as load_dura_components

def load_components(root,device,deps):
    blind,gate,dura,_=load_dura_components(root,device,deps)
    condition_reader,_=deps.core.teacher(root,device)
    return blind,gate,dura,condition_reader.eval().requires_grad_(False)

@torch.no_grad()
def input_noise(x,blind,dura,condition_reader,deps):
    anchor=dura(x);_,logvar=blind(x)
    features=condition_reader.features(anchor)
    logits=condition_reader.classify_features(features)
    if anchor.shape!=x.shape or logvar.shape!=(len(x),):
        raise ValueError('Matching frozen DURA output and blind noise required')
    return pack_context(anchor,features,logits),logvar

@torch.no_grad()
def predict_batch(x,model,blind,gate,dura,condition_reader,reader,deps):
    scores=gate(x).reshape(-1);anchor=dura(x)
    if scores.shape!=(len(x),) or anchor.shape!=x.shape or not all(torch.isfinite(v).all() for v in (x,scores,anchor)):
        raise FloatingPointError('Finite gate, input and DURA output required')
    active=scores.sigmoid()>.8
    output=anchor.clone()
    if active.any():
        selected=x[active];base=anchor[active];_,logvar=blind(selected)
        features=condition_reader.features(base);logits=condition_reader.classify_features(features)
        packed=pack_context(base,features,logits)
        output[active]=model(selected,packed,logvar)
    choices=[]
    for wave in (x,anchor,output):
        logits=reader.classify_features(reader.features(wave))
        if logits.shape!=(len(x),6) or not all(torch.isfinite(v).all() for v in (wave,logits)):
            raise FloatingPointError('Finite IQ and frozen-reader outputs required')
        choices.append(logits.argmax(1))
    return anchor,output,active,choices
