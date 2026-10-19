import torch
from torch.nn import functional as F
from . import design as d
from .models import feature_distance,posterior_distance
from experiments.pamf_feature_refine.target import readers


def load_teachers(args,device):
    models=readers(args,device)
    if tuple(models)!=d.TEACHER_SEEDS:raise ValueError('All three original frozen F readers required')
    return models


def capture(model,x):
    values={};handles=[]
    for name in ('t1','t2'):
        module=getattr(model.core.id_backbone,name)
        handles.append(module.register_forward_hook(lambda module,args,output,name=name:values.update({name:output})))
    try:embedding=model.features(x)
    finally:
        for handle in handles:handle.remove()
    if set(values)!={'t1','t2'} or any(not isinstance(v,torch.Tensor) or v.ndim!=3 or len(v)!=len(x) or v.is_complex() for v in values.values()):
        raise ValueError('Actual native t1/t2 real BCT features required')
    return embedding,values


def middle_distance(estimated,reference):
    if estimated.ndim!=3 or estimated.shape!=reference.shape:raise ValueError('Matched native BCT maps required')
    return 1-F.cosine_similarity(estimated.flatten(1),reference.detach().flatten(1),dim=1,eps=1e-8)


def guidance(models,estimate,references,arm,clean=None):
    config=d.configuration(arm)
    if tuple(models)!=d.TEACHER_SEEDS or references.shape!=(len(estimate),3,160):
        raise ValueError('Teacher-specific reference coordinates required')
    if config['middle_weight'] and (clean is None or clean.shape[1:]!=(2,256) or not len(clean) or len(estimate) not in (len(clean),2*len(clean))):
        raise ValueError('Same physical paired-clean source references required for middle loss')
    features=[];posteriors=[];middles=[]
    for i,model in enumerate(models.values()):
        reference=references[:,i].detach()
        if config['middle_weight']:
            embedding,hidden=capture(model,estimate)
            with torch.no_grad():_,target=capture(model,clean)
            repeats=len(estimate)//len(clean)
            middle=torch.stack([middle_distance(hidden[name],target[name].repeat(repeats,1,1)) for name in ('t1','t2')]).mean(0)
        else:
            embedding=model.features(estimate);middle=estimate.new_zeros(len(estimate))
        features.append(feature_distance(embedding,reference))
        posteriors.append(posterior_distance(model.classify_features(embedding),model.classify_features(reference)))
        middles.append(middle)
    feature=torch.stack(features).mean(0);posterior=torch.stack(posteriors).mean(0);middle=torch.stack(middles).mean(0)
    return feature+.3*posterior+config['middle_weight']*middle,feature,posterior,middle
