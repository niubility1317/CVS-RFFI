import math
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d
from .models import feature_distance,posterior_distance
from experiments.pamf_feature_refine.target import readers


def array_tensor(value,device):
    # Writable buffer avoids the unavailable NumPy C bridge in Torch2.1/NumPy2.
    value=np.asarray(value,dtype=np.float32)
    return torch.frombuffer(bytearray(value.tobytes(order='C')),dtype=torch.float32).reshape(value.shape).to(device)


def load_teachers(args,device):
    models=readers(args,device)
    if tuple(models)!=d.TEACHER_SEEDS:raise ValueError('All three original frozen F readers required')
    return models


def feature_projection(seed,device,dtype):
    generator=torch.Generator(device='cpu').manual_seed(d.PROJECTION_SEED+int(seed))
    signs=torch.randint(0,2,(d.PROBES,160),generator=generator,dtype=torch.int64)
    return ((2*signs-1).to(dtype)/math.sqrt(d.PROBES)).to(device)


def coordinates(model,embedding,arm,seed):
    if arm==d.ARMS[0]:
        return F.normalize(embedding,dim=1)@feature_projection(seed,embedding.device,embedding.dtype).T/math.sqrt(2)
    if arm==d.ARMS[1]:
        probabilities=(model.classify_features(embedding)/d.TEMPERATURE).softmax(1)
        return math.sqrt(2)*d.TEMPERATURE*probabilities.clamp_min(d.POSTERIOR_FLOOR).sqrt()
    raise ValueError('Registered tangent coordinates required')


def source_jacobians(models,clean,arm):
    d.configuration(arm)
    if tuple(models)!=d.TEACHER_SEEDS or clean.ndim!=3 or clean.shape[1:]!=(2,256) or clean.dtype!=torch.float32 or not torch.isfinite(clean).all():
        raise ValueError('Three original F and finite FP32 source-clean IQ required')
    features=[];jacobians=[]
    for seed,model in models.items():
        if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Frozen eval original F required')
        x=clean.detach().clone().requires_grad_(True)
        embedding=model.features(x);value=coordinates(model,embedding,arm,seed)
        if embedding.shape!=(len(x),160) or value.shape!=(len(x),d.PROBES):raise ValueError('Native160feature/six coordinates required')
        jac=torch.stack([torch.autograd.grad(value[:,j].sum(),x,retain_graph=j<d.PROBES-1)[0] for j in range(d.PROBES)],1)
        if not torch.isfinite(jac).all() or not torch.isfinite(embedding).all():raise FloatingPointError('Nonfinite source tangent')
        features.append(embedding.detach());jacobians.append(jac.detach())
    return torch.stack(features,1),torch.stack(jacobians,1)


def tangent_distance(estimate,clean,jacobians):
    if estimate.shape!=clean.shape or estimate.ndim!=3 or estimate.shape[1:]!=(2,256) or jacobians.shape!=(len(estimate),3,d.PROBES,2,256):
        raise ValueError('Aligned per-physical source-clean tangent required')
    if estimate.dtype!=torch.float32 or clean.dtype!=torch.float32 or jacobians.dtype!=torch.float32 or not torch.isfinite(jacobians).all():
        raise ValueError('Finite FP32 source tangent required')
    displacement=(estimate-clean.detach()).flatten(1)
    value=torch.bmm(jacobians.detach().reshape(len(estimate),3*d.PROBES,512),displacement[:,:,None])
    return value.reshape(len(estimate),3,d.PROBES).square().sum(2).T


def guidance(models,estimate,references,arm,clean,jacobians):
    d.configuration(arm)
    if tuple(models)!=d.TEACHER_SEEDS or references.shape!=(len(estimate),3,160):raise ValueError('Teacher-specific reference coordinates required')
    local=tangent_distance(estimate,clean,jacobians);features=[];posteriors=[]
    for i,model in enumerate(models.values()):
        reference=references[:,i].detach();embedding=model.features(estimate)
        if arm==d.ARMS[0]:
            features.append(local[i])
            posteriors.append(posterior_distance(model.classify_features(embedding),model.classify_features(reference)))
        else:
            features.append(feature_distance(embedding,reference));posteriors.append(local[i])
    feature=torch.stack(features).mean(0);posterior=torch.stack(posteriors).mean(0)
    return feature+d.configuration(arm)['posterior_weight']*posterior,feature,posterior
