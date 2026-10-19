import torch
from . import design as d
from .models import feature_distance,posterior_distance
from .identity import identity_terms
from experiments.pamf_feature_refine.target import readers


def load_teachers(args,device):
    models=readers(args,device)
    if tuple(models)!=d.TEACHER_SEEDS:raise ValueError('All three original frozen F readers required')
    return models


def guidance(models,estimate,references,arm,pseudo,accepted,prototypes,prototype_mask):
    n=len(pseudo)
    if (arm not in d.ARMS or tuple(models)!=d.TEACHER_SEEDS or references.shape!=(len(estimate),3,160) or
        n>len(estimate) or prototypes.shape!=(n,3,160)):
        raise ValueError('Separate teacher coordinates and observed-half identity targets required')
    features=[];posteriors=[];native=[];alignment=[]
    for i,model in enumerate(models.values()):
        reference=references[:,i].detach();embedding=model.features(estimate)
        logits=model.classify_features(embedding)
        features.append(feature_distance(embedding,reference))
        posteriors.append(posterior_distance(logits,model.classify_features(reference)))
        ce,cosine=identity_terms(embedding[:n],logits[:n],pseudo,accepted,prototypes[:,i],prototype_mask,arm)
        native.append(ce);alignment.append(cosine)
    feature=torch.stack(features).mean(0);posterior=torch.stack(posteriors).mean(0)
    return feature+.3*posterior,feature,posterior,torch.stack(native).mean(0),torch.stack(alignment).mean(0)
