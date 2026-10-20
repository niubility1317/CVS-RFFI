import torch
from . import design as d
from .models import feature_distance,posterior_distance
from experiments.pamf_feature_refine.target import readers


def load_teachers(args,device):
    models=readers(args,device)
    if tuple(models)!=d.TEACHER_SEEDS:raise ValueError('Three original frozen F/defaultheads required')
    return models


def guidance(models,estimate,references,arm):
    if arm!=d.ARMS[1] or tuple(models)!=d.TEACHER_SEEDS or references.shape!=(len(estimate),3,160):
        raise ValueError('Only guided scratch arm may run three-teacher guidance')
    features=[];posteriors=[]
    for column,model in enumerate(models.values()):
        reference=references[:,column].detach();embedding=model.features(estimate)
        features.append(feature_distance(embedding,reference))
        posteriors.append(posterior_distance(model.classify_features(embedding),model.classify_features(reference)))
    feature=torch.stack(features).mean(0);posterior=torch.stack(posteriors).mean(0)
    return feature+.3*posterior,feature,posterior
