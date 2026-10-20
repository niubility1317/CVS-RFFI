import torch
from . import design as d
from .models import feature_distance,posterior_distance
from experiments.pamf_feature_refine.target import readers


def load_teachers(args,device):
    models=readers(args,device)
    if tuple(models)!=d.TEACHER_SEEDS:raise ValueError('All three original frozen F readers required')
    return models


def combine(per_teacher,arm):
    if arm not in d.ARMS or per_teacher.ndim!=2 or per_teacher.shape[0]!=3:
        raise ValueError('Three per-teacher per-sample losses and registered aggregation required')
    return per_teacher.mean(0)


def guidance(models,estimate,references,arm):
    if tuple(models)!=d.TEACHER_SEEDS or references.shape!=(len(estimate),3,160):
        raise ValueError('Teacher-specific reference coordinates required')
    features=[];posteriors=[]
    for i,model in enumerate(models.values()):
        reference=references[:,i].detach()
        embedding=model.features(estimate)
        features.append(feature_distance(embedding,reference))
        posteriors.append(posterior_distance(model.classify_features(embedding),model.classify_features(reference)))
    # Keep each teacher graph separate for its input-gradient call.
    teacher=tuple(f+d.configuration(arm)['posterior_weight']*p for f,p in zip(features,posteriors))
    feature=torch.stack(features);posterior=torch.stack(posteriors)
    return combine(torch.stack(teacher),arm),feature.mean(0),posterior.mean(0),teacher
