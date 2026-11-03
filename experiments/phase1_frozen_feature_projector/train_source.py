"""Train independent frozen-feature IQ projection using paired source only."""
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dynamic_volterra import train_source as base
from experiments.phase1_equivariant_dictionary.objective import source_loss
from experiments.phase1_frozen_feature_projector.model import FeatureProjector

ROOT=Path('/home/szu2070436088/2510044040/CV-SincNet')
_current=[None]


def make_model():
 reader,_=core.teacher(ROOT,'cpu')
 _current[0]=FeatureProjector(reader)
 return _current[0]


def projector_objective(reader,output,anchor,clean,clean_features,groups):
 model=_current[0]
 if model is None or model.latest_target is None or model.latest_target.shape!=clean_features.shape:
  raise ValueError('Matching predicted source feature target required')
 primary,terms=source_loss(reader,output,clean,clean_features,groups)
 per=1-F.cosine_similarity(model.latest_target,clean_features.detach(),dim=1)
 feature_target=torch.stack([per[groups==g].mean() for g in groups.unique()]).mean()
 if not torch.isfinite(feature_target):raise FloatingPointError('Finite predicted-feature objective required')
 return primary+.5*feature_target,dict(terms,predicted_feature=feature_target)


def main():
 names=('DynamicVolterra','SCHEMA','LOG_PREFIX','source_objective','USES_DURA','extra_frozen_model_components')
 old=tuple(getattr(base,n) for n in names)
 new=(make_model,'phase1_frozen_feature_projector_v1','FROZEN_FEATURE_PROJECTOR',projector_objective,False,lambda model:(model.reader,))
 for name,value in zip(names,new):setattr(base,name,value)
 try:base.main()
 finally:
  for name,value in zip(names,old):setattr(base,name,value)
  _current[0]=None


if __name__=='__main__':main()
