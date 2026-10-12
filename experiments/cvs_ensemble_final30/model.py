import torch
from torch import nn
from experiments.cvs_stochastic_ensemble.model import build_member,groups,probabilities

class ProbabilityEnsemble(nn.Module):
 def __init__(self,models):
  super().__init__()
  if len(models)!=3:raise ValueError('Exactly3fixedmembersrequired')
  self.models=nn.ModuleList(models)
 def forward(self,x):return probabilities([m(x) for m in self.models]).clamp_min(1e-12).log()
