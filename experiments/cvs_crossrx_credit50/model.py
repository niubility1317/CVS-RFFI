"""Every control has exactly the same existing SIG_TIME/default classifier."""
from . import design as d
from experiments.cvs_feature_sequence50 import model as parent
source_domain_count=parent.source_domain_count
classifier=parent.classifier

def install(model,c):
 d.validate(c)
 return parent.install(model,d.structural_config(c['model_seed']))

def build(c,device):
 d.validate(c)
 return parent.build(d.structural_config(c['model_seed']),device)
