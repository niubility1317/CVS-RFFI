from . import design as d
from experiments.cvs_signature_stats50 import model as shared
source_domain_count=shared.source_domain_count
classifier=shared.classifier
def install(model,c):return shared.install(model,c,design=d)
def build(c,device='cpu'):return shared.build(c,device,design=d)
def groups(model,c):return shared.groups(model,c,design=d)
