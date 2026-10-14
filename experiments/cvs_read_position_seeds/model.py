from . import design as d
from experiments.cvs_read_position50 import model as shared
source_domain_count=shared.source_domain_count
classifier=shared.classifier
def install(model,c):d.validate(c);return shared.install(model,c)
def build(c,device='cpu',*,design=None):return shared.build(c,device,design=design or d)
