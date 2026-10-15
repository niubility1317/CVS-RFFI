from . import design as d
from experiments.cvs_evidence_responsibility_train20 import model as parent
source_domain_count=parent.source_domain_count
def install(model,c):
    d.validate(c)
    return parent.install(model,d.parent_config())
