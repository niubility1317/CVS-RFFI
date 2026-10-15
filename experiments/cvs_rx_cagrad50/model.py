from experiments.cvs_receiver_dro50.model import install,source_domain_count,classifier
from experiments.cvs_receiver_dro50.model import build as shared
from . import design as d
def build(c,device='cpu',*,design=None):return shared(c,device,design=design or d)
