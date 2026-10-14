from . import design as d
from experiments.cvs_sigtime_pair50.model import source_domain_count,classifier
from experiments.cvs_sigtime_pair50.model import install as shared_install,build as shared_build
def install(model,c):return shared_install(model,c,design=d)
def build(c,device='cpu',*,design=None):return shared_build(c,device,design=design or d)
