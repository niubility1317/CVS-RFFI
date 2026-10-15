from . import design as d
from experiments.cvs_sinc_units50.model import source_domain_count,classifier
from experiments.cvs_trusted_weight50.model import install as shared_install
from experiments.cvs_sinc_units50.model import build as shared_build
def install(model,c):
 d.validate(c);m=shared_install(model,c);m._source_neighbor_sampling=dict(c['source_neighbor_sampling'],feature='fixedSIG_TIME195041/defaultjointhead');return m
def build(c,device='cpu',*,design=None):return shared_build(c,device,design=design or d)
