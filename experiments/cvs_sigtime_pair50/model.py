from . import design as d
from experiments.cvs_trusted_weight50.model import source_domain_count,classifier
from experiments.cvs_trusted_weight50.model import install as signature_install,build as shared
def install(model,c,*,design=None):
 (design or d).validate(c);m=signature_install(model,c);m._source_neighbor_sampling=dict(c['source_neighbor_sampling'],feature='fixedSIG_TIME195041/defaultjointhead');return m
def build(c,device='cpu',*,design=None):
 m=shared(c,device,design=design or d);m._source_neighbor_sampling=dict(c['source_neighbor_sampling'],feature='fixedSIG_TIME195041/defaultjointhead');return m
