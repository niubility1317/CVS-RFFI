from . import design as d
from experiments.cvs_sigtime_rdrop_legacy import model as original
from experiments.cvs_sigtime_rdrop_masks20.model import attach
classifier=original.classifier;source_domain_count=original.source_domain_count
def install(m,c):d.validate(c);return attach(original.install(m,d.native_config(c)),c)
def build(c,device):d.validate(c);return attach(original.build(d.native_config(c),device),c)
