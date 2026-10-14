"""Reuse the exact reviewed intervention with an explicit fixed matrix context."""
from . import design as d
from experiments.cvs_identity_mixstyle50.model import installed as shared_installed,build as shared_build

def installed(c,training=False):return shared_installed(c,training,design=d)
def build(c,device='cpu'):return shared_build(c,device,design=d)
