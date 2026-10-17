"""Exact original raw-extra B00 F/default head; no model or loss changes."""
from . import design as d
from experiments.cvs_scene_cycle20 import model as original
classifier,source_domain_count,export=(getattr(original,k) for k in ('classifier','source_domain_count','export'))
def attach_report(m,c):
 m._L_order_config=c['batch_order'];m._feature_learning_report.update(L_batch_order=c['batch_order'],new_loss_terms=0,new_IQ_transform=False,new_deployment_state=False)
 return m
def build(c,device='cpu'):
 d.validate(c);return attach_report(original.build(d.legacy_config(),device),c)
def install(m,c):
 d.validate(c);return attach_report(original.install(m,d.legacy_config()),c)
