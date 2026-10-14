"""Expose the actual additive PA evidence for a source-only prototype."""
from experiments.cvs_evidence_relation.components import install,parts

def actual_parts(model,out,role='L_s'):
 if role!='L_s':raise ValueError('PA recomposition first prototype only consumes L_s')
 v=parts(model,out)
 # parts stops gradients only to the fusion gains; internal evidence tensors
 # retain their graph. Its reconstruction is checked against deployed joint160.
 return out['id_feat_cls'],v['PA'],v['ref']

def default_head(model):
 h=model.id_backbone.encoder.core.id_backbone.cls_head
 if list(h.weight.shape)!=[6,160] or float(h.scale)!=30.:raise ValueError('Original default ResidualPhysicalHead required')
 return h
