"""Reuse exactly the verified SAT18 AS_CE20 lineage and generic source trainer."""
from . import design as d
from experiments.cvs_noise_resample75.source import parent_payload as verified_parent
from experiments.cvs_crossview_supcon41 import source as backend

def parent_payload(c):
 d.validate(c);expected=d.old_config()
 if c['parent_source']!=expected['parent_source']:raise ValueError('Only shared SAT18 AS_CE20 source parent')
 return verified_parent(expected)

def segment_train(c,design=None,model_api=None,observer=None):
 previous=backend.parent_payload;backend.parent_payload=parent_payload
 try:return backend.segment_train(c,design=design,model_api=model_api,observer=observer)
 finally:backend.parent_payload=previous
