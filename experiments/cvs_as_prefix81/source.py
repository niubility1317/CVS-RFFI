"""Strict original B01 As20 default-head student+ownEMA;never LRaw/P78 states."""
from . import design as d
from experiments.cvs_lraw_tail30 import source as old
from experiments.cvs_crossview_supcon41 import source as backend
def parent_payload(c):
 d.validate(c)
 original=d.old.config(d.original_row())
 if c['parent_source']!=original['parent_source'] or c['checkpoint_sources']!=original['checkpoint_sources']:raise ValueError('Original As20 source differs')
 return old.parent_payload(original)
def segment_train(c,design=None,model_api=None,observer=None):
 previous=backend.parent_payload;backend.parent_payload=parent_payload
 try:return backend.segment_train(c,design=design,model_api=model_api,observer=observer)
 finally:backend.parent_payload=previous
