"""Strict original P43 B00/B01 own20 student/EMA sources, via committed loader."""
from . import design as d
from experiments.cvs_lraw_tail30 import source as old
from experiments.cvs_crossview_supcon41 import source as backend
def parent_payload(c):
 d.validate(c);row=next(r for r in d.rows() if r['row_id']==c['row_id']);original=d.old.config(row)
 if c['parent_source']!=original['parent_source'] or c['checkpoint_sources']!=original['checkpoint_sources']:raise ValueError('Original P43 own20 source differs')
 return old.parent_payload(original)
def segment_train(c,design=None,model_api=None,observer=None):
 previous=backend.parent_payload;backend.parent_payload=parent_payload
 try:return backend.segment_train(c,design=design,model_api=model_api,observer=observer)
 finally:backend.parent_payload=previous
