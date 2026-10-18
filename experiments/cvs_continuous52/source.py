from experiments.cvs_crossview_supcon41.source import segment_train as backend

def parent_payload(c):raise ValueError('Continuous30 has no parent checkpoint')
def segment_train(c,design=None,model_api=None,observer=None):
 if c['segment']!=20 or c['epochs']!=30 or c['parent_source'] or c['checkpoint_sources']:raise ValueError('Only single scratch30;20 is legacy attachment key,not epoch budget')
 return backend(c,design=design,model_api=model_api,observer=observer)
