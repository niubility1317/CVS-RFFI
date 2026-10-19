"""Bind historical, unchanged learning code to one new seed in an isolated worker."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
from pathlib import Path
from . import design as d
_CACHE={}

@contextmanager
def factory_seed(seed):
 full=importlib.import_module('experiments.cvs_fullsource_concat20.design')
 previous=full.SEED;full.SEED=seed
 try:
  if full.base_model_config()['model_seed']!=seed:raise ValueError('F factory does not use actual model seed')
  yield
 finally:full.SEED=previous

def _bind_design(module,r,stage20=False):
 original_config=module.config;original_args=module.make_args
 module.ROOT=d.ROOT;module.RUN=d.RUN;module.BASE=d.BASE;module.OWNER=d.OWNER;module.SEED=r['model_seed'];module.VIEWS=d.VIEWS
 def config(*args,**kwargs):
  c=deepcopy(original_config(*args,**kwargs))
  c['run_id']=d.RUN;c['model_seed']=r['model_seed'];c['launch_owner']=d.OWNER
  c['output_root']=str(d.BASE/c['row_id']/('stage20' if stage20 else ('stage20' if c.get('segment')==20 else 'source')))
  if r['family']=='pa':
   from experiments.cvs_As_replicate49 import design as p49
   parent=next(x for x in p49.rows() if x['arm']=='B01' and x['model_seed']==r['model_seed'])
   inherited=p49.config(parent,30);c['parent_source']=inherited['parent_source'];c['checkpoint_sources']=[c['parent_source']+'/final_ssdg.pth']
  c['test_views']=['clean']
  c['seed_completion']=dict(group=r['group'],model_seed=r['model_seed'],native_seed=r['model_seed'],F_factory_seed=r['model_seed'],physical_split=392005,training_recipe_changed=False,seed01_reused=True,test_scope='clean_only',selection='User-requested fixed replication after exposed seed01 result; no performance selection')
  return c
 def make_args(c,device='cuda:0'):
  a=original_args(c,device);a.seed=r['model_seed'];a.run_id=d.RUN;a.candidate_id=c['row_id'];a.output_dir=c['output_root']
  if a.teacher_ckpt or a.freeze_backbone:raise ValueError('No external teacher checkpoint or separately fitted head permitted')
  if (a.label_epochs,a.pseudo_epochs,a.epochs)!=(13,7,20) and (a.label_epochs,a.pseudo_epochs,a.epochs)!=(0,10,10):raise ValueError('Original fixed20+10 budget required')
  return a
 module.config=config;module.make_args=make_args
 return module

def configured(r):
 d.row_ok(r)
 if r['row_id'] in _CACHE:return _CACHE[r['row_id']]
 if _CACHE:raise ValueError('One row per process; historical module globals must not be shared')
 family=r['family'];ref=d.runtime()['references'][family];package='experiments.'+ref['package']
 if family in ('clock','fir'):
  parent_row=d.row('B01-s'+str(r['model_seed']))
  twenty=importlib.import_module('experiments.cvs_clock_factorial20.design')
  _bind_design(twenty,parent_row,stage20=True)
  tail=importlib.import_module('experiments.cvs_clock_factorial30.design')
  _bind_design(tail,parent_row)
  target=tail if family=='clock' else _bind_design(importlib.import_module(package+'.design'),r)
 else:target=_bind_design(importlib.import_module(package+'.design'),r)
 def config(segment=30):
  if segment not in (20,30):raise ValueError('Fixed segments20/30 only')
  if family in ('pa','fir') and segment!=30:raise ValueError('These requested variants only add their original ten-epoch tail')
  if family in ('clock','fir'):
   mod=twenty if segment==20 else target
   arm='B01' if segment==20 or family=='clock' else r['arm']
   row=next(x for x in mod.rows() if x['arm']==arm)
   c=mod.config(row)
  else:
   row=next(x for x in target.rows() if x['arm']==r['arm']);c=target.config(row,segment)
  if c['row_id']!=r['row_id'] or c['model_seed']!=r['model_seed'] or c['run_id']!=d.RUN:raise ValueError('Actual row identity differs')
  return c
 b=dict(row=r,design=target,package=package,config=config)
 if family in ('clock','fir'):b['twenty']=twenty
 _CACHE[r['row_id']]=b
 return b

def modules(r):
 b=configured(r);api=importlib.import_module(b['package']+'.model')
 if not getattr(api,'_selected_seed_wrapped',False):
  for name in ('build','install'):
   fn=getattr(api,name)
   def wrap(*args,_fn=fn,**kwargs):
    with factory_seed(r['model_seed']):model=_fn(*args,**kwargs)
    model._selected_factory_seed=r['model_seed'];return model
   setattr(api,name,wrap)
  api._selected_seed_wrapped=True
 if r['family'] in ('clock','fir'):
  early=importlib.import_module('experiments.cvs_clock_factorial20.model')
  if not getattr(early,'_selected_seed_wrapped',False):
   for name in ('build','install'):
    fn=getattr(early,name)
    def wrap_early(*args,_fn=fn,**kwargs):
     with factory_seed(r['model_seed']):model=_fn(*args,**kwargs)
     model._selected_factory_seed=r['model_seed'];return model
    setattr(early,name,wrap_early)
   early._selected_seed_wrapped=True
 if r['family']=='pa':
  source=importlib.import_module(b['package']+'.source')
  def parent_payload(c):
   b['design'].validate(c)
   from experiments.cvs_As_replicate49 import design as p49,source as oldsource
   row=next(x for x in p49.rows() if x['arm']=='B01' and x['model_seed']==r['model_seed'])
   expected=p49.config(row,30)
   if c['parent_source']!=expected['parent_source']:raise ValueError('Only own matching P49 scratch20 parent')
   cp=oldsource.parent_payload(expected)
   if cp['args']['seed']!=r['model_seed']:raise ValueError('PA parent actual seed differs')
   return cp
  source.parent_payload=parent_payload
 return b,api

def train(r):
 import gc,time,torch
 b,api=modules(r);timings=[]
 stages=(30,) if r['family'] in ('pa','fir') else (20,30)
 for segment in stages:
  c=b['config'](segment);start=time.monotonic()
  if r['family']=='clock':source=importlib.import_module('experiments.cvs_clock_factorial20.source' if segment==20 else 'experiments.cvs_clock_factorial30.source')
  else:source=importlib.import_module(b['package']+('.source' if r['family']=='fir' else '.observer'))
  source.train(c)
  timings.append(dict(segment=segment,epochs=c['epochs'],updates=c['updates'],seconds=time.monotonic()-start))
  gc.collect();torch.cuda.empty_cache()
 d.write(d.BASE/r['row_id']/'training_execution.json',dict(status='TRAINED',stages=timings,new_updates=sum(x['updates'] for x in timings),new_epochs=sum(x['epochs'] for x in timings),actual_factory_seed=r['model_seed'],recipe_changed=False,target_access=False))
 return timings
