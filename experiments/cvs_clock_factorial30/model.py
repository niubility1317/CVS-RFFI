"""Exact parent extractor/head; segment-local counters with cumulative IQ streams."""
from experiments.cvs_clock_factorial20 import model as clock
from experiments.cvs_fullsource_concat20 import model as ref
from . import design as d
from .clock import Clock
from .channel import Practical
classifier,source_domain_count,export=clock.classifier,clock.source_domain_count,clock.export
class Shifted:
 def __init__(self,obj):self.obj=obj
 def __getattr__(self,k):
  if k=='obj':raise AttributeError(k)
  return getattr(self.obj,k)
 def apply(self,x,meta,*args,**kw):
  if args:
   if len(args)!=2 or kw:raise ValueError('Clock epoch/step arguments differ')
   return self.obj.apply(x,meta,args[0]+20,args[1]+4440)
  return self.obj.apply(x,meta,epoch=kw['epoch']+20,step=kw['step']+4440)
def attach(m,c):
 if m._full_channel is not None:m._full_channel=Shifted(Practical('train'))
 if getattr(m,'_clock',None) is not None:m._clock=Shifted(Clock())
 m._feature_learning_report['E30_amendment']=dict(parent=c['parent_source'],new_epochs=10,cumulative_epochs=30,optimizer='resetcos10',EMA='ownparent',no_new_parameters=True)
 return m
def install(m,c):
 d.validate(c);p=d.parent_config(c['arm'])
 m=ref.install(m,p) if c['arm']=='FULL_AS' else clock.install(m,clock.d.config(next(r for r in clock.d.rows() if r['arm']==c['arm'])))
 return attach(m,c)
def build(c,device='cpu'):
 d.validate(c);p=d.parent_config(c['arm'])
 m=ref.build(p,device) if c['arm']=='FULL_AS' else clock.build(clock.d.config(next(r for r in clock.d.rows() if r['arm']==c['arm'])),device)
 return attach(m,c)
