"""Same deployed model and objective;replace only the training As quality input."""
from experiments.cvs_clock_factorial30.model import Shifted
from experiments.cvs_fullsource_concat20 import model as ref
from . import design as d
from .channel import Practical
classifier,source_domain_count,export=ref.classifier,ref.source_domain_count,ref.export
def attach(m,c):
 m._full_channel=Shifted(Practical('train'))
 m._feature_learning_report['As_quality']=dict(mode='finite_window_Hx_power',parent=c['parent_source'],new_epochs=10,cumulative_epochs=30,default_head_joint=True,extra_parameters=0,extra_neural_forwards=0)
 return m
def install(m,c):
 d.validate(c);return attach(ref.install(m,d.parent_config('FULL_AS')),c)
def build(c,device='cpu'):
 d.validate(c);return attach(ref.build(d.parent_config('FULL_AS'),device),c)
