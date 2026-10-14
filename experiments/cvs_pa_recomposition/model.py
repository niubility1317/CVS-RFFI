from . import design as d,features
from experiments.cvs_feature_multiscale50.model import build as original,source_domain_count,classifier
from experiments.cvs_feature_restitution.features import install as counter
def report(c):return dict(c['feature_extractor'],arm=c['arm'],default_head_only=True,separate_head_training=False,classifier_type='ResidualPhysicalHead',classifier_weight_shape=[6,160],classifier_scale=30.)
def install(model,c):
 d.validate(c);model=counter(model,c['model_seed'],norm='none',competition=False);features.install(model);model._feature_learning_report=report(c);return model
def build(c,device):return install(original(d.baseline_config(c['model_seed']),device),c)
def validate_report(v,c):
 if v!=report(c):raise ValueError('Actual PA/defaulthead differs')
