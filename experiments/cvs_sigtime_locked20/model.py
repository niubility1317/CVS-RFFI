from . import design as d
from .mechanism import install_dropout
from experiments.cvs_evidence_responsibility_train20 import model as base
classifier=base.classifier;source_domain_count=base.source_domain_count

def install(model,c):
    d.validate(c);model=base.install(model,d.previous.parent_config())
    report=install_dropout(model,c['arm'])
    model._feature_learning_report['t2_dropout_training']=report
    return model
