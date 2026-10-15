"""Exact current legacy SIG_TIME/RDrop, with parameter-free counters only."""
from types import MethodType
import torch
from . import design as d
from experiments.cvs_sigtime_rdrop_legacy import model as legacy
source_domain_count,classifier=legacy.source_domain_count,legacy.classifier


def features(self,x):
    name='_ordinary_train_packets' if self.training else '_ordinary_eval_packets'
    setattr(self,name,getattr(self,name)+len(x))
    return self._ordinary_original_features(x)


def hook(self,**kwargs):
    if self._ordinary_latest is not None:raise ValueError('Previous L metadata not drained')
    meta=kwargs['extra'][1];labels=kwargs['labels'].detach().cpu().tolist()
    rx,day=(meta[k].detach().cpu().tolist() for k in ('rx_i','day_i'))
    if len(labels)!=128 or set(rx)-set(self._ordinary_fit_RX):raise ValueError('Held RX entered L training')
    value=self._ordinary_original_hook(**kwargs)
    self._ordinary_latest=[(int(y),int(r),int(v)) for y,r,v in zip(labels,rx,day)]
    return value
hook._accepts_labeled_inputs=False


def attach(m,c,validator=None):
    (validator or d.validate)(c);m._ordinary_original_hook=m._source_labeled_feature_hook
    m._ordinary_fit_RX=tuple(c['fold']['fit_RX'])
    m._source_labeled_feature_hook=MethodType(hook,m);m._ordinary_latest=None
    e=m.id_backbone.encoder;e._ordinary_original_features=e.features
    e._ordinary_train_packets=0;e._ordinary_eval_packets=0;e.features=MethodType(features,e)
    m._feature_learning_report['ordinary_source_RX_control']=c['training_mechanism']
    return m


def install(m,c):return attach(legacy.install(m,d.parent_config()),c)
def build(c,device):return attach(legacy.build(d.parent_config(),device),c)
