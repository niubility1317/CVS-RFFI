from contextlib import contextmanager
from types import MethodType
import torch
from . import design as d,mechanism as f
from experiments.cvs_sigtime_rdrop_legacy import model as original
classifier=original.classifier;source_domain_count=original.source_domain_count
@contextmanager
def context(self):
 if not self.training:
  with self._mask_native_context():yield
  return
 state=self._pair_state;state.start()
 with self._mask_native_context():yield
 state.between()
def hook(self,**kw):
 if kw.get('model') is not self or not self.training or not torch.is_grad_enabled() or kw['optimizer_step']!=self._pair_state.audit['calls']+1:raise ValueError('Only own ordered student L pairs')
 self._pair_state.second();correction,metrics=original.hook(self,**kw);self._pair_state.finish()
 metrics={k.replace('legacy_rdrop/','paired_masks/'):v for k,v in metrics.items()};metrics['paired_masks/mask_pair_calls']=self._pair_state.audit['calls']
 return correction,metrics
hook._accepts_labeled_inputs=False
def attach(m,c):
 f.install(m,c['coupling']);m._mask_native_context=m._phase1_labeled_feature_context;m._phase1_labeled_feature_context=MethodType(context,m);m._source_labeled_feature_hook=MethodType(hook,m);m._feature_learning_report['RDrop_paired_masks']=c['training_mechanism'];return m
def install(m,c):d.validate(c);return attach(original.install(m,d.native_config(c)),c)
def build(c,device):d.validate(c);return attach(original.build(d.native_config(c),device),c)
