from types import MethodType
import torch
from . import design as d
from .supcon import attach as attach_supcon
from experiments.cvs_raw_anchor20 import model as raw
source_domain_count,classifier=raw.source_domain_count,raw.classifier

def hook(self,**kw):
    first=self._legacy_actual_L_input;primary=self._pair_primary;other=self._pair_secondary
    if primary is None or other is None or first is None or self._anchor_calls+1!=kw['optimizer_step'] or not torch.equal(first,primary):
        raise ValueError('One exact primary/secondary source IQ pair per update required')
    self._pair_primary=None;self._pair_secondary=None
    correction,metrics=self._anchor_original_hook(**kw)
    a=self.forward_identity_only(other,domain_labels=kw['domains'])['tx_logits']
    b=self.forward_identity_only(other,domain_labels=kw['domains'])['tx_logits']
    value,ce,kl,ce1,ce2=raw.base.legacy.terms(a,b,kw['labels'],1.,.01)
    original=kw['identity_loss']+correction;self._anchor_calls+=1;self._anchor_extra_packets+=2*len(other)
    metrics.update({'pair/primary_objective':original.detach(),'pair/secondary_objective':value.detach(),
        'pair/primary_CE':metrics['legacy_rdrop/paired_CE'],'pair/secondary_CE':ce.detach(),
        'pair/primary_weight':.5,'pair/secondary_weight':.5,'pair/L_forwards_per_sample':4})
    for key,v in [('paired_CE',ce),('symmetric_KL',kl),('CE1',ce1),('CE2',ce2)]:metrics['legacy_rdrop/'+key]=.5*(metrics['legacy_rdrop/'+key]+v.detach())
    metrics['pair/actual_native_primary_reused']=metrics.pop('legacy_rdrop/actual_augmented_input_reused')
    return .5*(correction+value-kw['identity_loss']),metrics
hook._accepts_labeled_inputs=False

def attach(m,c):
    m=raw.attach(m,c,validator=d.validate);m._source_labeled_feature_hook=MethodType(hook,m)
    m._pair_primary=None;m._pair_secondary=None
    m._feature_learning_report['baseline_CROSS_RX']=c['supcon']
    return attach_supcon(m,c)

def install(m,c):return attach(raw.F_configure(raw.base.legacy.install(m,d.parent_config()),c),c)
def build(c,device='cpu'):return attach(raw.F_configure(raw.base.legacy.build(d.parent_config(),'cpu'),c).to(device),c)
