from types import MethodType
import torch
from . import design as d
from experiments.cvs_ordinary_fold20 import model as base
from .f_base.coordinate import CoordinateSinc
from .f_base.x3_features import CompositePool
source_domain_count,classifier=base.source_domain_count,base.classifier

def hook(self,**kw):
    raw_first=self._legacy_actual_L_input
    if raw_first is None or self._anchor_calls+1!=kw['optimizer_step']:raise ValueError('One ordered native L input required')
    other=self._anchor_G
    if self._anchor_augmented!=(other is not None):raise ValueError('Exactly one ordinary draw required for this arm')
    expected=other if self._anchor_arm=='A1' else self._anchor_raw
    if expected is not None and not torch.equal(raw_first,expected):raise ValueError('Actual native first view differs from declared arm')
    self._anchor_G=None;self._anchor_raw=None
    correction,metrics=self._anchor_original_hook(**kw)
    view=raw_first if other is None else other
    first=self.forward_identity_only(view,domain_labels=kw['domains'])['tx_logits']
    second=self.forward_identity_only(view,domain_labels=kw['domains'])['tx_logits']
    value,ce,kl,ce1,ce2=base.legacy.terms(first,second,kw['labels'],1.,.01)
    original=kw['identity_loss']+correction
    self._anchor_calls+=1;self._anchor_extra_packets+=2*len(view)
    metrics.update({'anchor/pair1_objective':original.detach(),'anchor/pair2_objective':value.detach(),
        'anchor/pair1_CE':metrics['legacy_rdrop/paired_CE'],'anchor/pair1_KL':metrics['legacy_rdrop/symmetric_KL'],
        'anchor/pair2_CE':ce.detach(),'anchor/pair2_KL':kl.detach(),
        'anchor/combined_objective':(.5*(original+value)).detach(),'anchor/calls':self._anchor_calls,
        'anchor/extra_secondpair_packets':self._anchor_extra_packets,'anchor/L_forwards_per_sample':4,
        'anchor/raw_primary_weight':.5 if self._anchor_arm!='A1' else 0.})
    metrics['legacy_rdrop/paired_CE']=.5*(metrics['legacy_rdrop/paired_CE']+ce.detach())
    metrics['legacy_rdrop/symmetric_KL']=.5*(metrics['legacy_rdrop/symmetric_KL']+kl.detach())
    metrics['legacy_rdrop/CE1']=.5*(metrics['legacy_rdrop/CE1']+ce1.detach());metrics['legacy_rdrop/CE2']=.5*(metrics['legacy_rdrop/CE2']+ce2.detach())
    metrics['anchor/actual_native_first_input_reused']=metrics.pop('legacy_rdrop/actual_augmented_input_reused')
    return .5*(correction+value-kw['identity_loss']),metrics
hook._accepts_labeled_inputs=False

def attach(m,c,validator=None):
    validator=validator or d.validate
    validator(c);m=base.attach(m,c,validator=validator)
    m._anchor_original_hook=m._source_labeled_feature_hook;m._source_labeled_feature_hook=MethodType(hook,m)
    m._anchor_G=None;m._anchor_raw=None;m._anchor_calls=0;m._anchor_extra_packets=0
    m._anchor_augmented=c['training_mechanism']['ordinary_L'];m._anchor_arm=c['arm']
    m._feature_learning_report['raw_anchor_training_only']=c['training_mechanism']
    return m
def F_configure(m,c):
    ref=m.id_backbone.encoder
    ref.sequence_pool=CompositePool(ref.sequence_pool.channels,'SIG_TIME_X3',c['model_seed'])
    ref.core.id_backbone.sinc=CoordinateSinc(ref.core.id_backbone.sinc,'CENTER_25K')
    m._feature_learning_report.update(arm='SIG_TIME_X3',modes=['SIG_TIME']*3,parameter_matched=False,
        independent_local_projections=True,local_dimension_each=6,descriptor_width_each=28,
        extra_parameters=sum(p.numel() for p in ref.sequence_pool.parameters()),
        combination='Three independent local views jointly feed one actual joint160; no voting',
        sinc_coordinates='CENTER_25K',Sinc_optimizer='CENTER_UNITS',A_F_arm='F')
    return m

def install(m,c):return attach(F_configure(base.legacy.install(m,d.parent_config()),c),c)
def build(c,device):return attach(F_configure(base.legacy.build(d.parent_config(),'cpu'),c).to(device),c)
