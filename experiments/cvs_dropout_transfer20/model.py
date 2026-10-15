from types import MethodType
import torch
from torch.nn import functional as F
from . import design as d,mechanism
from experiments.cvs_evidence_responsibility_train20 import model as base
from experiments.cvs_rdrop_fold20.model import terms
classifier=base.classifier;source_domain_count=base.source_domain_count
def hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count,labeled_inputs):
 if self is not model or not self.training or not torch.is_grad_enabled() or optimizer_step!=self._feature_l_steps or optimizer_step!=self._exchange_calls+1:raise ValueError('Only ordered actual L student context permitted')
 if labeled_count!=128 or labels.shape!=(128,) or labeled_inputs.shape!=(128,2,256):raise ValueError('Native clean L128 input budget differs')
 from experiments.cvs_phase1_overlay.model import native_modules
 meta=native_modules()._meta_from_extra(extra)
 if meta is None:raise ValueError('L physical metadata required before paired forward')
 maps,n=mechanism.donors(labels,meta);donor=maps[self._exchange_arm]
 first=labeled_output;second=self.forward_identity_only(labeled_inputs.clone(),domain_labels=domains)
 _,paired,kl,ce1,ce2=terms(first['tx_logits'],second['tx_logits'],labels,1.)
 head=classifier(self);W=head.weight;z1,z2=(o['id_feat_joint'] for o in (first,second));virtual,audit=mechanism.loss(z1,z2,W,labels,donor,classify=head.classify)
 head_error=max(float((head.classify(o['id_feat_joint'])-o['tx_logits']).detach().abs().max()) for o in (first,second))
 if head_error>2e-5 or not torch.allclose(identity_loss.detach(),ce1.detach(),atol=1e-6,rtol=1e-5):raise ValueError('Virtual features must use actual native default cosine classifier')
 self._exchange_calls+=1;self._exchange_extra_packets+=128;self._exchange_cohort_packets+=n
 swapped=int((donor!=torch.arange(128,device=donor.device)).sum());self._exchange_swapped_packets+=swapped
 self._exchange_max_energy_error=max(self._exchange_max_energy_error,audit['energy_relative_error'])
 metrics={'exchange/calls':self._exchange_calls,'exchange/extra_L_packets':self._exchange_extra_packets,'exchange/cohort_packets':n,'exchange/swapped_packets':swapped,
  'exchange/cumulative_cohort_packets':self._exchange_cohort_packets,'exchange/cumulative_swapped_packets':self._exchange_swapped_packets,
  'exchange/CE_virtual':virtual.detach(),'exchange/CE_native_paired':paired.detach(),'exchange/symmetric_KL':kl.detach(),'exchange/CE1':ce1.detach(),'exchange/CE2':ce2.detach(),
  'exchange/native_head_max_abs_error':head_error,**{'exchange/'+k:v for k,v in audit.items()}}
 # Native closed loss already includes CE1. Replace that term once; original U remains.
 return virtual-identity_loss+kl,metrics
hook._accepts_labeled_inputs=True
def attach(model,c):
 model._exchange_arm=c['arm'];model._exchange_calls=0;model._exchange_extra_packets=0;model._exchange_cohort_packets=0;model._exchange_swapped_packets=0;model._exchange_max_energy_error=0.
 model._source_labeled_feature_hook=MethodType(hook,model);model._feature_learning_report['stochastic_residual_exchange']=c['training_mechanism'];return model
def install(model,c):d.validate(c);return attach(base.install(model,d.base_config()),c)
def build(c,device):d.validate(c);return attach(base.build(d.base_config(),device),c)
