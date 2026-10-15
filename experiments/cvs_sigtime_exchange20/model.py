from types import MethodType
import torch
from . import design as d,mechanism
from experiments.cvs_sigtime_rdrop_legacy import model as original
classifier=original.classifier;source_domain_count=original.source_domain_count
terms=original.terms

def hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count):
 if self is not model or not self.training or not torch.is_grad_enabled() or optimizer_step!=self._feature_l_steps or optimizer_step!=self._exchange_calls+1 or optimizer_step!=self._legacy_calls+1 or labeled_count!=128:raise ValueError('Only ordered actual L student context permitted')
 from experiments.cvs_phase1_overlay.model import native_modules
 meta=native_modules()._meta_from_extra(extra)
 if meta is None:raise ValueError('Visible L physical metadata required')
 maps,n=mechanism.donors(labels,meta);donor=maps[self._exchange_arm]
 x=self._legacy_actual_L_input;self._legacy_actual_L_input=None
 if x is None or x.shape!=(128,2,256):raise ValueError('Actual native augmented L input required')
 second=self.forward_identity_only(x,domain_labels=domains)
 _,paired,kl,ce1,ce2=original.terms(labeled_output['tx_logits'],second['tx_logits'],labels,1.)
 if not torch.allclose(identity_loss.detach(),ce1.detach(),atol=1e-6,rtol=1e-5):raise ValueError('Native LS.01 CE differs')
 head=classifier(self);W=head.weight
 virtual,audit=mechanism.loss(labeled_output['id_feat_joint'],second['id_feat_joint'],W,labels,donor,head.classify)
 head_error=max(float((head.classify(out['id_feat_joint'])-out['tx_logits']).detach().abs().max()) for out in (labeled_output,second))
 if head_error>2e-5:raise ValueError('Virtual features must use actual default cosine head')
 self._exchange_calls+=1;self._legacy_calls+=1;self._legacy_extra_packets+=128;self._legacy_positive_KL_calls+=int(float(kl.detach())>0)
 self._exchange_extra_packets+=128;self._exchange_cohort_packets+=n
 swapped=int((donor!=torch.arange(128,device=donor.device)).sum());self._exchange_swapped_packets+=swapped
 self._exchange_max_energy_error=max(self._exchange_max_energy_error,audit['energy_relative_error'])
 exact=self._exchange_arm=='SELF'
 correction=.5*(ce2-identity_loss)+kl if exact else virtual-identity_loss+kl
 metrics={'exchange/calls':self._exchange_calls,'exchange/extra_L_packets':self._exchange_extra_packets,'exchange/cohort_packets':n,'exchange/swapped_packets':swapped,
  'exchange/cumulative_cohort_packets':self._exchange_cohort_packets,'exchange/cumulative_swapped_packets':self._exchange_swapped_packets,
  'exchange/CE_virtual':virtual.detach(),'exchange/CE_used':(paired if exact else virtual).detach(),'exchange/CE_native_paired':paired.detach(),
  'exchange/symmetric_KL':kl.detach(),'exchange/CE1':ce1.detach(),'exchange/CE2':ce2.detach(),'exchange/actual_augmented_input_reused':1,
  'exchange/native_head_max_abs_error':head_error,**{'exchange/'+k:v for k,v in audit.items()}}
 return correction,metrics
hook._accepts_labeled_inputs=False

def attach(model,c):
 model._exchange_arm=c['arm'];model._exchange_calls=0;model._exchange_extra_packets=0;model._exchange_cohort_packets=0;model._exchange_swapped_packets=0;model._exchange_max_energy_error=0.
 model._source_labeled_feature_hook=MethodType(hook,model)
 model._feature_learning_report['conditional_residual_exchange']=c['training_mechanism']
 return model
def install(model,c):d.validate(c);return attach(original.install(model,d.native_config()),c)
def build(c,device):d.validate(c);return attach(original.build(d.native_config(),device),c)
