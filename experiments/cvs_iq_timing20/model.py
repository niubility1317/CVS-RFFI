"""The existing F graph and default head; timing is training-only input math."""
from types import MethodType
import torch
from . import design as d
from .augmentation import delayed,draw_delays
from .objective import objective
from experiments.cvs_rdrop_sinc_units import model as base

classifier,source_domain_count=base.classifier,base.source_domain_count


def counted(self,x):
    field='_timing_train_packets' if self.training else '_timing_eval_packets'
    setattr(self,field,getattr(self,field)+len(x))
    return self._timing_features(x)


def hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count,labeled_inputs):
    if model is not self or not self.training or not torch.is_grad_enabled() or labeled_count!=128 or optimizer_step!=self._feature_l_steps or optimizer_step!=self._timing_calls+1 or labeled_inputs.shape!=(128,2,256):
        raise ValueError('Exactly one ordered legal L batch required')
    x=labeled_inputs.detach();first=labeled_output['tx_logits']
    ce=torch.nn.functional.cross_entropy(first,labels,label_smoothing=.01)
    if not torch.allclose(ce.detach(),identity_loss.detach(),atol=1e-6,rtol=1e-5):raise ValueError('Native smoothed CE differs')
    self._timing_calls+=1
    g,detail=delayed(x,draw_delays(128,step=optimizer_step)) if self._timing_config['training_mechanism']['Ag'] else (x,dict(waveform_MSE=0.,max_relative_power_error=0.,max_abs_delay=0.))
    logits=[first,self.forward_identity_only(x,domain_labels=domains)['tx_logits']]
    for view in (g,x):
        # Each pair consumes the exact same generated IQ. No cross-view KL.
        logits.extend(self.forward_identity_only(view,domain_labels=domains)['tx_logits'] for _ in range(2))
    total,metrics=objective(logits,labels)
    self._timing_L_packets+=6*len(x)
    self._timing_details.append(dict(step=optimizer_step,**detail))
    metrics.update(calls=self._timing_calls,L_packets=self._timing_L_packets,total_L=total.detach(),same_IQ_pairs=1,**detail)
    return total-identity_loss,{'timing/'+k:v for k,v in metrics.items()}


hook._accepts_labeled_inputs=True


def attach(m,c):
    d.validate(c)
    # Keep the original L step context, replacing only the two-pass objective.
    m._phase1_labeled_feature_context=m._legacy_original_labeled_scope
    m._source_labeled_feature_hook=MethodType(hook,m)
    m._timing_config=c;m._timing_calls=0;m._timing_L_packets=0;m._timing_details=[]
    encoder=m.id_backbone.encoder
    encoder._timing_features=encoder.features;encoder._timing_train_packets=0;encoder._timing_eval_packets=0
    encoder.features=MethodType(counted,encoder)
    m._feature_learning_report.update(F_base=True,IQ_timing=c['training_mechanism'],no_deployment_change=True)
    return m


def install(m,c):return attach(base.install(m,d.base_model_config()),c)
def build(c,device='cpu'):return attach(base.build(d.base_model_config(),device),c)
