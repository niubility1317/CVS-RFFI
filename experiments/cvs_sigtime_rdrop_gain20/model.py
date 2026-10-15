from types import MethodType
import torch
from . import design as d,mechanism as f
from experiments.cvs_sigtime_rdrop_legacy import model as original
classifier,source_domain_count=original.classifier,original.source_domain_count

def hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count):
    if self is not model or not self.training or not torch.is_grad_enabled() or optimizer_step!=self._feature_l_steps or optimizer_step!=self._gain_calls+1 or optimizer_step!=self._legacy_calls+1 or labeled_count!=128:raise ValueError('Only ordered actual L student context')
    x=self._legacy_actual_L_input;self._legacy_actual_L_input=None
    if x is None or x.shape!=(128,2,256):raise ValueError('Actual captured augmented L128 required')
    first=labeled_output['tx_logits'];second_output=self.forward_identity_only(x,domain_labels=domains)
    second=second_output['tx_logits']
    _,paired,native,ce1,ce2=original.terms(first,second,labels,1.)
    if not torch.allclose(identity_loss.detach(),ce1.detach(),atol=1e-6,rtol=1e-5):raise ValueError('Original smoothed CE changed')
    head=classifier(self)
    if float(head.scale)!=30.:raise ValueError('Native head scale30 required')
    loss,parts=f.consistency(first,second,labeled_output['id_feat_joint'],second_output['id_feat_joint'],head.weight,self._gain_arm,self._gain_audit)
    aux=parts.pop('auxiliary_graph')
    self._gain_last_aux=aux if self._gain_capture_aux else None
    if self._gain_arm=='FULL' and not torch.equal(loss.detach(),native.detach()):raise ValueError('FULL must equal original KL')
    self._gain_calls+=1;self._legacy_calls+=1;self._legacy_extra_packets+=128
    self._legacy_positive_KL_calls+=int(float(native.detach())>0)
    metrics={'rdrop_gain/'+k:v for k,v in dict(parts,calls=self._gain_calls,extra_L_packets=self._legacy_extra_packets,CE1=ce1.detach(),CE2=ce2.detach(),paired_CE=paired.detach(),actual_augmented_input_reused=1).items()}
    return .5*(ce2-identity_loss)+loss,metrics

hook._accepts_labeled_inputs=False

def attach(m,c):
    m._gain_arm,m._gain_calls,m._gain_audit=c['arm'],0,f.new_audit()
    m._gain_capture_aux,m._gain_last_aux=False,None
    m._source_labeled_feature_hook=MethodType(hook,m)
    m._feature_learning_report['rdrop_gradient_gain']=c['training_mechanism']
    return m

def install(m,c):d.validate(c);return attach(original.install(m,d.native_config()),c)
def build(c,device):d.validate(c);return attach(original.build(d.native_config(),device),c)
