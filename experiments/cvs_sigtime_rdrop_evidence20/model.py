from types import MethodType
import torch
from . import design as d,mechanism as f
from experiments.cvs_sigtime_rdrop_legacy import model as original
classifier=original.classifier;source_domain_count=original.source_domain_count
def hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count):
 if self is not model or not self.training or not torch.is_grad_enabled() or optimizer_step!=self._feature_l_steps or optimizer_step!=self._evidence_calls+1 or optimizer_step!=self._legacy_calls+1 or labeled_count!=128 or domains is None or domains.shape!=(128,):raise ValueError('Only ordered actual L student context with domain metadata')
 mapping=self._evidence_domain_rx
 try:rx=torch.tensor([mapping[int(v)] for v in domains.detach().cpu().tolist()],dtype=torch.long,device=labels.device)
 except KeyError:raise ValueError('Native L domain not present in effective source dataset RX mapping')
 if not isinstance(extra,(tuple,list)) or len(extra)!=2 or not isinstance(extra[1],dict) or not torch.is_tensor(extra[1].get('rx_i')):raise ValueError('Actual native L receiver metadata required')
 actual_rx=extra[1]['rx_i'].to(device=labels.device,dtype=torch.long).view(-1)
 if not torch.equal(rx,actual_rx):raise ValueError('Compact domain RX decoding differs from actual batch metadata')
 x=self._legacy_actual_L_input;self._legacy_actual_L_input=None
 if x is None or x.shape!=(128,2,256):raise ValueError('Actual captured augmented L128 required')
 first=labeled_output['tx_logits'];second=self.forward_identity_only(x,domain_labels=domains)['tx_logits']
 _,paired,native,ce1,ce2=original.terms(first,second,labels,1.)
 if not torch.allclose(identity_loss.detach(),ce1.detach(),atol=1e-6,rtol=1e-5):raise ValueError('Original smoothed CE changed')
 loss,parts=f.consistency(first,second,labels,rx,self._evidence_arm,self._evidence_audit)
 if self._evidence_arm=='FULL' and not torch.equal(loss.detach(),native.detach()):raise ValueError('FULL must equal exact original KL')
 self._evidence_calls+=1;self._legacy_calls+=1;self._legacy_extra_packets+=128;self._legacy_positive_KL_calls+=int(float(loss.detach())>0)
 metrics={'rdrop_evidence/'+k:v for k,v in dict(parts,calls=self._evidence_calls,extra_L_packets=self._legacy_extra_packets,CE1=ce1.detach(),CE2=ce2.detach(),paired_CE=paired.detach(),consistency_loss=loss.detach(),actual_augmented_input_reused=1).items()}
 return .5*(ce2-identity_loss)+loss,metrics
hook._accepts_labeled_inputs=False
def attach(m,c):
 m._evidence_arm=c['arm'];m._evidence_calls=0;m._evidence_audit=f.new_audit();m._evidence_domain_rx={};m._source_labeled_feature_hook=MethodType(hook,m)
 m._feature_learning_report['rdrop_RX_evidence']=c['training_mechanism'];return m
def install(m,c):d.validate(c);return attach(original.install(m,d.native_config()),c)
def build(c,device):d.validate(c);return attach(original.build(d.native_config(),device),c)
