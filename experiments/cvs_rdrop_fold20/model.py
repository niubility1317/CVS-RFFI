from types import MethodType
import torch
from torch.nn import functional as F
from . import design as d
from experiments.cvs_evidence_responsibility_train20 import model as base
classifier=base.classifier;source_domain_count=base.source_domain_count

def terms(first,second,y,alpha):
 if alpha not in (0.,1.) or first.shape!=second.shape or first.shape!=(len(y),6) or y.dtype!=torch.long or not bool(((y>=0)&(y<6)).all()):
  raise ValueError('Visible six-class source labels and two default logits required')
 ce1=F.cross_entropy(first,y);ce2=F.cross_entropy(second,y)
 p,q=F.log_softmax(first,dim=1),F.log_softmax(second,dim=1)
 kl=.5*((p.exp()-q.exp())*(p-q)).sum(1).mean()
 paired=.5*(ce1+ce2)
 return paired+alpha*kl,paired,kl,ce1,ce2

def hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count,labeled_inputs):
 if self is not model or not self.training or not torch.is_grad_enabled() or optimizer_step!=self._feature_l_steps or optimizer_step!=self._rdrop_calls+1:
  raise ValueError('Only actual ordered L-student training steps may call paired objective')
 if labeled_count!=128 or labels.shape!=(128,) or labeled_inputs.shape!=(128,2,256):raise ValueError('Original clean L128 budget differs')
 first=labeled_output['tx_logits'];second=self.forward_identity_only(labeled_inputs.clone(),domain_labels=domains)['tx_logits']
 total,paired,kl,ce1,ce2=terms(first,second,labels,self._rdrop_alpha)
 if not torch.allclose(identity_loss.detach(),ce1.detach(),atol=1e-6,rtol=1e-5):raise ValueError('Native CE normalization changed')
 self._rdrop_calls+=1;self._rdrop_extra_packets+=len(labels)
 metrics={'rdrop/paired_CE':paired.detach(),'rdrop/symmetric_KL':kl.detach(),'rdrop/CE1':ce1.detach(),'rdrop/CE2':ce2.detach(),
    'rdrop/alpha':self._rdrop_alpha,'rdrop/calls':self._rdrop_calls,'rdrop/extra_L_packets':self._rdrop_extra_packets}
 # The native objective already includes CE1 with coefficient one.
 return .5*(ce2-identity_loss)+self._rdrop_alpha*kl,metrics
hook._accepts_labeled_inputs=True

def install(model,c):
 d.validate(c);model=base.install(model,d.base_config())
 model._rdrop_alpha=c['training_mechanism']['alpha'];model._rdrop_calls=0;model._rdrop_extra_packets=0
 model._source_labeled_feature_hook=MethodType(hook,model)
 model._feature_learning_report['paired_dropout_objective']=c['training_mechanism']
 return model
def build(c,device):
 d.validate(c)
 # base.build has already installed the original SIG_TIME/CENTER25K graph.
 m=base.build(d.base_config(),device)
 m._rdrop_alpha=c['training_mechanism']['alpha'];m._rdrop_calls=0;m._rdrop_extra_packets=0
 m._source_labeled_feature_hook=MethodType(hook,m);m._feature_learning_report['paired_dropout_objective']=c['training_mechanism']
 return m
