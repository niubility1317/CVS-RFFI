from types import MethodType
import torch
from . import design as d
from experiments.cvs_evidence_responsibility_train20 import model as base
classifier=base.classifier;source_domain_count=base.source_domain_count
def hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count,**unused):
 if self is not model or not self.training or not torch.is_grad_enabled() or optimizer_step!=self._feature_l_steps or optimizer_step!=self._teacher_calls+1:raise ValueError('Only ordered source L training context permitted')
 if labeled_count!=128 or labels.shape!=(128,) or not torch.isfinite(identity_loss):raise ValueError('Clean visible L128 CE required')
 from experiments.cvs_phase1_overlay.model import native_modules
 meta=native_modules()._meta_from_extra(extra)
 if meta is None:raise ValueError('Actual source L physical metadata required')
 values={k:meta[k].detach().cpu().tolist() for k in ('base_index','rx_i','day_i')};ys=labels.detach().cpu().tolist()
 if any(len(v)!=128 for v in values.values()):raise ValueError('One physical metadata row per labeled packet required')
 for b,rx,day,y in zip(values['base_index'],values['rx_i'],values['day_i'],ys):
  if self._teacher_allowed_L.get(int(b))!=(int(y),int(rx),int(day)):raise ValueError('Actual L packet/label/RX outside fixed teacher exposure')
 self._teacher_calls+=1;self._teacher_seen.update(map(int,values['base_index']))
 return identity_loss*0,{'teacher/verified_L_steps':self._teacher_calls,'teacher/unique_L_packets':len(self._teacher_seen)}
def attach(model,c):
 model._source_labeled_feature_hook=MethodType(hook,model);model._teacher_calls=0;model._teacher_seen=set();model._teacher_allowed_L={}
 model._feature_learning_report['teacher_exposure']=c['teacher_exposure'];return model
def install(model,c):d.validate(c);return attach(base.install(model,d.base_config()),c)
def build(c,device):d.validate(c);return attach(base.build(d.base_config(),device),c)
