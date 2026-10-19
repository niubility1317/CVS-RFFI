"""Original L/RDrop/SupCon unchanged, plus current-L-only meta feature objective."""
from collections.abc import Mapping
from types import MethodType
import torch
from experiments.cvs_f_supcon50.mechanism import schedule,RX_BY_DOMAIN
from .objective import objective,active,anchor_for

def training_hook(self,**kw):
 step=kw['optimizer_step'];scheduled=active(step);x=self._legacy_actual_L_input
 if scheduled and (x is None or x.shape!=(128,2,256)):raise ValueError('Reuse only actual post-augmentation current L IQ')
 correction,metrics=self._metarx_original_hook(**kw);detail=dict(eligible=False,extra_L_packets=0);raw=0.;weight=0.
 if scheduled:
  y=kw['labels'];domain=kw['domains'];extra=kw['extra']
  if not isinstance(extra,(tuple,list)) or len(extra)<2 or not isinstance(extra[1],Mapping) or 'rx_i' not in extra[1] or domain is None:raise ValueError('Current source L metadata required')
  rx=torch.as_tensor(extra[1]['rx_i'],device=y.device,dtype=torch.long)
  if not torch.equal(rx,torch.tensor(RX_BY_DOMAIN,device=domain.device)[domain]):raise ValueError('Source physical RX/domain disagreement')
  aux,detail=objective(self,x,y,rx,anchor_for(step),self._metarx_mode)
  if detail['eligible']:
   weight=schedule((step-1)//222+1);raw=float(aux.detach());correction=correction+weight*aux;self._metarx_applied+=1;self._metarx_packets+=detail['extra_L_packets']
 self._metarx_records.append(dict(step=step,scheduled=scheduled,weight=weight,loss=raw,**detail))
 metrics.update({'F_metarx/scheduled':int(scheduled),'F_metarx/applied':self._metarx_applied,'F_metarx/extra_L_packets':self._metarx_packets,'F_metarx/loss':raw,'F_metarx/weight':weight,'F_metarx/inner_grad_norm':detail.get('inner_gradient_norm',0.),'F_metarx/virtual_held_gain':detail.get('virtual_held_gain',0.),'F_metarx/inner_alpha':detail.get('inner_alpha',0.),'F_metarx/head_direct_aux_gradient':0})
 return correction,metrics
training_hook._accepts_labeled_inputs=False

def attach(m,c):
 if m._supcon_calls or hasattr(m,'_metarx_mode'):raise ValueError('Install once before actual L training')
 m._metarx_mode=c['treatment'];m._metarx_applied=0;m._metarx_packets=0;m._metarx_records=[];m._metarx_original_hook=m._source_labeled_feature_hook;m._source_labeled_feature_hook=MethodType(training_hook,m)
 m._feature_learning_report['weak_TX_meta_RX']=dict(mode=c['treatment'],weak_indices=[1,3],inner='class-balanced source L at one cyclic RX',outer='class-balanced weak L at otherRX',inner_alpha=.01 if c['treatment']=='META_RX' else 0.,every_steps=16,first_step=1120,calls=624,max_weight=.1,warmup=5,ramp=10,actual_default_W_aux_detached=True,default_head_main_joint=True,virtual_Sinc_update=False,actual_Sinc_outer_gradient=True,virtual_parameters='all identity dimensionless feature parameters; no W or two Sinc q leaves',extra_parameters=0,extra_optimizer_updates=0,source_L_only=True)
 return m
