"""Add current-L paired weak feature risk, keeping original RDrop/SupCon intact."""
from collections.abc import Mapping
from types import MethodType
import torch
from experiments.cvs_f_supcon50.mechanism import schedule,RX_BY_DOMAIN
from .loss import penalty

def training_hook(self,**kw):
 step=kw['optimizer_step'];epoch=(step-1)//222+1;weight=schedule(epoch);scheduled=weight>0;pending=self._supcon_pending
 if not self.training or not torch.is_grad_enabled() or step!=self._tail_calls+1:raise ValueError('Only ordered actual source L feature calls allowed')
 correction,metrics=self._tail_original_hook(**kw);raw=0.;detail=dict(eligible=False,valid_cells=0,available_anchor_objectives=0,selected_anchor_objectives=0,levels=[])
 if scheduled:
  y=kw['labels'];domain=kw['domains'];extra=kw['extra']
  if pending is None or domain is None or domain.shape!=y.shape or not isinstance(extra,(tuple,list)) or len(extra)<2 or not isinstance(extra[1],Mapping) or 'rx_i' not in extra[1]:raise ValueError('Only actual paired L maps and physical source RX metadata required')
  rx=torch.as_tensor(extra[1]['rx_i'],device=y.device,dtype=torch.long)
  if not torch.equal(rx,torch.tensor(RX_BY_DOMAIN,device=domain.device)[domain]):raise ValueError('Actual L physical RX/domain differs')
  aux,detail=penalty(pending['first'],pending['second'],y,rx,self._tail_mode)
  if detail['eligible']:
   raw=float(aux.detach());correction=correction+weight*aux;self._tail_applied+=1;self._tail_cell_total+=detail['valid_cells'];self._tail_available+=detail['available_anchor_objectives'];self._tail_selected+=detail['selected_anchor_objectives']
 self._tail_calls=step;self._tail_records.append(dict(step=step,epoch=epoch,scheduled=scheduled,weight=weight if scheduled else 0.,loss=raw,**detail))
 metrics.update({'F_tail/calls':step,'F_tail/applied':self._tail_applied,'F_tail/loss':raw,'F_tail/weight':weight if scheduled else 0.,'F_tail/valid_cells':detail['valid_cells'],'F_tail/available_anchor_objectives':detail['available_anchor_objectives'],'F_tail/selected_anchor_objectives':detail['selected_anchor_objectives'],'F_tail/extra_model_forwards':0,'F_tail/head_direct_aux_gradient':0})
 return correction,metrics
training_hook._accepts_labeled_inputs=False

def attach(m,c):
 if m._supcon_calls or hasattr(m,'_tail_mode'):raise ValueError('Install once before original source L training')
 m._tail_mode=c['treatment'];m._tail_calls=0;m._tail_applied=0;m._tail_cell_total=0;m._tail_available=0;m._tail_selected=0;m._tail_records=[];m._tail_original_hook=m._source_labeled_feature_hook;m._source_labeled_feature_hook=MethodType(training_hook,m)
 m._feature_learning_report['weak_TX_RX_feature_tail']=dict(mode=c['treatment'],weak_indices=[1,3],levels=['time_down','t2.pool'],positive='sameTX differentRX',negative='differentTX sameRX',temperature=.2,reduction='equal weakTX, equal observed RX; mean or ceil upperhalf per cell',pair_average_before_selection=True,tail_fraction=.5,max_weight=.1,warmup=5,ramp=10,source_L_only=True,default_head_main_joint=True,default_W_direct_aux_gradient=False,extra_parameters=0,extra_model_forwards=0,extra_IQ_reads=0,extra_optimizer_updates=0)
 return m
