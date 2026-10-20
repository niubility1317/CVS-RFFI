"""Keep F two-forward RDrop/SupCon, add source-only joint feature gap loss."""
from contextlib import contextmanager
from collections.abc import Mapping
from types import MethodType
import torch
from torch.nn import functional as F
from experiments.cvs_residual_identity.model import ResidualPhysicalHead

def feature_with_constant_head(head,parts,response_gain):
 """Same deployed values; only this auxiliary branch treats the whole head as constant."""
 if type(head) is not ResidualPhysicalHead or set(parts)!={'base_input','pa_input','response','sequence'}:raise ValueError('Exact native F fusion parts required')
 def norm(z,module):return F.layer_norm(z,module.normalized_shape,module.weight.detach(),module.bias.detach(),module.eps)
 core=norm(parts['base_input'],head.base_norm)+head.gain.detach().tanh()*norm(parts['pa_input'],head.pa_norm)
 return core+response_gain.tanh()*parts['response']+parts['sequence']
from experiments.cvs_f_supcon50.mechanism import schedule,RX_BY_DOMAIN
from .loss import penalty

@contextmanager
def labeled_scope(self):
 with self._gap_original_scope():
  if self._supcon_pending is not None:
   ref=self.id_backbone.encoder;head=ref.core.id_backbone.cls_head
   def capture_part(name):
    def capture(module,args,out):
     phase=self._supcon_phase;pending=self._supcon_pending
     if phase not in ('first','second') or pending is None:raise ValueError('F fusion capture escaped actual L/RDrop')
     parts=pending.setdefault(phase+'_gap_parts',{})
     if name in parts:raise ValueError('One actual fusion part per original forward')
     parts[name]=args[0] if name.endswith('_input') else out
    return capture
   for name,module in (('base_input',head.base_norm),('pa_input',head.pa_norm),('response',ref.response_projection),('sequence',ref.sequence_pool)):
    self._supcon_handles.append(module.register_forward_hook(capture_part(name)))
   def capture(module,args,out):
    phase=self._supcon_phase;pending=self._supcon_pending
    if phase not in ('first','second') or pending is None or not isinstance(out,Mapping) or out.get('feat_joint') is None:raise ValueError('Only actual original L/RDrop joint feature')
    key=phase+'_joint';z=out['feat_joint'];constant=feature_with_constant_head(head,pending.pop(phase+'_gap_parts'),ref.response_gain)
    if z.shape!=(128,160) or key in pending or not torch.equal(z.detach(),constant.detach()):raise ValueError('Auxiliary constant-head feature must exactly equal actual deployed joint160')
    pending[key]=constant
   self._supcon_handles.append(self.id_backbone.register_forward_hook(capture))
  yield

def training_hook(self,**kw):
 step=kw['optimizer_step'];epoch=(step-1)//222+1;weight=schedule(epoch);pending=self._supcon_pending;scheduled=weight>0
 if not self.training or not torch.is_grad_enabled() or step!=self._gap_calls+1:raise ValueError('Ordered actual L feature calls only')
 correction,metrics=self._gap_original_hook(**kw);raw=0.;detail=dict(eligible=False,valid_cells=0,active_cells=0,used_packets=0,cells=[])
 if scheduled:
  y=kw['labels'];domain=kw['domains'];extra=kw['extra']
  if pending is None or kw['labeled_count']!=128 or not isinstance(extra,(tuple,list)) or len(extra)<2 or not isinstance(extra[1],Mapping):raise ValueError('Actual original L128 feature/metadata required')
  meta=extra[1]
  if set(meta.get('split_source',[]))!={'ssdg_labeled_tx_visible'} or not bool(torch.as_tensor(meta.get('tx_label_visible',False)).all()) or domain is None or domain.dtype!=torch.long or domain.shape!=y.shape or not bool(((domain>=0)&(domain<15)).all()):raise ValueError('Visible original source L only, no U/V/target')
  rx=torch.as_tensor(meta.get('rx_i'),device=y.device,dtype=torch.long);day=torch.as_tensor(meta.get('day_i'),device=y.device,dtype=torch.long)
  if not torch.equal(rx,torch.tensor(RX_BY_DOMAIN,device=y.device)[domain]) or not torch.equal(day,domain%3+1):raise ValueError('Actual physical RX/day differs from source domain map')
  aux,detail=penalty(pending['first_joint'],pending['second_joint'],y,domain,meta.get('physical_id'),self._gap_mode)
  if detail['eligible']:
   correction=correction+weight*aux;raw=float(aux.detach());self._gap_eligible+=1;self._gap_active+=int(detail['active_cells']>0);self._gap_used_packets+=detail['used_packets']
   for cell in detail['cells']:self._gap_cell_calls[cell['domain']]+=1
 self._gap_calls=step;self._gap_scheduled+=int(scheduled);record=dict(step=step,epoch=epoch,scheduled=scheduled,weight=weight if scheduled else 0.,loss=raw,**detail);self._gap_records.append(record)
 metrics.update({'F_gap/calls':step,'F_gap/scheduled_calls':self._gap_scheduled,'F_gap/eligible_calls':self._gap_eligible,'F_gap/active_calls':self._gap_active,'F_gap/loss':raw,'F_gap/weight':weight if scheduled else 0.,'F_gap/valid_cells':detail['valid_cells'],'F_gap/used_current_L_packets':detail['used_packets'],'F_gap/extra_model_forwards':0,'F_gap/head_direct_aux_gradient':0})
 return correction,metrics
training_hook._accepts_labeled_inputs=False

def attach(m,c):
 if m._supcon_calls or hasattr(m,'_gap_mode'):raise ValueError('Attach disjoint gap once before scratch training')
 m._gap_mode=c['treatment'];m._gap_calls=0;m._gap_scheduled=0;m._gap_eligible=0;m._gap_active=0;m._gap_used_packets=0;m._gap_cell_calls=[0]*15;m._gap_records=[];m._gap_original_scope=m._phase1_labeled_feature_context;m._gap_original_hook=m._source_labeled_feature_hook;m._phase1_labeled_feature_context=MethodType(labeled_scope,m);m._source_labeled_feature_hook=MethodType(training_hook,m)
 m._feature_learning_report['weak_TX_disjoint_gap']=dict(mode=c['treatment'],feature='actualjoint160',weak_indices=[1,3],groups='same sourceRX and day',minimum_per_class=2,partition='disjoint alternating physical source packets; twoRDrop views averaged per packet first',margin=.4,max_weight=.1,warmup=5,ramp=10,scheduled=9990,default_head_main_joint=True,head_direct_aux_gradient=False,aux_feature='exact same joint160 values reconstructed from original fusion inputs; all cls_head parameters detached only in auxiliary tensor algebra',extra_parameters=0,extra_model_forwards=0,extra_IQ_reads=0,extra_optimizer_steps=0,limitation='Cross product is signed; disjoint finite-pool minibatch halves are not independent IID draws, so no unbiasedness or target benefit claimed')
 return m
