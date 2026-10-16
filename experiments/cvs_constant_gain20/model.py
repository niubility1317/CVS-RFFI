"""F/default head, common-pair SupCon and a separately generated constant-gain slot."""
from collections import Counter
from collections.abc import Mapping
from types import MethodType
import torch
from . import design as d
from experiments.cvs_crossrx_satmix20 import supcon
from .gain import apply as gain_view
from experiments.cvs_crossrx_satmix20.channel import PairedChannel
from experiments.cvs_iq_timing20.objective import rd
from experiments.cvs_rdrop_sinc_units import model as base
classifier,source_domain_count,export=base.classifier,base.source_domain_count,base.export

def counted(self,x):
    name='_view_train_packets' if self.training else '_view_eval_packets'
    setattr(self,name,getattr(self,name)+len(x));return self._view_features(x)

def hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count,labeled_inputs):
    if model is not self or not self.training or not torch.is_grad_enabled() or labeled_count!=128 or optimizer_step!=self._feature_l_steps or optimizer_step!=self._joint_calls+1 or labeled_inputs.shape!=(128,2,256):raise ValueError('One ordered real L128 step required')
    if not isinstance(extra,(tuple,list)) or len(extra)<2 or not isinstance(extra[1],Mapping):raise ValueError('Native (domain,metadata) required')
    rx=torch.as_tensor(extra[1]['rx_i'],device=labels.device,dtype=torch.long)
    if domains is None or domains.shape!=labels.shape or not bool(((domains>=0)&(domains<15)).all()) or not torch.equal(rx,torch.tensor(supcon.RX_BY_DOMAIN,device=domains.device)[domains]) or bool((rx==1).any()):raise ValueError('Actual fit-source RX/domain mapping differs')
    epoch=(optimizer_step-1)//222+1;weight=supcon.schedule(epoch);pending=self._supcon_pending
    if weight>0 and (pending is None or tuple(pending['first'])!=supcon.LEVELS):raise ValueError('Actual ordinary L feature maps required')
    ordinary=self._legacy_actual_L_input;self._legacy_actual_L_input=None
    if ordinary is None or ordinary.shape!=labeled_inputs.shape:raise ValueError('Same actual ordinary IQ must be reused')
    first=labeled_output['tx_logits'];ce=torch.nn.functional.cross_entropy(first,labels,label_smoothing=.01)
    if not torch.allclose(ce.detach(),identity_loss.detach(),atol=1e-6,rtol=1e-5):raise ValueError('Native smoothed CE differs')
    try:
        if weight>0:self._supcon_phase='second'
        second=self.forward_identity_only(ordinary,domain_labels=domains)['tx_logits'];self._supcon_phase=None
        common,common_ce,common_kl=rd(first,second,labels);details=[]
        if weight>0:
            if tuple(pending['second'])!=supcon.LEVELS:raise ValueError('Second ordinary feature maps missing')
            penalties=[]
            for phase in ('first','second'):
                value,records=supcon.penalty(pending[phase],labels,rx,'CROSS_RX');penalties.append(value);details.extend(dict(phase=phase,**v) for v in records)
            aux=torch.stack(penalties).mean();self._supcon_applied+=1
        else:aux=common*0.
    finally:supcon.clear(self)
    raw=labeled_inputs.detach()
    raw_logits=[self.forward_identity_only(raw,domain_labels=domains)['tx_logits'] for _ in range(2)]
    raw_loss,raw_ce,raw_kl=rd(*raw_logits,labels)
    if self._joint_config['training_mechanism']['Ag']:
        satellite,gains,power_error=gain_view(raw,optimizer_step);self._gain_packets+=len(raw)
    else:satellite=raw;gains=[];power_error=0.
    observations=[]
    sat_logits=[self.forward_identity_only(satellite,domain_labels=domains)['tx_logits'] for _ in range(2)]
    sat_loss,sat_ce,sat_kl=rd(*sat_logits,labels)
    total=.25*common+.5*raw_loss+.25*sat_loss+weight*aux
    self._joint_calls+=1;self._joint_L_packets+=6*len(raw);self._legacy_calls+=1;self._legacy_extra_packets+=len(raw);self._supcon_calls=optimizer_step
    record=dict(gain_values=gains,gain_packets=len(gains),gain_power_error=power_error,step=optimizer_step,epoch=epoch,total_L=float(total.detach()),raw_CE=float(raw_ce.detach()),ordinary_CE=float(common_ce.detach()),s_slot_CE=float(sat_ce.detach()),raw_KL=float(raw_kl.detach()),ordinary_KL=float(common_kl.detach()),s_slot_KL=float(sat_kl.detach()),supcon=float(aux.detach()),supcon_weight=weight,supcon_details=details,ordinary_raw_MSE=float((ordinary-raw).square().mean()),sat_raw_MSE=float((satellite-raw).square().mean()),As_packets=len(observations),scene_counts=dict(Counter(o['scene'] for o in observations)),state_counts=dict(Counter(o['state'] for o in observations)),lock_counts=dict(Counter(o['lock'] for o in observations)),all_L_supervised=True,same_IQ_pairs=True)
    self._joint_details.append(record)
    return total-identity_loss,{'satmix/'+k:record[k] for k in ('total_L','raw_CE','ordinary_CE','s_slot_CE','supcon','supcon_weight','As_packets','ordinary_raw_MSE','sat_raw_MSE')}

hook._accepts_labeled_inputs=True

def attach(m,c):
    d.validate(c);supcon.attach(m,dict(baseline_reuse=False,arm='CROSS_RX'))
    m._source_labeled_feature_hook=MethodType(hook,m);m._joint_config=c;m._joint_calls=0;m._joint_L_packets=0;m._joint_As_packets=0;m._gain_packets=0;m._joint_details=[];m._joint_coverage=Counter();m._joint_channel=PairedChannel()
    e=m.id_backbone.encoder;e._view_features=e.features;e._view_train_packets=0;e._view_eval_packets=0;e.features=MethodType(counted,e)
    m._feature_learning_report.update(shared_cross_RX=c['supcon'],matched_IQ_views=c['training_mechanism'],no_deployment_change=True)
    return m
def build(c,device='cpu'):return attach(base.build(d.base_model_config(),device),c)
def install(m,c):return attach(base.install(m,d.base_model_config()),c)
