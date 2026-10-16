from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from types import MethodType
import random,torch
from . import design as d
from experiments.cvs_additive_as20 import model as parent
from experiments.cvs_crossrx_satmix20 import supcon
from experiments.cvs_iq_timing20.objective import rd
classifier,source_domain_count,export=parent.classifier,parent.source_domain_count,parent.export

def retained_hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count,labeled_inputs):
    if model is not self or not self.training or not torch.is_grad_enabled() or labeled_count!=128 or optimizer_step!=self._feature_l_steps or optimizer_step!=self._joint_calls+1 or labeled_inputs.shape!=(128,2,256):raise ValueError('One ordered real L128 step required')
    if not isinstance(extra,(tuple,list)) or len(extra)<2 or not isinstance(extra[1],Mapping):raise ValueError('Native (domain,metadata) required')
    rx=torch.as_tensor(extra[1]['rx_i'],device=labels.device,dtype=torch.long)
    if domains is None or domains.shape!=labels.shape or not bool(((domains>=0)&(domains<15)).all()) or not torch.equal(rx,torch.tensor(supcon.RX_BY_DOMAIN,device=domains.device)[domains]) or any(int(v) not in d.FIT for v in rx.detach().cpu().tolist()):raise ValueError('Actual fit-source RX/domain mapping differs')
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
    if self._joint_config['training_mechanism']['As']:
        satellite,observations=self._joint_channel.apply(raw,extra[1],namespace='train/epoch%d/step%d'%(epoch,optimizer_step))
        self._joint_As_packets+=len(raw)
        for y,o in zip(labels.detach().cpu().tolist(),observations):self._joint_coverage[(int(y),o['RX'],o['day'],o['scene'],o['state'],o['lock'])]+=1
    else:satellite=raw;observations=[]
    sat_logits=[self.forward_identity_only(satellite,domain_labels=domains)['tx_logits'] for _ in range(2)]
    sat_loss,sat_ce,sat_kl=rd(*sat_logits,labels)
    total=.25*common+.5*raw_loss+.25*sat_loss+weight*aux
    self._joint_calls+=1;self._joint_L_packets+=6*len(raw);self._legacy_calls+=1;self._legacy_extra_packets+=len(raw);self._supcon_calls=optimizer_step
    record=dict(step=optimizer_step,epoch=epoch,total_L=float(total.detach()),raw_CE=float(raw_ce.detach()),ordinary_CE=float(common_ce.detach()),s_slot_CE=float(sat_ce.detach()),raw_KL=float(raw_kl.detach()),ordinary_KL=float(common_kl.detach()),s_slot_KL=float(sat_kl.detach()),supcon=float(aux.detach()),supcon_weight=weight,supcon_details=details,ordinary_raw_MSE=float((ordinary-raw).square().mean()),sat_raw_MSE=float((satellite-raw).square().mean()),As_packets=len(observations),scene_counts=dict(Counter(o['scene'] for o in observations)),state_counts=dict(Counter(o['state'] for o in observations)),lock_counts=dict(Counter(o['lock'] for o in observations)),all_L_supervised=True,same_IQ_pairs=True)
    self._joint_details.append(record)
    return total-identity_loss,{'satmix/'+k:record[k] for k in ('total_L','raw_CE','ordinary_CE','s_slot_CE','supcon','supcon_weight','As_packets','ordinary_raw_MSE','sat_raw_MSE')}


def hook(self,**kwargs):
 labels=kwargs['labels'];raw=kwargs['labeled_inputs'].detach();domains=kwargs['domains'];extra=kwargs['extra']
 if labels.shape!=(128,) or bool(((labels<0)|(labels>5)).any()):raise ValueError('VisibleL128 labels only;never U truth')
 if self._joint_config['training_mechanism']['As']:raise ValueError('Common third pair must remain raw')
 correction,metrics=retained_hook(self,**kwargs)
 base=correction+kwargs['identity_loss'];record=self._joint_details[-1];epoch=record['epoch'];step=record['step'];c=self._additive_config;weight=c['additive']['extra_weight'];observations=[]
 raw_before=raw.clone();cpu_rng=torch.get_rng_state();py_rng=random.getstate()
 if c['arm']=='D_As':
  view,observations=self._joint_channel.apply(raw,extra[1],namespace='train/epoch%d/step%d'%(epoch,step))
  if len(observations)!=len(raw):raise ValueError('All legalL As observations required')
  self._joint_As_packets+=len(raw)
  for y,o in zip(labels.detach().cpu().tolist(),observations):self._joint_coverage[(int(y),o['RX'],o['day'],o['scene'],o['state'],o['lock'])]+=1
 else:view=raw
 if not torch.equal(raw,raw_before) or not torch.equal(cpu_rng,torch.get_rng_state()) or py_rng!=random.getstate():raise ValueError('As must preserve raw and global CPU/Python random streams')
 bn_before=sum(int(self.get_submodule(n).num_batches_tracked) for n in self._additive_bn_names)
 logits=[self.forward_identity_only(view,domain_labels=domains)['tx_logits'] for _ in range(2)]
 loss,ce,kl=rd(*logits,labels);total=base+weight*loss
 bn_delta=sum(int(self.get_submodule(n).num_batches_tracked) for n in self._additive_bn_names)-bn_before
 self._joint_L_packets+=2*len(raw);self._additive_calls+=1;self._additive_L_packets+=2*len(raw)
 record.update(base_total_L=float(base.detach()),extra_pair_CE=float(ce.detach()),extra_pair_KL=float(kl.detach()),extra_pair_loss=float(loss.detach()),extra_pair_weight=weight,extra_pair_weighted=float((weight*loss).detach()),total_L=float(total.detach()),extra_raw_MSE=float((view-raw).square().mean()),extra_input='As(raw)' if observations else 'raw',extra_As_packets=len(observations),extra_BN_batch_updates=bn_delta,extra_view_once=True,extra_same_IQ_pair=True,base_raw_total_weight=.75,base_ordinary_weight=.25,total_pair_weight=1.+weight)
 metrics.update({'satmix/'+k:record[k] for k in ('base_total_L','extra_pair_CE','extra_pair_KL','extra_pair_loss','extra_pair_weight','extra_pair_weighted','total_L','extra_As_packets','extra_raw_MSE','extra_BN_batch_updates')})
 return total-kwargs['identity_loss'],metrics
hook._accepts_labeled_inputs=True


def attach(m,c):
 d.validate(c);m._source_labeled_feature_hook=MethodType(hook,m);m._additive_config=c;m._additive_calls=0;m._additive_L_packets=0
 m._joint_config=deepcopy(m._joint_config);m._joint_config['fold']=deepcopy(c['fold'])
 m._feature_learning_report.update(additive_As_controls=c['additive'],source_fold=c['fold'],deployed_structure_unchanged=True)
 return m
def build(c,device='cpu'):return attach(parent.build(d.template(),device),c)
def install(m,c):return attach(parent.install(m,d.template()),c)
