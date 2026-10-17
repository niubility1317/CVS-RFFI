"""Original fixed F/defaulthead;only fourth IQ pair propagation dependence differs."""
from types import MethodType
from collections import Counter
import random,torch
from . import design as d
from .channel import GroupedPractical
from experiments.cvs_f_additive_rx6 import model as parent
from experiments.cvs_iq_timing20.objective import rd
classifier,source_domain_count,export=parent.classifier,parent.source_domain_count,parent.export

def hook(self,**kw):
 raw=kw['labeled_inputs'].detach();labels=kw['labels'];domains=kw['domains'];meta=kw['extra'][1]
 if raw.shape!=(128,2,256) or labels.shape!=(128,) or bool(((labels<0)|(labels>5)).any()) or self._joint_config['training_mechanism']['As']:raise ValueError('Original commonbase andallvisibleL128 required')
 correction,metrics=parent.retained_hook(self,**kw);base=correction+kw['identity_loss'];record=self._joint_details[-1];step=record['step'];epoch=record['epoch'];saved=raw.clone();ts=torch.get_rng_state();ps=random.getstate();shadow,observed=self._joint_channel.apply(raw,meta,namespace='train/epoch%d/step%d'%(epoch,step));arm=self._group_config['arm'];view=raw if arm=='RAW' else shadow
 if len(observed)!=128 or not torch.equal(raw,saved) or not torch.equal(ts,torch.get_rng_state()) or ps!=random.getstate():raise ValueError('All legalL generatedonce/privateRNG/rawimmutable required')
 self._group_steps+=1;self._group_generated+=128
 for yy,o in zip(labels.detach().cpu().tolist(),observed):
  key=(int(yy),o['RX'],o['day'],o['scene'],o['state'],o['lock']);self._group_coverage[key]+=1
  if arm!='RAW':self._joint_coverage[key]+=1
 if arm!='RAW':self._joint_As_packets+=128
 logits=[self.forward_identity_only(view,domain_labels=domains)['tx_logits'] for _ in range(2)];loss,ce,kl=rd(*logits,labels);total=base+.25*loss;self._joint_L_packets+=256;self._additive_calls+=1;self._additive_L_packets+=256
 record.update(base_total_L=float(base.detach()),extra_pair_CE=float(ce.detach()),extra_pair_KL=float(kl.detach()),extra_pair_loss=float(loss.detach()),extra_pair_weight=.25,extra_pair_weighted=float((.25*loss).detach()),total_L=float(total.detach()),extra_raw_MSE=float((view-raw).square().mean()),shadow_raw_MSE=float((shadow-raw).square().mean()),extra_input=arm,extra_As_packets=0 if arm=='RAW' else 128,extra_BN_batch_updates=0,extra_view_once=True,extra_same_IQ_pair=True,base_raw_total_weight=.75,base_ordinary_weight=.25,total_pair_weight=1.25,group_shadow_generated=128,group_policy=self._joint_channel.policy,group_session_count=len({o['session'] for o in observed}),group_scene_counts=dict(Counter(o['scene'] for o in observed)),group_state_counts=dict(Counter(o['state'] for o in observed)),group_lock_counts=dict(Counter(o['lock'] for o in observed)))
 for name in ('session','scene','state','lock','environment_seed','packet_seed','noise_stream_seed','kernel_energy','packet_residual_hz','packet_coarse_error_hz','packet_phase','packet_tracking_z'):record['group_'+name]=[o[name] for o in observed]
 metrics.update({'propgroup/'+k:record[k] for k in ('base_total_L','extra_pair_CE','extra_pair_KL','total_L','extra_raw_MSE','shadow_raw_MSE')});return total-kw['identity_loss'],metrics
hook._accepts_labeled_inputs=True

def attach(m,c):
 d.validate(c)
 if m._additive_bn_names:raise ValueError('Fixed0BN architecture')
 m._joint_channel=GroupedPractical(c['channel_group']['policy']);m._source_labeled_feature_hook=MethodType(hook,m);m._group_config=c;m._group_steps=0;m._group_generated=0;m._group_coverage=Counter();m._feature_learning_report.update(propagation_group=c['channel_group'],new_parameters=0,deployment_structure_unchanged=True)
 return m
def build(c,device='cpu'):return attach(parent.build(d.template(),device),c)
def install(m,c):return attach(parent.install(m,d.template()),c)
