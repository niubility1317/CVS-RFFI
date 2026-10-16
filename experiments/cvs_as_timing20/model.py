"""Fixed deployed F/defaulthead;extra labeled pair only follows fixed As calendar."""
from types import MethodType
from collections import Counter
import json,random
import torch
from . import design as d
from experiments.cvs_additive_as20 import model as additive
from experiments.cvs_crossrx_satmix20 import model as original
from experiments.cvs_iq_timing20.objective import rd
classifier,source_domain_count,export=original.classifier,original.source_domain_count,original.export
def hook(self,**kwargs):
 labels=kwargs['labels'];raw=kwargs['labeled_inputs'].detach();domains=kwargs['domains'];extra=kwargs['extra']
 if labels.shape!=(128,) or bool(((labels<0)|(labels>5)).any()) or self._joint_config['training_mechanism']['As']:raise ValueError('VisibleL only;whole base remains unmodified')
 correction,metrics=original.hook(self,**kwargs);base=correction+kwargs['identity_loss'];record=self._joint_details[-1];epoch=record['epoch'];step=record['step'];before=raw.clone();cpu=torch.get_rng_state();py=random.getstate()
 # Generate the identical original physical observation in every arm/epoch,
 # including shadows. The only choice is which IQ enters the existing pair.
 satellite,observed=self._joint_channel.apply(raw,extra[1],namespace='train/epoch%d/step%d'%(epoch,step))
 if len(observed)!=128 or not torch.equal(before,raw) or not torch.equal(cpu,torch.get_rng_state()) or py!=random.getstate():raise ValueError('FullL/private channel/unchangedraw required')
 use=d.active(self._timing_config['arm'],epoch);view=satellite if use else raw
 self._timing_calls+=1;self._timing_generated+=128
 for y,o in zip(labels.detach().cpu().tolist(),observed):
  key=(int(y),o['RX'],o['day'],o['scene'],o['state'],o['lock']);self._timing_generated_coverage[key]+=1
  if use:self._joint_coverage[key]+=1
 if use:self._joint_As_packets+=128
 logits=[self.forward_identity_only(view,domain_labels=domains)['tx_logits'] for _ in range(2)];loss,ce,kl=rd(*logits,labels);total=base+.25*loss
 self._joint_L_packets+=256;self._additive_calls+=1;self._additive_L_packets+=256
 record.update(base_total_L=float(base.detach()),extra_pair_CE=float(ce.detach()),extra_pair_KL=float(kl.detach()),extra_pair_loss=float(loss.detach()),extra_pair_weight=.25,extra_pair_weighted=float((.25*loss).detach()),total_L=float(total.detach()),extra_raw_MSE=float((view-raw).square().mean()),extra_input='As(raw)' if use else 'raw',extra_As_packets=128 if use else 0,extra_BN_batch_updates=0,extra_view_once=True,extra_same_IQ_pair=True,base_raw_total_weight=.75,base_ordinary_weight=.25,total_pair_weight=1.25,As_shadow_generated=128,As_active=use,shadow_raw_MSE=float((satellite-raw).square().mean()),shadow_scene_counts=dict(Counter(o['scene'] for o in observed)),shadow_state_counts=dict(Counter(o['state'] for o in observed)),shadow_lock_counts=dict(Counter(o['lock'] for o in observed)))
 metrics.update({'satmix/'+k:record[k] for k in ('base_total_L','extra_pair_CE','extra_pair_KL','extra_pair_loss','extra_pair_weight','extra_pair_weighted','total_L','extra_As_packets','extra_raw_MSE','extra_BN_batch_updates')});return total-kwargs['identity_loss'],metrics
hook._accepts_labeled_inputs=True
def attach(m,c):
 d.validate(c)
 if m._additive_bn_names:raise ValueError('Fixed architecture has no BatchNorm')
 m._source_labeled_feature_hook=MethodType(hook,m);m._timing_config=c;m._timing_calls=0;m._timing_generated=0;m._timing_generated_coverage=Counter();m._feature_learning_report.update(As_timing=c['As_timing'],deployed_structure_unchanged=True,new_parameters=0)
 return m
def build(c,device='cpu'):return attach(additive.build(d.template(),device),c)
def install(m,c):return attach(additive.install(m,d.template()),c)
