"""Existing full CVS/defaulthead jointly trained;new IQ only in existing fourth pair."""
from types import MethodType
from collections import Counter
import random,torch
from . import design as d
from .iq import apply,choices
from experiments.cvs_additive_as20 import model as additive
from experiments.cvs_crossrx_satmix20 import model as original
from experiments.cvs_iq_timing20.objective import rd
classifier,source_domain_count,export=original.classifier,original.source_domain_count,original.export

def hook(self,**kw):
 raw=kw['labeled_inputs'].detach();ordinary=self._legacy_actual_L_input;labels=kw['labels'];domains=kw['domains'];meta=kw['extra'][1]
 if ordinary is None or ordinary.shape!=raw.shape or self._joint_config['training_mechanism']['As']:raise ValueError('Actualoriginalordinarycache andunchanged noAs base required')
 correction,metrics=original.hook(self,**kw);base=correction+kw['identity_loss'];record=self._joint_details[-1];step=record['step'];saved=raw.clone();ts=torch.get_rng_state();ps=random.getstate();k=choices(len(raw),d.MASK_SEED+step);masked,gain,info=apply(raw,k)
 if not torch.equal(raw,saved) or not torch.equal(ts,torch.get_rng_state()) or ps!=random.getstate():raise ValueError('Private operator changesraw/global RNG')
 arm=self._availability_config['arm'];view={'RAW':raw,'GAIN':gain,'ORDINARY':ordinary,'MASK':masked}[arm];logits=[self.forward_identity_only(view,domain_labels=domains)['tx_logits'] for _ in range(2)];loss,ce,kl=rd(*logits,labels);total=base+.25*loss
 self._joint_L_packets+=256;self._additive_calls+=1;self._additive_L_packets+=256;self._availability_calls+=1;self._availability_generated+=128
 for i,(yy,kk) in enumerate(zip(labels.detach().cpu().tolist(),k)):
  rx=int(meta['rx_i'][i]);day=int(meta['day_i'][i]);bid=int(meta['base_index'][i]);self._availability_coverage[int(yy),rx,day,kk]+=1;self._availability_packets[bid,kk]+=1
 record.update(base_total_L=float(base.detach()),extra_pair_CE=float(ce.detach()),extra_pair_KL=float(kl.detach()),extra_pair_loss=float(loss.detach()),extra_pair_weight=.25,total_L=float(total.detach()),extra_raw_MSE=float((view-raw).square().mean()),extra_input=arm,extra_As_packets=0,extra_view_once=True,extra_same_IQ_pair=True,base_raw_total_weight=.75,base_ordinary_weight=.25,total_pair_weight=1.25,extra_BN_batch_updates=0,availability_shadow_generated=128,availability_pairs=k,availability_cap=info['cap'],availability_relative_RMS=info['relative_IQ_RMS'],availability_matched_gain=info['matched_gain'],availability_power_match_error=info['power_match_relative_error'],mask_shadow_MSE=float((masked-raw).square().mean()),gain_shadow_MSE=float((gain-raw).square().mean()))
 metrics.update({'availability/'+n:record[n] for n in ('base_total_L','extra_pair_CE','extra_pair_KL','extra_pair_loss','extra_pair_weight','total_L','extra_raw_MSE','mask_shadow_MSE','gain_shadow_MSE')});return total-kw['identity_loss'],metrics
hook._accepts_labeled_inputs=True

def attach(m,c):
 d.validate(c)
 if m._additive_bn_names:raise ValueError('Fixedarchitecture has0BN')
 m._source_labeled_feature_hook=MethodType(hook,m);m._availability_config=c;m._availability_calls=0;m._availability_generated=0;m._availability_coverage=Counter();m._availability_packets=Counter();m._feature_learning_report.update(band_availability=c['band_availability'],new_parameters=0,deployment_structure_unchanged=True)
 return m
def build(c,device='cpu'):return attach(additive.build(d.template(),device),c)
def install(m,c):return attach(additive.install(m,d.template()),c)
