"""No deployed changes;two additional IQ pairs with unchanged within-IQ RD."""
from types import MethodType
import random
import torch
from . import design as d
from experiments.cvs_additive_as20 import model as additive
from experiments.cvs_crossrx_satmix20 import model as original
from experiments.cvs_iq_timing20.objective import rd
classifier,source_domain_count,export=original.classifier,original.source_domain_count,original.export

def hook(self,**kw):
 labels=kw['labels'];raw=kw['labeled_inputs'].detach();domains=kw['domains'];meta=kw['extra'][1]
 if labels.shape!=(128,) or bool(((labels<0)|(labels>5)).any()) or self._joint_config['training_mechanism']['As']:raise ValueError('All visibleL128,trueTX,raw third common pair required')
 correction,metrics=original.hook(self,**kw);base=correction+kw['identity_loss'];record=self._joint_details[-1];step=record['step'];c=self._noise_config
 if self._noise_runtime is None:raise ValueError('Training-only noise runtime required')
 raw_before=raw.clone();cpu_rng=torch.get_rng_state();py_rng=random.getstate();observations=[]
 if c['arm']=='RAW10':views=(raw,raw)
 else:views,observations=self._noise_runtime.channel.apply(raw,meta,step=step,mode=c['arm'])
 if not torch.equal(raw,raw_before) or not torch.equal(cpu_rng,torch.get_rng_state()) or py_rng!=random.getstate():raise ValueError('View generation changed raw or global random streams')
 if observations:
  if len(observations)!=2*len(raw):raise ValueError('All labeled noise views required,no confidence filtering')
  # Records are packet-major then view;labels belong to original anchor.
  for i,o in enumerate(observations):
   tx=int(labels[i//2]);o['TX']=tx;self._joint_coverage[(tx,o['RX'],o['day'],o['scene'],o['state'],o['lock'])]+=1
   self._noise_runtime.trace.write(__import__('json').dumps(o,allow_nan=False)+'\n')
  self._joint_As_packets+=len(observations)
 pairs=[];CE=[];KL=[]
 for view in views:
  a=self.forward_identity_only(view,domain_labels=domains)['tx_logits'];b=self.forward_identity_only(view,domain_labels=domains)['tx_logits'];loss,ce,kl=rd(a,b,labels);pairs.append(loss);CE.append(ce);KL.append(kl)
 extra=torch.stack(pairs).mean();ce=torch.stack(CE).mean();kl=torch.stack(KL).mean();total=base+.25*extra
 self._noise_calls+=1;self._noise_physical_presentations+=len(raw);self._additive_calls+=1;self._additive_L_packets+=4*len(raw);self._joint_L_packets+=4*len(raw)
 record.update(base_total_L=float(base.detach()),extra_pair_CE=float(ce.detach()),extra_pair_KL=float(kl.detach()),extra_pair_loss=float(extra.detach()),extra_pair0_CE=float(CE[0].detach()),extra_pair1_CE=float(CE[1].detach()),extra_pair0_KL=float(KL[0].detach()),extra_pair1_KL=float(KL[1].detach()),extra_pair0_loss=float(pairs[0].detach()),extra_pair1_loss=float(pairs[1].detach()),extra_pair_weight=.25,extra_pair_weighted=float((.25*extra).detach()),total_L=float(total.detach()),extra_input=c['arm'],extra_As_packets=len(observations),extra_raw_MSE=float(torch.stack([(v-raw).square().mean() for v in views]).mean()),noise_view_pair_MSE=float((views[0]-views[1]).square().mean()),extra_BN_batch_updates=0,base_raw_total_weight=.75,base_ordinary_weight=.25,total_pair_weight=1.25,L_forwards_per_sample=10,extra_pair_count=2,extra_same_IQ_pair=True,cross_noise_KL=False)
 metrics.update({'satmix/'+k:record[k] for k in ('base_total_L','extra_pair_CE','extra_pair_KL','extra_pair_loss','extra_pair_weighted','total_L','extra_As_packets','extra_raw_MSE','noise_view_pair_MSE')})
 return total-kw['identity_loss'],metrics
hook._accepts_labeled_inputs=True

def attach(m,c):
 d.validate(c);m._source_labeled_feature_hook=MethodType(hook,m);m._noise_config=c;m._noise_runtime=None;m._noise_calls=0;m._noise_physical_presentations=0;m._additive_calls=0;m._additive_L_packets=0
 if any(isinstance(v,torch.nn.modules.batchnorm._BatchNorm) for v in m.modules()):raise ValueError('Fixed current F hasno BatchNorm;do not change deployed structure')
 m._feature_learning_report.update(noise_pair_training=c['noise_pair'],deployed_structure_unchanged=True,classification_head_unchanged=True)
 return m

def build(c,device='cpu'):return attach(additive.build(d.template(),device),c)
def install(m,c):return attach(additive.install(m,d.template()),c)
