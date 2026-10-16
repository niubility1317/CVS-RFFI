"""Whole original F objective plus matched two extra RDrop pairs; deployment unchanged."""
from collections import Counter
from types import MethodType
import random
import torch
from . import design as d
from experiments.cvs_crossrx_satmix20 import model as original
from experiments.cvs_iq_timing20.objective import rd
classifier,source_domain_count,export=original.classifier,original.source_domain_count,original.export

def hook(self,**kw):
 raw=kw['labeled_inputs'].detach();labels=kw['labels'];domains=kw['domains'];meta=kw['extra'][1]
 if labels.shape!=(128,) or bool(((labels<0)|(labels>5)).any()) or self._joint_config['training_mechanism']['As']:raise ValueError('VisibleL128 and unchanged raw commonpair required')
 correction,metrics=original.hook(self,**kw);base=correction+kw['identity_loss'];record=self._joint_details[-1];mode=self._MC2_config['mc2']['mode'];epoch,step=record['epoch'],record['step'];before=raw.clone();cpu_rng=torch.get_rng_state();py_rng=random.getstate();generated=[];views=[]
 if mode=='raw':views=[raw,raw]
 else:
  first,obs=self._joint_channel.apply(raw,meta,namespace='train/epoch%d/step%d/mc0'%(epoch,step));generated.append((0,first,obs));views.append(first)
  if mode=='repeat':views.append(first)
  elif mode=='independent':
   second,obs=self._joint_channel.apply(raw,meta,namespace='train/epoch%d/step%d/mc1'%(epoch,step));generated.append((1,second,obs));views.append(second)
  else:raise ValueError('Unregistered MC2 view mode')
 if not torch.equal(raw,before) or not torch.equal(cpu_rng,torch.get_rng_state()) or py_rng!=random.getstate():raise ValueError('As must preserve raw and globalCPU/Python RNG')
 draw_records=[]
 for slot,x,observed in generated:
  if len(observed)!=128 or x.shape!=raw.shape or not bool(torch.isfinite(x).all()):raise ValueError('Full finite allL practical view required;no filtering')
  self._joint_As_packets+=128;self._MC2_generated[slot]+=128
  for y,o in zip(labels.detach().cpu().tolist(),observed):
   key=(int(y),o['RX'],o['day'],o['scene'],o['state'],o['lock']);self._joint_coverage[key]+=1;self._MC2_coverage[(slot,*key)]+=1
  draw_records.append(dict(slot=slot,namespace='train/epoch%d/step%d/mc%d'%(epoch,step,slot),packets=128,seeds=[int(o['seed']) for o in observed],scenes=[o['scene'] for o in observed],sessions=[o['session'] for o in observed],scene_counts=dict(Counter(o['scene'] for o in observed)),state_counts=dict(Counter(o['state'] for o in observed)),lock_counts=dict(Counter(o['lock'] for o in observed)),raw_MSE=float((x-raw).square().mean())))
 bn_before=sum(int(self.get_submodule(n).num_batches_tracked) for n in self._MC2_bn_names);losses=[];pairs=[]
 for slot,x in enumerate(views):
  logits=[self.forward_identity_only(x,domain_labels=domains)['tx_logits'] for _ in range(2)];loss,ce,kl=rd(*logits,labels);losses.append(loss);pairs.append(dict(pair=slot,input='raw' if mode=='raw' else 'As%d'%(slot if mode=='independent' else 0),CE=float(ce.detach()),KL=float(kl.detach()),loss=float(loss.detach()),weight=.125,generated_once=True,same_IQ_within_pair=True))
 extra=torch.stack(losses).mean();total=base+.25*extra;bn_delta=sum(int(self.get_submodule(n).num_batches_tracked) for n in self._MC2_bn_names)-bn_before
 self._joint_L_packets+=4*128;self._MC2_calls+=1;self._MC2_L_packets+=4*128;self._MC2_As_supervised_pair_observations+=0 if mode=='raw' else 256
 record.update(base_total_L=float(base.detach()),extra_pair_mean_loss=float(extra.detach()),extra_pair_mean_CE=sum(v['CE'] for v in pairs)/2,extra_pair_mean_KL=sum(v['KL'] for v in pairs)/2,extra_total_weight=.25,extra_weighted=float((.25*extra).detach()),total_L=float(total.detach()),extra_mode=mode,extra_pairs=pairs,extra_draws=draw_records,extra_generated_packets=128*len(generated),extra_As_supervised_pair_observations=0 if mode=='raw' else 256,extra_BN_batch_updates=bn_delta,extra_between_MSE=float((views[0]-views[1]).square().mean()),base_raw_total_weight=.75,base_ordinary_weight=.25,total_pair_weight=1.25,no_IQ_averaging=True,no_crossview_consistency=True,all_extra_L_supervised=True)
 metrics.update({'satmix/'+k:record[k] for k in ('base_total_L','extra_pair_mean_CE','extra_pair_mean_KL','extra_pair_mean_loss','extra_total_weight','extra_weighted','total_L','extra_generated_packets','extra_As_supervised_pair_observations','extra_BN_batch_updates','extra_between_MSE')})
 return total-kw['identity_loss'],metrics
hook._accepts_labeled_inputs=True

def attach(m,c):
 d.validate(c);m._source_labeled_feature_hook=MethodType(hook,m);m._MC2_config=c;m._MC2_calls=0;m._MC2_L_packets=0;m._MC2_generated=[0,0];m._MC2_coverage=Counter();m._MC2_As_supervised_pair_observations=0;m._MC2_bn_names=tuple(n for n,mod in m.named_modules() if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm) and mod.num_batches_tracked is not None)
 m._feature_learning_report.update(practical_MC2=c['mc2'],deployed_structure_unchanged=True,tracked_BN_modules=len(m._MC2_bn_names));return m

def build(c,device='cpu'):return attach(original.build(d.original_config(),device),c)
def install(m,c):return attach(original.install(m,d.original_config()),c)
