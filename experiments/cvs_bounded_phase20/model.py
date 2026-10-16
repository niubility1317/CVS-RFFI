"""Whole originalF objective plus oneextra matchedpair;deployed structure unchanged."""
from collections import Counter
from types import MethodType
import random,time
import torch
from . import design as d,phase
from experiments.cvs_crossrx_satmix20 import model as original
from experiments.cvs_iq_timing20.objective import rd
classifier,source_domain_count,export=original.classifier,original.source_domain_count,original.export

def hook(self,**kw):
 labels=kw['labels'];raw=kw['labeled_inputs'].detach();domains=kw['domains'];meta=kw['extra'][1];ordinary=self._legacy_actual_L_input
 if labels.shape!=(128,) or bool(((labels<0)|(labels>5)).any()) or self._joint_config['training_mechanism']['As'] or ordinary is None or ordinary.shape!=raw.shape:raise ValueError('VisibleL128,actualordinary and unchangedraw commonpair required')
 correction,metrics=original.hook(self,**kw);base=correction+kw['identity_loss'];record=self._joint_details[-1];mode=self._phase_config['isolated_phase']['mode'];saved=raw.clone();savedordinary=ordinary.clone();cpu=torch.get_rng_state();py=random.getstate();cuda=torch.cuda.get_rng_state(raw.device) if raw.is_cuda else None
 generation_seconds=0.
 if mode=='raw':view=raw;info={}
 elif mode=='ordinary':view=ordinary;info={}
 elif mode=='phase':
  started=time.perf_counter();view,info=phase.apply(raw,record['step']);generation_seconds=time.perf_counter()-started;self._phase_generated+=128
 else:raise ValueError('Unregistered isolatedphase mode')
 if not torch.equal(raw,saved) or not torch.equal(ordinary,savedordinary) or not torch.equal(cpu,torch.get_rng_state()) or py!=random.getstate() or (cuda is not None and not torch.equal(cuda,torch.cuda.get_rng_state(raw.device))):raise ValueError('Operator must preserve raw,ordinary and globalRNG')
 if mode=='phase':
  for y,r,day in zip(labels.detach().cpu().tolist(),meta['rx_i'].tolist(),meta['day_i'].tolist()):self._phase_coverage[(int(y),int(r),int(day))]+=1
 bn_before=sum(int(self.get_submodule(n).num_batches_tracked) for n in self._phase_bn_names);z=[self.forward_identity_only(view,domain_labels=domains)['tx_logits'] for _ in range(2)];loss,ce,kl=rd(*z,labels);total=base+.25*loss;bn_delta=sum(int(self.get_submodule(n).num_batches_tracked) for n in self._phase_bn_names)-bn_before
 self._joint_L_packets+=2*128;self._phase_calls+=1;self._phase_L_packets+=2*128
 record.update(base_total_L=float(base.detach()),extra_pair_CE=float(ce.detach()),extra_pair_KL=float(kl.detach()),extra_pair_loss=float(loss.detach()),extra_pair_weight=.25,extra_pair_weighted=float((.25*loss).detach()),total_L=float(total.detach()),extra_raw_MSE=float((view-raw).square().mean()),extra_input=mode,extra_BN_batch_updates=bn_delta,extra_view_once=True,extra_same_IQ_pair=True,base_raw_total_weight=.75,base_ordinary_weight=.25,total_pair_weight=1.25,phase_generated_packets=128 if mode=='phase' else 0,phase=info,phase_generation_seconds=generation_seconds,all_extra_L_supervised=True)
 metrics.update({'satmix/'+k:record[k] for k in ('base_total_L','extra_pair_CE','extra_pair_KL','extra_pair_loss','extra_pair_weight','extra_pair_weighted','total_L','extra_raw_MSE','extra_BN_batch_updates','phase_generated_packets')});return total-kw['identity_loss'],metrics
hook._accepts_labeled_inputs=True

def attach(m,c):
 d.validate(c);m._source_labeled_feature_hook=MethodType(hook,m);m._phase_config=c;m._phase_calls=0;m._phase_L_packets=0;m._phase_generated=0;m._phase_coverage=Counter();m._phase_bn_names=tuple(n for n,mod in m.named_modules() if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm) and mod.num_batches_tracked is not None);m._feature_learning_report.update(bounded_nonaffine_phase=c['isolated_phase'],deployed_structure_unchanged=True,tracked_BN_modules=len(m._phase_bn_names));return m

def build(c,device='cpu'):return attach(original.build(d.original_config(),device),c)
def install(m,c):return attach(original.install(m,d.original_config()),c)
