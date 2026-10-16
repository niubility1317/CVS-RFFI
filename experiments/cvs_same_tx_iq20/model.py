"""Unchanged deployed F, whole original objective and one matched extra IQ pair."""
from types import MethodType
import random
import torch
from . import design as d,bank
from experiments.cvs_crossrx_satmix20 import model as original
from experiments.cvs_iq_timing20.objective import rd
classifier,source_domain_count,export=original.classifier,original.source_domain_count,original.export

def hook(self,**kw):
 labels=kw['labels'];raw=kw['labeled_inputs'].detach();domains=kw['domains'];metadata=kw['extra'][1]
 if labels.shape!=(128,) or bool(((labels<0)|(labels>5)).any()) or self._joint_config['training_mechanism']['As']:raise ValueError('Only visible L128 and unchanged raw common pair allowed')
 correction,metrics=original.hook(self,**kw);base=correction+kw['identity_loss'];record=self._joint_details[-1];c=self._IQmix_config;mode=c['iqmix']['mode'];raw_before=raw.clone();rng=torch.get_rng_state();py_rng=random.getstate()
 view,details=bank.current().apply(raw,labels,metadata,mode,record['step'])
 if not torch.equal(raw,raw_before) or not torch.equal(rng,torch.get_rng_state()) or py_rng!=random.getstate():raise ValueError('IQmix must preserve raw and globalCPU/Python RNG')
 bn_before=sum(int(self.get_submodule(n).num_batches_tracked) for n in self._IQmix_bn_names)
 logits=[self.forward_identity_only(view,domain_labels=domains)['tx_logits'] for _ in range(2)];loss,ce,kl=rd(*logits,labels);total=base+.25*loss
 bn_delta=sum(int(self.get_submodule(n).num_batches_tracked) for n in self._IQmix_bn_names)-bn_before
 self._joint_L_packets+=2*len(raw);self._IQmix_calls+=1;self._IQmix_L_packets+=2*len(raw)
 record.update(base_total_L=float(base.detach()),extra_pair_CE=float(ce.detach()),extra_pair_KL=float(kl.detach()),extra_pair_loss=float(loss.detach()),extra_pair_weight=.25,extra_pair_weighted=float((.25*loss).detach()),total_L=float(total.detach()),extra_raw_MSE=float((view-raw).square().mean()),extra_input=mode,extra_BN_batch_updates=bn_delta,extra_view_once=True,extra_same_IQ_pair=True,base_raw_total_weight=.75,base_ordinary_weight=.25,total_pair_weight=1.25,IQmix=details)
 metrics.update({'satmix/'+k:record[k] for k in ('base_total_L','extra_pair_CE','extra_pair_KL','extra_pair_loss','extra_pair_weight','extra_pair_weighted','total_L','extra_raw_MSE','extra_BN_batch_updates')})
 return total-kw['identity_loss'],metrics
hook._accepts_labeled_inputs=True

def attach(m,c):
 d.validate(c);m._source_labeled_feature_hook=MethodType(hook,m);m._IQmix_config=c;m._IQmix_calls=0;m._IQmix_L_packets=0;m._IQmix_bn_names=tuple(n for n,mod in m.named_modules() if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm) and mod.num_batches_tracked is not None)
 m._feature_learning_report.update(sameTX_IQmix=c['iqmix'],deployed_structure_unchanged=True,tracked_BN_modules=len(m._IQmix_bn_names),bank_in_model_state=False);return m

def build(c,device='cpu'):return attach(original.build(d.original_config(),device),c)
def install(m,c):return attach(original.install(m,d.original_config()),c)
