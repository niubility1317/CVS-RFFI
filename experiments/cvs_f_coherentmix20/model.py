"""Unchanged deployed F; fourth IQ pair is anchored coherent source interpolation."""
from types import MethodType
import random
import torch
from . import design as d
from experiments.cvs_crossrx_satmix20 import model as original
from experiments.cvs_additive_as20 import model as additive
from experiments.cvs_iq_timing20.objective import rd
classifier,source_domain_count,export=original.classifier,original.source_domain_count,original.export

def hook(self,**kwargs):
 labels=kwargs['labels'];raw=kwargs['labeled_inputs'].detach();domains=kwargs['domains'];extra=kwargs['extra']
 if labels.shape!=(128,) or bool(((labels<0)|(labels>5)).any()):raise ValueError('VisibleL128 labels only;never U truth')
 if self._joint_config['training_mechanism']['As']:raise ValueError('Common third pair must remain raw')
 correction,metrics=original.hook(self,**kwargs)
 base=correction+kwargs['identity_loss'];record=self._joint_details[-1];epoch=record['epoch'];step=record['step'];c=self._additive_config;weight=c['additive']['extra_weight'];observations=[]
 raw_before=raw.clone();cpu_rng=torch.get_rng_state();py_rng=random.getstate()
 if self._mix_runtime is None:raise ValueError('Actual filtered-L bank must attach after scratch initialization')
 from .iqmix import coherent_mix
 partner,partner_domains,pairs=self._mix_runtime.bank.select(labels,extra[1],step,c['arm'],raw)
 view,observations=coherent_mix(raw,partner.to(raw.device));self._mix_calls+=1;self._mix_packets+=len(raw)
 for p,o in zip(pairs,observations):
  self._mix_max=max(self._mix_max,o['relative_L2']);self._mix_nontrivial+=int(o['relative_L2']>=.005);self._mix_runtime.trace.write(__import__('json').dumps(dict(step=step,epoch=epoch,**p,**o),allow_nan=False)+'\n')
 if not torch.equal(raw,raw_before) or not torch.equal(cpu_rng,torch.get_rng_state()) or py_rng!=random.getstate():raise ValueError('IQ interpolation must preserve raw and global CPU/Python random streams')
 bn_before=sum(int(self.get_submodule(n).num_batches_tracked) for n in self._additive_bn_names)
 logits=[self.forward_identity_only(view,domain_labels=domains)['tx_logits'] for _ in range(2)]
 loss,ce,kl=rd(*logits,labels);total=base+weight*loss
 bn_delta=sum(int(self.get_submodule(n).num_batches_tracked) for n in self._additive_bn_names)-bn_before
 self._joint_L_packets+=2*len(raw);self._additive_calls+=1;self._additive_L_packets+=2*len(raw)
 record.update(base_total_L=float(base.detach()),extra_pair_CE=float(ce.detach()),extra_pair_KL=float(kl.detach()),extra_pair_loss=float(loss.detach()),extra_pair_weight=weight,extra_pair_weighted=float((weight*loss).detach()),total_L=float(total.detach()),extra_raw_MSE=float((view-raw).square().mean()),extra_input='coherent_'+c['arm'],extra_As_packets=0,coherent_packets=len(observations),extra_BN_batch_updates=bn_delta,extra_view_once=True,extra_same_IQ_pair=True,base_raw_total_weight=.75,base_ordinary_weight=.25,total_pair_weight=1.+weight)
 metrics.update({'satmix/'+k:record[k] for k in ('base_total_L','extra_pair_CE','extra_pair_KL','extra_pair_loss','extra_pair_weight','extra_pair_weighted','total_L','extra_As_packets','extra_raw_MSE','extra_BN_batch_updates')})
 return total-kwargs['identity_loss'],metrics
hook._accepts_labeled_inputs=True

def attach(m,c):
 d.validate(c);m._source_labeled_feature_hook=MethodType(hook,m);m._additive_config=c;m._additive_calls=0;m._additive_L_packets=0
 m._additive_bn_names=tuple(name for name,mod in m.named_modules() if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm) and mod.num_batches_tracked is not None)
 m._mix_runtime=None;m._mix_calls=0;m._mix_packets=0;m._mix_max=0.;m._mix_nontrivial=0
 m._feature_learning_report.update(coherent_source_IQ=c['coherent_mix'],additive_As_controls=c['additive'],deployed_structure_unchanged=True,tracked_BN_modules=len(m._additive_bn_names))
 return m
def build(c,device='cpu'):return attach(additive.build(d.template(),device),c)
def install(m,c):return attach(additive.install(m,d.template()),c)
