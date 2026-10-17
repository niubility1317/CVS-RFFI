"""Same F/default head/eight L forwards;route existing SupCon without adding a term."""
from types import MethodType
from collections import Counter
from collections.abc import Mapping
import random
import numpy as np
import torch
from . import design as d
from experiments.cvs_scene_cycle20 import model as legacy
from experiments.cvs_crossrx_satmix20 import supcon
from experiments.cvs_iq_timing20.objective import rd
classifier,source_domain_count,export=legacy.classifier,legacy.source_domain_count,legacy.export

def positions(policy,step):
 if policy not in ('ordinary','satellite','mixed') or not isinstance(step,int) or not 1<=step<=4440:raise ValueError('Fixed route policy/real step required')
 if policy=='ordinary':return []
 if policy=='satellite':return list(range(128))
 return sorted(np.random.default_rng(d.ROUTING_SEED+step).permutation(128)[:64].tolist())

def route_maps(ordinary,satellite,selected):
 if tuple(ordinary)!=supcon.LEVELS or tuple(satellite)!=supcon.LEVELS or len(set(selected))!=len(selected) or any(not isinstance(v,int) or not 0<=v<128 for v in selected):raise ValueError('Complete matching layer maps and legal appearance indices required')
 out={}
 for level in supcon.LEVELS:
  a,b=ordinary[level],satellite[level]
  if a.shape!=b.shape or a.ndim!=3 or len(a)!=128:raise ValueError('Actual128 matching featuremaps required')
  mask=torch.zeros(128,dtype=torch.bool,device=a.device);mask[selected]=True;out[level]=torch.where(mask[:,None,None],b,a)
 return out

def hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count,labeled_inputs):
 if model is not self or not self.training or not torch.is_grad_enabled() or labeled_count!=128 or optimizer_step!=self._feature_l_steps or optimizer_step!=self._joint_calls+1 or labeled_inputs.shape!=(128,2,256):raise ValueError('One native ordered L128 step required')
 if not isinstance(extra,(tuple,list)) or len(extra)<2 or not isinstance(extra[1],Mapping):raise ValueError('Native domain/metadata required')
 rx=torch.as_tensor(extra[1]['rx_i'],device=labels.device,dtype=torch.long)
 if labels.dtype!=torch.long or labels.shape!=(128,) or bool(((labels<0)|(labels>5)).any()) or domains is None or domains.shape!=labels.shape or not bool(((domains>=0)&(domains<15)).all()) or not torch.equal(rx,torch.tensor(supcon.RX_BY_DOMAIN,device=domains.device)[domains]) or bool((rx==1).any()):raise ValueError('VisiblefitL labels/RX only')
 epoch=(optimizer_step-1)//222+1;weight=supcon.schedule(epoch);pending=self._supcon_pending;policy=self._route_config['supcon_view_routing']['policy'];selected=positions(policy,optimizer_step) if weight>0 else []
 ordinary=self._legacy_actual_L_input;self._legacy_actual_L_input=None
 if ordinary is None or ordinary.shape!=labeled_inputs.shape or (weight>0 and (pending is None or tuple(pending['first'])!=supcon.LEVELS)):raise ValueError('Actual ordinary IQ/maps required')
 first=labeled_output['tx_logits'];ce=torch.nn.functional.cross_entropy(first,labels,label_smoothing=.01)
 if not torch.allclose(ce.detach(),identity_loss.detach(),atol=1e-6,rtol=1e-5):raise ValueError('Native CE differs')
 try:
  if weight>0:self._supcon_phase='second'
  second=self.forward_identity_only(ordinary,domain_labels=domains)['tx_logits'];self._supcon_phase=None
  common,common_ce,common_kl=rd(first,second,labels)
  if weight>0 and tuple(pending['second'])!=supcon.LEVELS:raise ValueError('SameIQ RDrop ordinary maps required')
 finally:supcon.clear(self)
 raw=labeled_inputs.detach();raw_loss,raw_ce,raw_kl=rd(*[self.forward_identity_only(raw,domain_labels=domains)['tx_logits'] for _ in range(2)],labels)
 third_loss,third_ce,third_kl=rd(*[self.forward_identity_only(raw,domain_labels=domains)['tx_logits'] for _ in range(2)],labels)
 before=raw.clone();cpu=torch.get_rng_state();py=random.getstate();view,observations=self._joint_channel.apply(raw,extra[1],namespace='train/epoch%d/step%d'%(epoch,optimizer_step))
 if len(observations)!=128 or not torch.equal(raw,before) or not torch.equal(cpu,torch.get_rng_state()) or py!=random.getstate():raise ValueError('AllL fullAs/unchanged raw and global RNG required')
 self._joint_As_packets+=128
 for y,o in zip(labels.detach().cpu().tolist(),observations):self._joint_coverage[(int(y),o['RX'],o['day'],o['scene'],o['state'],o['lock'])]+=1
 maps={'first':{},'second':{}};handles=[];phase=[None];logits=[];bn0=sum(int(self.get_submodule(n).num_batches_tracked) for n in self._additive_bn_names)
 try:
  if weight>0 and policy!='ordinary':
   for level in supcon.LEVELS:
    def capture(module,args,out,level=level):
     ph=phase[0]
     if ph not in maps or level in maps[ph] or not self.training or not torch.is_grad_enabled():raise ValueError('As map escaped paired L forward')
     maps[ph][level]=out
    handles.append(self.get_submodule(supcon.PREFIX+level).register_forward_hook(capture))
   self._route_handles=handles
  for ph in ('first','second'):
   phase[0]=ph;logits.append(self.forward_identity_only(view,domain_labels=domains)['tx_logits'])
 finally:
  for handle in handles:handle.remove()
  self._route_handles=[];phase[0]=None
 extra_loss,extra_ce,extra_kl=rd(*logits,labels);details=[]
 if weight>0:
  penalties=[]
  for ph in ('first','second'):
   chosen=pending[ph] if policy=='ordinary' else maps[ph] if policy=='satellite' else route_maps(pending[ph],maps[ph],selected)
   value,dd=supcon.penalty(chosen,labels,rx,'CROSS_RX');penalties.append(value);details.extend(dict(phase=ph,**v) for v in dd)
  aux=torch.stack(penalties).mean();self._supcon_applied+=1;self._route_applied+=1;self._route_sat_appearances+=len(selected);self._route_ord_appearances+=128-len(selected)
 else:aux=common*0.
 base=.25*common+.5*raw_loss+.25*third_loss+weight*aux;total=base+.25*extra_loss
 self._joint_calls+=1;self._joint_L_packets+=1024;self._legacy_calls+=1;self._legacy_extra_packets+=128;self._supcon_calls=optimizer_step;self._additive_calls+=1;self._additive_L_packets+=256;self._route_calls+=1
 bn_delta=sum(int(self.get_submodule(n).num_batches_tracked) for n in self._additive_bn_names)-bn0
 record=dict(step=optimizer_step,epoch=epoch,total_L=float(total.detach()),base_total_L=float(base.detach()),raw_CE=float(raw_ce.detach()),ordinary_CE=float(common_ce.detach()),s_slot_CE=float(third_ce.detach()),raw_KL=float(raw_kl.detach()),ordinary_KL=float(common_kl.detach()),s_slot_KL=float(third_kl.detach()),supcon=float(aux.detach()),supcon_weight=weight,supcon_details=details,ordinary_raw_MSE=float((ordinary-raw).square().mean()),sat_raw_MSE=0.,As_packets=0,scene_counts={},state_counts={},lock_counts={},all_L_supervised=True,same_IQ_pairs=True,extra_pair_CE=float(extra_ce.detach()),extra_pair_KL=float(extra_kl.detach()),extra_pair_loss=float(extra_loss.detach()),extra_pair_weight=.25,extra_pair_weighted=float((.25*extra_loss).detach()),extra_input='As(raw)',extra_raw_MSE=float((view-raw).square().mean()),extra_As_packets=128,extra_BN_batch_updates=bn_delta,extra_view_once=True,extra_same_IQ_pair=True,base_raw_total_weight=.75,base_ordinary_weight=.25,total_pair_weight=1.25,extra_scene_counts=dict(Counter(o['scene'] for o in observations)),scene_schedule_mode='iid',channel_generation_wall_seconds=self._joint_channel.training_wall_seconds,route_policy=policy,route_active=weight>0,route_selected_As_positions=selected,route_ordinary_appearances=128-len(selected) if weight>0 else 0,route_As_appearances=len(selected),route_new_NN_forwards=0,route_new_loss_terms=0)
 self._joint_details.append(record)
 return total-identity_loss,{'satmix/'+k:record[k] for k in ('total_L','base_total_L','raw_CE','ordinary_CE','s_slot_CE','supcon','supcon_weight','As_packets','ordinary_raw_MSE','sat_raw_MSE','extra_pair_CE','extra_pair_KL','extra_pair_loss','extra_pair_weight','extra_pair_weighted','extra_As_packets','extra_raw_MSE','extra_BN_batch_updates')}
hook._accepts_labeled_inputs=True

def attach(m,c):
 d.validate(c);m._source_labeled_feature_hook=MethodType(hook,m);m._route_config=c;m._additive_config=c;m._route_handles=[];m._route_calls=0;m._route_applied=0;m._route_sat_appearances=0;m._route_ord_appearances=0;m._feature_learning_report.update(shared_cross_RX=c['supcon'],existing_supcon_view_routing=c['supcon_view_routing'],no_new_deployment_modules=True)
 return m
def build(c,device='cpu'):return attach(legacy.build(d.legacy_config(),device),c)
def install(m,c):return attach(legacy.install(m,d.legacy_config()),c)
