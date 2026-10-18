"""Only mixed-slot membership changes; keep original F, loss, U and default head."""
from collections import Counter
from collections.abc import Mapping
from types import MethodType
from pathlib import Path
import torch
from . import design as d,relations
from .protocol import slot_plan,check_checkpoint,check_source_metadata
from .channel import Shifted,Practical
from experiments.cvs_noise_shape_factorial43 import model as original
from experiments.cvs_crossview_supcon41 import model as base
from experiments.cvs_fullsource_concat20 import supcon
classifier,source_domain_count=original.classifier,original.source_domain_count
terms=base.terms
capture_forward=base.capture_forward

def mixed_view(channel,raw,ordinary,meta,epoch,step):
 plan=slot_plan(epoch,step)
 if raw.shape!=(128,2,256) or ordinary.shape!=raw.shape:raise ValueError('Same actual128 IQ inputs required')
 if plan['on']:
  view,observed=channel.apply(raw,meta,epoch=epoch,step=step)
  if view.shape!=raw.shape or len(observed)!=128:raise ValueError('Every on-step128 labeled satellite must be supervised')
 else:view,observed=ordinary.clone(),[]
 return view,observed,plan

def hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count,labeled_inputs):
 if model is not self or not self.training or not torch.is_grad_enabled() or labeled_count!=128 or labels.dtype!=torch.long or labels.shape!=(128,) or not bool(((labels>=0)&(labels<6)).all()) or labeled_inputs.shape!=(128,2,256) or optimizer_step!=self._feature_l_steps or optimizer_step!=self._full_calls+1:raise ValueError('Only ordered trueL128 native default-head training')
 if not isinstance(extra,(tuple,list)) or len(extra)<2 or not isinstance(extra[1],Mapping):raise ValueError('Original physical L metadata required')
 rx=torch.as_tensor(extra[1]['rx_i'],device=labels.device,dtype=torch.long)
 if domains is None or domains.shape!=labels.shape or not bool(((domains>=0)&(domains<15)).all()) or not torch.equal(rx,torch.tensor(supcon.RX_BY_DOMAIN,device=domains.device)[domains]):raise ValueError('All5 original RX/domain mapping required')
 step=optimizer_step;epoch=(step-1)//222+1;plan=slot_plan(epoch,step);weight=supcon.schedule(epoch);pending=self._supcon_pending;ordinary=self._legacy_actual_L_input;self._legacy_actual_L_input=None
 if ordinary is None or ordinary.shape!=(128,2,256):raise ValueError('Same actual ordinary IQ must be reused')
 first=labeled_output['tx_logits']
 if not torch.allclose(torch.nn.functional.cross_entropy(first,labels,label_smoothing=.01).detach(),identity_loss.detach(),atol=1e-6,rtol=1e-5):raise ValueError('Original CE scale differs')
 if weight and (pending is None or tuple(pending['first'])!=supcon.LEVELS):raise ValueError('Actual first ordinary maps absent')
 first_maps=pending['first'] if weight else {};supcon.clear(self)
 second,second_maps=capture_forward(self,ordinary,domains,weight>0);common,ce,kl,_,_=terms(first,second,labels,1.)
 raw=labeled_inputs.detach();view,observed,plan=mixed_view(self._full_channel,raw,ordinary,extra[1],epoch,step)
 days=torch.as_tensor(extra[1]['day_i']).tolist()
 for y,r,day,o in zip(labels.tolist(),rx.tolist(),days,observed):
  self._full_coverage[(int(y),int(r),o['scene'],o['state_end'],o['receiver_lock_status'])]+=1
  self._stepalt_coverage[(int(y),int(r),int(day),o['scene'],o['state_end'],o['receiver_lock_status'])]+=1
 # ORD128 uses only the two existing ordinary passes; mixed passes need no maps.
 z0,_=capture_forward(self,view,domains,False);z1,_=capture_forward(self,view,domains,False);mixed,mixed_ce,mixed_kl,_,_=terms(z0,z1,labels,1.)
 details=[]
 if weight:
  values=[]
  for phase,maps in [('first',first_maps),('second',second_maps)]:
   value,records=relations.penalty(maps,labels,rx);values.append(value);details.extend(dict(phase=phase,**r) for r in records)
  aux=torch.stack(values).mean();self._supcon_applied+=1
 else:aux=common*0
 raw_view,noise_meta=original.noise.apply(raw,plan['cumulative_step'],'NONE')
 raw0=self.forward_identity_only(raw_view,domain_labels=domains)['tx_logits'];raw1=self.forward_identity_only(raw_view,domain_labels=domains)['tx_logits'];raw_rd,raw_ce,raw_kl,_,_=terms(raw0,raw1,labels,1.)
 total=.5*common+.5*mixed+.25*raw_rd+weight*aux
 self._full_calls+=1;self._full_L_packets+=512;self._shape_NN+=256;self._supcon_calls=step;self._legacy_calls+=1;self._legacy_extra_packets+=128;self._legacy_positive_KL_calls+=int(float(kl.detach())>0)
 self._shape_records.append(dict(step=step,cumulative_step=plan['cumulative_step'],shape_RD=float(raw_rd.detach()),shape_CE=float(raw_ce.detach()),shape_KL=float(raw_kl.detach()),weight=.25,NN=256,same_IQ_each_pair=True,**noise_meta))
 keys=('eq_i','rx_i','day_i','sig_i','base_index')
 metadata={k:extra[1][k].detach().cpu().tolist() if torch.is_tensor(extra[1][k]) else list(extra[1][k]) for k in keys}
 if any(len(v)!=128 for v in metadata.values()):raise ValueError('All physical L128 exposures required')
 self._stepalt_records.append(dict(step=step,**plan,physical_ids=list(map(list,zip(*(metadata[k] for k in keys)))),TX=labels.tolist(),RX=rx.tolist(),day=days,scenes=[o['scene'] for o in observed],sample_ids=[o['sample_id'] for o in observed]))
 record=dict(step=step,epoch=epoch,total_L=float(total.detach()),base_L_before_shape=float((total-.25*raw_rd).detach()),common_RD=float(common.detach()),common_CE=float(ce.detach()),common_KL=float(kl.detach()),extra_RD=float(mixed.detach()),extra_CE=float(mixed_ce.detach()),extra_KL=float(mixed_kl.detach()),SupCon=float(aux.detach()),SupCon_weight=weight,SupCon_details=details,L_forwards=6,Ag_packets=0,As_packets=plan['As_packets'],Ag_mode='NONE',extra_control_packets=128,ordinary_raw_MSE=float((ordinary-labeled_inputs).square().mean()),second_raw_MSE=float((view-raw).square().mean()),mixed_ordinary_MSE=float((view-ordinary).square().mean()),same_IQ_each_pair=True,ordinary_redraws=0,scene_counts=dict(Counter(o['scene'] for o in observed)),state_counts=dict(Counter(o['state_end'] for o in observed)),all_satellite_L_supervised=True,crossview_mode='ORD128',new_losses=0,extra_U_NN=0,stepALT=plan)
 if not plan['on'] and record['mixed_ordinary_MSE']!=0.:raise ValueError('Off-step must reuse the exact same ordinary IQ')
 self._full_records.append(record)
 return total-identity_loss,{'concat/'+k:record[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')}
hook._accepts_labeled_inputs=True

def original_config(c):
 d.validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 cfg=b.config(next(r for r in b.rows() if r['row_id']==c['row_id']),30)
 if cfg!=d.reference_config(30):raise ValueError('Actual pinned original recipe differs')
 return cfg

def attach(m,c):
 d.validate(c);m._full_channel=Shifted(Practical('train'));m._source_labeled_feature_hook=MethodType(hook,m);m._stepalt_records=[];m._stepalt_coverage=Counter()
 m._feature_learning_report['stepALT']=dict(parent=c['parent_source'],new_epochs=10,cumulative_epochs=30,As_per_two_steps=128,As_per_epoch=14208,As_per_segment=142080,per_step_L_NN=768,total_L_loss_scale=1.25,ORD128='actual two ordinary maps;same original formula',ordinary_RDrop=True,mixed_RDrop=True,raw_RDrop=True,default_head_joint=True,no_new_parameters=True,no_U_change=True)
 return m

def install(m,c):return attach(original.install(m,original_config(c)),c)
def build(c,device='cpu'):return attach(original.build(original_config(c),device),c)
def own_model(c,device='cpu'):
 d.validate(c)
 from .source import parent_payload
 parent_payload(c);out=Path(c['output_root'])
 check_source_metadata(d.read(out/'completion.json'),d.read(out/'source_frozen.json'),d.read(out/'initialization.json'),d.read(out/'source_contract.json'),c,d.read(d.SOURCE)['role_ids'])
 cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
 check_checkpoint(cp,run=d.RUN,row=c['row_id'],epoch=10,updates=2220,baseline=c['parent_source']+'/final_ssdg.pth')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
