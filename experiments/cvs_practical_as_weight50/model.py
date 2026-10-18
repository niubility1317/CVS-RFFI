"""Unchanged original graph/6L order/IQ; only existing mixed-group loss weights differ."""
from collections import Counter
from collections.abc import Mapping
from types import MethodType
from pathlib import Path
import torch
from . import design as d
from .protocol import check_checkpoint,check_source_metadata
from .channel import Shifted,Practical
from .weighted import grouped
from experiments.cvs_noise_shape_factorial43 import model as original
from experiments.cvs_practical_as_weight30 import model as weighted30
from experiments.cvs_crossview_supcon41 import model as base
classifier,source_domain_count=original.classifier,original.source_domain_count
supcon,relations,terms,capture_forward=base.supcon,base.relations,base.terms,base.capture_forward

def base_hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count,labeled_inputs):
 if model is not self or not self.training or not torch.is_grad_enabled() or labeled_count!=128 or labels.dtype!=torch.long or labels.shape!=(128,) or not bool(((labels>=0)&(labels<6)).all()) or labeled_inputs.shape!=(128,2,256) or optimizer_step!=self._feature_l_steps or optimizer_step!=self._full_calls+1:raise ValueError('Only ordered true L128 default-head training')
 if not isinstance(extra,(tuple,list)) or len(extra)<2 or not isinstance(extra[1],Mapping):raise ValueError('Original physical L metadata required')
 rx=torch.as_tensor(extra[1]['rx_i'],device=labels.device,dtype=torch.long)
 if domains is None or domains.shape!=labels.shape or not bool(((domains>=0)&(domains<15)).all()) or not torch.equal(rx,torch.tensor(supcon.RX_BY_DOMAIN,device=domains.device)[domains]):raise ValueError('All5 original RX/domain mapping required')
 step=optimizer_step;epoch=(step-1)//222+1;weight=supcon.schedule(epoch);pending=self._supcon_pending;ordinary=self._legacy_actual_L_input;self._legacy_actual_L_input=None
 if ordinary is None or ordinary.shape!=(128,2,256):raise ValueError('Same actual ordinary IQ must be reused')
 first=labeled_output['tx_logits']
 if not torch.allclose(torch.nn.functional.cross_entropy(first,labels,label_smoothing=.01).detach(),identity_loss.detach(),atol=1e-6,rtol=1e-5):raise ValueError('Original CE scale differs')
 if weight and (pending is None or tuple(pending['first'])!=supcon.LEVELS):raise ValueError('Actual first ordinary maps absent')
 first_maps=pending['first'] if weight else {};supcon.clear(self)
 second,second_maps=capture_forward(self,ordinary,domains,weight>0);common,ce,kl,_,_=terms(first,second,labels,1.)
 raw=labeled_inputs.detach();view=ordinary.clone();smask=(torch.arange(128,device=labels.device)+step).remainder(2)!=0
 meta={k:v[smask.to(v.device)] if torch.is_tensor(v) and v.ndim>0 and len(v)==128 else v for k,v in extra[1].items()};satellite,observed=self._full_channel.apply(raw[smask],meta,epoch=epoch,step=step)
 if len(observed)!=64:raise ValueError('Every labeled satellite packet must receive unchanged CE')
 view[smask]=satellite
 for y,r,o in zip(labels[smask].tolist(),rx[smask].tolist(),observed):self._full_coverage[(int(y),int(r),o['scene'],o['state_end'],o['receiver_lock_status'])]+=1
 z0,sat0=capture_forward(self,view,domains,weight>0);z1,sat1=capture_forward(self,view,domains,weight>0);extra_loss,extra_ce,extra_kl,_,_=terms(z0,z1,labels,1.)
 weighted,ord_group,selected_group=grouped(z0,z1,labels,smask,terms)
 details=[]
 if weight:
  vals=[]
  for phase,a,b in [('first',first_maps,sat0),('second',second_maps,sat1)]:
   value,records=relations.penalty(a,b,labels,rx,smask,self._crossview_mode);vals.append(value);details.extend(dict(phase=phase,**r) for r in records)
  aux=torch.stack(vals).mean();self._supcon_applied+=1
 else:aux=common*0
 if getattr(self,'_crossview_smoke_capture',False):self._crossview_debug=dict(aux=aux,ordinary0=first_maps,ordinary1=second_maps,satellite0=sat0,satellite1=sat1,mask=smask)
 total=.5*common+weighted[0]+weight*aux
 self._full_calls+=1;self._full_L_packets+=512;self._supcon_calls=step;self._legacy_calls+=1;self._legacy_extra_packets+=128;self._legacy_positive_KL_calls+=int(float(kl.detach())>0)
 record=dict(step=step,epoch=epoch,total_L=float(total.detach()),common_RD=float(common.detach()),common_CE=float(ce.detach()),common_KL=float(kl.detach()),extra_RD=float(extra_loss.detach()),extra_CE=float(extra_ce.detach()),extra_KL=float(extra_kl.detach()),SupCon=float(aux.detach()),SupCon_weight=weight,SupCon_details=details,L_forwards=4,Ag_packets=0,As_packets=64 if self._shape_As else 0,selected_packets=64,mixed_weighted_RD=float(weighted[0].detach()),mixed_weighted_CE=float(weighted[1].detach()),mixed_weighted_KL=float(weighted[2].detach()),ordinary64_RD=float(ord_group[0].detach()),ordinary64_CE=float(ord_group[1].detach()),ordinary64_KL=float(ord_group[2].detach()),selected64_RD=float(selected_group[0].detach()),selected64_CE=float(selected_group[1].detach()),selected64_KL=float(selected_group[2].detach()),ordinary_group_weight=.125,selected_group_weight=.375,selected_actual_view='As_raw' if self._shape_As else 'raw',selected_ordinary_MSE=float((view[smask]-ordinary[smask]).square().mean()),selected_raw_MSE=float((view[smask]-raw[smask]).square().mean()),ordinary_raw_MSE=float((ordinary-labeled_inputs).square().mean()),second_raw_MSE=float((view-raw).square().mean()),same_IQ_each_pair=True,ordinary_redraws=0,scene_counts=dict(Counter(o['scene'] for o in observed)),state_counts=dict(Counter(o['state_end'] for o in observed)),all_satellite_L_supervised=True,crossview_mode=self._crossview_mode,new_losses=0,extra_U_NN=0)
 self._weight_records.append(dict(selected_actual_view=record['selected_actual_view'],selected_ordinary_MSE=record['selected_ordinary_MSE'],selected_raw_MSE=record['selected_raw_MSE'],selected_positions=smask.nonzero(as_tuple=False).flatten().tolist(),selected_TX=labels[smask].tolist(),selected_RX=rx[smask].tolist(),selected_day=torch.as_tensor(extra[1]['day_i'])[smask.to(torch.as_tensor(extra[1]['day_i']).device)].tolist(),step=step,cumulative_step=step+self._shape_offset,As_packets=record['As_packets'],selected_packets=64,ordinary_group_weight=.125,selected_group_weight=.375,mixed_weighted_RD=record['mixed_weighted_RD'],mixed_weighted_CE=record['mixed_weighted_CE'],mixed_weighted_KL=record['mixed_weighted_KL'],ordinary64_RD=record['ordinary64_RD'],ordinary64_CE=record['ordinary64_CE'],ordinary64_KL=record['ordinary64_KL'],selected64_RD=record['selected64_RD'],selected64_CE=record['selected64_CE'],selected64_KL=record['selected64_KL'],unweighted_mixed_RD=record['extra_RD'],unweighted_mixed_CE=record['extra_CE'],unweighted_mixed_KL=record['extra_KL']))
 self._full_records.append(record)
 return total-identity_loss,{'concat/'+k:record[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')}
base_hook._accepts_labeled_inputs=True


def hook(self,**kw):
 delta,info=base_hook(self,**kw);step=kw['optimizer_step']+self._shape_offset;labels=kw['labels'];raw=kw['labeled_inputs'].detach();view,meta=original.noise.apply(raw,step,'NONE')
 z0=self.forward_identity_only(view,domain_labels=kw['domains'])['tx_logits'];z1=self.forward_identity_only(view,domain_labels=kw['domains'])['tx_logits'];rd,ce,kl,_,_=terms(z0,z1,labels,1.)
 self._shape_NN+=256;self._shape_records.append(dict(step=kw['optimizer_step'],cumulative_step=step,shape_RD=float(rd.detach()),shape_CE=float(ce.detach()),shape_KL=float(kl.detach()),weight=.25,NN=256,same_IQ_each_pair=True,**meta))
 record=self._full_records[-1];record['base_L_before_shape']=record['total_L'];record['total_L']+=.25*float(rd.detach());record['L_forwards']=6;record['Ag_packets']=0;record['extra_control_packets']=128;record['Ag_mode']='NONE';record['same_IQ_each_pair']=True
 # Every native metric below is actually emitted. total_L always means full6L.
 info={'concat/'+k:record[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')}
 info.update({'weight/'+k:record[k] for k in ('mixed_weighted_RD','mixed_weighted_CE','mixed_weighted_KL','ordinary64_RD','ordinary64_CE','ordinary64_KL','selected64_RD','selected64_CE','selected64_KL','ordinary_group_weight','selected_group_weight')})
 info.update({'weight/raw_RD':float(rd.detach()),'weight/raw_CE':float(ce.detach()),'weight/raw_KL':float(kl.detach())})
 return delta+.25*rd,info
hook._accepts_labeled_inputs=True

def original_config(c):d.validate(c);return d.parent_config(c)

def attach(m,c):
 d.validate(c);m._full_channel=Shifted(Practical('train')) if c['As'] else original.NoAs();m._source_labeled_feature_hook=MethodType(hook,m);m._weight_records=[];m._shape_offset=6660
 m._feature_learning_report['Asweight']=dict(parent=c['parent_source'],new_epochs=20,cumulative_epochs=50,mixed_group_weights=[.125,.375],primary_ordinary_weight=.5,raw_weight=.25,As_per_step=64 if c['As'] else 0,per_step_L_NN=768,total_L_loss_scale=1.25,ORD128='original actual formula/maps/mask',default_head_joint=True,no_new_parameters=True,no_U_change=True,unchanged_forward_order=True)
 return m

def install(m,c):return attach(weighted30.install(m,original_config(c)),c)
def build(c,device='cpu'):return attach(weighted30.build(original_config(c),device),c)
def own_model(c,device='cpu'):
 d.validate(c)
 from .source import parent_payload
 parent_payload(c);out=Path(c['output_root'])
 check_source_metadata(d.read(out/'completion.json'),d.read(out/'source_frozen.json'),d.read(out/'initialization.json'),d.read(out/'source_contract.json'),c,d.read(d.SOURCE)['role_ids'],20,4440)
 cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
 check_checkpoint(cp,run=d.RUN,row=c['row_id'],epoch=20,updates=4440,baseline=c['parent_source']+'/final_ssdg.pth')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
