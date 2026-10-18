"""Minimal originalP41+P43 L hook;only selectedAs CE target smoothing differs."""
from collections import Counter
from collections.abc import Mapping
from types import MethodType
from pathlib import Path
import torch
from . import design as d
from .protocol import check_checkpoint,check_source_metadata
from .schema import CONCAT_FIELDS,RISK_FIELDS
from .risk import correction
from experiments.cvs_noise_shape_factorial43 import model as original,noise
from experiments.cvs_crossview_supcon41 import model as p41
capture_forward=p41.capture_forward;terms=p41.terms;relations=p41.relations;supcon=p41.supcon
classifier,source_domain_count=original.classifier,original.source_domain_count

def hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count,labeled_inputs):
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
 z0,sat0=capture_forward(self,view,domains,weight>0);z1,sat1=capture_forward(self,view,domains,weight>0);extra_loss,extra_ce,extra_kl,mixed_ce0,mixed_ce1=terms(z0,z1,labels,1.)
 smooth_delta,sr=correction(z0,z1,labels,smask,mode=self._risk_mode,step=step,enabled=self._shape_As,old_CE_means=(mixed_ce0,mixed_ce1));old_extra_loss=extra_loss;old_extra_ce=extra_ce;extra_loss=extra_loss+2*smooth_delta;extra_ce=extra_ce+2*smooth_delta
 details=[]
 if weight:
  vals=[]
  for phase,a,b in [('first',first_maps,sat0),('second',second_maps,sat1)]:
   value,records=relations.penalty(a,b,labels,rx,smask,self._crossview_mode);vals.append(value);details.extend(dict(phase=phase,**r) for r in records)
  aux=torch.stack(vals).mean();self._supcon_applied+=1
 else:aux=common*0
 if getattr(self,'_crossview_smoke_capture',False):self._crossview_debug=dict(aux=aux,ordinary0=first_maps,ordinary1=second_maps,satellite0=sat0,satellite1=sat1,mask=smask)
 total=.5*common+.5*extra_loss+weight*aux
 sr.update(step=step,epoch=epoch,cumulative_step=step+4440,cumulative_epoch=epoch+20,selected_L_positions=smask.nonzero(as_tuple=False).flatten().tolist(),TX=labels[smask].tolist(),RX=rx[smask].tolist(),day=meta['day_i'].tolist(),sample_ids=[o['sample_id'] for o in observed],scene_counts=dict(Counter(o['scene'] for o in observed)),As_packets=len(observed),mixed_KL=float(extra_kl.detach()),mixed_RD_old=float(old_extra_loss.detach()),mixed_RD_new=float(extra_loss.detach()),same_IQ_each_pair=True,ordinary_redraws=0)
 self._full_calls+=1;self._full_L_packets+=512;self._supcon_calls=step;self._legacy_calls+=1;self._legacy_extra_packets+=128;self._legacy_positive_KL_calls+=int(float(kl.detach())>0)
 record=dict(step=step,epoch=epoch,total_L=float(total.detach()),common_RD=float(common.detach()),common_CE=float(ce.detach()),common_KL=float(kl.detach()),extra_RD=float(extra_loss.detach()),extra_CE=float(extra_ce.detach()),extra_KL=float(extra_kl.detach()),SupCon=float(aux.detach()),SupCon_weight=weight,SupCon_details=details,L_forwards=4,Ag_packets=0,As_packets=64,ordinary_raw_MSE=float((ordinary-labeled_inputs).square().mean()),second_raw_MSE=float((view-raw).square().mean()),same_IQ_each_pair=True,ordinary_redraws=0,scene_counts=dict(Counter(o['scene'] for o in observed)),state_counts=dict(Counter(o['state_end'] for o in observed)),all_satellite_L_supervised=True,crossview_mode=self._crossview_mode,new_losses=0,extra_U_NN=0)
 self._full_records.append(record);self._hardrisk_records.append(sr)
 return total-identity_loss,{}
hook._accepts_labeled_inputs=True

base_hook=hook

def hook(self,**kw):
 delta,unused=base_hook(self,**kw);step=kw['optimizer_step']+self._shape_offset;labels=kw['labels'];raw=kw['labeled_inputs'].detach();view,meta=noise.apply(raw,step,self._shape_mode)
 z0=self.forward_identity_only(view,domain_labels=kw['domains'])['tx_logits'];z1=self.forward_identity_only(view,domain_labels=kw['domains'])['tx_logits'];rd,ce,kl,_,_=terms(z0,z1,labels,1.)
 self._shape_NN+=256;self._shape_packets+=meta['applied'];self._shape_records.append(dict(step=kw['optimizer_step'],cumulative_step=step,shape_RD=float(rd.detach()),shape_CE=float(ce.detach()),shape_KL=float(kl.detach()),weight=.25,NN=256,same_IQ_each_pair=True,**meta))
 record=self._full_records[-1];record['base_L_before_shape']=record['total_L'];record['total_L']+=.25*float(rd.detach());record['L_forwards']=6;record['Ag_packets']=meta['applied'];record['extra_control_packets']=128;record['As_packets']=64 if self._shape_As else 0;record['Ag_mode']=self._shape_mode;record['same_IQ_each_pair']=True
 sr=self._hardrisk_records[-1];sr.update(total_L=record['total_L'],base_total_L=record['total_L']-sr['loss_delta'],raw_RD=float(rd.detach()),common_RD=record['common_RD'],SupCon=record['SupCon'],SupCon_weight=record['SupCon_weight'],total_L_scale=1.25)
 if abs(sr['mixed_CE_old']-float(old_extra_ce_for_audit(record,sr)))>3e-6:raise ValueError('Existing mixed CE01 scalar differs')
 info={'concat/'+k:record[k] for k in CONCAT_FIELDS};info.update({'hardrisk/'+k:sr[k] for k in RISK_FIELDS})
 return delta+.25*rd,info
hook._accepts_labeled_inputs=True

def old_extra_ce_for_audit(record,sr):return record['extra_CE']-sr['mixed_CE_delta']

def original_config(c):
 d.validate(c);return d.reference_config(next(r for r in d.rows() if r['row_id']==c['row_id']),30)

def attach(m,c):
 d.validate(c)
 if m._crossview_mode!='ORD128' or not m._shape_As or m._shape_mode!='NONE':raise ValueError('OriginalP43 ORD128/As/rawNONE required')
 m._hardrisk_records=[];m._risk_mode=c['risk_mode'];m._source_labeled_feature_hook=MethodType(hook,m)
 m._feature_learning_report['within_TX_risk']=dict(c['within_TX_risk'],parent=c['parent_source'],new_epochs=10,cumulative_epochs=30,pair_weights=[.5,.5,.25],SupCon='unchanged ORD128',per_step_L_NN=768,total_L_loss_scale=1.25,default_head_joint=True,extra_U_NN=0,no_new_parameters=True)
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
