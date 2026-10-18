"""Original6L/ORD128;only bankAs64 IQ/true metadata and additive risk weights differ."""
from collections import Counter
from collections.abc import Mapping
from types import MethodType
from pathlib import Path
import time
import torch
from . import design as d
from .protocol import check_checkpoint,check_source_metadata
from .schema import CONCAT_FIELDS,BANK_FIELDS
from .bank import Sampler,FIELDS
from .losses import weighted_terms
from experiments.cvs_noise_shape_factorial43 import model as original,noise
from experiments.cvs_crossview_supcon41 import model as p41
capture_forward=p41.capture_forward;terms=p41.terms;relations=p41.relations;supcon=p41.supcon
classifier,source_domain_count=original.classifier,original.source_domain_count

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
 iq,bmeta,bank=self._full_channel.draw(epoch,step);copy_start=time.perf_counter();satellite=torch.from_numpy(iq).to(device=raw.device,dtype=raw.dtype);bank_y=torch.as_tensor(bmeta['TX'],device=labels.device,dtype=torch.long);bank_domains=torch.as_tensor(bmeta['domain'],device=domains.device,dtype=torch.long)
 if satellite.shape!=(64,2,256) or bank_y.shape!=(64,) or not torch.isfinite(satellite).all() or not bool(((bank_y>=0)&(bank_y<6)).all()) or not bool(((bank_domains>=0)&(bank_domains<15)).all()) or not torch.equal(torch.tensor(supcon.RX_BY_DOMAIN,device=domains.device)[bank_domains],torch.as_tensor(bmeta['RX'],device=domains.device)):raise ValueError('Bank IQ/own true labels/RXdomain metadata must match')
 mixed_y=labels.clone();mixed_y[smask]=bank_y;mixed_domains=domains.clone();mixed_domains[smask]=bank_domains;view[smask]=satellite;weights=torch.ones(128,device=labels.device,dtype=first.dtype);weights[smask]=torch.as_tensor(bank['weights'],device=labels.device,dtype=first.dtype);bank['cache_transfer_cpu_wall_seconds']=time.perf_counter()-copy_start
 for y,r,s,t,l in zip(bmeta['TX'],bmeta['RX'],bmeta['scene'],bmeta['state'],bmeta['lock']):self._full_coverage[(int(y),int(r),str(s),str(t),str(l))]+=1
 z0,sat0=capture_forward(self,view,mixed_domains,weight>0);z1,sat1=capture_forward(self,view,mixed_domains,weight>0);extra_loss,extra_ce,extra_kl,per,loss_seconds=weighted_terms(z0,z1,mixed_y,weights)
 details=[]
 if weight:
  vals=[]
  for phase,a,b in [('first',first_maps,sat0),('second',second_maps,sat1)]:
   value,records=relations.penalty(a,b,labels,rx,smask,self._crossview_mode);vals.append(value);details.extend(dict(phase=phase,**r) for r in records)
  aux=torch.stack(vals).mean();self._supcon_applied+=1
 else:aux=common*0
 if getattr(self,'_crossview_smoke_capture',False):self._crossview_debug=dict(aux=aux,ordinary0=first_maps,ordinary1=second_maps,satellite0=sat0,satellite1=sat1,mask=smask,mixed_logits=(z0,z1),mixed_y=mixed_y,mixed_domains=mixed_domains,weights=weights,mixed_input=view,bank_input=satellite)
 total=.5*common+.5*extra_loss+weight*aux
 self._full_calls+=1;self._full_L_packets+=512;self._supcon_calls=step;self._legacy_calls+=1;self._legacy_extra_packets+=128;self._legacy_positive_KL_calls+=int(float(kl.detach())>0)
 observed_scene=list(map(str,bmeta['scene']));observed_state=list(map(str,bmeta['state']));observed_lock=list(map(str,bmeta['lock']))
 record=dict(step=step,epoch=epoch,total_L=float(total.detach()),common_RD=float(common.detach()),common_CE=float(ce.detach()),common_KL=float(kl.detach()),extra_RD=float(extra_loss.detach()),extra_CE=float(extra_ce.detach()),extra_KL=float(extra_kl.detach()),SupCon=float(aux.detach()),SupCon_weight=weight,SupCon_details=details,L_forwards=4,Ag_packets=0,As_packets=64,ordinary_raw_MSE=float((ordinary-labeled_inputs).square().mean()),second_raw_MSE=float((view-raw).square().mean()),same_IQ_each_pair=True,ordinary_redraws=0,scene_counts=dict(Counter(observed_scene)),state_counts=dict(Counter(observed_state)),all_satellite_L_supervised=True,crossview_mode=self._crossview_mode,new_losses=0,extra_U_NN=0)
 all_meta={k:v.detach().cpu().tolist() if hasattr(v,'detach') else list(v) for k,v in extra[1].items() if k in FIELDS};sample_ids=[tuple(int(all_meta[k][i]) for k in FIELDS) for i in range(128)];positions=smask.nonzero().flatten().tolist();ordinary_positions=(~smask).nonzero().flatten().tolist()
 def scalar(v):return float(v.detach())
 bank.update(step=step,epoch=epoch,selected_L_positions=positions,ordinary_L_positions=ordinary_positions,current_L_TX=labels.detach().cpu().tolist(),current_L_RX=rx.detach().cpu().tolist(),current_L_day=all_meta['day_i'],current_L_sample_ids=sample_ids,bank_TX=bmeta['TX'].astype(int).tolist(),bank_RX=bmeta['RX'].astype(int).tolist(),bank_day=bmeta['day'].astype(int).tolist(),bank_domain=bmeta['domain'].astype(int).tolist(),bank_sample_ids=bmeta['sample_ids'].astype(int).tolist(),sample_id_fields=list(FIELDS),bank_source_L_index=bmeta['source_L_index'].astype(int).tolist(),scene=observed_scene,state=observed_state,lock=observed_lock,signal_to_added_noise_db=bmeta['signal_to_added_noise_db'].astype(float).tolist(),quality_snr_db=bmeta['quality_snr_db'].astype(float).tolist(),bank_context_epoch=bmeta['cumulative_epoch'].astype(int).tolist(),bank_replica=bmeta['replica'].astype(int).tolist(),bank_context_step=bmeta['context_step'].astype(int).tolist(),mixed_labels=mixed_y.detach().cpu().tolist(),mixed_domains=mixed_domains.detach().cpu().tolist(),ordinary_CE=scalar(per['CE'][~smask].mean()),ordinary_KL=scalar(per['KL'][~smask].mean()),bank_unweighted_CE=scalar(per['CE'][smask].mean()),bank_unweighted_KL=scalar(per['KL'][smask].mean()),bank_weighted_CE=scalar((per['CE'][smask]*weights[smask]).sum()/64),bank_weighted_KL=scalar((per['KL'][smask]*weights[smask]).sum()/64),mixed_weighted_CE=scalar(extra_ce),mixed_weighted_KL=scalar(extra_kl),mixed_weighted_RD=scalar(extra_loss),applied_bank_weights=weights[smask].detach().cpu().tolist(),Torch_bank_weight_sum=scalar(weights[smask].sum()),ordinary_weight_sum=scalar(weights[~smask].sum()),weighted_loss_cpu_wall_seconds=loss_seconds,CE_operator_widths=[128,128],baseline_CE_operator_widths=[128,128],extra_CE_operator_calls=0,CE_logit_packets=256,per_packet_CE_a=per['CE_a'].detach().cpu().tolist(),per_packet_CE_b=per['CE_b'].detach().cpu().tolist(),per_packet_KL=per['KL'].detach().cpu().tolist(),new_weight_multiplication_packets=128,As_packets=64,bank_cache_reads=64,propagation_simulations=0,same_IQ_each_pair=True,ordinary_redraws=0,bank_redraws_in_second_forward=0,scene_counts=dict(Counter(observed_scene)),state_counts=dict(Counter(observed_state)),mixed_label_replacements=sum(int(a)!=int(b) for a,b in zip(labels[smask].tolist(),bmeta['TX'])),currentL_bank_PID_matches=sum(tuple(sample_ids[p])==tuple(pid) for p,pid in zip(positions,bmeta['sample_ids'].tolist())),CPU_timing='perf_counter CPUwall;no explicit CUDA synchronization;sampling/cachegather/transfer/loss scopes disjoint;device compute not isolated')
 for i,n in enumerate(bank['bin_counts']):bank['bin'+str(i)]=n
 self._full_records.append(record);self._bank_records.append(bank)
 return total-identity_loss,{}
base_hook._accepts_labeled_inputs=True

def hook(self,**kw):
 delta,unused=base_hook(self,**kw);step=kw['optimizer_step']+self._shape_offset;labels=kw['labels'];raw=kw['labeled_inputs'].detach();view,meta=noise.apply(raw,step,self._shape_mode)
 z0=self.forward_identity_only(view,domain_labels=kw['domains'])['tx_logits'];z1=self.forward_identity_only(view,domain_labels=kw['domains'])['tx_logits'];rd,ce,kl,_,_=terms(z0,z1,labels,1.)
 self._shape_NN+=256;self._shape_packets+=meta['applied'];self._shape_records.append(dict(step=kw['optimizer_step'],cumulative_step=step,shape_RD=float(rd.detach()),shape_CE=float(ce.detach()),shape_KL=float(kl.detach()),weight=.25,NN=256,same_IQ_each_pair=True,**meta))
 record=self._full_records[-1];record['base_L_before_shape']=record['total_L'];record['total_L']+=.25*float(rd.detach());record['L_forwards']=6;record['Ag_packets']=meta['applied'];record['extra_control_packets']=128;record['Ag_mode']=self._shape_mode
 bank=self._bank_records[-1];bank.update(total_L=record['total_L'],total_L_scale=1.25,L_forwards=6,L_NN=768)
 info={'concat/'+k:record[k] for k in CONCAT_FIELDS};info.update({'bank/'+k:bank[k] for k in BANK_FIELDS})
 return delta+.25*rd,info
hook._accepts_labeled_inputs=True

def original_config(c):
 d.validate(c)
 from experiments.cvs_noise_shape_factorial43 import design as b
 cfg=b.config(next(r for r in b.rows() if r['row_id']==d.REFERENCE_ROW),30);row=next(r for r in d.rows() if r['row_id']==c['row_id'])
 if cfg!=d.reference_config(row,30):raise ValueError('Actual pinned original B01 recipe differs')
 return cfg

def attach(m,c):
 d.validate(c)
 if m._crossview_mode!='ORD128' or not m._shape_As or m._shape_mode!='NONE':raise ValueError('OriginalP43 ORD128/As/rawNONE required')
 m._full_channel=Sampler(c['quality_risk']['mode'],c['quality_risk']['bank']);m._bank_records=[];m._source_labeled_feature_hook=MethodType(hook,m)
 m._feature_learning_report['quality_risk']=dict(c['quality_risk'],parent=c['parent_source'],new_epochs=10,cumulative_epochs=30,pair_weights=[.5,.5,.25],SupCon='unchanged ORD128 on currentL only',per_step_L_NN=768,total_L_loss_scale=1.25,default_head_joint=True,extra_U_NN=0,no_new_parameters=True)
 return m

def install(m,c):return attach(original.install(m,original_config(c)),c)
def build(c,device='cpu'):return attach(original.build(original_config(c),device),c)
def own_model(c,device='cpu'):
 d.validate(c)
 from .source import parent_payload
 parent_payload(c);out=Path(c['output_root']);check_source_metadata(d.read(out/'completion.json'),d.read(out/'source_frozen.json'),d.read(out/'initialization.json'),d.read(out/'source_contract.json'),c,d.read(d.SOURCE)['role_ids'])
 cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False);check_checkpoint(cp,run=d.RUN,row=c['row_id'],epoch=10,updates=2220,baseline=c['parent_source']+'/final_ssdg.pth')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
