"""Same F/default head, original CE/RDrop/U; relocate existing SupCon view coverage."""
from collections import Counter
from collections.abc import Mapping
from types import MethodType
from pathlib import Path
import torch
from . import design as d,relations
from experiments.cvs_clock_factorial20 import model as old
from experiments.cvs_clock_factorial30.model import Shifted
from experiments.cvs_clock_factorial30.channel import Practical as TailPractical
from experiments.cvs_fullsource_concat20 import supcon
from experiments.cvs_sigtime_rdrop_legacy.model import terms
classifier,source_domain_count=old.classifier,old.source_domain_count

def old_config():return old.d.config(next(r for r in old.d.rows() if r['arm']=='B01'))
def capture_forward(m,x,domains,active):
 maps={};handles=[]
 if active:
  for level in supcon.LEVELS:
   def capture(module,args,out,level=level):
    if level in maps or out.ndim!=3 or len(out)!=128:raise ValueError('One actual L128 map per existing forward')
    maps[level]=out
   handles.append(m.get_submodule(supcon.PREFIX+level).register_forward_hook(capture))
 try:z=m.forward_identity_only(x,domain_labels=domains)['tx_logits']
 finally:
  for h in handles:h.remove()
 if active and tuple(maps)!=supcon.LEVELS:raise ValueError('Missing actual shallow/middle maps')
 return z,maps

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
 z0,sat0=capture_forward(self,view,domains,weight>0);z1,sat1=capture_forward(self,view,domains,weight>0);extra_loss,extra_ce,extra_kl,_,_=terms(z0,z1,labels,1.)
 details=[]
 if weight:
  vals=[]
  for phase,a,b in [('first',first_maps,sat0),('second',second_maps,sat1)]:
   value,records=relations.penalty(a,b,labels,rx,smask,self._crossview_mode);vals.append(value);details.extend(dict(phase=phase,**r) for r in records)
  aux=torch.stack(vals).mean();self._supcon_applied+=1
 else:aux=common*0
 if getattr(self,'_crossview_smoke_capture',False):self._crossview_debug=dict(aux=aux,ordinary0=first_maps,ordinary1=second_maps,satellite0=sat0,satellite1=sat1,mask=smask)
 total=.5*common+.5*extra_loss+weight*aux
 self._full_calls+=1;self._full_L_packets+=512;self._supcon_calls=step;self._legacy_calls+=1;self._legacy_extra_packets+=128;self._legacy_positive_KL_calls+=int(float(kl.detach())>0)
 record=dict(step=step,epoch=epoch,total_L=float(total.detach()),common_RD=float(common.detach()),common_CE=float(ce.detach()),common_KL=float(kl.detach()),extra_RD=float(extra_loss.detach()),extra_CE=float(extra_ce.detach()),extra_KL=float(extra_kl.detach()),SupCon=float(aux.detach()),SupCon_weight=weight,SupCon_details=details,L_forwards=4,Ag_packets=0,As_packets=64,ordinary_raw_MSE=float((ordinary-labeled_inputs).square().mean()),second_raw_MSE=float((view-raw).square().mean()),same_IQ_each_pair=True,ordinary_redraws=0,scene_counts=dict(Counter(o['scene'] for o in observed)),state_counts=dict(Counter(o['state_end'] for o in observed)),all_satellite_L_supervised=True,crossview_mode=self._crossview_mode,new_losses=0,extra_U_NN=0)
 self._full_records.append(record)
 return total-identity_loss,{'concat/'+k:record[k] for k in ('total_L','common_CE','common_KL','SupCon','SupCon_weight','L_forwards','As_packets')}
hook._accepts_labeled_inputs=True

def attach(m,c):
 d.validate(c)
 if c['segment']==30:m._full_channel=Shifted(TailPractical('train'))
 if getattr(m,'_source_unlabeled_selection_hook',None) is not None or getattr(m,'_source_unlabeled_feature_hook',None) is not None:raise ValueError('No new U objective or forward permitted')
 m._source_labeled_feature_hook=MethodType(hook,m);m._crossview_mode=c['arm'];m._feature_learning_report['crossviewSupCon41']=c['crossview'];return m

def install(m,c):return attach(old.install(m,old_config()),c)
def build(c,device='cpu'):return attach(old.build(old_config(),device),c)
def own_model(c,device='cpu'):
 d.validate(c)
 if c['segment']!=30:raise ValueError('Formal predictions require own fixed cumulative30')
 b=Path(c['output_root']);done=d.read(b/'completion.json');frozen=d.read(b/'source_frozen.json');parent=d.config(next(r for r in d.rows() if r['row_id']==c['row_id']),20)
 for cfg,epoch,updates in [(parent,20,4440),(c,10,2220)]:
  p=Path(cfg['output_root']);ok=d.read(p/'completion.json');init=d.read(p/'initialization.json')
  if ok['config']!=cfg or ok['epoch']!=epoch or ok['optimizer_steps']!=updates or ok['target_access'] or d.read(p/'source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids'] or init['target_contact'] or init['checkpoint_sources']!=cfg['checkpoint_sources']:raise ValueError('Own source inheritance differs')
 if frozen['config']!=c or frozen['selection']!='final_only' or frozen['target_access']:raise ValueError('Own fixed freeze differs')
 cp=torch.load(b/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if cp['epoch']!=10 or cp['a1_ema_successful_updates']!=2220 or cp['run_id']!=d.RUN or cp['candidate_id']!=c['row_id'] or cp['args']['baseline_ckpt']!=c['parent_source']+'/final_ssdg.pth' or cp['args']['teacher_ckpt']:raise ValueError('Only own fixed final state allowed')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
