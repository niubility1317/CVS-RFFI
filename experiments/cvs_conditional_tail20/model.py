from types import MethodType
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d
from .loss import objective
from .channel import Practical,SecondPractical
from experiments.cvs_F_nontrue20 import model as base
af=base.af
classifier,source_domain_count=base.classifier,base.source_domain_count

def hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count,labeled_inputs):
 if self is not model or not self.training or not torch.is_grad_enabled() or optimizer_step!=self._feature_l_steps or optimizer_step!=self._nontrue_steps+1 or labeled_count!=128 or labeled_inputs.shape!=(128,2,256) or labels.shape!=(128,) or not isinstance(extra,(tuple,list)) or len(extra)<2:raise ValueError('Actual ordered legal L128 update required')
 first=F.cross_entropy(labeled_output['tx_logits'],labels,label_smoothing=.01)
 if not torch.allclose(first.detach(),identity_loss.detach(),atol=1e-6,rtol=1e-5):raise ValueError('Native ordinary CE differs')
 ep=(optimizer_step-1)//222+1;raw=labeled_inputs.detach()
 x,a=self._nontrue_channel.apply(raw,extra[1],epoch=ep,step=optimizer_step)
 y,b=self._second_channel.apply(raw,extra[1],epoch=ep,step=optimizer_step)
 if len(a)!=128 or len(b)!=128 or any(q['sample_id']!=r['sample_id'] or q['session_id']!=r['session_id'] or q['namespace']==r['namespace'] for q,r in zip(a,b)):raise ValueError('All physical pairs and disjoint draw namespaces required')
 z0=self.forward_identity_only(x,domain_labels=domains)['tx_logits']
 devices=[] if x.device.type=='cpu' else [x.device.index or 0]
 with torch.random.fork_rng(devices=devices):
  torch.manual_seed(d.SECOND_DROPOUT_SEED+optimizer_step)
  z1=self.forward_identity_only(y,domain_labels=domains)['tx_logits']
 asloss,info=objective(z0,z1,labels,self._tail_config['arm']);total=.5*(identity_loss+asloss)
 self._nontrue_steps+=1;self._nontrue_L_packets+=384
 rec=dict(step=optimizer_step,epoch=ep,ordinary_CE=float(first.detach()),total_L=float(total.detach()),L_forwards=3,As_packets=256,teacher_packets=0,extra_CE=float(asloss.detach()),TX=labels.detach().tolist(),same_physical_pairs=128,second_draw_namespace_step=4441-optimizer_step,**info)
 self._nontrue_records.append(rec)
 return total-identity_loss,{'conditional_tail/ordinary_CE':first.detach(),'conditional_tail/As_CE':asloss.detach(),'conditional_tail/total_L':total.detach()}
hook._accepts_labeled_inputs=True

def attach(m,c):
 d.validate(c);m._source_labeled_feature_hook=MethodType(hook,m);m._tail_config=c
 m._nontrue_steps=0;m._nontrue_L_packets=0;m._nontrue_records=[];m._teacher_packets=0;m._nontrue_channel=Practical('train');m._second_channel=SecondPractical('train');m.__dict__['_class_teacher']=None
 e=m.id_backbone.encoder;e._counted_features=e.features;e._full_train_packets=0;e._full_eval_packets=0;e.features=MethodType(base.counted,e)
 m._feature_learning_report.update(fixed_F='SIG_TIME_X3+CENTER_25K+CENTER_UNITS',CROSS_RX=False,RDrop=False,conditional_channel_risk=c['training_mechanism'],new_deployment_parameters=0,head_fit_updates=0)
 return m

def install(m,c):return attach(af.configure(af.original.old.install(m,af.original.d.legacy_config(c['model_seed'])),d.model_config()),c)

def build(c,device='cpu'):return attach(af.configure(af.original.old.build(af.original.d.legacy_config(c['model_seed']),'cpu'),d.model_config()),c).to(device)

def own_model(c,device='cpu'):
 d.validate(c);out=Path(c['output_root']);done=d.read(out/'completion.json');f=d.read(out/'source_frozen.json');init=d.read(out/'initialization.json');contract=d.read(out/'source_contract.json')
 if done['config']!=c or done['optimizer_steps']!=4440 or done['epoch']!=20 or done['target_access'] or done['head_fit_updates'] or f['config']!=c or f['selection']!='final_only' or f['target_access'] or not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or contract['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('Own source scratch20 lineage differs')
 cp=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if cp['epoch']!=20 or cp['a1_ema_successful_updates']!=4440 or cp['args']['baseline_ckpt'] or cp['args']['teacher_ckpt'] or cp['checkpoint_selection']!='final_only' or cp['args']['candidate_id']!=c['row_id']:raise ValueError('Own default checkpoint identity differs')
 m=build(c,device);m.load_state_dict(cp['model'],strict=True);return m.eval().requires_grad_(False)
