"""Original F/L objective plus one private-stream U readout; no new parameters."""
from contextlib import contextmanager
from types import MethodType
import gzip,json
from pathlib import Path
import torch
from experiments.cvs_clock_factorial20 import model as old
from experiments.cvs_clock_factorial30.model import Shifted
from experiments.cvs_clock_factorial30.channel import Practical as TailPractical
from experiments.cvs_U_teacher38.soft_transfer import transfer_loss
from . import design as d
from .channel import UPractical
from .protocol import supervision_weight
classifier,source_domain_count=old.classifier,old.source_domain_count

from .rng import private_rng

def select(self,*,model,teacher,x_u,teacher_logits,student_logits,pseudo,base_mask,strong_mask,extra,domains,optimizer_step):
 if model is not self or not self.training or teacher is self or teacher.training or self._u39_pending is not None or optimizer_step!=self._feature_l_steps:raise ValueError('Ordered hidden-U own-EMA context required')
 n=len(x_u);step=optimizer_step+self._u39_config['step_offset'];epoch=(step-1)//222+1
 if epoch not in range(14,31) or x_u.shape!=(n,2,256) or not 0<n<=256 or teacher_logits.shape!=(n,6) or student_logits.shape!=(n,6) or domains.shape!=(n,) or not bool(((domains>=0)&(domains<15)).all()):raise ValueError('Original source U dimensions/domain differ')
 meta={k:extra[1][k] for k in ('eq_i','rx_i','day_i','sig_i','base_index')}
 from experiments.cvs_fullsource_concat20.supcon import RX_BY_DOMAIN
 rx=torch.as_tensor(meta['rx_i'],device=domains.device,dtype=torch.long)
 if not torch.equal(rx,torch.tensor(RX_BY_DOMAIN,device=domains.device)[domains]):raise ValueError('Physical U RX and domain differ')
 if self._u39_quality is None:
  from .quality import check
  with private_rng(x_u,step):self._u39_quality=check(self,teacher,Path(self._u39_config['output_root'])/'U_qualification.json')
 self._u39_pending=dict(x=x_u.detach().clone(),meta=meta,step=step,local_step=optimizer_step,teacher=teacher_logits.detach(),original_mask=(base_mask&strong_mask).detach().clone())
 return base_mask&strong_mask,{'U39/original_mask_unchanged':1.}

def extra_hook(self,*,model,teacher,student_output,teacher_output,extra,domains,optimizer_step,identity_loss,pseudo_loss,entropy_loss):
 p=self._u39_pending
 if model is not self or not self.training or p is None or p['local_step']!=optimizer_step or not torch.equal(p['teacher'],teacher_output['tx_logits'].detach()):raise ValueError('Matching native teacher/U step required')
 x=p['x'];n=len(x);step=p['step'];epoch=(step-1)//222+1;mode=self._u39_config['arm'];observed=[]
 try:
  view=x
  if mode=='UAS':
   view,observed=self._u39_channel.apply(x,p['meta'],epoch,step)
   if len(observed)!=n:raise ValueError('No source U packet may be dropped/rejected by urban confidence')
   with gzip.open(Path(self._u39_config['output_root'])/'U_physics.jsonl.gz','xt' if self._u39_physics is None else 'at',encoding='utf8',compresslevel=3) as f:f.write(json.dumps(dict(step=step,observations=observed),allow_nan=False)+'\n')
   self._u39_physics=True
   for session,hw in self._u39_channel.sessions.items():
    lhw=self._full_channel.sessions.get(session)
    if lhw is not None and lhw!=hw:raise ValueError('L/U hardware must match the same physical RX/day/eq session')
  with private_rng(x,step):z=self.forward_identity_only(view,domain_labels=domains)['tx_logits']
  value,info=transfer_loss(z,teacher_output['tx_logits'],torch.full((n,),-1,device=z.device,dtype=torch.long),tau=.95)
  enabled=self._u39_quality['source_qualified'];weight=supervision_weight(mode,epoch,enabled)
  self._u39_packets+=n
  r=dict(step=step,epoch=epoch,packets=n,mode=mode,source_qualified=enabled,weight=weight,KL=float(value.detach()),weighted_KL=weight*float(value.detach()),q_mean=float(info['q'].mean()),positive_q=int((info['q']>0).sum()),original_selected=int(p['original_mask'].sum()),U_TX_truth_read=False,extra_teacher_forwards=0,urban_confidence_filter=False)
  self._u39_records.append(r)
  return weight*value,{'U39/'+k:r[k] for k in ('weight','KL','weighted_KL','q_mean','positive_q','packets')}
 finally:self._u39_pending=None

def attach(m,c):
 d.validate(c)
 if c['segment']==30:m._full_channel=Shifted(TailPractical('train'))
 if getattr(m,'_source_unlabeled_selection_hook',None) is not None or getattr(m,'_source_unlabeled_feature_hook',None) is not None:raise ValueError('Original U must have no competing extra method')
 m._u39_config=c;m._u39_pending=None;m._u39_records=[];m._u39_packets=0;m._u39_physics=None;m._u39_channel=UPractical('train') if c['arm']=='UAS' else None
 m._u39_quality=d.read(Path(c['parent_source'])/'U_qualification.json') if c['segment']==30 else None
 m._source_unlabeled_selection_hook=MethodType(select,m);m._source_unlabeled_feature_hook=MethodType(extra_hook,m)
 m._feature_learning_report['U39']=c['U_transfer'];return m

def old_config():return old.d.config(next(r for r in old.d.rows() if r['arm']=='B01'))
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
