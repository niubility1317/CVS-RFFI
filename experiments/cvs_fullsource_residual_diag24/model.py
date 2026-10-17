"""Load only the three new full-source FINAL20 default heads; no replacement."""
import torch
from . import design as d
def load(arm,device):
 from experiments.cvs_fullsource_concat22 import model as api,design as prior
 path=d.REFERENCES[arm];done=d.read(path/'completion.json');init=d.read(path/'initialization.json');frozen=d.read(path/'source_frozen.json');c=done['config']
 run=d.RAW_RUN if arm=='RAW_PRAC' else d.REFERENCE_RUN
 if c['run_id']!=run or c['row_id']!=arm+'-s'+str(d.SEED) or c['model_seed']!=d.SEED or c['epochs']!=20 or c['source_roles']!=dict(L_s=6300,U_s=56700,V=27000) or c['held_RX'] is not None:raise ValueError('Registered full-source checkpoint config differs')
 if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['head_fit_updates'] or done['target_access'] or not init['scratch_only'] or init['checkpoint_sources'] or init['ancestors'] or init['target_contact'] or init['source_roles']!='EXACT_MATCH' or frozen['epoch']!=20 or frozen['selection']!='final_only' or frozen['target_access'] or frozen['config']!=c:raise ValueError('Checkpoint provenance/freeze differs')
 if d.read(path/'source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('Checkpoint original role handles differ')
 ck=torch.load(path/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if ck['epoch']!=20 or ck['a1_ema_successful_updates']!=4440 or ck['candidate_id']!=c['row_id'] or ck['run_id']!=run or not ck['args']['from_scratch'] or ck['args']['baseline_ckpt'] or ck['args']['teacher_ckpt']:raise ValueError('Scratch-only final checkpoint payload differs')
 # RAW uses exactly the B01 deployed structure. Its own provenance is checked above.
 reference=prior.config(prior.rows()[0 if arm=='B00' else 1]);m=api.build(reference,device);m.load_state_dict(ck['model'],strict=True);m.eval().requires_grad_(False)
 head=m.id_backbone.encoder.core.id_backbone.cls_head
 if head.weight.shape!=(6,160) or float(head.scale)!=30. or sum(p.numel() for p in m.parameters())!=900722 or len(m.state_dict())!=206:raise ValueError('Fixed F/joint160/cos30 structure differs')
 return m,dict(arm=arm,source=str(path),checkpoint=str(path/'final_ssdg.pth'),model_seed=d.SEED,training_epochs=20,training_updates=4440,diagnostic_updates=0,head_fit_updates=0,scratch_origin=True,source_role_handles='EXACT_MATCH',selection='FINAL20',trained_before_target=True,formal_scores_not_consumed=True)
def modules(m):
 e=m.id_backbone.encoder;b=e.core.id_backbone
 return dict(time=b.t_proj,frequency=b.f_proj,PA=b.pa_proj,reference=e.response_projection,signature=e.sequence_pool)
def gates(m):
 e=m.id_backbone.encoder;b=e.core.id_backbone
 return dict(PA=b.cls_head.gain.detach().tanh().cpu().tolist(),reference=e.response_gain.detach().tanh().cpu().tolist(),signature=(.2*e.sequence_pool.gain_raw.detach().tanh()).cpu().tolist(),kind='learned static coordinate gains; not invented per-packet gate weights')
def group(n):
 if '.sinc.' in n:return 'Sinc_Hz'
 if '.sequence_pool.' in n:return 'signature'
 if '.response_projection.' in n or '.response_gain' in n:return 'reference'
 if '.cls_head.weight' in n:return 'default_head'
 if '.pa_' in n or '.cls_head.gain' in n:return 'PA'
 if any('.'+k+'.' in n for k in ('time_stem','time_down','t1','t2','t_proj')):return 'time'
 if '.f_proj.' in n or '.freq' in n:return 'frequency'
 return 'remaining'
def gradient(loss,named):
 values=torch.autograd.grad(loss,[p for n,p in named],allow_unused=True)
 result={}
 for (n,p),g in zip(named,values):
  if g is None:g=torch.zeros_like(p)
  if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite readonly CE gradient')
  g=g.detach().double().flatten()
  if n.endswith(('.sinc.low_coordinate','.sinc.band_coordinate')):g=g/25000.
  result.setdefault(group(n),[]).append(g)
 return {k:torch.cat(v) for k,v in result.items()}
def compare_grad(a,b):
 out={}
 for k in a:
  aa=float(a[k].square().sum());bb=float(b[k].square().sum());dot=float((a[k]*b[k]).sum())
  out[k]=dict(clean_norm=aa**.5,satellite_norm=bb**.5,dot=dot,cosine=dot/(aa*bb)**.5 if aa*bb else None,finite=True)
 aa=sum(float(v.square().sum()) for v in a.values());bb=sum(float(v.square().sum()) for v in b.values());dot=sum(float((a[k]*b[k]).sum()) for k in a)
 out['whole']=dict(clean_norm=aa**.5,satellite_norm=bb**.5,dot=dot,cosine=dot/(aa*bb)**.5 if aa*bb else None,finite=True)
 return out
