"""Real checkpoint no-query smoke, then real-F synthetic extra-U gradient route."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import torch
from . import design as d,model

def smoke():
 source=Path(d.SMOKE_SOURCE);done=d.read(source/'completion.json');init=d.read(source/'initialization.json')
 if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['target_access'] or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact'] or d.read(source/'source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids']:raise ValueError('Compatibility checkpoint must match original scratch source contract')
 cp=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if cp['args']['baseline_ckpt'] or cp['args']['teacher_ckpt'] or cp['checkpoint_selection']!='final_only':raise ValueError('Compatibility inheritance differs')
 m=model.old.build(model.old_config(),'cpu').eval();m.load_state_dict(cp['model'],strict=True)
 with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
 if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Actual default CP no-query smoke failed')
 d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',strict_load=True,NN=2,real_IQ=0,target_query=False,training_initializer=False,optimizer_updates=0))
 records=[]
 for row in d.rows():
  c=d.config(row,20);m=model.build(c,'cpu').train();test=d.BASE/'synthetic'/row['arm'];test.mkdir(parents=True,exist_ok=False);m._u39_config=deepcopy(c);m._u39_config['output_root']=str(test);m._u39_quality=dict(source_qualified=True,synthetic_only=True);m._feature_l_steps=2887
  n=16;g=torch.Generator().manual_seed(731);x=torch.randn(n,2,256,generator=g);domains=torch.arange(n)%15
  from experiments.cvs_fullsource_concat20.supcon import RX_BY_DOMAIN
  meta=dict(eq_i=torch.ones(n,dtype=torch.long),rx_i=torch.tensor(RX_BY_DOMAIN)[domains],day_i=domains%3+1,sig_i=torch.arange(n)+2000,base_index=torch.arange(n));extra=(domains,meta)
  t=torch.zeros(n,6);t[torch.arange(n),torch.arange(n)%6]=12.;old_s=torch.zeros_like(t);teacher=SimpleNamespace(training=False);base=(torch.arange(n)%3==0);strong=(torch.arange(n)%2==0)
  initial=torch.get_rng_state().clone();mask,_=m._source_unlabeled_selection_hook(model=m,teacher=teacher,x_u=x,teacher_logits=t,student_logits=old_s,pseudo=t.argmax(1),base_mask=base,strong_mask=strong,extra=extra,domains=domains,optimizer_step=2887)
  loss,_=m._source_unlabeled_feature_hook(model=m,teacher=teacher,student_output=dict(tx_logits=old_s),teacher_output=dict(tx_logits=t),extra=extra,domains=domains,optimizer_step=2887,identity_loss=torch.tensor(0.),pseudo_loss=torch.tensor(0.),entropy_loss=torch.tensor(0.));loss.backward()
  h=m.id_backbone.encoder.core.id_backbone.cls_head.weight;sinc=m.id_backbone.encoder.core.id_backbone.sinc
  if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in m.parameters()):raise ValueError('Nonfinite added-U gradient')
  head_norm=float(h.grad.abs().sum());sinc_norm=sum(float(p.grad.abs().sum()) for p in sinc.parameters() if p.grad is not None)
  if not torch.equal(mask,base&strong) or not torch.equal(initial,torch.get_rng_state()) or m._u39_packets!=n or m._u39_pending is not None or not torch.isfinite(loss):raise ValueError('Real synthetic U mask/RNG/count mismatch')
  if row['arm']!='B0' and (head_norm<=0 or sinc_norm<=0):raise ValueError('Extra U objective did not reach original joint head and Sinc')
  if row['arm']=='UAS' and m._u39_channel.packets!=n:raise ValueError('All16 full practical synthetic U required')
  if Path(c['output_root']).exists():raise ValueError('Synthetic must not occupy real source output')
  records.append(dict(arm=row['arm'],NN=n,head_gradient_L1=head_norm,Sinc_gradient_L1=sinc_norm,old_mask_unchanged=True,RNG_restored=True,U_As=m._u39_channel.packets if m._u39_channel else 0,optimizer_updates=0,IQ_source='synthetic only;highconfidence logits synthetic;real CP smoke separately above'))
 d.write(d.BASE/'synthetic/complete.json',dict(status='PASS',records=records,NN=48,real_IQ=0,optimizer_updates=0,target_access=False,real_outputs_absent=True))
 return dict(status='PASS',NN=50,CP_NN=2,synthetic_NN=48,optimizer_updates=0)
