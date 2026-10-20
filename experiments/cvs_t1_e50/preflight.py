"""Actual source CP noquery first; six existing F gradient and budget probes."""
import torch
from . import design as d,model
from experiments.cvs_crossview_supcon41 import model as base,design as bd

def smoke():
 torch.set_num_threads(2)
 from .source import original_parent
 c=d.config(d.rows()[0]);cp=original_parent();m=model.build(c,'cpu');m.load_state_dict(cp['model'],strict=True);m.eval().requires_grad_(False);before={k:v.clone() for k,v in m.state_dict().items()}
 with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
 if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Real source CP defaulthead smoke failed')
 d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',NN=2,source_IQ=0,target_query=False,checkpoint=str(d.PARENT_SOURCE/'final_ssdg.pth'),training_initializer=False,same_source_contract=True,optimizer_updates=0))
 del m,before
 initial=None;records=[]
 for row in d.rows():
  torch.manual_seed(6007);c=d.config(row);m=model.build(c,'cpu').train();current={k:v.clone() for k,v in m.state_dict().items()}
  if initial is None:initial=current
  if sum(p.numel() for p in m.parameters())!=900722 or len(current)!=206 or any(not torch.equal(v,initial[k]) for k,v in current.items()):raise ValueError('Same fixed F/defaulthead initialization differs')
  m._feature_l_steps=1333;m._full_calls=1332;g=torch.Generator().manual_seed(6008);x=torch.randn((128,2,256),generator=g);y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(base.supcon.RX_BY_DOMAIN)[domains];meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=domains%3+1,sig_i=torch.arange(128)+2000,base_index=torch.arange(128))
  first,maps=base.capture_forward(m,x,domains,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=x.clone();ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
  extra,_=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(domains,meta),domains=domains,optimizer_step=1333,identity_loss=ce,labeled_count=128,labeled_inputs=x)
  total=ce+extra;total.backward();head=m.id_backbone.encoder.core.id_backbone.cls_head.weight;sinc=m.id_backbone.encoder.core.id_backbone.sinc
  if not torch.isfinite(total) or head.grad is None or float(head.grad.abs().sum())<=0 or sum(float(p.grad.abs().sum()) for p in sinc.parameters() if p.grad is not None)<=0 or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise ValueError('Full joint defaulthead/Sinc finite gradient failed')
  r=m._full_records[-1];shape=m._shape_records[-1]
  if len(m._temperature_records)!=1 or m._temperature_records[0]['T']!=c['temperature']['T'] or m.id_backbone.encoder._full_train_packets!=768 or m._shape_NN!=256 or len(m._shape_records)!=1 or m._full_channel.packets!=(64 if c['As'] else 0) or r['As_packets']!=(64 if c['As'] else 0) or r['L_forwards']!=6 or m._supcon_pending is not None or m._supcon_handles or any(not torch.equal(v,m.state_dict()[k]) for k,v in current.items()) or getattr(m,'_source_unlabeled_feature_hook',None) is not None:raise ValueError('SixL/actualAs/ownshape/state/defaultU budget differs')
  if abs(r['total_L']-float(total.detach()))>3e-6 or abs(r['total_L']-r['base_L_before_shape']-.25*shape['shape_RD'])>1e-6 or not shape['same_IQ_each_pair']:raise ValueError('Prospective normalized1.25 loss scale differs')
  records.append(dict(arm=row['arm'],NN=768,As_IQ=r['As_packets'],Ag_IQ=shape['applied'],head_joint_gradient=True,Sinc_joint_gradient=True,same_initial=True,optimizer_updates=0,noise=shape))
 d.write(d.BASE/'synthetic_smoke.json',dict(status='PASS',records=records,NN=1536,real_checkpoint_NN=2,real_IQ=0,optimizer_updates=0,head_fit_updates=0,target_access=False))
 print('REAL_CP_NO_QUERY+SIX_REAL_F_SYNTHETIC_SMOKE PASS1538NN;continue training',flush=True)
 return dict(status='PASS',NN=1538)
