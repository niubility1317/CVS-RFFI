"""One real As20 no-query smoke and two actual F L-steps; PASS immediately trains."""
import torch
from . import design as d,model,relations
from .checks import run as focused_checks
from .source import parent_payload
from experiments.cvs_crossview_supcon41 import model as base

def ord128_equivalence():
 g=torch.Generator().manual_seed(6010)
 maps={k:torch.randn((128,4,8),generator=g,requires_grad=True) for k in relations.LEVELS}
 satellite={k:torch.randn((128,4,8),generator=g) for k in relations.LEVELS}
 y=torch.arange(128)%6;rx=torch.tensor([1,3,4,6,8])[torch.arange(128)%5];mask=(torch.arange(128)%2)==0
 old,old_details=base.relations.penalty(maps,satellite,y,rx,mask,'ORD128');new,new_details=relations.penalty(maps,y,rx)
 old_grad=torch.autograd.grad(old,tuple(maps.values()),retain_graph=True);new_grad=torch.autograd.grad(new,tuple(maps.values()))
 if not torch.equal(old,new) or old_details!=new_details or any(not torch.equal(a,b) for a,b in zip(old_grad,new_grad)):raise ValueError('Actual original ORD128 scalar/details/gradient changed')
 return dict(status='PASS',actual_ordinary_anchors=128,loss_and_gradients_exact=True,NN=0)

def smoke():
 d.write(d.BASE/'focused_checks.json',focused_checks());c=d.config(d.rows()[0]);parent=parent_payload(c)
 m=model.build(c,'cpu');m.load_state_dict(parent['model'],strict=True);m.eval();before={k:v.clone() for k,v in m.state_dict().items()}
 with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
 if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Real original As20 default-head no-query smoke failed')
 d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',NN=2,source_IQ=0,target_query=False,checkpoint=c['parent_source']+'/final_ssdg.pth',training_initializer=True,same_source_contract=True,optimizer_updates=0))
 del m,before
 equivalent=ord128_equivalence();records=[];schedule=base.supcon.schedule;base.supcon.schedule=lambda ep:schedule(int(ep)+20)
 try:
  torch.manual_seed(6007);m=model.build(c,'cpu');m.load_state_dict(parent['model'],strict=True);m.train();initial={k:v.clone() for k,v in m.state_dict().items()};g=torch.Generator().manual_seed(6008)
  if sum(p.numel() for p in m.parameters())!=900722 or len(initial)!=206:raise ValueError('Fixed real F/default head shape differs')
  for step in (1,2):
   m.zero_grad(set_to_none=True);m._feature_l_steps=step;x=torch.randn((128,2,256),generator=g);ordinary=x+.01;y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(base.supcon.RX_BY_DOMAIN)[domains];meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=domains%3+1,sig_i=torch.arange(128)+2000,base_index=torch.arange(128))
   first,maps=base.capture_forward(m,ordinary,domains,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=ordinary.clone();ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
   extra,_=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(domains,meta),domains=domains,optimizer_step=step,identity_loss=ce,labeled_count=128,labeled_inputs=x)
   total=ce+extra;total.backward();head=m.id_backbone.encoder.core.id_backbone.cls_head.weight;sinc=m.id_backbone.encoder.core.id_backbone.sinc
   if not torch.isfinite(total) or head.grad is None or float(head.grad.abs().sum())<=0 or sum(float(p.grad.abs().sum()) for p in sinc.parameters() if p.grad is not None)<=0 or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise ValueError('Joint default head/Sinc/backbone finite gradient failed')
   r=m._full_records[-1];shape=m._shape_records[-1];expected_As=128 if step==1 else 0
   if m.id_backbone.encoder._full_train_packets!=step*768 or m._shape_NN!=step*256 or m._full_channel.packets!=128 or m._full_channel.calls!=1 or r['As_packets']!=expected_As or r['L_forwards']!=6 or r['SupCon_weight']!=.1 or shape['cumulative_step']!=4440+step or m._supcon_pending is not None or m._supcon_handles or any(not torch.equal(v,m.state_dict()[k]) for k,v in initial.items()) or getattr(m,'_source_unlabeled_feature_hook',None) is not None:raise ValueError('Actual128/0,6L,cumulativeSupCon/state/U cost differs')
   if abs(r['total_L']-float(total.detach()))>3e-6 or abs(r['total_L']-r['base_L_before_shape']-.25*shape['shape_RD'])>1e-6 or not shape['same_IQ_each_pair'] or (step==2 and r['mixed_ordinary_MSE']!=0):raise ValueError('1.25 loss/sameIQ/offordinary reuse differs')
   records.append(dict(step=step,NN=768,As_IQ=expected_As,cumulative_epoch=21,cumulative_step=4440+step,full_ordinary_SupCon=True,head_joint_gradient=True,Sinc_joint_gradient=True,optimizer_updates=0))
 finally:base.supcon.schedule=schedule
 d.write(d.BASE/'synthetic_smoke.json',dict(status='PASS',records=records,ORD128_equivalence=equivalent,NN=1536,real_checkpoint_NN=2,real_IQ=0,optimizer_updates=0,head_fit_updates=0,target_access=False))
 print('REAL_AS20_NO_QUERY_AND_TWO_STEPALT_F_SYNTHETIC_SMOKE PASS1538NN;continue training',flush=True)
 return dict(status='PASS',NN=1538)
