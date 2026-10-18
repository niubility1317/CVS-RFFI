"""One real fixed FINAL30 no-query smoke, then actual F synthetic continuation hooks."""
import torch
from . import design as d,model
from .checks import run as focused_checks
from .source import parent_payload,ancestor_payload
from experiments.cvs_crossview_supcon41 import model as base
from experiments.cvs_noise_shape_factorial43 import model as original

def smoke():
 d.write(d.BASE/'focused_checks.json',focused_checks())
 for row in d.rows():parent_payload(d.config(row))
 c=d.config(d.rows()[0]);old=model.original_config(c);m=original.own_model(old,'cpu');before={k:v.clone() for k,v in m.state_dict().items()}
 with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
 if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Real fixed FINAL30 default-head smoke failed')
 d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',NN=2,source_IQ=0,target_query=False,checkpoint=c['parent_source']+'/final_ssdg.pth',training_initializer=True,same_source_contract=True,optimizer_updates=0))
 del m,before
 initial=None;records=[];schedule=base.supcon.schedule;base.supcon.schedule=lambda epoch:schedule(int(epoch)+30)
 try:
  for row in d.rows():
   torch.manual_seed(6007);c=d.config(row);m=model.build(c,'cpu').train();current={k:v.clone() for k,v in m.state_dict().items()}
   if initial is None:initial=current
   if sum(p.numel() for p in m.parameters())!=900722 or len(current)!=206 or any(not torch.equal(v,initial[k]) for k,v in current.items()):raise ValueError('Same F/default-head graph differs across fixed pair')
   m._feature_l_steps=1;g=torch.Generator().manual_seed(6008);x=torch.randn((128,2,256),generator=g);y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(base.supcon.RX_BY_DOMAIN)[domains];meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=domains%3+1,sig_i=torch.arange(128)+2000,base_index=torch.arange(128))
   first,maps=base.capture_forward(m,x,domains,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=x.clone();ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
   extra,_=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(domains,meta),domains=domains,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
   total=ce+extra;total.backward();head=m.id_backbone.encoder.core.id_backbone.cls_head.weight;sinc=m.id_backbone.encoder.core.id_backbone.sinc
   if not torch.isfinite(total) or head.grad is None or float(head.grad.abs().sum())<=0 or sum(float(p.grad.abs().sum()) for p in sinc.parameters() if p.grad is not None)<=0 or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise ValueError('Joint F/defaulthead/Sinc finite gradient failed')
   r=m._full_records[-1];shape=m._shape_records[-1]
   if m.id_backbone.encoder._full_train_packets!=768 or m._shape_NN!=256 or m._full_channel.packets!=(64 if c['As'] else 0) or r['As_packets']!=(64 if c['As'] else 0) or r['L_forwards']!=6 or r['SupCon_weight']!=.1 or shape['cumulative_step']!=6661 or m._supcon_pending is not None or m._supcon_handles or any(not torch.equal(v,m.state_dict()[k]) for k,v in current.items()) or getattr(m,'_source_unlabeled_feature_hook',None) is not None:raise ValueError('SixL/cumulativechannel/fullSupCon/defaultU cost differs')
   if abs(r['total_L']-float(total.detach()))>3e-6 or abs(r['total_L']-r['base_L_before_shape']-.25*shape['shape_RD'])>1e-6 or not shape['same_IQ_each_pair']:raise ValueError('Normalized1.25 sameIQ RDrop objective differs')
   if c['As']:
    if m._full_channel.obj.last_namespace!='FcrossRX/train/e31/step6661/L_S':raise ValueError('Channel namespace differs')
   records.append(dict(arm=row['arm'],NN=768,As_IQ=r['As_packets'],Ag_IQ=shape['applied'],full_SupCon_at_first_step=True,cumulative_epoch=31,cumulative_step=6661,head_joint_gradient=True,Sinc_joint_gradient=True,optimizer_updates=0))
 finally:base.supcon.schedule=schedule
 d.write(d.BASE/'synthetic_smoke.json',dict(status='PASS',records=records,NN=1536,real_checkpoint_NN=2,real_IQ=0,optimizer_updates=0,head_fit_updates=0,target_access=False))
 print('REAL_FINAL30_CP_NO_QUERY+TWO_REAL_F_SYNTHETIC_SMOKE PASS1538NN;continue training',flush=True)
 return dict(status='PASS',NN=1538)
