"""Real original As20 default-head smoke and exact-six-forward synthetic paths."""
import torch
from types import SimpleNamespace
from . import design as d,model
from .source import parent_payload
from .checks import run as focused_checks
from experiments.cvs_crossview_supcon41 import model as base

def smoke():
 d.write(d.BASE/'focused_checks.json',focused_checks());records=[];schedule=base.supcon.schedule;base.supcon.schedule=lambda ep:schedule(int(ep)+20)
 try:
  parent=parent_payload(d.config(d.rows()[0]))
  for row in d.rows():
   c=d.config(row);m=model.build(c,'cpu');m.load_state_dict(parent['model'],strict=True);m.eval();before={k:v.clone() for k,v in m.state_dict().items()}
   with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
   if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Real original As20 no-query smoke failed')
   del m,before;torch.manual_seed(6007);m=model.build(c,'cpu');m.load_state_dict(parent['model'],strict=True);m.train();initial={k:v.clone() for k,v in m.state_dict().items()};g=torch.Generator().manual_seed(6008)
   m._feature_l_steps=1;m._crossview_smoke_capture=True;x=torch.randn((128,2,256),generator=g);y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(base.supcon.RX_BY_DOMAIN)[domains];meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=domains%3+1,sig_i=torch.arange(128)+2000,base_index=torch.arange(128))
   from experiments.cvs_phase1_overlay.model import native_modules
   native=native_modules();parent_args=SimpleNamespace(**parent['args']);aug_cfg=native.build_aug_base_cfg(parent_args);aug=native.make_augmentor(aug_cfg);native.configure_augmentor_for_epoch(aug,aug_cfg,13,parent_args);ordinary=aug(x,labels=y,no_pa=False)
   first,maps=base.capture_forward(m,ordinary,domains,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=ordinary.clone();ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
   extra,info=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(domains,meta),domains=domains,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
   total=ce+extra;total.backward();head=m.id_backbone.encoder.core.id_backbone.cls_head.weight;sinc=m.id_backbone.encoder.core.id_backbone.sinc
   if not torch.isfinite(total) or head.grad is None or float(head.grad.abs().sum())<=0 or sum(float(p.grad.abs().sum()) for p in sinc.parameters() if p.grad is not None)<=0 or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise ValueError('Joint default head/Sinc/backbone finite gradients required')
   r=m._full_records[-1];shape=m._shape_records[-1]
   if m.id_backbone.encoder._full_train_packets!=768 or m._shape_NN!=256 or m._full_channel.packets!=64 or m._full_channel.calls!=1 or len(m._full_channel.records)!=1 or r['As_packets']!=64 or r['L_forwards']!=6 or r['SupCon_weight']!=.1 or shape['cumulative_step']!=4441 or m._supcon_pending is not None or m._supcon_handles or any(not torch.equal(v,m.state_dict()[k]) for k,v in initial.items()) or getattr(m,'_source_unlabeled_feature_hook',None) is not None:raise ValueError('Exact6L/fullAs64/originalU budget differs')
   expected=.5*r['common_RD']+.5*r['extra_RD']+.25*shape['shape_RD']+.1*r['SupCon']
   if abs(r['total_L']-float(total.detach()))>3e-6 or abs(r['total_L']-expected)>3e-6 or not shape['same_IQ_each_pair']:raise ValueError('Original loss or sameIQ RDrop differs')
   # Actual final cumulative epoch boundary:channel only,no extra network/update.
   finalmeta={k:v[:64] for k,v in meta.items()};m._full_channel.apply(x[:64],finalmeta,epoch=10,step=2220)
   records.append(dict(row_id=row['row_id'],NN=770,real_checkpoint_NN=2,synthetic6L_NN=768,synthetic_As_generated=128,masked=sum(q['enabled'] for q in m._full_channel.records[0]['records']),first_cumulative_step=4441,last_cumulative_step=6660,last_cumulative_epoch=30,default_head_joint_gradients=True,same_IQ_RDrop=True,no_new_loss=True,optimizer_updates=0,real_IQ=0,target_query=False))
 finally:base.supcon.schedule=schedule
 d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',records=records,NN=1540,real_checkpoint_loads=1,source_IQ=0,target_query=False,optimizer_updates=0,head_fit_updates=0))
 print('REAL_AS20_NO_QUERY_AND_MASK6L PASS;continue training',flush=True)
 return dict(status='PASS',NN=1540)
