"""Real own-parent no-query readout and actual6L probes; PASS directly trains."""
import torch
from . import design as d,model
from . import relations
from .checks import run as focused_checks
from .source import parent_payload
from experiments.cvs_crossview_supcon41 import model as base

def relation_equivalence():
 g=torch.Generator().manual_seed(6010);y=torch.arange(128)%6;rx=torch.tensor((1,3,4,6,8))[torch.arange(128)%5];mask=(torch.arange(128)%2)==0
 ordinary={level:torch.randn((128,3,7),generator=g,dtype=torch.float64,requires_grad=True) for level in relations.LEVELS}
 mixed={level:torch.randn((128,3,7),generator=g,dtype=torch.float64,requires_grad=True) for level in relations.LEVELS}
 new,detail=relations.penalty(ordinary,mixed,y,rx,mask,'MIXED128',actual_As=True)
 previous,old_detail=base.relations.penalty(mixed,mixed,y,rx,mask,'ORD128')
 if not torch.allclose(new,previous,atol=1e-12,rtol=1e-12):raise ValueError('Membership-only formula differs from existing ORD128 on same maps')
 gg=torch.autograd.grad(new,tuple(ordinary.values())+tuple(mixed.values()),retain_graph=True,allow_unused=True);old_grad=torch.autograd.grad(previous,tuple(mixed.values()))
 if any(v is not None for v in gg[:2]) or any(v is None or not torch.isfinite(v).all() or float(v.abs().sum())<=0 for v in gg[2:]) or any(not torch.allclose(a,b,atol=1e-12,rtol=1e-12) for a,b in zip(gg[2:],old_grad)):raise ValueError('Only existing mixed maps receive unchanged SupCon gradients')
 if any(r['total_anchors']!=128 or r['ordinary_anchors']!=64 or r['additional_anchors']!=0 or r['true_satellite_anchors']!=64 or r['same_RX_positive_pairs']!=0 for r in detail):raise ValueError('Single-input128 relation metadata differs')
 return dict(status='PASS',same_maps_original_ORD128_formula_and_gradients=True,mixed_maps_only=True,precision='float64',absolute_tolerance=1e-12,NN=0)

def smoke():
 d.write(d.BASE/'focused_checks.json',focused_checks());equivalent=relation_equivalence();records=[];real_records=[];schedule=base.supcon.schedule;base.supcon.schedule=lambda ep:schedule(int(ep)+20)
 try:
  for row in d.rows():
   c=d.config(row);parent=parent_payload(c);m=model.build(c,'cpu');m.load_state_dict(parent['model'],strict=True);m.eval();before={k:v.clone() for k,v in m.state_dict().items()}
   with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
   if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Real own original20 default-head no-query smoke failed')
   real_records.append(dict(row_id=row['row_id'],NN=2,source_IQ=0,target_query=False,checkpoint=c['parent_source']+'/final_ssdg.pth',optimizer_updates=0));del m,before
   torch.manual_seed(6007);m=model.build(c,'cpu');m.load_state_dict(parent['model'],strict=True);m.train();initial={k:v.clone() for k,v in m.state_dict().items()};g=torch.Generator().manual_seed(6008)
   if sum(p.numel() for p in m.parameters())!=900722 or len(initial)!=206:raise ValueError('Fixed F/default head shape differs')
   m._crossview_smoke_capture=True;m._feature_l_steps=1;x=torch.randn((128,2,256),generator=g);ordinary=x+.01;y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(base.supcon.RX_BY_DOMAIN)[domains];meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=domains%3+1,sig_i=torch.arange(128)+2000,base_index=torch.arange(128))
   first,maps=base.capture_forward(m,ordinary,domains,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=ordinary.clone();ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
   extra,info=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(domains,meta),domains=domains,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
   debug=m._crossview_debug;ordinary_maps=tuple(debug['ordinary0'].values())+tuple(debug['ordinary1'].values());mixed_maps=tuple(debug['satellite0'].values())+tuple(debug['satellite1'].values());route=torch.autograd.grad(debug['aux'],ordinary_maps+mixed_maps,allow_unused=True,retain_graph=True)
   if any(v is not None for v in route[:4]) or any(v is None or not torch.isfinite(v).all() or float(v.abs().sum())<=0 for v in route[4:]):raise ValueError('Real6L SupCon must use only corresponding existing mixed maps')
   total=ce+extra;total.backward();head=m.id_backbone.encoder.core.id_backbone.cls_head.weight;sinc=m.id_backbone.encoder.core.id_backbone.sinc;sinc_ids={id(p) for p in sinc.parameters()}
   backbone_gradient=sum(float(p.grad.abs().sum()) for p in m.parameters() if p is not head and id(p) not in sinc_ids and p.grad is not None)
   if not torch.isfinite(total) or head.grad is None or float(head.grad.abs().sum())<=0 or sum(float(p.grad.abs().sum()) for p in sinc.parameters() if p.grad is not None)<=0 or backbone_gradient<=0 or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise ValueError('Joint default head/Sinc/backbone finite gradients failed')
   r=m._full_records[-1];shape=m._shape_records[-1];contrast=m._contrast_records[-1];expected_As=64 if c['As'] else 0
   if m.id_backbone.encoder._full_train_packets!=768 or m._shape_NN!=256 or m._full_channel.packets!=expected_As or m._full_channel.calls!=1 or r['As_packets']!=expected_As or r['L_forwards']!=6 or r['SupCon_weight']!=.1 or shape['cumulative_step']!=4441 or m._supcon_pending is not None or m._supcon_handles or any(not torch.equal(v,m.state_dict()[k]) for k,v in initial.items()) or getattr(m,'_source_unlabeled_feature_hook',None) is not None:raise ValueError('Unchanged6L/selected64/actualAs64or0/defaultU budget differs')
   if r['selected_actual_view']!=('As_raw' if c['As'] else 'raw') or (not c['As'] and r['selected_raw_MSE']!=0.):raise ValueError('Original NoAs selected raw64 versus As(raw64) differs')
   from .schema import INFO_KEYS
   if set(info)!=set(INFO_KEYS) or info['concat/total_L']!=r['total_L'] or info['concat/L_forwards']!=6:raise ValueError('Actual emitted metric schema differs')
   expected=.5*r['common_RD']+.5*r['extra_RD']+.25*shape['shape_RD']+.1*r['SupCon']
   if abs(r['total_L']-float(total.detach()))>3e-6 or abs(r['total_L']-expected)>3e-6 or not shape['same_IQ_each_pair']:raise ValueError('Original1.25 sameIQ loss differs')
   if contrast['mode']!='MIXED128' or contrast['anchor_positions']!=list(range(128)) or contrast['anchors_per_term']!=128 or contrast['anchors_per_physical_input']!=1 or contrast['map_terms']!=4 or len(contrast['pass_maps'])!=4 or contrast['true_As_anchors_per_term']!=expected_As or contrast['extra_NN'] or contrast['ordinary_maps_used'] or contrast['raw_maps_used']:raise ValueError('Existing mixed128 one-position membership differs')
   records.append(dict(row_id=row['row_id'],NN=768,As_IQ=expected_As,selected64=64,cumulative_epoch=21,cumulative_step=4441,mixed128_SupCon=True,anchors_per_input_position=1,map_terms=4,ordinary_maps_SupCon_gradient=False,mixed_maps_SupCon_gradient=True,head_joint_gradient=True,Sinc_joint_gradient=True,backbone_joint_gradient=True,optimizer_updates=0))
 finally:base.supcon.schedule=schedule
 d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',records=real_records,NN=4,real_IQ=0,target_query=False,optimizer_updates=0))
 d.write(d.BASE/'synthetic_smoke.json',dict(status='PASS',records=records,relation_equivalence=equivalent,NN=1536,real_checkpoint_NN=4,total_NN=1540,real_IQ=0,optimizer_updates=0,head_fit_updates=0,target_access=False))
 print('REAL_OWN20_NO_QUERY_AND_TWO_MIXEDCONTRAST_F_PROBES PASS1540NN;continue training',flush=True)
 return dict(status='PASS',NN=1540)
