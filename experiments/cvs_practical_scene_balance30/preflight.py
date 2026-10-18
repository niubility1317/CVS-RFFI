"""Real own-parent no-query readout and actual6L probes; PASS directly trains."""
import torch
from . import design as d,model
from .checks import run as focused_checks
from .source import parent_payload
from experiments.cvs_crossview_supcon41 import model as base

def smoke():
 d.write(d.BASE/'focused_checks.json',focused_checks());records=[];real_records=[];schedule=base.supcon.schedule;base.supcon.schedule=lambda ep:schedule(int(ep)+20)
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
   if any(v is None or not torch.isfinite(v).all() or float(v.abs().sum())<=0 for v in route[:4]) or any(v is not None for v in route[4:]):raise ValueError('Unchanged ORD128 SupCon must use only original ordinary pass maps')
   total=ce+extra;total.backward();head=m.id_backbone.encoder.core.id_backbone.cls_head.weight;sinc=m.id_backbone.encoder.core.id_backbone.sinc;sinc_ids={id(p) for p in sinc.parameters()}
   backbone_gradient=sum(float(p.grad.abs().sum()) for p in m.parameters() if p is not head and id(p) not in sinc_ids and p.grad is not None)
   if not torch.isfinite(total) or head.grad is None or float(head.grad.abs().sum())<=0 or sum(float(p.grad.abs().sum()) for p in sinc.parameters() if p.grad is not None)<=0 or backbone_gradient<=0 or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise ValueError('Joint default head/Sinc/backbone finite gradients failed')
   r=m._full_records[-1];shape=m._shape_records[-1];balance=m._scene_balance_records[-1];expected_As=64 if c['As'] else 0
   if m.id_backbone.encoder._full_train_packets!=768 or m._shape_NN!=256 or m._full_channel.packets!=expected_As or m._full_channel.calls!=1 or r['As_packets']!=expected_As or r['L_forwards']!=6 or r['SupCon_weight']!=.1 or shape['cumulative_step']!=4441 or m._supcon_pending is not None or m._supcon_handles or any(not torch.equal(v,m.state_dict()[k]) for k,v in initial.items()) or getattr(m,'_source_unlabeled_feature_hook',None) is not None:raise ValueError('Unchanged6L/selected64/actualAs64or0/defaultU budget differs')
   from .schema import INFO_KEYS
   if set(info)!=set(INFO_KEYS) or info['concat/total_L']!=r['total_L'] or info['concat/L_forwards']!=6:raise ValueError('Actual emitted metric schema differs')
   expected=.5*r['common_RD']+.5*r['extra_RD']+.25*shape['shape_RD']+.1*r['SupCon']
   if abs(r['total_L']-float(total.detach()))>3e-6 or abs(r['total_L']-expected)>3e-6 or not shape['same_IQ_each_pair']:raise ValueError('Original1.25 sameIQ loss differs')
   expected_counts=[22,21,21]
   if r['crossview_mode']!='ORD128' or [balance['scene_counts'][s] for s in d.TRAIN_SCENES]!=expected_counts or balance['extra_scene_index']!=0 or balance['selected_packets']!=64 or balance['As_packets']!=64 or balance['assignment_permutations']!=1 or balance['selected_L_positions']!=list(range(0,128,2)) or balance['extra_NN']:raise ValueError('Actual one-call balanced64/E21phase/unchangedmask differs')
   records.append(dict(row_id=row['row_id'],NN=768,As_IQ=expected_As,selected64=64,cumulative_epoch=21,cumulative_step=4441,full_ordinary_SupCon=True,scene_counts=balance['scene_counts'],assignment_permutations=1,extra_NN=0,head_joint_gradient=True,Sinc_joint_gradient=True,backbone_joint_gradient=True,optimizer_updates=0))
 finally:base.supcon.schedule=schedule
 d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',records=real_records,NN=2,real_IQ=0,target_query=False,optimizer_updates=0))
 d.write(d.BASE/'synthetic_smoke.json',dict(status='PASS',records=records,NN=768,real_checkpoint_NN=2,total_NN=770,real_IQ=0,optimizer_updates=0,head_fit_updates=0,target_access=False))
 print('REAL_OWN20_NO_QUERY_AND_ONE_SCENEBALANCE_F_PROBE PASS770NN;continue training',flush=True)
 return dict(status='PASS',NN=770)
