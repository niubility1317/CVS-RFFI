"""Real-parent synthetic no-query L/U/VJP smoke; PASS directly trains."""
import torch
from . import design as d,model,core
from .checks import run as focused_checks
from .source import parent_payload
from experiments.cvs_crossview_supcon41 import model as p41
from experiments.cvs_rdrop_sinc_units import optimizer as units
def smoke():
 from .core_checks import run as core_checks,actual_torch_smoke
 d.write(d.BASE/'focused_checks.json',focused_checks());math=core_checks();toy=actual_torch_smoke();d.write(d.BASE/'optimizer_toy_smoke.json',toy);records=[];real_records=[];schedule=p41.supcon.schedule;p41.supcon.schedule=lambda ep:schedule(int(ep)+20)
 try:
  for row in d.rows():
   c=d.config(row);parent=parent_payload(c);m=model.build(c,'cpu');m.load_state_dict(parent['model'],strict=True);m.eval();before={k:v.clone() for k,v in m.state_dict().items()}
   with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
   if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Real own20 default-head no-query smoke failed')
   real_records.append(dict(row_id=row['row_id'],NN=2,source_IQ=0,target_query=False,checkpoint=c['parent_source']+'/final_ssdg.pth',optimizer_updates=0));del m,before
   torch.manual_seed(6007);m=model.build(c,'cpu');m.load_state_dict(parent['model'],strict=True);m.train();initial={k:v.clone() for k,v in m.state_dict().items()};g=torch.Generator().manual_seed(6008)
   opt=units.build(m,m.parameters(),'CENTER_UNITS',lr=2e-4,weight_decay=1e-4);core.attach_optimizer(opt,m,amp=False);opt.zero_grad(set_to_none=True)
   m._feature_l_steps=1;x=torch.randn((128,2,256),generator=g);ordinary=x+.01;y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(p41.supcon.RX_BY_DOMAIN)[domains];meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=domains%3+1,sig_i=torch.arange(128)+2000,base_index=torch.arange(128))
   first,maps=p41.capture_forward(m,ordinary,domains,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=ordinary.clone();ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
   extra,info=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(domains,meta),domains=domains,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
   # Synthetic U exercises the real teacher and student with hidden TX. This is
   # a smoke tensor only; production keeps the native old-U gate/loss verbatim.
   teacher=model.build(c,'cpu');teacher.load_state_dict(parent['ema_model'],strict=True);teacher.eval();u=torch.randn((8,2,256),generator=g)
   with torch.no_grad():tlogits=teacher.forward_identity_only(u)['tx_logits'];conf,pseudo=tlogits.softmax(-1).max(-1)
   strong=u+.015*torch.randn((8,2,256),generator=g);ulogits=m.forward_identity_only(strong)['tx_logits'];mask=conf>=.95;uprob=ulogits.softmax(-1);entropy=-(uprob*uprob.clamp_min(1e-12).log()).sum(-1).mean();pseudo_loss=torch.nn.functional.cross_entropy(ulogits[mask],pseudo[mask]) if bool(mask.any()) else ulogits.sum()*0.;U=.16*pseudo_loss+.01*entropy;total=ce+extra+U
   component=m._sa_as_pending[1]
   m._source_optimizer_update_probe_hook(model=m,optimizer=opt,epoch=0,batch_idx=0,phase='pseudo',optimizer_step=1,total_loss=total,unlabeled_loss=U,labeled_count=128,unlabeled_count=8)
   if m._sa_as_pending is not None or m._sa_vjp_calls!=2 or m._sa_native_records[-1]['total_loss']!=float(total.detach()):raise ValueError('Actual native full L+synthetic U capture failed')
   total.backward();params=list(m.parameters());norm=units.clip_physical(opt,params,5.);core.after_clip(m,params,5.,norm)
   head=m.id_backbone.encoder.core.id_backbone.cls_head.weight;sinc=m.id_backbone.encoder.core.id_backbone.sinc;sinc_ids={id(p) for p in sinc.parameters()};backbone=sum(float(p.grad.abs().sum()) for p in m.parameters() if p is not head and id(p) not in sinc_ids and p.grad is not None)
   if not torch.isfinite(total) or head.grad is None or float(head.grad.abs().sum())<=0 or sum(float(p.grad.abs().sum()) for p in sinc.parameters() if p.grad is not None)<=0 or backbone<=0 or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise ValueError('Joint default head/Sinc/backbone finite gradients failed')
   r=m._full_records[-1];shape=m._shape_records[-1];component_record=m._sa_component_records[-1];expected_As=64 if c['As'] else 0
   if m.id_backbone.encoder._full_train_packets!=776 or teacher.id_backbone.encoder._full_eval_packets!=8 or m._shape_NN!=256 or m._full_channel.packets!=expected_As or m._full_channel.calls!=1 or r['As_packets']!=expected_As or r['L_forwards']!=6 or r['SupCon_weight']!=.1 or shape['cumulative_step']!=4441 or m._supcon_pending is not None or m._supcon_handles or any(not torch.equal(v,m.state_dict()[k]) for k,v in initial.items()) or getattr(m,'_source_unlabeled_feature_hook',None) is not None or m._crossview_mode!='ORD128':raise ValueError('Original6L/As64/ORD128/defaultU budget differs')
   from .schema import INFO_KEYS
   expected=.5*r['common_RD']+.5*r['extra_RD']+.25*shape['shape_RD']+.1*r['SupCon']
   if set(info)!=set(INFO_KEYS) or info['concat/total_L']!=r['total_L'] or abs(r['total_L']-float((ce+extra).detach()))>3e-6 or abs(r['total_L']-expected)>3e-6 or not shape['same_IQ_each_pair'] or component_record['native_RD_calls']!=3 or component_record['extra_NN'] or (not c['As'] and float(component.detach())!=0.):raise ValueError('Unchanged original1.25L/sameIQ loss or captured component differs')
   if m._sa_steps or m._sa_clips!=1 or m._sa_ready is None or m._sa_pending is not None or opt._sinc_unit_steps or opt._sinc_physical_clip_calls!=1 or opt.state:raise ValueError('No actual optimizer update or moment mutation permitted in real smoke')
   records.append(dict(row_id=row['row_id'],NN=784,L_NN=768,U_student_NN=8,U_teacher_NN=8,As_IQ=expected_As,cumulative_epoch=21,cumulative_step=4441,ORD128_SupCon=True,old_U_production_hook_absent=True,synthetic_U_TX_truth=False,native_full_loss_capture=True,mode=c['optimizer_mode'],extra_VJP_calls=2,one_physical_clip=True,head_joint_gradient=True,Sinc_joint_gradient=True,backbone_joint_gradient=True,optimizer_updates=0,As_RD=float(component.detach())))
   del m,teacher,opt,initial
 finally:p41.supcon.schedule=schedule
 d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',records=real_records,NN=6,real_IQ=0,target_query=False,optimizer_updates=0))
 d.write(d.BASE/'synthetic_smoke.json',dict(status='PASS',records=records,optimizer_math=math,optimizer_toy=toy,NN=2352,real_checkpoint_NN=6,total_NN=2358,extra_VJP_calls=6,real_IQ=0,optimizer_updates=0,head_fit_updates=0,target_access=False))
 print('REAL_OWN20_NO_QUERY_AND_THREE_SPLIT_ADAM_F_PROBES PASS2358NN;continue training',flush=True);return dict(status='PASS',NN=2358)
