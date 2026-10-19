"""Real ownAs20/default head and frozen CSIL; full first/last-step no-query smoke."""
import torch
from . import design as d,model
from .source import parent_payload
base=model.original.base
def smoke():
 records=[];old=base.supcon.schedule;base.supcon.schedule=lambda ep:old(int(ep)+20)
 try:
  for row,step in [(r,1) for r in d.rows()]+[(d.rows()[-1],2220)]:
   c=d.config(row);a=d.make_args(c,'cpu');parent=parent_payload(c);torch.manual_seed(1119);m=model.build(c,'cpu');m.load_state_dict(parent['model'],strict=True);m.eval()
   with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
   if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Real As20 defaulthead smoke failed')
   m.train();before={k:v.clone() for k,v in m.state_dict().items()};m._feature_l_steps=step;m._full_calls=step-1;g=torch.Generator().manual_seed(1120);x=torch.randn((128,2,256),generator=g);ordinary=x+.01;y=(torch.arange(128)//2)%6;dom=torch.arange(128)%15;rx=torch.tensor(base.supcon.RX_BY_DOMAIN)[dom];meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=dom%3+1,sig_i=torch.arange(128)+2000,base_index=torch.arange(128))
   first,maps=base.capture_forward(m,ordinary,dom,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=ordinary.clone();ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
   delta,info=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(dom,meta),domains=dom,optimizer_step=step,identity_loss=ce,labeled_count=128,labeled_inputs=x);total=ce+delta;total.backward();teacher=m.__dict__['_class_teacher']
   if len(m.state_dict())!=206 or sum(p.numel() for p in m.parameters())!=900722 or m.id_backbone.encoder._full_train_packets!=768 or m._distill_NN!=64 or m._full_channel.packets!=64:raise ValueError('Six student forwards and64 teacher budget differs')
   if teacher.training or any(p.requires_grad or p.grad is not None for p in teacher.parameters()) or any(not torch.equal(v,teacher.state_dict()[k].cpu()) for k,v in m.__dict__['_class_teacher_initial'].items()):raise ValueError('Teacher updated or entered optimizer')
   if any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()) or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise ValueError('Nonfinite gradient or state update')
   head=m.id_backbone.encoder.core.id_backbone.cls_head.weight
   if head.grad is None or not bool(head.grad.abs().sum()>0) or not torch.isfinite(total):raise ValueError('Defaulthead joint learning failed')
   r=m._full_records[-1];shape=m._shape_records[-1];rec=m._distill_records[-1]
   expected=.5*r['common_RD']+.5*r['extra_RD']+.25*shape['shape_RD']+.1*r['SupCon']+(.25*6*rec['KD'] if c['distill_mode']=='CSIL' else 0)
   if abs(float(total)-expected)>2e-5 or not rec['same_actual_IQ'] or not rec['all_As_CE']:raise ValueError('Full formula or same-IQ transfer differs')
   records.append(dict(row=row['row_id'],step=step,student_NN=770,teacher_NN=64,updates=0,target_access=False,defaulthead_joint=True))
 finally:base.supcon.schedule=old
 d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',records=records,NN=2502,optimizer_updates=0,query_access=False));print('SAME_IQ_CSIL REAL_CP_FULL_PATH_SMOKE PASS2502NN;continue',flush=True)
