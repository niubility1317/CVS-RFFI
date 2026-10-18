"""Real own-parent no-query readout and actual6L probes; PASS directly trains."""
import torch
from . import design as d,model
from .weighted import grouped
from .checks import run as focused_checks
from .source import parent_payload
from experiments.cvs_crossview_supcon41 import model as base

def grouped_equivalence():
 g=torch.Generator().manual_seed(6010);first=torch.randn((128,6),generator=g,dtype=torch.float64,requires_grad=True);second=torch.randn((128,6),generator=g,dtype=torch.float64,requires_grad=True);y=torch.arange(128)%6;mask=(torch.arange(128)%2)==0
 original=base.terms(first,second,y,1.);equal,_,_=grouped(first,second,y,mask,base.terms,.25,.25)
 for actual,expected in zip(equal,original[:3]):
  if not torch.allclose(actual,.5*expected,atol=1e-12,rtol=1e-12):raise ValueError('Equal group means do not reproduce original CE/KL/RD')
 old_grad=torch.autograd.grad(.5*original[0],(first,second),retain_graph=True);equal_grad=torch.autograd.grad(equal[0],(first,second),retain_graph=True)
 if any(not torch.allclose(a,b,atol=1e-12,rtol=1e-12) for a,b in zip(old_grad,equal_grad)):raise ValueError('Original mixed RD gradient not reproduced')
 weighted,ordinary,selected=grouped(first,second,y,mask,base.terms)
 if not torch.allclose(weighted[0],weighted[1]+weighted[2],atol=1e-12,rtol=1e-12) or any(not torch.allclose(v,.125*a+.375*b,atol=1e-12,rtol=1e-12) for v,a,b in zip(weighted,ordinary,selected)):raise ValueError('CE and symmetricKL must share fixed group weights')
 torch.autograd.grad(weighted[0],(first,second))
 if model.relations is not base.relations:raise ValueError('Existing ORD128 implementation changed')
 return dict(status='PASS',equal_0p25_reproduces_original_loss_CE_KL_and_gradients=True,precision='float64',absolute_tolerance=1e-12,NN=0)

def smoke():
 d.write(d.BASE/'focused_checks.json',focused_checks());equivalent=grouped_equivalence();records=[];real_records=[];schedule=base.supcon.schedule;base.supcon.schedule=lambda ep:schedule(int(ep)+20)
 try:
  for row in d.rows():
   c=d.config(row);parent=parent_payload(c);m=model.build(c,'cpu');m.load_state_dict(parent['model'],strict=True);m.eval();before={k:v.clone() for k,v in m.state_dict().items()}
   with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
   if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Real own original20 default-head no-query smoke failed')
   real_records.append(dict(row_id=row['row_id'],NN=2,source_IQ=0,target_query=False,checkpoint=c['parent_source']+'/final_ssdg.pth',optimizer_updates=0));del m,before
   torch.manual_seed(6007);m=model.build(c,'cpu');m.load_state_dict(parent['model'],strict=True);m.train();initial={k:v.clone() for k,v in m.state_dict().items()};g=torch.Generator().manual_seed(6008)
   if sum(p.numel() for p in m.parameters())!=900722 or len(initial)!=206:raise ValueError('Fixed F/default head shape differs')
   m._feature_l_steps=1;x=torch.randn((128,2,256),generator=g);ordinary=x+.01;y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(base.supcon.RX_BY_DOMAIN)[domains];meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=domains%3+1,sig_i=torch.arange(128)+2000,base_index=torch.arange(128))
   first,maps=base.capture_forward(m,ordinary,domains,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=ordinary.clone();ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
   extra,info=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(domains,meta),domains=domains,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
   total=ce+extra;total.backward();head=m.id_backbone.encoder.core.id_backbone.cls_head.weight;sinc=m.id_backbone.encoder.core.id_backbone.sinc;sinc_ids={id(p) for p in sinc.parameters()}
   backbone_gradient=sum(float(p.grad.abs().sum()) for p in m.parameters() if p is not head and id(p) not in sinc_ids and p.grad is not None)
   if not torch.isfinite(total) or head.grad is None or float(head.grad.abs().sum())<=0 or sum(float(p.grad.abs().sum()) for p in sinc.parameters() if p.grad is not None)<=0 or backbone_gradient<=0 or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise ValueError('Joint default head/Sinc/backbone finite gradients failed')
   r=m._full_records[-1];shape=m._shape_records[-1];w=m._weight_records[-1];expected_As=64 if c['As'] else 0
   if m.id_backbone.encoder._full_train_packets!=768 or m._shape_NN!=256 or m._full_channel.packets!=expected_As or m._full_channel.calls!=1 or r['As_packets']!=expected_As or r['L_forwards']!=6 or r['SupCon_weight']!=.1 or shape['cumulative_step']!=4441 or m._supcon_pending is not None or m._supcon_handles or any(not torch.equal(v,m.state_dict()[k]) for k,v in initial.items()) or getattr(m,'_source_unlabeled_feature_hook',None) is not None:raise ValueError('Unchanged6L/selected64/actualAs64or0/defaultU budget differs')
   if r['selected_actual_view']!=('As_raw' if c['As'] else 'raw') or (not c['As'] and r['selected_raw_MSE']!=0.):raise ValueError('Original NoAs selected raw64 versus As(raw64) differs')
   from .schema import INFO_KEYS
   if set(info)!=set(INFO_KEYS) or info['concat/total_L']!=r['total_L'] or info['concat/L_forwards']!=6:raise ValueError('Actual emitted metric schema differs')
   expected=.5*r['common_RD']+r['mixed_weighted_RD']+.25*shape['shape_RD']+.1*r['SupCon']
   if abs(r['total_L']-float(total.detach()))>3e-6 or abs(r['total_L']-expected)>3e-6 or abs(r['mixed_weighted_RD']-.125*w['ordinary64_RD']-.375*w['selected64_RD'])>1e-6 or not shape['same_IQ_each_pair']:raise ValueError('Normalized1.25 sameIQ weighted loss differs')
   records.append(dict(row_id=row['row_id'],NN=768,As_IQ=expected_As,selected64=64,cumulative_epoch=21,cumulative_step=4441,full_ordinary_SupCon=True,head_joint_gradient=True,Sinc_joint_gradient=True,backbone_joint_gradient=True,optimizer_updates=0))
 finally:base.supcon.schedule=schedule
 d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',records=real_records,NN=4,real_IQ=0,target_query=False,optimizer_updates=0))
 d.write(d.BASE/'synthetic_smoke.json',dict(status='PASS',records=records,grouped_equivalence=equivalent,NN=1536,real_checkpoint_NN=4,total_NN=1540,real_IQ=0,optimizer_updates=0,head_fit_updates=0,target_access=False))
 print('REAL_OWN20_NO_QUERY_AND_TWO_ASWEIGHT_F_PROBES PASS1540NN;continue training',flush=True)
 return dict(status='PASS',NN=1540)
