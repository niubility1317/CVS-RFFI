"""One real parent no-query + two actual6L/nativeU probes;PASS immediately trains."""
from copy import deepcopy
from types import SimpleNamespace
import torch
from . import design as d,model
from .source import parent_payload
from .schema import INFO_KEYS
from .u_scope import NativeUScope
from .u_routing import Router
from experiments.cvs_crossview_supcon41 import model as base

def router_boundary_toys():
 """No model/channel queries:exercise production batch sizes even if native mask is empty."""
 class ToyChannel:
  def __init__(self):self.calls=0;self.packets=0
  def apply(self,x,meta,epoch,step):
   self.calls+=1;self.packets+=len(x)
   observed=[dict(sample_id=tuple(int(meta[k][i]) for k in ('eq_i','rx_i','day_i','sig_i','base_index')),scene='practical_low_urban',session_id='synthetic',namespace='synthetic_nonNN',state_end='LOS',receiver_lock_status='LOCKED') for i in range(len(x))]
   return x+7,observed
  def summary(self):return dict(calls=self.calls,packets=self.packets,synthetic_only=True)
 checks=[]
 for row in d.rows():
  for n in (256,124,8):
   for kind in ('empty','single','sparse','all'):
    raw=torch.arange(n*512,dtype=torch.float32).reshape(n,2,256)/1000;strong=raw+.02
    mask=torch.zeros(n,dtype=torch.bool)
    if kind=='single':mask[-1]=True
    elif kind=='sparse':mask[::3]=True
    elif kind=='all':mask[:]=True
    meta=dict(eq_i=torch.ones(n,dtype=torch.long),rx_i=torch.tensor(d.RX)[torch.arange(n)%5],day_i=torch.arange(n)%3+1,sig_i=torch.arange(n)+2000,base_index=torch.arange(n));channel=ToyChannel();router=Router(d.config(row),channel)
    student,r=router.route(raw,strong,meta,mask,epoch=21,step=4441);positions=r['selected_positions'];k=int(mask.sum())
    if r['qualification_count']!=k or r['selected_slots']!=k or r['As_packets']!=k or channel.calls!=int(k>0) or channel.packets!=k or (student is strong)!=(k==0) or not torch.equal(student[r['Gaussian_positions']],strong[r['Gaussian_positions']]):raise ValueError('Synthetic zero/all/last-batch route differs')
    if positions and not torch.equal(student[positions],raw[positions]+7):raise ValueError('Synthetic route must receive raw rather than Gaussian IQ')
    if row['arm']=='Q01' and positions!=torch.nonzero(mask,as_tuple=False).reshape(-1).tolist():raise ValueError('Q must overwrite exactly the existing qualified positions')
    checks.append(dict(row_id=row['row_id'],batch=n,mask=kind,qualification_count=k,As_packets=k,channel_calls=channel.calls))
 return dict(status='PASS',NN=0,real_IQ=0,practical_simulations=0,cases=checks)

def smoke():
 from experiments.cvs_phase1_overlay.model import native_modules
 native=native_modules();records=[];real=[];parents={};schedule=base.supcon.schedule;base.supcon.schedule=lambda ep:schedule(int(ep)+20)
 try:
  for row in d.rows():
   c=d.config(row);args=d.make_args(c,'cpu');ref=c['reference_row_id']
   if ref not in parents:parents[ref]=parent_payload(c)
   parent=parents[ref];torch.manual_seed(6007);m=model.build(c,'cpu');m.load_state_dict(parent['model'],strict=True);m.eval();initial={k:v.clone() for k,v in m.state_dict().items()}
   if ref not in {v['reference_row_id'] for v in real}:
    with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
    if z.shape!=(2,6) or not torch.isfinite(z).all():raise ValueError('Actual corresponding own20 defaulthead no-query smoke failed')
    real.append(dict(reference_row_id=ref,NN=2,real_IQ=0,checkpoint=c['parent_source']+'/final_ssdg.pth',optimizer_updates=0))
   m.train();g=torch.Generator().manual_seed(6008);m._crossview_smoke_capture=True;m._feature_l_steps=1
   if sum(p.numel() for p in m.parameters())!=900722 or len(initial)!=206:raise ValueError('Fixed F/default head shape differs')
   x=torch.randn((128,2,256),generator=g);ordinary=x+.01;y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(base.supcon.RX_BY_DOMAIN)[domains];meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=domains%3+1,sig_i=torch.arange(128)+2000,base_index=torch.arange(128))
   first,maps=base.capture_forward(m,ordinary,domains,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=ordinary.clone();ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
   extra,info=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(domains,meta),domains=domains,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
   debug=m._crossview_debug;ordmaps=tuple(debug['ordinary0'].values())+tuple(debug['ordinary1'].values());mixmaps=tuple(debug['satellite0'].values())+tuple(debug['satellite1'].values());route=torch.autograd.grad(debug['aux'],ordmaps+mixmaps,allow_unused=True,retain_graph=True)
   if any(v is None or not torch.isfinite(v).all() or float(v.abs().sum())<=0 for v in route[:4]) or any(v is not None for v in route[4:]):raise ValueError('Original ORD128 must use only ordinary pass maps')
   total=ce+extra;total.backward();head=m.id_backbone.encoder.core.id_backbone.cls_head.weight;sinc=m.id_backbone.encoder.core.id_backbone.sinc;sinc_ids={id(p) for p in sinc.parameters()};backbone_gradient=sum(float(p.grad.abs().sum()) for p in m.parameters() if p is not head and id(p) not in sinc_ids and p.grad is not None)
   if not torch.isfinite(total) or head.grad is None or float(head.grad.abs().sum())<=0 or sum(float(p.grad.abs().sum()) for p in sinc.parameters() if p.grad is not None)<=0 or backbone_gradient<=0 or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise ValueError('Default head/Sinc/backbone joint finite gradients failed')
   r=m._full_records[-1];sh=m._shape_records[-1];As=64 if c['As'] else 0
   if m.id_backbone.encoder._full_train_packets!=768 or m._shape_NN!=256 or m._full_channel.packets!=As or m._full_channel.calls!=1 or r['As_packets']!=As or r['L_forwards']!=6 or sh['cumulative_step']!=4441 or m._supcon_pending is not None or m._supcon_handles or set(info)!=set(INFO_KEYS) or info['concat/total_L']!=r['total_L']:raise ValueError('Original6L/native total/cumulative/channel budget differs')
   if abs(r['total_L']-float(total.detach()))>3e-6 or abs(r['total_L']-(.5*r['common_RD']+.5*r['extra_RD']+.25*sh['shape_RD']+.1*r['SupCon']))>3e-6 or not sh['same_IQ_each_pair']:raise ValueError('Original1.25 sameIQ loss differs')
   teacher=model.build(c,'cpu');teacher.load_state_dict(parent['ema_model'],strict=True);teacher.eval().requires_grad_(False);t0=teacher.id_backbone.encoder._full_eval_packets
   o=SimpleNamespace(model=m,pending_U=None,u_records=[],u_router=Router(c),smoke_rng_check=True);scope=NativeUScope(o,native);scope.install();scope.teacher_created(teacher)
   try:
    ux=x[:8].clone();ud=domains[:8].clone();um={k:v[:8].clone() for k,v in meta.items()};um['tx_label_visible']=torch.zeros(8,dtype=torch.bool);o.pending_U=dict(meta=um,count=8,hidden_y=True)
    with torch.no_grad():
     weak=teacher(ux,y_tx=None,grl_lambda=1.,return_aux=True,domain_labels=ud);conf,pseudo=weak['tx_logits'].softmax(1).max(1);cm=native._threshold_mask(conf,ud,args);tm=native._temporal_mask_tensor(pseudo,conf,(ud,um),args,torch.device('cpu'));mask=cm&tm
    student_iq=native._strong_augment(ux,args.strong_noise_std);out=m(student_iq,y_tx=None,grl_lambda=1.,return_aux=True,domain_labels=ud);logits=out['tx_logits'];lu=torch.nn.functional.cross_entropy(logits[mask],pseudo[mask]) if bool(mask.any()) else logits.sum()*0.;pr=logits.softmax(1);ent=-(pr*pr.clamp_min(1e-8).log()).sum(1).mean();feature_hook=getattr(m,'_source_unlabeled_feature_hook',None)
    if feature_hook is None or not bool((torch.full((8,),-1,dtype=torch.long)==-1).all()):raise ValueError('Actual native hiddenU model feature hook must exist')
    adjustment,feature_metrics=feature_hook(model=m,teacher=teacher,student_output=out,teacher_output=weak,extra=(ud,um),domains=ud,optimizer_step=1,identity_loss=total,pseudo_loss=lu,entropy_loss=ent)
    if adjustment is not None or feature_metrics:raise ValueError('Read-only native U hook must add no loss/node/metrics')
    uobjective=args.lambda_u*lu+args.lambda_ent*ent;uobjective.backward();ur=o.u_records[0];expectedUAs=int(mask.sum());qualified=torch.nonzero(mask,as_tuple=False).reshape(-1).tolist()
    if ur['As_packets']!=expectedUAs or ur['selected_slots']!=expectedUAs or ur['qualification_count']!=expectedUAs or ur['qualification_positions']!=qualified or ur['teacher_NN']!=8 or ur['student_NN']!=8 or ur['CE_denominator']!=expectedUAs or ur['entropy_denominator']!=8 or not ur['teacher_qualification_completed_before_strong'] or o.pending_U is not None or o.u_router.channel.packets!=expectedUAs or o.u_router.channel.calls!=int(expectedUAs>0) or not torch.isfinite(uobjective) or teacher.id_backbone.encoder._full_eval_packets-t0!=8 or m.id_backbone.encoder._full_train_packets!=776:raise ValueError('Actual nativeU teacher/qualification/loss/count interface differs')
    if ur['C_returns_original_strong_object']!=(expectedUAs==0) or ur['strong_agreement'] or ur['U_truth_used'] or ur['extra_NN'] or ur['same_future_Ascount_claim'] or (c['arm']=='Q01' and ur['selected_positions']!=qualified):raise ValueError('Matched U route semantics differ')
   finally:scope.restore()
   if any(not torch.equal(v,m.state_dict()[k]) for k,v in initial.items()):raise ValueError('Smoke must perform zero optimizer updates/state changes')
   records.append(dict(row_id=c['row_id'],reference_row_id=ref,L_NN=768,L_As=As,U_teacher_NN=8,U_student_NN=8,U_As=expectedUAs,NN=784,total_L_scale=1.25,L_original_hook=True,ORD128=True,same_IQ_each_RD_pair=True,teacher_only_qualification=True,teacher_qualification_before_student_view=True,U_raw_to_practical=True,Gaussian_full_draw_before_overwrite=True,U_truth_used=False,head_joint_gradient=True,Sinc_joint_gradient=True,backbone_joint_gradient=True,optimizer_updates=0,production_U_batches=[256,124],U_practical_count='actual existing teacher qualification_count per batch',synthetic_U_record=ur))
 finally:base.supcon.schedule=schedule
 toys=router_boundary_toys();realNN=sum(r['NN'] for r in real);synNN=sum(r['NN'] for r in records)
 if len(real)!=1 or len(records)!=2 or realNN+synNN!=1570:raise ValueError('Fixed one-parent/two-row smoke forward budget differs')
 d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',records=real,NN=realNN,real_IQ=0,target_query=False,optimizer_updates=0))
 d.write(d.BASE/'synthetic_smoke.json',dict(status='PASS',records=records,NN=synNN,real_checkpoint_NN=realNN,total_NN=realNN+synNN,real_IQ=0,optimizer_updates=0,head_fit_updates=0,target_access=False,training_continues_on_PASS=True,router_boundary_toys=toys))
 print('REAL_OWN_AS20_NO_QUERY_AND_TWO_ORIGINAL6L_NATIVE_U_PROBES PASS '+str(realNN+synNN)+'NN;continue training',flush=True)
 return dict(status='PASS',NN=realNN+synNN)
