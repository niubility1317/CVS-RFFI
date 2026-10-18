"""One real parent/no-query smoke first, then the actual late-tail sixL path."""
from types import MethodType
import torch
from . import design as d,model

def synthetic():
 torch.set_num_threads(2)
 from experiments.cvs_crossview_supcon41 import model as base
 c=d.config(d.rows()[0]);torch.manual_seed(6007);m=model.build(c,'cpu').train();before={k:v.clone() for k,v in m.state_dict().items()};seen=[];forward=m.forward_identity_only
 def record(self,x,*args,**kwargs):seen.append(x.detach().clone());return forward(x,*args,**kwargs)
 m.forward_identity_only=MethodType(record,m);m._feature_l_steps=2220;m._full_calls=2219
 generator=torch.Generator().manual_seed(6008);x=torch.randn((128,2,256),generator=generator);y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(base.supcon.RX_BY_DOMAIN)[domains];meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=domains%3+1,sig_i=torch.arange(128)+2000,base_index=torch.arange(128))
 schedule=base.supcon.schedule;base.supcon.schedule=lambda epoch:schedule(epoch+20)
 try:
  first,maps=base.capture_forward(m,x,domains,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=x.clone();ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
  extra,_=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(domains,meta),domains=domains,optimizer_step=2220,identity_loss=ce,labeled_count=128,labeled_inputs=x)
 finally:base.supcon.schedule=schedule
 total=ce+extra;total.backward();head=m.id_backbone.encoder.core.id_backbone.cls_head;sinc=m.id_backbone.encoder.core.id_backbone.sinc;r=m._full_records[-1];shape=m._shape_records[-1]
 if sum(p.numel() for p in m.parameters())!=900722 or len(before)!=206 or tuple(head.weight.shape)!=(6,160) or head.scale!=30.:raise ValueError('Actual F/default cosine classifier changed')
 if len(seen)!=6 or any(not torch.equal(seen[a],seen[b]) for a,b in [(0,1),(2,3),(4,5)]):raise ValueError('Six real forwards must reuse the same generated IQ in each RDrop pair')
 if not torch.isfinite(total) or head.weight.grad is None or float(head.weight.grad.abs().sum())<=0 or sum(float(p.grad.abs().sum()) for p in sinc.parameters() if p.grad is not None)<=0 or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise ValueError('Joint defaulthead/Sinc finite gradient failed')
 if m.id_backbone.encoder._full_train_packets!=768 or m._shape_NN!=256 or m._full_channel.packets!=64 or r['L_forwards']!=6 or r['As_packets']!=64 or shape['applied'] or not shape['same_IQ_each_pair'] or m._supcon_pending is not None or m._supcon_handles or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()) or getattr(m,'_source_unlabeled_feature_hook',None) is not None:raise ValueError('SixL/defaultU/realAs/state counter differs')
 if abs(r['total_L']-float(total.detach()))>3e-6 or abs(r['total_L']-r['base_L_before_shape']-.25*shape['shape_RD'])>1e-6 or r['SupCon_weight']!=.1:raise ValueError('Original normalized1.25 objective or cumulative SupCon schedule differs')
 return dict(status='PASS',NN=768,As_IQ=64,Ag_IQ=0,same_generated_IQ_each_pair=True,default_head_joint_gradient=True,Sinc_joint_gradient=True,all_parameters_unchanged_without_optimizer=True,cumulative_step=6660,cumulative_epoch=30,scene_prior=m._full_channel.audit(),channel_counts=m._full_channel.summary(),new_losses=0,extra_U_NN=0,optimizer_updates=0,source_IQ=0,target_access=False)

def smoke():
 torch.set_num_threads(2)
 from .source import parent_payload
 c=d.config(d.rows()[0]);parent=parent_payload(c);m=model.build(c,'cpu');m.load_state_dict(parent['model'],strict=True);m.eval();before={k:v.clone() for k,v in m.state_dict().items()}
 with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
 if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Real exact As20 parent/noquery smoke failed')
 d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',NN=2,checkpoint=c['parent_source']+'/final_ssdg.pth',source_IQ=0,target_query=False,optimizer_updates=0,head_fit_updates=0,training_initializer=True,same_source_contract=True,strict_own_EMA_present=parent.get('ema_model') is not None));del m,before,parent
 result=synthetic();d.write(d.BASE/'synthetic_smoke.json',result);print('REAL_PARENT_NO_QUERY+LATE_E30_REAL_SIXL PASS770NN;continue training',flush=True);return dict(status='PASS',NN=770)
