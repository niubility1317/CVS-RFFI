"""Real checkpoint first;real F synthetic coverage/gradient smoke then immediate training."""
from pathlib import Path
import torch
from . import design as d,model,relations

def smoke():
 from experiments.cvs_Uurban39 import model as previous,design as pd
 c=pd.config(next(r for r in pd.rows() if r['arm']=='B0'));m=previous.own_model(c,'cpu');state={k:v.clone() for k,v in m.state_dict().items()}
 with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
 if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in state.items()):raise ValueError('Actual checkpoint/defaulthead smoke differs')
 d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',NN=2,source_IQ=0,target_query=False,checkpoint=c['output_root']+'/final_ssdg.pth',training_initializer=False,optimizer_updates=0,same_source_contract=True,fixed_parent_not_target_selected=True))
 del m,state
 records=[];initial=None
 for row in d.rows():
  torch.manual_seed(6007);c=d.config(row,20);m=model.build(c,'cpu').train();current={k:v.clone() for k,v in m.state_dict().items()}
  if initial is None:initial=current
  if sum(p.numel() for p in m.parameters())!=900722 or len(current)!=206 or any(not torch.equal(v,initial[k]) for k,v in current.items()):raise ValueError('Fixed same initial F/default head differs')
  m._crossview_smoke_capture=True;m._feature_l_steps=1333;m._full_calls=1332;g=torch.Generator().manual_seed(6008);x=torch.randn(128,2,256,generator=g);y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(model.supcon.RX_BY_DOMAIN)[domains];meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=domains%3+1,sig_i=torch.arange(128)+2000,base_index=torch.arange(128));extra=(domains,meta)
  first,maps=model.capture_forward(m,x,domains,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=x.clone();ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
  value,info=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=extra,domains=domains,optimizer_step=1333,identity_loss=ce,labeled_count=128,labeled_inputs=x)
  dbg=m._crossview_debug;head=m.id_backbone.encoder.core.id_backbone.cls_head.weight;targets=tuple(dbg['satellite0'][level] for level in relations.LEVELS)
  grads=torch.autograd.grad(dbg['aux'],targets+(head,),allow_unused=True,retain_graph=True);sat_norm=sum(float(v.abs().sum()) for v in grads[:-1] if v is not None)
  if grads[-1]is not None or (row['arm']=='SAT192')!=(sat_norm>0):raise ValueError('Existing auxiliary path/defaultW/view membership gradient differs')
  if row['arm']=='ORD128':
   ref=torch.stack([model.supcon.penalty(dbg[key],y,rx,'CROSS_RX')[0] for key in ('ordinary0','ordinary1')]).mean()
   if not torch.allclose(ref,dbg['aux'],atol=1e-6,rtol=1e-6):raise ValueError('ORD128 must equal original exact SupCon formula')
  (ce+value).backward();sinc=m.id_backbone.encoder.core.id_backbone.sinc
  if not torch.isfinite(ce+value) or not bool(torch.isfinite(head.grad).all()) or float(head.grad.abs().sum())<=0 or sum(float(p.grad.abs().sum()) for p in sinc.parameters() if p.grad is not None)<=0 or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise ValueError('Actual defaulthead/Sinc joint gradient failed')
  if m.id_backbone.encoder._full_train_packets!=512 or m._full_channel.packets!=64 or m._supcon_pending is not None or m._supcon_handles or any(not torch.equal(v,m.state_dict()[k]) for k,v in current.items()) or getattr(m,'_source_unlabeled_feature_hook',None) is not None or getattr(m,'_source_unlabeled_selection_hook',None) is not None:raise ValueError('Same4L/sameIQ/read-onlysynthetic/noextraU budget differs')
  r=m._full_records[-1]
  if r['As_packets']!=64 or not r['all_satellite_L_supervised'] or r['ordinary_redraws'] or not r['same_IQ_each_pair']:raise ValueError('L CE/input policy changed')
  records.append(dict(arm=row['arm'],NN=512,satellite_IQ=64,satellite_aux_gradient_L1=sat_norm,head_fit_updates=0,optimizer_updates=0,same_initial=True,extra_U_NN=0,details=r['SupCon_details']))
 d.write(d.BASE/'synthetic/complete.json',dict(status='PASS',records=records,NN=1536,backward_calls=3,auxiliary_VJP_calls=3,real_IQ=0,target_access=False,optimizer_updates=0))
 return dict(status='PASS',NN=1538,real_checkpoint_NN=2,synthetic_NN=1536,optimizer_updates=0)
