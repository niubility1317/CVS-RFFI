"""One real own source checkpoint/noquery smoke;two actual synthetic L hooks."""
import torch
from . import design as d,model
from experiments.cvs_noise_shape_factorial43 import model as old,design as od
from experiments.cvs_crossview_supcon41 import model as shared

def smoke():
 c=od.config(next(r for r in od.rows() if r['arm']=='B01'));m=old.own_model(c,'cpu');before={k:v.clone() for k,v in m.state_dict().items()}
 with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
 if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Real CP noquery smoke failed')
 d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',NN=2,source_IQ=0,target_query=False,checkpoint=c['output_root']+'/final_ssdg.pth',training_initializer=False,same_source_contract=True,optimizer_updates=0))
 del m,before
 initial=None;records=[];original_terms=shared.terms
 for row in d.rows():
  torch.manual_seed(6007);c=d.config(row,20);m=model.build(c,'cpu').train();state={k:v.clone() for k,v in m.state_dict().items()}
  if initial is None:initial=state
  if sum(p.numel() for p in m.parameters())!=900722 or len(state)!=206 or any(not torch.equal(v,initial[k]) for k,v in state.items()):raise ValueError('Fixed F initial state differs')
  m._feature_l_steps=1333;m._full_calls=1332;g=torch.Generator().manual_seed(6008);x=torch.randn((128,2,256),generator=g);y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(shared.supcon.RX_BY_DOMAIN)[domains];meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=domains%3+1,sig_i=torch.arange(128)+2000,base_index=torch.arange(128))
  first,maps=shared.capture_forward(m,x,domains,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=x.clone();native_CE=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
  extra,_=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(domains,meta),domains=domains,optimizer_step=1333,identity_loss=native_CE,labeled_count=128,labeled_inputs=x)
  total=native_CE+extra;total.backward();head=m.id_backbone.encoder.core.id_backbone.cls_head.weight;sinc=m.id_backbone.encoder.core.id_backbone.sinc
  if not torch.isfinite(total) or head.grad is None or float(head.grad.abs().sum())<=0 or sum(float(p.grad.abs().sum()) for p in sinc.parameters() if p.grad is not None)<=0 or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise ValueError('Joint defaulthead/Sinc derivative invalid')
  rec=m._hard_records[-1]
  if shared.terms is not original_terms or len(rec['pairs'])!=3 or any(r['L_smoothing']!=0. or r['NN'] for r in rec['pairs']) or m.id_backbone.encoder._full_train_packets!=768 or m._full_channel.packets!=(64 if c['As'] else 0) or m._shape_packets or any(not torch.equal(v,m.state_dict()[k]) for k,v in state.items()):raise ValueError('Only L CE changed;NN/IQ/U/state differs')
  if abs(float(total.detach())-rec['L_total'])>3e-6:raise ValueError('Native CE01 must cancel before returning total hard L objective')
  records.append(dict(arm=row['arm'],NN=768,L_As=64 if c['As'] else 0,L_smoothing=0.,native_U_smoothing=.01,U_function_restored=True,total_L=float(total.detach()),head_joint_gradient=True,Sinc_joint_gradient=True,optimizer_updates=0))
 d.write(d.BASE/'synthetic_smoke.json',dict(status='PASS',records=records,NN=1536,real_checkpoint_NN=2,real_IQ=0,optimizer_updates=0,head_fit_updates=0,target_access=False))
 print('REAL_CP_NO_QUERY+HARD_L_ONLY_SMOKE PASS1538NN;continue training',flush=True)
 return dict(status='PASS',NN=1538)
