"""One shared real parent plus four actual synthetic six-L training graphs."""
import torch
from . import design as d,model
from .source import parent_payload
from experiments.cvs_crossview_supcon41 import model as base

def _smoke():
 torch.set_num_threads(2);c0=d.config(d.rows()[0]);cp=parent_payload(c0)
 m=model.build(c0,'cpu').eval();m.load_state_dict(cp['model'],strict=True)
 ema=model.build(c0,'cpu').eval();ema.load_state_dict(cp['ema_model'],strict=True)
 with torch.no_grad():
  for net in (m,ema):
   z=net.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
   if z.shape!=(2,6) or not bool(torch.isfinite(z).all()):raise ValueError('Strict real parent/ownEMA defaulthead smoke failed')
 del m,ema
 records=[];physics=[]
 for row in d.rows():
  c=d.config(row);parent_payload(c);m=model.build(c,'cpu');m.load_state_dict(cp['model'],strict=True);m.train()
  before={k:v.clone() for k,v in m.state_dict().items()}
  if sum(p.numel() for p in m.parameters())!=900722 or len(before)!=206:raise ValueError('Fixed F/defaulthead architecture differs')
  m._feature_l_steps,m._full_calls=1333,1332
  x=torch.randn(128,2,256,generator=torch.Generator().manual_seed(76007));y,domains=torch.arange(128)%6,torch.arange(128)%15;rx=torch.tensor(base.supcon.RX_BY_DOMAIN)[domains]
  meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=domains%3+1,sig_i=torch.arange(128)+76000,base_index=torch.arange(128))
  first,maps=base.capture_forward(m,x,domains,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=x.clone()
  ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01);one=[];model.physics_sink(m,one.append)
  delta,_=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(domains,meta),domains=domains,optimizer_step=1333,identity_loss=ce,labeled_count=128,labeled_inputs=x)
  model.physics_sink(m);loss=ce+delta;loss.backward();r=m._prop_records[-1]
  head=m.id_backbone.encoder.core.id_backbone.cls_head.weight;sinc=m.id_backbone.encoder.core.id_backbone.sinc
  if not bool(torch.isfinite(loss)) or head.grad is None or float(head.grad.abs().sum())<=0 or sum(float(p.grad.abs().sum()) for p in sinc.parameters() if p.grad is not None)<=0 or any(not bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None) or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()) or m.id_backbone.encoder._full_train_packets!=768 or m._full_channel.packets!=64 or m._prop_generated!=64 or m._prop_slots!=128 or m._pair_generated!=64 or len(one)!=1 or len(one[0]['view0'])!=64 or len(one[0]['view1'])!=64 or m._supcon_pending is not None or m._supcon_handles or getattr(m,'_source_unlabeled_feature_hook',None) is not None or abs(m._full_records[-1]['total_L']-float(loss.detach()))>3e-6:raise ValueError('Actual six-L/default-head/As graph smoke failed')
  if abs(r['As_loss_weighted']-(.05*r['As_loss_raw'] if c['As_contrast'] else 0.))>1e-6:raise ValueError('Actual extra As coefficient differs')
  records.append(dict(row=c['row_id'],NN=768,actual_snapshots=64,record=r,state_unchanged=True,head_Sinc_gradients=True));physics.append(one[0])
 result=dict(status='PASS',NN=d.SMOKE_NN,real_checkpoint_NN=4,synthetic_remote_NN=3072,physical_snapshots=256,
  checkpoint=c0['parent_source']+'/final_ssdg.pth',student_and_ownEMA=True,source_IQ=0,target_query=False,optimizer_updates=0,head_fit_updates=0,rows=records)
 d.write(d.BASE/'real_checkpoint_smoke.json',result);d.write(d.BASE/'synthetic_physics.json',dict(records=physics,source_IQ=0,target_query=False))
 print('ACTUAL_AS_CE20_OWNEMA+FOUR_PROP_CONTRAST_CELLS PASS3076NN;continue training',flush=True);return result


def smoke():
 # Match the already verified source observer: local tail epochs are cumulative+20.
 original_schedule=base.supcon.schedule
 base.supcon.schedule=lambda epoch:original_schedule(epoch+20)
 try:return _smoke()
 finally:base.supcon.schedule=original_schedule
