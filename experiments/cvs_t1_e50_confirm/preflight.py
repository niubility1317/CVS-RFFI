import torch
from . import design as d,model
from .source import original_parent
from experiments.cvs_crossview_supcon41 import model as base

def smoke():
 torch.set_num_threads(2);records=[];real_done=False
 from .dispatch import reused_complete
 reused_complete()
 for row in d.rows():
  if d.is_reused(row):continue
  c=d.config(row);a=d.make_args(c,'cpu');seed=row['model_seed'];cp=original_parent(seed)
  if a.seed!=seed or a.baseline_ckpt!=str(d.parent_source(seed)/'final_ssdg.pth') or a.epochs!=20 or a.label_epochs!=0 or a.pseudo_epochs!=20 or a.from_scratch or a.a1_scratch_only or a.teacher_ckpt:
   raise ValueError('Actual native seed/budget/parent args differ')
  for key in ('temperature','training_mechanism','crossview','U_transfer','As','Ag_mode','noise','total_L_loss_scale'):
   if c[key]!=d.TEMPLATE[key]:raise ValueError('Seed expansion changed scientific recipe: '+key)
  torch.manual_seed(seed);m=model.build(c,'cpu');m.load_state_dict(cp['model'],strict=True)
  if not real_done:
   m.eval();before={k:v.clone() for k,v in m.state_dict().items()}
   with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
   if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Real same-seed parent no-query smoke failed')
   d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',checkpoint=str(d.parent_source(seed)/'final_ssdg.pth'),seed=seed,NN=2,target_query=False,source_IQ=0,optimizer_updates=0))
   real_done=True
  # Fresh object so smoke packet counters have no residue from the real-CP probe.
  m=model.build(c,'cpu');m.load_state_dict(cp['model'],strict=True);m.train()
  m._feature_l_steps=1333;m._full_calls=1332;g=torch.Generator().manual_seed(6008);x=torch.randn(128,2,256,generator=g);y=torch.arange(128)%6
  domains=torch.arange(128)%15;rx=torch.tensor(base.supcon.RX_BY_DOMAIN)[domains]
  meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=domains%3+1,sig_i=torch.arange(128)+2000,base_index=torch.arange(128))
  first,maps=base.capture_forward(m,x,domains,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=x.clone()
  ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
  extra,_=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(domains,meta),domains=domains,optimizer_step=1333,identity_loss=ce,labeled_count=128,labeled_inputs=x)
  total=ce+extra;total.backward();head=m.id_backbone.encoder.core.id_backbone.cls_head.weight
  if not torch.isfinite(total) or head.grad is None or float(head.grad.abs().sum())<=0 or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise ValueError('Same-seed T1 joint-head finite gradient failed')
  if m.id_backbone.encoder._full_train_packets!=768 or m._full_channel.packets!=64 or len(m._temperature_records)!=1 or m._temperature_records[0]['As_KL']!=0 or m._temperature_records[0]['T']!=1. or m._supcon_pending is not None or m._supcon_handles:
   raise ValueError('Unchanged six-L/As KL0/SupCon budget differs')
  records.append(dict(seed=seed,parent_native_seed=cp['args']['seed'],As_KL=0,T=1,NN=768,As_IQ=64,head_joint_gradient=True,optimizer_updates=0,target_access=False))
 d.write(d.BASE/'synthetic_smoke.json',dict(status='PASS',records=records,NN=1536,real_checkpoint_NN=2,head_fit_updates=0,target_access=False))
 print('REAL_CP_NO_QUERY+SAME_SEED_PARENT/SIX-L_SMOKE PASS1538NN;start both new E20 tails',flush=True)
