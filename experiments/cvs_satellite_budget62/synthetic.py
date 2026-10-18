"""Two actual F synthetic training forwards, no IQ/checkpoint/query or updates."""
import json,tempfile,copy
from pathlib import Path
from types import SimpleNamespace
import torch
from . import design as d,model
from .observer import Observer
from experiments.cvs_crossview_supcon41 import model as base

def run():
 initial=None;records=[];rng=[];ordinary_logits=[];schedule=base.supcon.schedule;base.supcon.schedule=lambda epoch:schedule(int(epoch)+30)
 try:
  for row in d.rows():
   torch.manual_seed(6007);c=d.config(row);m=model.build(c,'cpu').train();current={k:v.clone() for k,v in m.state_dict().items()}
   if initial is None:initial=current
   assert sum(p.numel() for p in m.parameters())==900722 and len(current)==206 and all(torch.equal(v,initial[k]) for k,v in current.items())
   m._feature_l_steps=1;g=torch.Generator().manual_seed(6008);x=torch.randn((128,2,256),generator=g);y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(base.supcon.RX_BY_DOMAIN)[domains];meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=domains%3+1,sig_i=torch.arange(128)+2000,base_index=torch.arange(128))
   first,maps=base.capture_forward(m,x,domains,True);ordinary_logits.append(first.detach().clone());m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=x.clone();ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
   extra,_=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(domains,meta),domains=domains,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
   total=ce+extra;total.backward();rng.append(torch.get_rng_state().clone());head=m.id_backbone.encoder.core.id_backbone.cls_head.weight;sinc=m.id_backbone.encoder.core.id_backbone.sinc
   assert torch.isfinite(total) and head.grad is not None and float(head.grad.abs().sum())>0 and sum(float(p.grad.abs().sum()) for p in sinc.parameters() if p.grad is not None)>0 and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
   r=m._full_records[-1];shape=m._shape_records[-1];probe=m._sat_steps[-1]
   assert m.id_backbone.encoder._full_train_packets==768 and m._shape_NN==256 and m._full_channel.packets==64 and r['As_packets']==64 and r['L_forwards']==6 and r['SupCon_weight']==.1 and shape['cumulative_step']==6661 and m._supcon_pending is None and not m._supcon_handles and all(torch.equal(v,m.state_dict()[k]) for k,v in current.items()) and getattr(m,'_source_unlabeled_feature_hook',None) is None
   assert abs(r['total_L']-float(total.detach()))<3e-6 and abs(r['total_L']-r['base_L_before_shape']-.25*shape['shape_RD'])<1e-6 and shape['same_IQ_each_pair']
   assert m._full_channel.obj.last_namespace=='FcrossRX/train/e31/step6661/L_S' and probe['mode']==row['arm'] and probe['As_supervised']==64 and probe['extra_NN']==0
   if row['arm']=='AS_DET':assert probe['satellite_pair_logit_maxdiff']<=1e-5 and probe['bypassed_elements']>0
   else:assert probe['bypassed_elements']==0 and probe['satellite_pair_logit_maxdiff']>1e-5
   # Exercise the real observer's critical native lifecycle without reading data.
   with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
    o=Observer.__new__(Observer);o.c=c;o.out=Path(tmp);o.model=m;o.optimizer=None;o.parent={'ema_model':{k:v.detach().clone()+(.001 if v.is_floating_point() else 0) for k,v in current.items()}};o.scratch_args={'epochs':20,'a1_ema_startup_average':False}
    captured=[];native=SimpleNamespace(deepcopy=copy.deepcopy,configure_augmentor_for_epoch=lambda obj,cfg,epoch,args:captured.append((epoch,args.epochs)),_resolve_phase1_terminal_status=lambda **kw:kw)
    orig=(native.deepcopy,native.configure_augmentor_for_epoch,native._resolve_phase1_terminal_status);base_schedule=base.supcon.schedule
    with o.native_scope(native):
     teacher=native.deepcopy(m);native.configure_augmentor_for_epoch(None,None,20,SimpleNamespace(epochs=20))
     assert captured==[(13,20)] and all(torch.equal(v,o.parent['ema_model'][k]) for k,v in teacher.state_dict().items())
     assert native._resolve_phase1_terminal_status(mechanism_gates_required=True)['mechanism_gates_required'] is False
     assert base.supcon.schedule(1)==base_schedule(31)
     opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.0002,weight_decay=.0001)
     assert not opt.state and o.optimizer is opt
    assert orig==(native.deepcopy,native.configure_augmentor_for_epoch,native._resolve_phase1_terminal_status) and base.supcon.schedule is base_schedule
   records.append(dict(arm=row['arm'],NN=768,As_IQ=r['As_packets'],real_IQ=0,CP=0,Ag_IQ=shape['applied'],cumulative_epoch=31,cumulative_step=6661,head_joint_gradient=True,Sinc_joint_gradient=True,optimizer_updates=0,satellite=probe,ordinary_schedule_scratch20_E13=True,ownEMA_strict=True,resetAdamW=True))
 finally:base.supcon.schedule=schedule
 assert torch.equal(rng[0],rng[1]) and torch.equal(ordinary_logits[0],ordinary_logits[1])
 return dict(status='PASS',records=records,NN=1536,real_IQ=0,real_CP=0,optimizer_updates=0,head_fit_updates=0,target_access=False,global_RNG_matched=True,ordinary_logits_matched=True)
if __name__=='__main__':print(json.dumps(run()))
