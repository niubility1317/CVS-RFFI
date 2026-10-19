"""First-step real own20 default-head/physical-HQ/8forward smoke,then train."""
import torch
from . import design as d,model
from .source import parent_payload
base=model.original.base
def smoke():
 records=[];real=[];old=base.supcon.schedule;base.supcon.schedule=lambda ep:old(int(ep)+20);reference_HQ=None
 try:
  for row,localstep in [(r,1) for r in d.rows()]+[(d.rows()[-1],2220)]:
   c=d.config(row);args=d.make_args(c,'cpu');p=parent_payload(c);m=model.build(c,'cpu');m.load_state_dict(p['model'],strict=True);m.eval();before={k:v.clone() for k,v in m.state_dict().items()}
   with torch.no_grad():z=m.forward_identity_only(torch.zeros(2,2,256))['tx_logits']
   if z.shape!=(2,6) or not torch.isfinite(z).all() or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()):raise ValueError('Own20 no-query defaulthead smoke failed')
   real.append(dict(row=row['row_id'],boundary_step=localstep,NN=2,updates=0,source_IQ=0,target_access=False));del m
   torch.manual_seed(6107);m=model.build(c,'cpu');m.load_state_dict(p['model'],strict=True);m.train();before={k:v.clone() for k,v in m.state_dict().items()};m._extra_smoke_debug=True;m._feature_l_steps=localstep;m._full_calls=localstep-1;g=torch.Generator().manual_seed(6108);x=torch.randn((128,2,256),generator=g);ordinary=x+.01;y=(torch.arange(128)//2)%6;dom=torch.arange(128)%15;rx=torch.tensor(base.supcon.RX_BY_DOMAIN)[dom];meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=dom%3+1,sig_i=torch.arange(128)+2000,base_index=torch.arange(128))
   first,maps=base.capture_forward(m,ordinary,dom,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=ordinary.clone();ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01);delta,info=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(dom,meta),domains=dom,optimizer_step=localstep,identity_loss=ce,labeled_count=128,labeled_inputs=x);total=ce+delta;total.backward()
   debug=m._extra_debug;rec=m._extraCE_records[-1];r=m._full_records[-1];shape=m._shape_records[-1];mask=debug['mask'];head=m.id_backbone.encoder.core.id_backbone.cls_head.weight
   if sum(p.numel() for p in m.parameters())!=900722 or len(m.state_dict())!=206 or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items()) or head.grad is None or not torch.isfinite(head.grad).all() or float(head.grad.abs().sum())<=0 or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):raise ValueError('Joint defaulthead/backbone/state smoke failed')
   if m.id_backbone.encoder._full_train_packets!=896 or m._full_L_packets!=512 or m._shape_NN!=256 or m._extra_NN!=128 or r['L_forwards']!=8 or not torch.equal(debug['raw'],x[mask]) or m._full_channel.calls!=1 or m._full_channel.packets!=(64 if c['As'] else 0) or m._supcon_pending is not None or m._supcon_handles:raise ValueError('Original6L plus exactly2CE64 budget/physical selection differs')
   expected=.5*r['common_RD']+.5*r['extra_RD']+.25*shape['shape_RD']+.1*r['SupCon']+.25*rec['CE']
   if abs(float(total.detach())-expected)>1e-5 or abs(r['total_L']-expected)>1e-5 or rec['extra_KL'] or rec['extra_teacher_NN'] or not rec['same_IQ_two_CE'] or not rec['all_extra_labels_supervised']:raise ValueError('Pure extraCE formula/normalization differs')
   if c['As']:
    if len(debug['HQ_records'])!=64 or any(abs(z['quality_delta_db']-6)>1e-8 or not z['receiver_quality_recomputed'] or not z['compensation_recomputed'] or not z['AGC_recomputed'] for z in debug['HQ_records']):raise ValueError('Actual native HQ complete receiver failed')
    if c['extra_view']=='RAW':reference_HQ=debug['HQ'].detach().clone()
    elif localstep==1 and (reference_HQ is None or not torch.equal(reference_HQ,debug['view'])):raise ValueError('HQ shadow/control and candidate must share exact physical view')
   if c['extra_view']=='RAW' and not torch.equal(debug['view'],debug['raw']):raise ValueError('Extra raw control input differs')
   if c['extra_view']=='HQ' and torch.equal(debug['view'],debug['raw']):raise ValueError('HQ candidate never reached actual input')
   records.append(dict(row=row['row_id'],localstep=localstep,cumulative_step=4440+localstep,NN=896,source_IQ=0,updates=0,extra_CE_NN=128,extra_teacher_NN=0,extra_KL=0,fullAs_packets=64 if c['As'] else 0,HQ_shadow_packets=rec['HQ_shadow'],HQ_consumed=rec['HQ_consumed'],extra_physical_snapshots=rec['physical_snapshots_extra'],total_L_nominal_scale=1.5,default_head_joint=True,physicalquality_delta=6 if c['As'] else None,full_original_path_retained=True))
 finally:base.supcon.schedule=old
 d.write(d.BASE/'real_checkpoint_smoke.json',dict(status='PASS',records=real,NN=8,updates=0,target_access=False));d.write(d.BASE/'synthetic_smoke.json',dict(status='PASS',records=records,NN=3584,real_checkpoint_NN=8,total_NN=3592,optimizer_updates=0,head_fit_updates=0,target_access=False));print('SIGNAL_BRIDGE_REAL_CP_AND_PHYSICAL_SMOKE PASS3592NN;continue',flush=True)
 return dict(status='PASS',NN=3592)
