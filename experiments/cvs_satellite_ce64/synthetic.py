"""Two actual F six-view synthetic gradient probes;1536 NN, no real IQ/CP/update."""
import json,random,time
import numpy as np
import torch
from . import design as d,model
from experiments.cvs_crossview_supcon41 import model as base
from experiments.cvs_phase1_overlay.model import native_modules

def arm(row):
    started=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(6007);c=d.config(row);m=model.build(c,'cpu').train();before={k:v.clone() for k,v in m.state_dict().items()}
    assert sum(p.numel() for p in m.parameters())==900722 and len(before)==206
    args=d.old.make_args(d.old.config(next(r for r in d.old.rows() if r['arm']=='B01'),20),'cpu');native=native_modules();cfg=native.build_aug_base_cfg(args);aug=native.make_augmentor(cfg);native.configure_augmentor_for_epoch(aug,cfg,13,args)
    x=torch.randn((128,2,256),generator=torch.Generator().manual_seed(6008));y=torch.arange(128)%6;domains=torch.arange(128)%15;rx=torch.tensor(base.supcon.RX_BY_DOMAIN)[domains];meta=dict(eq_i=torch.ones(128,dtype=torch.long),rx_i=rx,day_i=domains%3+1,sig_i=torch.arange(128)+2000,base_index=torch.arange(128))
    random.seed(6009);np.random.seed(6009);torch.manual_seed(6009);ordinary=aug(x,labels=y,no_pa=False);ordinary_saved=ordinary.clone();raw_saved=x.clone();m._feature_l_steps=1
    original_replacement=model.replacement;captured={};schedule=base.supcon.schedule;base.supcon.schedule=lambda epoch:schedule(int(epoch)+20)
    def replacement(first,second,labels,which,objective):
        rng=torch.get_rng_state().clone();loss,info=original_replacement(first,second,labels,which,objective);captured.update(first=first,second=second,loss=loss,rng_preserved=torch.equal(rng,torch.get_rng_state()));return loss,info
    model.replacement=replacement
    try:
        first,maps=base.capture_forward(m,ordinary,domains,True);m._supcon_pending=dict(first=maps,second={});m._legacy_actual_L_input=ordinary
        ce=torch.nn.functional.cross_entropy(first,y,label_smoothing=.01)
        delta,_=m._source_labeled_feature_hook(model=m,labeled_output=dict(tx_logits=first),labels=y,extra=(domains,meta),domains=domains,optimizer_step=1,identity_loss=ce,labeled_count=128,labeled_inputs=x)
        changes=torch.autograd.grad(captured['loss'],(captured['first'],captured['second']),retain_graph=True,allow_unused=False)
        total=ce+delta;total.backward()
    finally:model.replacement=original_replacement;base.supcon.schedule=schedule
    r=m._full_records[-1];p=m._satellite_ce_records[-1];shape=m._shape_records[-1];head=m.id_backbone.encoder.core.id_backbone.cls_head.weight;sinc=m.id_backbone.encoder.core.id_backbone.sinc
    grads={name:float(value.grad.norm()) for name,value in m.named_parameters() if value.grad is not None}
    assert torch.isfinite(total) and head.grad is not None and float(head.grad.abs().sum())>0 and sum(float(v.grad.abs().sum()) for v in sinc.parameters() if v.grad is not None)>0 and all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None)
    assert all(torch.isfinite(g).all() and float(g.abs().sum())>0 for g in changes)
    assert m.id_backbone.encoder._full_train_packets==768 and m._full_L_packets==512 and m._shape_NN==256 and m._full_channel.packets==64 and r['As_packets']==64 and r['L_forwards']==6 and r['SupCon_weight']==.1
    assert m._supcon_pending is None and not m._supcon_handles and getattr(m,'_source_unlabeled_feature_hook',None) is None and all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert torch.equal(x,raw_saved) and torch.equal(ordinary,ordinary_saved) and captured['rng_preserved'] and p['all_satellite_CE_retained'] and p['extra_NN']==0 and p['satellite_slots']==64 and p['passes']==2 and p['gradient_alive']
    assert abs(r['total_L']-float(total.detach()))<3e-6 and abs(r['base_L_before_ce']-r['base_L_before_shape']-.25*shape['shape_RD'])<1e-6 and abs(r['total_L']-r['base_L_before_ce']-p['satellite_CE_correction'])<1e-6
    assert p['step']==1 and p['cumulative_step']==4441 and p['forward_count']==5 and p['channel_calls']==1 and r['ordinary_redraws']==0
    compare=dict(ordinary=ordinary.detach(),first=first.detach(),satellite0=captured['first'].detach(),satellite1=captured['second'].detach(),rng=torch.get_rng_state(),sequence={k:r[k] for k in ('step','epoch','common_RD','common_CE','common_KL','extra_RD','extra_CE','extra_KL','SupCon','SupCon_weight','ordinary_raw_MSE','second_raw_MSE','scene_counts','state_counts','base_L_before_shape','base_L_before_ce')})
    result=dict(status='PASS',arm=c['arm'],NN=768,real_IQ=0,CP=0,optimizer_updates=0,head_fit_updates=0,target_access=False,wall_seconds=time.perf_counter()-started,actual_parent_ordinary_args_epochs=args.epochs,ordinary_frozen_epoch=13,ordinary_changed_packets=int((ordinary!=x).flatten(1).any(1).sum()),ordinary_IQ_unchanged_by_hook=True,raw_IQ_unchanged=True,objective_RNG_unchanged=captured['rng_preserved'],L_forwards=6,As_supervised=64,cumulative_epoch=21,cumulative_step=4441,all_CE_retained=True,objective=p,satellite_logit_correction_gradient_norms=[float(g.norm()) for g in changes],default_head_gradient_norm=float(head.grad.norm()),default_head_scale=m.id_backbone.encoder.core.id_backbone.cls_head.scale,default_head_shape=list(head.shape),state_keys=206,parameter_count=900722,Sinc_gradient_norms={name:value for name,value in grads.items() if '.sinc.' in name},all_joint_gradients_finite=True,new_parameters=0,new_U_hook=False,ordinary_redraws=r['ordinary_redraws'])
    return result,compare

def run():
    started=time.perf_counter();results=[];reference=None
    for row in d.rows():
        result,observed=arm(row);results.append(result)
        if reference is None:reference=observed
        else:
            assert observed['sequence']==reference['sequence']
            assert all(torch.equal(observed[k],reference[k]) for k in ('ordinary','first','satellite0','satellite1','rng'))
    return dict(status='PASS',NN=sum(r['NN'] for r in results),rows=results,real_IQ=0,CP=0,optimizer_updates=0,wall_seconds=time.perf_counter()-started,two_arm_sequence_and_ordinary_fingerprints_identical=True,all_default_head_parameters_shapes_keys_preserved=True)
if __name__=='__main__':print(json.dumps(run()))
