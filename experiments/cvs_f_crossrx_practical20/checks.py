import copy,io,math
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d,model as api,supcon
from .channel import Practical
from .source import Observer
from experiments.cvs_raw_anchor20 import model as reference

def reject(call):
    try:call()
    except (ValueError,RuntimeError):return
    raise AssertionError('Invalid protocol/view accepted')

def run(output):
    from SSDG import train_ssdg as native
    torch.set_num_threads(2);torch.manual_seed(20261016429);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    x=torch.randn(128,2,256);y=torch.arange(128)%6
    rx=torch.tensor([3,4,6,8]*32);day=torch.arange(128)%3+1
    domains=torch.tensor([{3:3,4:6,6:9,8:12}[int(r)]+int(v)-1 for r,v in zip(rx,day)])
    meta=dict(rx_i=rx,day_i=day,eq_i=torch.ones(128,dtype=torch.long),base_index=torch.arange(128),sig_i=torch.arange(128))
    inputs={};initial=None;total_packets=0;updates=0;primary_losses={};records=[]
    for row in d.rows():
        c=d.config(row);a=d.make_args(c,'cpu');m=api.build(c,'cpu');m.train()
        state={k:v.clone() for k,v in m.state_dict().items()}
        if initial is None:
            initial=state;base=reference.build(d.old.config(d.old.rows()[1]),'cpu')
            assert state.keys()==base.state_dict().keys() and all(torch.equal(v,base.state_dict()[k]) for k,v in state.items())
        else:assert all(torch.equal(v,initial[k]) for k,v in state.items())
        ema=copy.deepcopy(m);assert ema._source_labeled_feature_hook.__self__ is ema and ema._supcon_original_hook.__self__ is ema
        # Start at the first active SupCon step without simulating preceding updates.
        # This is a synthetic interface test, not an E6 trained checkpoint.
        for key in ('_feature_l_steps','_legacy_calls','_anchor_calls','_supcon_calls'):setattr(m,key,1110)
        o=Observer(c);o.out=output/row['arm'];o.out.mkdir();o.m=o.model=m;o.initialize(m,c,o.out)
        o._current_L_meta=meta;o._current_L_labels=y.tolist();received=[];apply=o.channel.apply
        def seen(value,*args,**kw):received.append(value.detach().clone());return apply(value,*args,**kw)
        o.channel.apply=seen
        maps={level:[] for level in supcon.LEVELS};handles=[]
        for level in supcon.LEVELS:
            handles.append(m.get_submodule(supcon.PREFIX+level).register_forward_hook(lambda mod,args,out,level=level:maps[level].append(out)))
        captured=[];original_penalty=supcon.penalty
        def penalty(values,*args):
            result=original_penalty(values,*args);captured.append((dict(values),result[0]));return result
        supcon.penalty=penalty
        try:
            before=torch.get_rng_state().clone()
            with o.native_scope(native):
                cfg=native.build_aug_base_cfg(a);aug=native.make_augmentor(cfg);native.configure_augmentor_for_epoch(aug,cfg,6,a)
                primary=aug(x,labels=y,no_pa=not a.aug_enable_pa_normal);secondary=m._pair_secondary.clone()
                assert torch.equal(before,torch.get_rng_state())
                if c['As']:assert len(received)==1 and torch.equal(received[0],x) and not torch.equal(secondary,primary)
                else:assert not received and torch.equal(secondary,primary)
                inputs[row['arm']]=primary.detach().clone()
                with m._phase1_labeled_feature_context():out=m(primary,y_tx=y,return_aux=True)
                ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
                correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(domains,meta),domains=domains,
                    optimizer_step=1111,identity_loss=ce,labeled_count=128)
                full=ce+correction;aux=torch.stack([v[1] for v in captured]).mean()
                assert math.isclose(metrics['F_supcon/weight'],.01,rel_tol=1e-14)
                assert torch.allclose(full,.5*(metrics['pair/primary_objective']+metrics['pair/secondary_objective'])+.01*aux,atol=1e-6)
                for level in supcon.LEVELS:
                    assert len(maps[level])==4 and all(captured[j][0][level] is maps[level][j] for j in (0,1))
                head=dict(m.named_parameters())['id_backbone.encoder.core.id_backbone.cls_head.weight']
                assert torch.autograd.grad(aux,head,allow_unused=True,retain_graph=True)[0] is None
                full.backward()
                assert all(dict(m.named_parameters())[k].grad is not None and torch.isfinite(dict(m.named_parameters())[k].grad).all() and float(dict(m.named_parameters())[k].grad.norm())>0 for k in o.feature_parameter_names)
                optimizer=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.0001)
                torch.nn.utils.clip_grad_norm_(m.parameters(),5.);optimizer.step();updates+=1
                assert o.optimizer is optimizer and optimizer._sinc_unit_steps==optimizer._sinc_physical_clip_calls==1
                assert m._supcon_handles==[] and m._supcon_pending is None and m._pair_primary is None and m._pair_secondary is None
                assert m.id_backbone.encoder._ordinary_train_packets==512;total_packets+=512
                primary_losses[row['arm']]=float(metrics['pair/primary_objective'])
                records.append(dict(arm=row['arm'],primary_secondary_same_IQ=not c['As'],actual_S_packets=o.channel.packets,
                    actual_S_scenes=sorted({r[2] for r in o.channel.coverage}),CROSS_RX_weight=metrics['F_supcon/weight'],all_L_supervised=True))
        finally:
            supcon.penalty=original_penalty
            for handle in handles:handle.remove()
        assert all(torch.equal(v,ema.state_dict()[k]) for k,v in initial.items())
    assert torch.equal(inputs['C0'],inputs['C1'])
    # Physical channel probes use no network and no TX labels.
    small={k:v[:12] for k,v in meta.items()};channel=Practical('train')
    one,m1=channel.apply(x[:12],small,1,1);two,m2=channel.apply(x[:12],dict(small,sig_i=torch.arange(12).flip(0)),1,1)
    three,m3=channel.apply(x[:12],small,2,223)
    assert torch.equal(one,two) and not torch.equal(one,three)
    assert m1[0]['receiver']==m3[0]['receiver'] and m1[0]['namespace']!=m3[0]['namespace']
    validation=Practical('validation');vm=dict(small,rx_i=torch.ones(12,dtype=torch.long))
    four,_=validation.apply(x[:12],vm,scene=d.SCENES[-1]);five,_=validation.apply(x[:12],vm,scene=d.SCENES[-1])
    assert torch.equal(four,five) and not torch.equal(four,one)
    p,den,valid=supcon.masks(y,rx,'CROSS_RX');assert not bool((p&torch.eye(128,dtype=torch.bool)).any())
    assert torch.equal(p,(y[:,None]==y[None,:])&(rx[:,None]!=rx[None,:]))
    assert not bool((den&(y[:,None]==y[None,:])&(rx[:,None]==rx[None,:])).any())
    zero,stats=supcon.loss(torch.randn(128,8),y,torch.full_like(rx,3),'CROSS_RX');assert float(zero)==0 and stats['valid_anchors']==0
    weights=[supcon.schedule(e) for e in range(1,21)];assert weights[:5]==[0.]*5 and weights[14:]==[.1]*6
    negatives=0
    for change in (dict(epochs=50),dict(test_views=['clean']),dict(checkpoint_sources=['old']),dict(supcon=dict(d.SUPCON,temperature=.1))):reject(lambda:d.validate(dict(d.config(d.rows()[0]),**change)));negatives+=1
    for labels,receivers in ((torch.full_like(y,-1),rx),(y,torch.zeros_like(rx))):reject(lambda:supcon.masks(labels,receivers,'CROSS_RX'));negatives+=1
    reject(lambda:channel.apply(x[:12],dict(small,rx_i=torch.ones(12,dtype=torch.long)),1,1));negatives+=1
    reject(lambda:validation.apply(x[:12],small,scene=d.SCENES[-1]));negatives+=1
    reject(lambda:channel.apply(x[:12],small,1,1,scene=d.SCENES[-1]));negatives+=1
    o.sequence=io.StringIO();reject(lambda:o.record('U_s',(x,y,domains,meta)));negatives+=1
    return dict(status='PASS',synthetic_model_packets=total_packets,synthetic_optimizer_updates=updates,
        synthetic_practical_packets=128+36+24,protocol_negatives=negatives,initial_state_tensors=len(initial),matched_original_F_initialization=True,
        primary_IQ_same_across_rows=True,primary_two_RDrop_only_SupCon=True,no_satellite_feature_alignment=True,
        four_L_forwards=True,default_head_SIG3_Sinc_joint_gradients=True,extra_head_or_model_forwards=0,
        unchanged_own_EMA=True,session_stability=True,named_stream_refresh_and_separation=True,
        real_IQ_packets=0,checkpoint_loads=0,target_access=False,rows=records)
