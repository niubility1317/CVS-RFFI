import copy
import io
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .source import Observer,NativeView
from experiments.cvs_ordinary_fold20.source import Observer as OriginalObserver
from experiments.cvs_ordinary_fold20.streams import stream,PrivateAugment

def run(reference_builder):
    from SSDG import train_ssdg as native
    torch.set_num_threads(2);torch.manual_seed(20261016057)
    x=torch.randn(128,2,256);y=torch.arange(128)%6
    meta=dict(rx_i=torch.tensor([3,4,6,8]*32),day_i=torch.arange(128)%3+1)
    initial=None;generated=None;records=[];negative=0;updates=0
    for row in d.rows():
        c=d.config(row);a=d.make_args(c,'cpu');m=api.build(c,'cpu');o=Observer(c);o.m=m
        state={k:v.clone() for k,v in m.state_dict().items()}
        if initial is None:initial=state
        else:assert all(torch.equal(v,initial[k]) for k,v in state.items())
        reference=reference_builder()
        assert set(state)==set(reference.state_dict()) and all(torch.equal(v,reference.state_dict()[k]) for k,v in state.items())
        assert type(m.id_backbone.encoder.core.id_backbone.sinc).__name__=='CoordinateSinc'
        assert len(m.id_backbone.encoder.sequence_pool.views)==3
        ema=copy.deepcopy(m)
        assert ema._source_labeled_feature_hook.__self__ is ema and ema._anchor_original_hook.__self__ is ema
        assert ema.id_backbone.encoder._ordinary_original_features.__self__ is ema.id_backbone.encoder
        before=torch.get_rng_state().clone();actual=x
        if row['arm']!='A0':
            cfg=native.build_aug_base_cfg(a)
            with stream(20261016051,0):inner=native.make_augmentor(cfg)
            private=PrivateAugment(inner,20261016051);aug=NativeView(private,o)
            native.configure_augmentor_for_epoch(aug,cfg,13,a)
            actual=aug(x.clone(),labels=y,no_pa=not a.aug_enable_pa_normal)
            assert private._calls==1 and private._packets==128 and aug.p_pa>0
            if generated is None:generated=m._anchor_G.clone()
            else:assert torch.equal(generated,m._anchor_G)
            assert torch.equal(actual,generated if row['arm']=='A1' else x)
        assert torch.equal(before,torch.get_rng_state())
        m.train();seen=[];original=m.forward_identity_only
        def tracked(v,*args,**kw):seen.append(v.detach().clone());return original(v,*args,**kw)
        m.forward_identity_only=tracked
        with m._phase1_labeled_feature_context():out=m(actual,y_tx=y,return_aux=True)
        loss=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
        correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(None,meta),domains=torch.zeros(128,dtype=torch.long),optimizer_step=1,identity_loss=loss,labeled_count=128)
        total=loss+correction;total.backward()
        second=x if row['arm']=='A0' else generated
        assert len(seen)==3 and torch.equal(seen[0],actual) and all(torch.equal(v,second) for v in seen[1:])
        assert m._legacy_calls==1 and m._anchor_calls==1 and m._anchor_extra_packets==256 and m.id_backbone.encoder._ordinary_train_packets==512
        assert m._anchor_G is None and m._anchor_raw is None and m._legacy_actual_L_input is None
        assert abs(float(total.detach()-metrics['legacy_rdrop/paired_CE']-metrics['legacy_rdrop/symmetric_KL']))<1e-6
        assert abs(float(total.detach()-.5*(metrics['anchor/pair1_objective']+metrics['anchor/pair2_objective'])))<1e-6
        for suffix in ('cls_head.weight','cls_head.gain',*[f'sequence_pool.views.{i}.local_projection.weight' for i in range(3)],'sinc.low_coordinate','sinc.band_coordinate'):
            p=next(p for k,p in m.named_parameters() if k.endswith(suffix));assert p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm())>0
        assert all(torch.equal(v,ema.state_dict()[k]) for k,v in initial.items())
        # One synthetic update per arm, through the actual native optimizer scope.
        from .f_base.optimizer import native_scope,build,clip_physical
        reference_optimizer=build(reference,list(reference.parameters()),'CENTER_UNITS',lr=.001,weight_decay=.0001)
        for k,p in reference.named_parameters():p.grad=dict(m.named_parameters())[k].grad.clone() if dict(m.named_parameters())[k].grad is not None else None
        expected_norm=clip_physical(reference_optimizer,reference.parameters(),5.)
        o.model=m
        with native_scope(o,'CENTER_UNITS'):
            actual_optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
            actual_norm=torch.nn.utils.clip_grad_norm_(m.parameters(),5.)
            assert torch.equal(actual_norm,expected_norm)
            actual_optimizer.step();reference_optimizer.step()
        assert actual_optimizer._sinc_unit_steps==actual_optimizer._sinc_physical_clip_calls==1
        assert actual_optimizer._sinc_unit_lr_records==reference_optimizer._sinc_unit_lr_records
        assert all(torch.equal(v,reference.state_dict()[k]) for k,v in m.state_dict().items())
        updates+=2
        assert o.expected_L_packets==2273280
        records.append(dict(arm=row['arm'],L_forwards=512,ordinary_draws=int(row['arm']!='A0'),first_view=c['training_mechanism']['first_pair'],second_view=c['training_mechanism']['second_pair']))
    assert OriginalObserver(d.old.config(d.old.rows()[0])).expected_L_packets==1136640
    for changes in (dict(epochs=50),dict(test_views=['clean']),dict(checkpoint_sources=['old'])):
        try:d.validate(dict(d.config(d.rows()[0]),**changes));raise AssertionError('Illegal config accepted')
        except ValueError:negative+=1
    c=d.config(d.rows()[0]);o=Observer(c);o.sequence=io.StringIO()
    for labels,rx in ((y,meta['rx_i']),(torch.full((128,),-1),torch.ones(128,dtype=torch.long))):
        try:o.record('U_s',(x,labels,None,dict(meta,rx_i=rx)));raise AssertionError('Illegal U/heldRX accepted')
        except ValueError:negative+=1
    # No model forward is needed to test invalid cache/declared-view paths.
    for other,raw,arm in ((None,None,'A1'),(torch.zeros_like(x),torch.zeros_like(x),'A2')):
        fake=SimpleNamespace(_legacy_actual_L_input=x,_anchor_calls=0,_anchor_G=other,_anchor_raw=raw,_anchor_augmented=True,_anchor_arm=arm)
        try:api.hook(fake,optimizer_step=1);raise AssertionError('Invalid view cache accepted')
        except ValueError:negative+=1
    common=dict(held_V=dict(count=5400,accuracy=.8,macro_f1=.8),seen_V=dict(source_val_accuracy=.98),target_access=False,U_truth_read=False,head_fit_updates=0,L_forward_packets=2273280,complete_health=True)
    results={a:copy.deepcopy(common) for a in d.ARMS};results['A2']['held_V'].update(accuracy=.82,macro_f1=.82)
    assert d.qualify(results)['status']=='PASS'
    for change in ('accuracy','macro_f1'):
        z=copy.deepcopy(results);z['A1']['held_V'][change]=.83;assert d.qualify(z)['status']=='FAIL';negative+=1
    z=copy.deepcopy(results);z['A2']['seen_V']['source_val_accuracy']=.974;assert d.qualify(z)['status']=='FAIL';negative+=1
    return dict(status='PASS',rows=records,initial_state_tensors=len(initial),synthetic_network_forwards=1536,
        synthetic_augmentation_packets=256,protocol_negatives=negative,same_three_initial_states=True,
        same_A1_A2_generated_IQ=True,same_IQ_within_each_RDpair=True,mean_two_pair_objective=True,
        default_head_SIG_gradients=True,F_three_projection_and_coordinate_gradients=True,
        exact_authoritative_F_initial_state=True,physical_clip_and_optimizer_reference_match=True,
        old_two_forward_counter_regression=True,U_truth_and_held_RX_rejected=True,
        checkpoint_loads=0,real_IQ_packets=0,synthetic_optimizer_updates=updates,target_access=False)
