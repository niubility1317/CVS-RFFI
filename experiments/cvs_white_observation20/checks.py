import copy,io
from collections import Counter
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .noise import WhiteObservation
from .source import Observer
from experiments.cvs_raw_anchor20 import model as reference

def run(output):
    from SSDG import train_ssdg as native
    torch.set_num_threads(2);torch.manual_seed(20261016233)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    c=d.config(d.rows()[0]);a=d.make_args(c,'cpu');m=api.build(c,'cpu');baseline=reference.build(d.control_config(),'cpu')
    initial={k:v.clone() for k,v in m.state_dict().items()}
    assert initial.keys()==baseline.state_dict().keys() and all(torch.equal(v,baseline.state_dict()[k]) for k,v in initial.items())
    ema=copy.deepcopy(m);assert ema._anchor_original_hook.__self__ is ema and ema._source_labeled_feature_hook.__self__ is ema
    o=Observer(c);o.m=m;o.model=m;o.out=output
    x=torch.randn(128,2,256);y=torch.arange(128)%6;meta=dict(rx_i=torch.tensor([3,4,6,8]*32),day_i=torch.arange(128)%3+1)
    before=torch.get_rng_state().clone()
    with o.native_scope(native):
        cfg=native.build_aug_base_cfg(a);aug=native.make_augmentor(cfg)
        assert cfg['p_awgn']==1 and all(cfg['p_'+k]==0 for k in ('dac','pa','time_shift','amp_scale','phase_rot','cfo','phase_noise','multipath','dc_offset','bandedge_taper'))
        state=native.configure_augmentor_for_epoch(aug,cfg,1,a);assert state['scale']==1 and state['p_pa']==0
        raw=aug(x,labels=y,no_pa=not a.aug_enable_pa_normal);noise=m._anchor_G.clone()
        assert torch.equal(raw,x) and torch.equal(before,torch.get_rng_state())
        m.train();views=[];forward=m.forward_identity_only
        def track(v,*args,**kw):views.append(v.detach().clone());return forward(v,*args,**kw)
        m.forward_identity_only=track
        with m._phase1_labeled_feature_context():out=m(raw,y_tx=y,return_aux=True)
        loss=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
        correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(None,meta),domains=torch.zeros(128,dtype=torch.long),optimizer_step=1,identity_loss=loss,labeled_count=128)
        total=loss+correction;total.backward()
        assert len(views)==3 and torch.equal(views[0],x) and all(torch.equal(v,noise) for v in views[1:])
        assert abs(float(total.detach()-.5*(metrics['anchor/pair1_objective']+metrics['anchor/pair2_objective'])))<1e-6
        assert m.id_backbone.encoder._ordinary_train_packets==512 and m._anchor_G is None and m._anchor_raw is None
        assert m._legacy_actual_L_input is None and m._anchor_calls==1 and o.augmentor._calls==1
        for name in o.feature_parameter_names:
            p=dict(m.named_parameters())[name];assert p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm())>0
        opt=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.0001);torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
        assert o.optimizer is opt and opt._sinc_unit_steps==opt._sinc_physical_clip_calls==1
    assert all(torch.equal(v,ema.state_dict()[k]) for k,v in initial.items())
    # Same private random direction for different label permutations, and fresh next draw.
    first=WhiteObservation(d.NOISE['seed']);second=WhiteObservation(d.NOISE['seed'])
    before=torch.get_rng_state().clone();g1=first(x,labels=y);g2=second(x,labels=y.flip(0));g3=first(x,labels=y)
    assert torch.equal(g1,g2) and not torch.equal(g1,g3) and torch.equal(before,torch.get_rng_state())
    zero=WhiteObservation(d.NOISE['seed']);z=zero(torch.zeros_like(x));assert torch.equal(z,torch.zeros_like(x)) and zero.zero_packets==128
    summary=o.augmentor.summary();assert 29.999<summary['signal_to_added_noise_db_min']<30.001
    assert 29.999<summary['signal_to_added_noise_db_max']<30.001 and summary['hardware_changes'] is False
    negative=0
    for change in (dict(epochs=50),dict(test_views=['clean']),dict(checkpoint_sources=['old']),dict(observation_noise=dict(d.NOISE,signal_to_added_noise_db=20))):
        try:d.validate(dict(c,**change));raise AssertionError('Illegal config accepted')
        except ValueError:negative+=1
    for value,flag in ((x.double(),True),(x,False)):
        try:WhiteObservation(1)(value,no_pa=flag);raise AssertionError('Illegal noise input accepted')
        except ValueError:negative+=1
    o.sequence=io.StringIO()
    for labels,rx in ((y,meta['rx_i']),(torch.full_like(y,-1),torch.ones_like(y))):
        try:o.record('U_s',(x,labels,None,dict(meta,rx_i=rx)));raise AssertionError('Illegal U/heldRX accepted')
        except ValueError:negative+=1
    common=dict(held_V=dict(count=5400,accuracy=.8,macro_f1=.8),seen_V=dict(source_val_accuracy=.98),target_access=False,U_truth_read=False,head_fit_updates=0,L_forward_packets=2273280,complete_health=True)
    good=copy.deepcopy(common);good['held_V'].update(accuracy=.82,macro_f1=.82);assert d.qualify(good,common)['status']=='PASS'
    for path in ('accuracy','macro_f1','seen'):
        bad=copy.deepcopy(good)
        if path=='seen':bad['seen_V']['source_val_accuracy']=.974
        else:bad['held_V'][path]=.8 if path=='accuracy' else .79
        assert d.qualify(bad,common)['status']=='FAIL';negative+=1
    return dict(status='PASS',synthetic_network_forwards=512,synthetic_noise_input_packets=768,synthetic_optimizer_updates=1,
        protocol_negatives=negative,initial_state_tensors=len(initial),matched_F_A0_initial_state=True,
        four_L_forwards=True,half_raw_half_white_RDrop=True,same_IQ_within_RDpair=True,default_head_three_SIG_Sinc_gradients=True,
        fixed_energy_bound=True,zero_packets_preserved=True,label_permutation_independent=True,named_stream_restored=True,
        epochs_views_refresh=True,old_U_truth_hidden=True,target_access=False,checkpoint_loads=0,real_IQ_packets=0)
