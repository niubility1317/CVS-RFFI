import copy,io
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .jitter import OriginJitter
from .source import Observer
from experiments.cvs_raw_anchor20 import model as reference

def run(output):
    from SSDG import train_ssdg as native
    from experiments.cvs_f_practical_source20.checks import reject
    from experiments.cvs_urban_sigtime.channel import imports
    imports()
    from leo_practical.equalization import linear_filter
    torch.set_num_threads(2);torch.manual_seed(20261016323);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    c=d.config(d.rows()[0]);a=d.make_args(c,'cpu');m=api.build(c,'cpu');baseline=reference.build(d.control_config(),'cpu')
    initial={k:v.clone() for k,v in m.state_dict().items()}
    assert initial.keys()==baseline.state_dict().keys() and all(torch.equal(v,baseline.state_dict()[k]) for k,v in initial.items())
    ema=copy.deepcopy(m);assert ema._anchor_original_hook.__self__ is ema and ema._source_labeled_feature_hook.__self__ is ema
    o=Observer(c);o.m=m;o.model=m;o.out=output
    x=torch.randn(128,2,256);y=torch.arange(128)%6;meta=dict(rx_i=torch.tensor([3,4,6,8]*32),day_i=torch.arange(128)%3+1)
    delays=torch.arange(25);shifted=OriginJitter.transform(x[:25],delays)
    for j,delay in enumerate(range(25)):
        h=np.zeros(delay+1,complex);h[delay]=1.;wave=np.asarray(x[j,0].tolist())+1j*np.asarray(x[j,1].tolist());expected,_=linear_filter(wave,h,'reflect')
        actual=np.asarray(shifted[j,0].tolist())+1j*np.asarray(shifted[j,1].tolist());assert np.array_equal(actual,expected.astype(np.complex64))
    assert torch.equal(shifted[0],x[0])
    before=torch.get_rng_state().clone()
    with o.native_scope(native):
        cfg=native.build_aug_base_cfg(a);aug=native.make_augmentor(cfg)
        assert cfg['p_time_shift']==1 and all(cfg['p_'+k]==0 for k in ('dac','pa','amp_scale','phase_rot','cfo','phase_noise','awgn','multipath','dc_offset','bandedge_taper'))
        state=native.configure_augmentor_for_epoch(aug,cfg,1,a);assert state['scale']==1 and state['max_time_shift']==24 and state['p_awgn']==0
        raw=aug(x,labels=y,no_pa=not a.aug_enable_pa_normal);jitter=m._anchor_G.clone()
        assert torch.equal(raw,x) and torch.equal(before,torch.get_rng_state())
        m.train();views=[];forward=m.forward_identity_only
        def track(value,*args,**kw):views.append(value.detach().clone());return forward(value,*args,**kw)
        m.forward_identity_only=track
        with m._phase1_labeled_feature_context():out=m(raw,y_tx=y,return_aux=True)
        loss=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
        correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(None,meta),domains=torch.zeros(128,dtype=torch.long),optimizer_step=1,identity_loss=loss,labeled_count=128)
        total=loss+correction;total.backward()
        assert len(views)==3 and torch.equal(views[0],x) and all(torch.equal(value,jitter) for value in views[1:])
        assert abs(float(total.detach()-.5*(metrics['anchor/pair1_objective']+metrics['anchor/pair2_objective'])))<1e-6
        assert m.id_backbone.encoder._ordinary_train_packets==512 and m._anchor_G is None and m._anchor_raw is None and m._legacy_actual_L_input is None
        for name in o.feature_parameter_names:
            p=dict(m.named_parameters())[name];assert p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.norm())>0
        opt=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.0001);torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
        assert o.optimizer is opt and opt._sinc_unit_steps==opt._sinc_physical_clip_calls==1
    assert all(torch.equal(value,ema.state_dict()[key]) for key,value in initial.items())
    first=OriginJitter(d.ORIGIN['seed']);second=OriginJitter(d.ORIGIN['seed']);before=torch.get_rng_state().clone()
    one=first(x,labels=y);two=second(x,labels=y.flip(0));three=first(x,labels=y)
    assert torch.equal(one,two) and not torch.equal(one,three) and torch.equal(before,torch.get_rng_state())
    zero=OriginJitter(d.ORIGIN['seed'])(torch.zeros_like(x));assert torch.equal(zero,torch.zeros_like(x))
    summary=o.augmentor.summary();assert sum(summary['delay_counts'])==128 and not summary['circular_roll'] and not summary['post_normalization']
    negatives=0
    for change in (dict(epochs=50),dict(test_views=['clean']),dict(checkpoint_sources=['old']),dict(origin_jitter=dict(d.ORIGIN,delay_max=32))):reject(lambda:d.validate(dict(c,**change)));negatives+=1
    for value,flag in ((x.double(),True),(x,False)):reject(lambda:OriginJitter(1)(value,no_pa=flag));negatives+=1
    for delay in (-1,25):reject(lambda:OriginJitter.transform(x,torch.full((128,),delay)));negatives+=1
    o.sequence=io.StringIO()
    for labels,rx in ((y,meta['rx_i']),(torch.full_like(y,-1),torch.ones_like(y))):
        reject(lambda:o.record('U_s',(x,labels,None,dict(meta,rx_i=rx))));negatives+=1
    common=dict(held_V=dict(count=5400,accuracy=.8,macro_f1=.8),seen_V=dict(source_val_accuracy=.98),target_access=False,U_truth_read=False,head_fit_updates=0,L_forward_packets=2273280,complete_health=True)
    good=copy.deepcopy(common);good['held_V'].update(accuracy=.82,macro_f1=.82);assert d.qualify(good,common)['status']=='PASS'
    for path in ('accuracy','macro_f1','seen'):
        bad=copy.deepcopy(good)
        if path=='seen':bad['seen_V']['source_val_accuracy']=.974
        else:bad['held_V'][path]=.8 if path=='accuracy' else .79
        assert d.qualify(bad,common)['status']=='FAIL';negatives+=1
    return dict(status='PASS',synthetic_network_forwards=512,valid_synthetic_jitter_packets=640,explicit_formula_packets=25,synthetic_optimizer_updates=1,
        protocol_negatives=negatives,initial_state_tensors=len(initial),matched_F_A0_initial_state=True,four_L_forwards=True,half_raw_half_jitter=True,
        same_IQ_within_RDpair=True,default_head_three_SIG_Sinc_gradients=True,full_25_origins_match_original_FIR=True,zero_packets_preserved=True,
        labels_independent=True,named_stream_restored=True,steps_views_refresh=True,old_U_truth_hidden=True,target_access=False,checkpoint_loads=0,real_IQ_packets=0)
