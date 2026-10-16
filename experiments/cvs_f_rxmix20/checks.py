import copy,io,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .transform import Sessions,mix,inverse
from .source import Observer
from experiments.cvs_f_crossrx_practical20 import model as control
from experiments.cvs_f_crossrx_practical20 import supcon
from experiments.cvs_ordinary_fold20.streams import PrivateAugment,stream
def reject(call):
    try:call()
    except (ValueError,RuntimeError):return
    raise AssertionError('Invalid source/view accepted')
def run(output):
    from SSDG import train_ssdg as native
    from cvsrffi.schedule import make_augmentor
    torch.set_num_threads(2);torch.manual_seed(20261016941);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    c=d.config(d.rows()[0]);a=d.make_args(c,'cpu');m=api.build(c,'cpu').train()
    initial={k:v.clone() for k,v in m.state_dict().items()};baseline=control.build(d.prior.config(d.prior.rows()[0]),'cpu')
    assert initial.keys()==baseline.state_dict().keys() and all(torch.equal(v,baseline.state_dict()[k]) for k,v in initial.items())
    ema=copy.deepcopy(m);assert ema._source_labeled_feature_hook.__self__ is ema and ema._supcon_original_hook.__self__ is ema
    x=torch.randn(128,2,256);y=torch.arange(128)%6;rx=torch.tensor([3,4,6,8]*32);day=torch.arange(128)%3+1
    domains=torch.tensor([{3:3,4:6,6:9,8:12}[int(r)]+int(v)-1 for r,v in zip(rx,day)])
    meta=dict(rx_i=rx,day_i=day,eq_i=torch.ones(128,dtype=torch.long),base_index=torch.arange(128),sig_i=y.clone())
    bank=np.asarray([.003,.006,.009,.012,.016,.02]);s=Sessions(d.MIX_SEED,bank);other=Sessions(d.MIX_SEED,bank)
    for epoch in range(1,21):
        before=torch.get_rng_state().clone();py=np.random.get_state()
        v,ids,k=s.apply(x,meta,epoch);w,_,j=other.apply(x,dict(meta,sig_i=torch.flip(meta['sig_i'],[0])),epoch)
        assert k==j and torch.equal(v,w) and torch.equal(before,torch.get_rng_state())
        after=np.random.get_state();assert py[0]==after[0] and np.array_equal(py[1],after[1]) and py[2:]==after[2:]
        assert s.parameter(epoch,3,1)==s.parameter(epoch,3,1)
    assert len(s.parameters)==240
    for r in (3,4,6,8):
        for date in (1,2,3):assert {s.parameter(epoch,r,date)[2] for epoch in range(1,21)}==set(range(6))
    b=torch.tensor([[.012,.016]]*128,dtype=x.dtype);z=mix(x,b)
    complex_x=torch.complex(x[:,0],x[:,1]);complex_b=torch.complex(b[:,0,None],b[:,1,None])
    expected=(complex_x+complex_b*complex_x.conj())/(1+complex_b.abs().square()).sqrt()
    assert torch.allclose(torch.complex(z[:,0],z[:,1]),expected,atol=1e-6,rtol=1e-6) and torch.allclose(inverse(z,b),x,atol=1e-6,rtol=1e-6)
    for key in ('_feature_l_steps','_legacy_calls','_anchor_calls','_supcon_calls'):setattr(m,key,1110)
    o=Observer(c,bank=bank);o.out=output/'RXMIX';o.out.mkdir();o.m=o.model=m;o.initialize(m,c,o.out)
    cfg=native.build_aug_base_cfg(a)
    with stream(20261016051,0):inner=make_augmentor(cfg)
    expected_aug=PrivateAugment(inner,20261016051);native.configure_augmentor_for_epoch(expected_aug,cfg,6,a)
    packets=0;ordinary_IQ=0
    with o.native_scope(native):
        optimizer=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.0001)
        aug=native.make_augmentor(cfg);native.configure_augmentor_for_epoch(aug,cfg,6,a)
        for step in (1111,1112):
            o._current_L_meta=meta;o._current_L_labels=y.tolist();before=torch.get_rng_state().clone()
            primary=aug(x,labels=y,no_pa=not a.aug_enable_pa_normal);secondary=m._pair_secondary.clone()
            target=expected_aug(x,labels=y,no_pa=not a.aug_enable_pa_normal)
            assert torch.equal(primary,target) and torch.equal(before,torch.get_rng_state()) and not torch.equal(primary,secondary);ordinary_IQ+=256
            maps={level:[] for level in supcon.LEVELS};handles=[];captured=[];original_penalty=supcon.penalty
            for level in supcon.LEVELS:handles.append(m.get_submodule(supcon.PREFIX+level).register_forward_hook(lambda mod,args,out,level=level:maps[level].append(out)))
            def penalty(values,*args):
                result=original_penalty(values,*args);captured.append((dict(values),result[0]));return result
            supcon.penalty=penalty
            try:
                with m._phase1_labeled_feature_context():out=m(primary,y_tx=y,return_aux=True)
                ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
                correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(domains,meta),domains=domains,optimizer_step=step,identity_loss=ce,labeled_count=128)
                full=ce+correction;aux=torch.stack([v[1] for v in captured]).mean()
                assert math.isclose(metrics['F_supcon/weight'],.01,rel_tol=1e-14)
                assert torch.allclose(full,.5*(metrics['pair/primary_objective']+metrics['pair/secondary_objective'])+.01*aux,atol=1e-6)
                for level in supcon.LEVELS:assert len(maps[level])==4 and all(captured[j][0][level] is maps[level][j] for j in (0,1))
                head=dict(m.named_parameters())['id_backbone.encoder.core.id_backbone.cls_head.weight'];assert torch.autograd.grad(aux,head,allow_unused=True,retain_graph=True)[0] is None
                optimizer.zero_grad();full.backward()
                assert all(dict(m.named_parameters())[k].grad is not None and torch.isfinite(dict(m.named_parameters())[k].grad).all() and float(dict(m.named_parameters())[k].grad.norm())>0 for k in o.feature_parameter_names)
                torch.nn.utils.clip_grad_norm_(m.parameters(),5.);optimizer.step();packets+=512
                native._detach_log_mapping(dict(metrics,**{'train/loss':full.detach()}))
                assert optimizer._sinc_unit_steps==optimizer._sinc_physical_clip_calls==step-1110
                assert m._supcon_handles==[] and m._supcon_pending is None and m._pair_primary is None and m._pair_secondary is None
            finally:
                supcon.penalty=original_penalty
                for h in handles:h.remove()
    assert m.id_backbone.encoder._ordinary_train_packets==1024 and all(torch.equal(v,ema.state_dict()[k]) for k,v in initial.items())
    negatives=0
    for change in (dict(epochs=50),dict(test_views=['clean']),dict(checkpoint_sources=['old']),dict(rxmix=dict(c['rxmix'],b_abs_max=.03)),dict(As=True)):
        reject(lambda:d.validate(dict(c,**change)));negatives+=1
    for bad in (torch.full((128,2),.03),torch.full((128,2),float('nan')),torch.ones(127,2)):
        reject(lambda:mix(x,bad));negatives+=1
    reject(lambda:mix(x*float('nan'),b));negatives+=1
    for change in (dict(rx_i=torch.ones(128,dtype=torch.long)),dict(rx_i=torch.full((128,),2,dtype=torch.long)),dict(eq_i=torch.zeros(128,dtype=torch.long)),dict(day_i=torch.zeros(128,dtype=torch.long))):
        reject(lambda:Sessions(d.MIX_SEED,bank).apply(x,dict(meta,**change),1));negatives+=1
    reject(lambda:s.parameter(21,3,1));negatives+=1
    reject(lambda:Sessions(d.MIX_SEED,[.03]*6));negatives+=1
    o.sequence=io.StringIO();reject(lambda:o.record('U_s',(x,y,domains,meta)));negatives+=1
    return dict(status='PASS',synthetic_model_packets=packets,synthetic_optimizer_updates=2,synthetic_transform_packets=128*40,synthetic_ordinary_IQ_packets=ordinary_IQ,protocol_negatives=negatives,
        initial_state_tensors=len(initial),matched_original_F_initialization=True,primary_IQ_exact_C0=True,exact_widely_linear_formula=True,bounded_inverse_and_power=True,
        stable_RX_day_session_all_TX=True,all20epoch12session_coverage=True,session_seed_not_TX=True,primary_two_RDrop_only_SupCon=True,four_L_forwards=True,default_head_SIG3_Sinc_joint_gradients=True,
        extra_model_forwards=0,unchanged_own_EMA=True,global_RNG_preserved=True,real_IQ_packets=0,checkpoint_loads=0,target_access=False)
