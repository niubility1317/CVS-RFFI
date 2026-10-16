import copy,gzip,io,json,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import design as d,model as api,channel as ch
from .policy import Policy,aggregate,eligible,FIT
from .source import Observer
from experiments.cvs_f_crossrx_folds20.channel import Practical as Nominal
from experiments.cvs_f_crossrx_folds20 import model as control
from experiments.cvs_f_crossrx_practical20 import supcon
from experiments.cvs_ordinary_fold20.streams import PrivateAugment,stream
def reject(f):
    try:f()
    except (ValueError,RuntimeError,KeyError):return
    raise AssertionError('Invalid course/source input accepted')
def fixture():
    cells=[];i=0
    for t in range(6):
        for r in FIT:
            for dd in (1,2,3):
                n=394+int(i<48);i+=1;cells.append(dict(TX=t,RX=r,day=dd,count=n,correct=int(.9*n),margin_sum=.1*n,CE_sum=.3*n))
    return cells,aggregate(cells,1.)
def metadata(n):
    rx=torch.tensor([FIT[i%4] for i in range(n)]);day=torch.arange(n)%3+1;y=torch.arange(n)%6
    return y,dict(rx_i=rx,day_i=day,eq_i=torch.ones(n,dtype=torch.long),base_index=torch.arange(n),sig_i=y.clone())
def run(output):
    from SSDG import train_ssdg as native
    from cvsrffi.schedule import make_augmentor
    torch.set_num_threads(2);torch.manual_seed(20261016661);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cells,good=fixture();policy=Policy()
    for e in range(1,21):
        before=policy.eta;r=policy.advance(e,good)
        assert r['eta_used']==before and 0<=r['eta_next']<=.5 and (e>5 or before==0.)
    assert policy.history[4]['eta_next']==.1 and policy.summary()['max_eta']==.5
    held=Policy()
    for e in range(1,6):held.advance(e,good)
    held.advance(6,dict(good,gradient=10.));assert held.eta==.1
    assert not eligible('practical_low_urban',dict(signal_to_added_noise_db=10.,receiver_lock_status='unlocked'))
    assert eligible('practical_low_urban',dict(signal_to_added_noise_db=10.,receiver_lock_status='locked'))
    c=d.config(d.rows()[0]);a=d.make_args(c,'cpu');m=api.build(c,'cpu').train();initial={k:v.clone() for k,v in m.state_dict().items()};baseline=control.build(d.original('C1'),'cpu')
    assert initial.keys()==baseline.state_dict().keys() and all(torch.equal(v,baseline.state_dict()[k]) for k,v in initial.items())
    ema=copy.deepcopy(m);assert ema._source_labeled_feature_hook.__self__ is ema and ema._supcon_original_hook.__self__ is ema
    x=torch.randn(128,2,256);y,meta=metadata(128);domains=torch.tensor([{1:0,3:3,4:6,8:12}[int(r)]+int(dd)-1 for r,dd in zip(meta['rx_i'],meta['day_i'])])
    nominal=Nominal('train',6);zero=ch.Practical(6,Policy());before=torch.get_rng_state().clone();np_before=np.random.get_state()
    expected,reference=nominal.apply(x[:8],{k:v[:8] for k,v in meta.items()},1,1);got,records=zero.apply(x[:8],{k:v[:8] for k,v in meta.items()},1,1)
    assert torch.equal(expected,got) and all(reference[i][k]==records[i][k] for i in range(8) for k in reference[i])
    assert zero.proposal_packets==zero.packets==8 and zero.hard_requested==0
    # Force absence of an eligible proposal: fallback is exact original nominal, never the final failed draw.
    pp=Policy()
    for e in range(1,6):pp.advance(e,good)
    fallback=ch.Practical(6,pp);expected2,_=Nominal('train',6).apply(x,meta,6,1111);original_eligible=ch.eligible;ch.eligible=lambda scene,m:False
    try:got2,rr=fallback.apply(x,meta,6,1111)
    finally:ch.eligible=original_eligible
    assert torch.equal(got2,expected2) and fallback.hard_requested>0 and fallback.hard_accepted==0 and fallback.hard_fallback==fallback.hard_requested
    assert all(v['curriculum']['selected_index']==0 for v in rr) and fallback.proposal_packets==128+4*fallback.hard_requested
    assert torch.equal(before,torch.get_rng_state());np_after=np.random.get_state();assert np_before[0]==np_after[0] and np.array_equal(np_before[1],np_after[1]) and np_before[2:]==np_after[2:]
    for key in ('_feature_l_steps','_legacy_calls','_anchor_calls','_supcon_calls','_course_calls'):setattr(m,key,1110)
    o=Observer(c);o.out=output/'C2';o.out.mkdir();o.m=o.model=m;o.initialize(m,c,o.out)
    for e in range(1,6):o.policy.advance(e,good)
    cfg=native.build_aug_base_cfg(a)
    with stream(20261016051,0):inner=make_augmentor(cfg)
    expected_aug=PrivateAugment(inner,20261016051);native.configure_augmentor_for_epoch(expected_aug,cfg,6,a);packets=0
    with o.native_scope(native):
        optimizer=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.0001);aug=native.make_augmentor(cfg);native.configure_augmentor_for_epoch(aug,cfg,6,a)
        for step in (1111,1112):
            o._current_L_meta=meta;o._current_L_labels=y.tolist();before=torch.get_rng_state().clone()
            primary=aug(x,labels=y,no_pa=not a.aug_enable_pa_normal);target=expected_aug(x,labels=y,no_pa=not a.aug_enable_pa_normal)
            assert torch.equal(primary,target) and torch.equal(before,torch.get_rng_state())
            maps={level:[] for level in supcon.LEVELS};handles=[];captured=[];original_penalty=supcon.penalty
            for level in supcon.LEVELS:handles.append(m.get_submodule(supcon.PREFIX+level).register_forward_hook(lambda mod,args,out,level=level:maps[level].append(out)))
            def penalty(values,*args):
                v=original_penalty(values,*args);captured.append((dict(values),v[0]));return v
            supcon.penalty=penalty
            try:
                with m._phase1_labeled_feature_context():out=m(primary,y_tx=y,return_aux=True)
                ce=F.cross_entropy(out['tx_logits'],y,label_smoothing=.01)
                correction,metrics=m._source_labeled_feature_hook(model=m,labeled_output=out,labels=y,extra=(domains,meta),domains=domains,optimizer_step=step,identity_loss=ce,labeled_count=128)
                full=ce+correction;aux=torch.stack([v[1] for v in captured]).mean();assert math.isclose(metrics['F_supcon/weight'],.01,rel_tol=1e-14)
                assert torch.allclose(full,.5*(metrics['pair/primary_objective']+metrics['pair/secondary_objective'])+.01*aux,atol=1e-6)
                for level in supcon.LEVELS:assert len(maps[level])==4 and all(captured[j][0][level] is maps[level][j] for j in (0,1))
                head=dict(m.named_parameters())['id_backbone.encoder.core.id_backbone.cls_head.weight'];assert torch.autograd.grad(aux,head,allow_unused=True,retain_graph=True)[0] is None
                optimizer.zero_grad();full.backward();assert all(dict(m.named_parameters())[k].grad is not None and torch.isfinite(dict(m.named_parameters())[k].grad).all() and float(dict(m.named_parameters())[k].grad.norm())>0 for k in o.feature_parameter_names)
                torch.nn.utils.clip_grad_norm_(m.parameters(),5.);optimizer.step();packets+=512;native._detach_log_mapping(dict(metrics,**{'train/loss':full.detach()}))
                assert optimizer._sinc_unit_steps==optimizer._sinc_physical_clip_calls==step-1110 and m._course_calls==step
                assert m._supcon_handles==[] and m._supcon_pending is None and m._pair_primary is None and m._pair_secondary is None
            finally:
                supcon.penalty=original_penalty
                for h in handles:h.remove()
    assert sum(v[0] for v in m._course_cells.values())==256 and ema._course_cells=={} and all(torch.equal(v,ema.state_dict()[k]) for k,v in initial.items())
    with gzip.open(o.out/'curriculum_draws.jsonl.gz','rt',encoding='utf-8') as f:draws=[json.loads(line) for line in f]
    assert o.channel.hard_requested>0 and o.channel.proposal_packets>=256 and len(draws)==2
    assert all(v['selected_index']==0 or v['proposals'][v['selected_index']]['eligible'] for r in draws for v in r['records'])
    negatives=0
    for change in (dict(epochs=50),dict(held_RX=1),dict(checkpoint_sources=['old']),dict(curriculum=dict(c['curriculum'],eta_max=.6)),dict(As=False)):
        reject(lambda:d.validate(dict(c,**change)));negatives+=1
    for change in (dict(rx_i=torch.full((128,),6,dtype=torch.long)),dict(rx_i=torch.full((128,),2,dtype=torch.long)),dict(eq_i=torch.zeros(128,dtype=torch.long)),dict(day_i=torch.zeros(128,dtype=torch.long))):
        reject(lambda:ch.Practical(6,Policy()).apply(x,dict(meta,**change),1,1));negatives+=1
    bad=copy.deepcopy(cells);bad[0]['RX']=6;reject(lambda:aggregate(bad,1.));negatives+=1
    reject(lambda:aggregate(cells[:-1],1.));negatives+=1
    reject(lambda:Policy().advance(2,good));negatives+=1
    reject(lambda:ch.Practical(6,Policy()).apply(x,meta,1,1,scene='practical_low_urban'));negatives+=1
    o.sequence=io.StringIO();reject(lambda:o.record('U_s',(x,y,domains,meta)));negatives+=1
    return dict(status='PASS',synthetic_model_packets=packets,synthetic_optimizer_updates=2,protocol_negatives=negatives,initial_state_tensors=len(initial),initial_F_matches_C1=True,
        eta0_exact_original_C1_IQ_and_physics=True,finite_fallback_exact_nominal=True,old_unlocked_selector_counterexample=True,all20_policy_progress_bound=True,
        source_fitL_only_progress=True,stable_RX_day_hardware=True,primary_C0_IQ_exact=True,four_L_forwards=True,primary_only_SupCon=True,no_extra_loss=True,default_head_SIG3_Sinc_joint_gradients=True,
        private_RNG_preserved=True,own_EMA_unchanged=True,synthetic_selected_channel_packets=nominal.packets+zero.packets+128+fallback.packets+o.channel.packets,
        synthetic_full_channel_proposals=nominal.packets+zero.proposal_packets+128+fallback.proposal_packets+o.channel.proposal_packets,
        extra_model_forwards=0,real_IQ_packets=0,checkpoint_loads=0,target_access=False)
