import copy
import pytest
import torch
from experiments.cvs_urban_ce import design as d
from experiments.cvs_urban_ce.augmentation import bounded_echo,satellite_keep

def test_matrix_and_no_hidden_teacher_or_losses():
    assert len(d.rows())==24
    assert len({d.config(r)['output_root'] for r in d.rows()})==24
    for arm in d.ARMS:
        c=d.config(next(r for r in d.rows() if r['arm']==arm));a=d.make_args(c,'cpu')
        assert not any((a.use_ema_teacher,a.use_muse_ssdg,a.fasttrust_rc4,a.use_unlabeled,a.use_adv3b02_daot_stn,a.use_mixstyle,a.use_sat_consistency))
        assert not a.baseline_ckpt and not a.teacher_ckpt and a.from_scratch
        assert a.sat_cons_start_epoch==80
        assert all(v==0 for k,v in vars(a).items() if k.startswith('lambda_') and k!='lambda_sat_cls')
        if c['urban']:assert a.sat_view_schedule==d.URBAN_SCHEDULE
        bad=copy.deepcopy(c);bad['echo_l1_max']=.5
        with pytest.raises(ValueError):d.validate(bad)

def test_echo_bound_replay_energy_clean_and_gradient():
    x=torch.randn(128,2,256,requires_grad=True)
    kwargs=dict(seed=97,epoch=100,step=7)
    y,info=bounded_echo(x,**kwargs);again,_=bounded_echo(x,**kwargs)
    assert torch.equal(y,again)
    assert 0<info['dg_selected']<128 and info['dg_actual_max']<=.150001
    assert torch.allclose(x.square().mean((1,2)),y.square().mean((1,2)),atol=1e-6)
    unchanged=(x==y).all(2).all(1)
    assert int(unchanged.sum())==128-info['dg_selected']
    y.square().mean().backward();assert torch.isfinite(x.grad).all()
    initial,_=bounded_echo(x,seed=97,epoch=1,step=7)
    assert torch.equal(x,initial)

def test_thinning_has_both_paths_and_unbiased_auxiliary_weight():
    draws=[satellite_keep(17001,100,i,.5) for i in range(10000)]
    assert .48<sum(draws)/len(draws)<.52
    assert .96<sum(2*int(v) for v in draws)/len(draws)<1.04
    assert all(satellite_keep(17001,100,i,1.) for i in range(8))

def test_native_skip_has_no_satellite_renderer_and_preserves_clean():
    from experiments.cvs_urban_ce.runtime import installed
    c=d.config(next(r for r in d.rows() if r['arm']=='urban_dg_thin'));a=d.make_args(c,'cpu')
    import tempfile
    class Renderer:
        def transform(self,*args,**kwargs):raise AssertionError('Unused satellite renderer invoked')
    with tempfile.TemporaryDirectory() as out:
        a.output_dir=out
        with installed(c) as n:
            x=torch.randn(8,2,256);y=torch.arange(8)%6;dom=torch.zeros(8,dtype=torch.long)
            for epoch,step in [(1,0),(100,next(i for i in range(100) if not satellite_keep(c['augmentation_seed'],100,i,.5)))]:
                value,labels,domains,view,info=n._prepare_concat_sat_batch_for_training(Renderer(),x,y,dom,args=a,epoch=epoch,batch_idx=step)
                assert value.shape==x.shape and labels is y and domains is dom and view is None
                assert a.concat_sat_ce_weight==0 and not a.use_sat_consistency

def test_pairs_cover_all_fixed_arms():
    from experiments.cvs_urban_ce.evaluate import paired_results,VIEWS
    results=[dict(arm=a,model_seed=s,view=v,dimension='overall',accuracy=.5,macro_f1=.4) for a in d.ARMS for s in d.SEEDS for v in VIEWS]
    pairs=paired_results(results)
    assert len(pairs)==7*7*2 and all(p['mean_difference']==0 for p in pairs)
