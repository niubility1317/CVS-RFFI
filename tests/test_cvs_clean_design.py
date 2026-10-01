import math
import torch
import pytest
from experiments.cvs_clean_design.model import build, VARIANTS, OrthogonalMemoryLift, MomentPool
from experiments.cvs_clean_design.source import validate_config


def rotate(x, angle):
    c,s=math.cos(angle),math.sin(angle)
    return torch.stack((c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1]),1)


def test_orthogonal_lift_phase_geometry_and_gradients():
    torch.manual_seed(7);x=torch.randn(3,2,256,requires_grad=True)
    lift=OrthogonalMemoryLift(memory_depth=1)
    b=lift(x).reshape(3,3,2,256)
    rotated=lift(rotate(x,.7)).reshape(3,3,2,256)
    expected=torch.stack([rotate(b[:,j],.7) for j in range(3)],1)
    torch.testing.assert_close(rotated,expected,atol=3e-5,rtol=3e-5)
    for a,c in ((0,1),(0,2),(1,2)):
        assert (b[:,a]*b[:,c]).sum((1,2)).abs().max()<.02
    b.square().mean().backward();assert torch.isfinite(x.grad).all()
    for value in (0.,1.):
        assert torch.isfinite(lift(torch.full((2,2,256),value))).all()


def test_moment_pool_preserves_shape_initial_mean_and_variance_information():
    p=MomentPool(1);x=torch.tensor([[[0.,2.,0.,2.]],[[1.,1.,1.,1.]]])
    torch.testing.assert_close(p(x),x.mean(-1,keepdim=True))
    p.mix.data.fill_(.5);assert p(x)[0].item()>p(x)[1].item()
    p(x).sum().backward();assert torch.isfinite(p.mix.grad).all()


def test_shared_complex_is_phase_invariant_and_sample_local():
    torch.manual_seed(4);model=build('shared_complex').eval();x=torch.randn(3,2,256)
    with torch.no_grad():
        y=model(x)
        torch.testing.assert_close(y,model(rotate(x,1.1)),atol=3e-4,rtol=3e-4)
        torch.testing.assert_close(y[:1],model(x[:1]),atol=3e-4,rtol=3e-4)


@pytest.mark.parametrize('variant',VARIANTS)
def test_real_ce_backward_and_light_capacity(variant):
    torch.set_num_threads(2);torch.manual_seed(11);model=build(variant)
    reference=sum(p.numel() for p in build('native').parameters())
    params=sum(p.numel() for p in model.parameters())
    assert params<=reference*1.1
    if variant=='shared_complex':assert params<reference
    x=torch.randn(4,2,256);scores=model(x)
    assert scores.shape==(4,6)
    loss=torch.nn.functional.cross_entropy(scores,torch.tensor([0,1,2,3]));loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0002)
    optimizer.step();assert torch.isfinite(model(x)).all()


def test_clean_contract_rejects_target_and_augmentation():
    c=dict(method='cvs_clean_design',epochs=200,batch_size=128,lr=.0002,lr_min=1e-6,
        weight_decay=.0001,drop_last=False,augmentation=False,domain_backbone=False,
        extra_losses=[],selection='fixed_last_epoch',split_seed=392005,variant='native')
    assert validate_config(c)==c
    for key,value in [('augmentation',True),('target_truth','truth.json'),('p1_capsule','target'),('checkpoint','old.pt'),('extra_losses',['domain'])]:
        with pytest.raises(ValueError):validate_config(dict(c,**{key:value}))


def test_source_selection_cannot_use_target_scores_and_keeps_all_baselines():
    from experiments.cvs_clean_design.dispatch import select_source_candidate
    from experiments.cvs_clean_design.model import BASELINES
    records=[]
    for variant in VARIANTS:
        for seed in (2026092701,2026092702,2026092703,2026092704):
            records.append(dict(variant=variant,seed=seed,accuracy=.9,worst_rx=.85,
                parameters=100 if variant=='shared_complex' else 200,macs=100 if variant=='shared_complex' else 200,
                target_accuracy=0. if variant=='shared_complex' else 1.))
    selection=select_source_candidate(records)
    assert selection['selected_variant']=='shared_complex'
    assert set(BASELINES)<=set(selection['test_variants'])
    assert selection['target_access'] is False and selection['target_score_used'] is False
    with pytest.raises(ValueError):select_source_candidate(records[:-1])


def test_fresh_source_entry_and_resource_trace():
    import subprocess,sys
    from pathlib import Path
    from experiments.cvs_clean_design.profile import resource_profile
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'-m','experiments.cvs_clean_design.source','--help'],cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    torch.set_num_threads(2);model=build('shared_complex').eval()
    before={k:v.clone() for k,v in model.state_dict().items()}
    measured=resource_profile(model,torch.device('cpu'))
    assert measured['conv_linear_macs_per_sample']>0 and measured['training_batch128_ms']>0
    assert measured['benchmark_state_used_for_training'] is False
    for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
