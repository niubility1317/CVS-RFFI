import pytest
import torch

from experiments.phase1_dura_orthogonal.dura_circular_recovery import DuraCircularRecovery


def rotate(x, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c*x[:, 0]-s*x[:, 1], s*x[:, 0]+c*x[:, 1]), 1)


def test_initial_dura_anchor_and_joint_phase_equivariance():
    torch.manual_seed(9)
    model=DuraCircularRecovery(width=4)
    x,anchor,innovation=(torch.randn(3,2,256) for _ in range(3))
    variance=torch.ones(3)*.2
    assert torch.equal(model(x,anchor,innovation,variance),anchor)
    for p in model.head.parameters():
        torch.nn.init.normal_(p,std=.1)
    angle=torch.tensor(.73)
    expected=rotate(model(x,anchor,innovation,variance),angle)
    actual=model(rotate(x,angle),rotate(anchor,angle),rotate(innovation,angle),variance)
    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=2e-5)
    model(x,anchor,innovation,variance).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
               for p in (model.head.real,model.head.imag))


def test_zero_innovation_is_finite_and_bad_variance_rejected():
    model=DuraCircularRecovery(width=4)
    x=torch.randn(2,2,256)
    assert torch.isfinite(model(x,x,torch.zeros_like(x),torch.ones(2))).all()
    with pytest.raises(ValueError):
        model(x,x,torch.zeros_like(x),torch.tensor([1.,-1.]))


def test_ground_requires_fixed_dura_circular_training_metadata(tmp_path, monkeypatch):
    import json
    import numpy as np
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.direct_circular_recovery import WINDOWS
    from experiments.phase1_dura_orthogonal.phase_equivariant_dura import RESIDUAL_SCALE
    folder=tmp_path/'runs'/'source'
    folder.mkdir(parents=True)
    common=dict(commit='test',epochs=5,architecture='dura_circular_recovery',objective='wide_dura_circular',
                source_fit_size=32768,DURA_training_reads=1,DURA_frozen=True,TX_labels_read=0,
                F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,
                dura_circular_windows=list(WINDOWS),dura_circular_feature='local_second_moment_times_conjugate',
                dura_circular_inputs='received_anchor_innovation',dura_circular_residual_scale=RESIDUAL_SCALE)
    summary=dict(common,status='SOURCE_TRAINED',fit_ids=32768)
    state=dict(common,schema=ground.SCHEMA,fit_ids=np.arange(32768),width=4,
               model=DuraCircularRecovery(width=4).state_dict())
    (folder/'source_summary.json').write_text(json.dumps(summary),encoding='utf8')
    torch.save(state,folder/'final.pt')
    monkeypatch.setattr(ground,'frozen_dura',lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core,'teacher',lambda *a: (torch.nn.Identity(),None))
    assert isinstance(ground.load_fixed(tmp_path,'source','test')[0],DuraCircularRecovery)
    summary['target_reads']=1
    (folder/'source_summary.json').write_text(json.dumps(summary),encoding='utf8')
    with pytest.raises(ValueError,match='source-only'):
        ground.load_fixed(tmp_path,'source','test')
