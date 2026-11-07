import torch

from experiments.phase1_dura_orthogonal.phase_predictability_source import oracle_angle, phase_features, fit_ridge, angle_nmse
from experiments.phase1_dura_orthogonal.conditional_phase_oracle_probe import rotate


def test_oracle_and_waveform_error_use_correct_rotation():
    torch.manual_seed(1)
    anchor=torch.randn(7,2,256)
    clean=rotate(anchor,torch.full((7,),.3))
    angle=oracle_angle(anchor,clean)
    torch.testing.assert_close(angle,torch.full((7,),.3))
    assert (angle_nmse(anchor,clean,angle)<1e-12).all()
    assert (angle_nmse(anchor,clean,torch.zeros(7))>0).all()


def test_ridge_learns_heldout_linear_phase_without_using_holdout_targets():
    torch.manual_seed(2)
    train=torch.randn(128,7)
    held=torch.randn(48,7)
    truth=.2*train[:,0]-.1*train[:,1]+.03
    expected=.2*held[:,0]-.1*held[:,1]+.03
    predicted=fit_ridge(train,truth,held,ridge=0.)
    torch.testing.assert_close(predicted,expected,atol=1e-5,rtol=1e-5)


def test_feature_shapes_and_zero_innovation_remain_finite():
    torch.manual_seed(3)
    x=torch.randn(8,2,256)
    base,moments,cross=phase_features(x,x,torch.zeros_like(x),torch.full((8,),.1))
    assert base.shape==(8,7) and moments.shape==(8,19) and cross.shape==(8,25)
    assert torch.isfinite(cross).all()
