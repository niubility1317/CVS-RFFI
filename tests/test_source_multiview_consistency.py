import torch
from experiments.phase1_dura_orthogonal.source_multiview_consistency import metrics


def test_shared_error_is_not_removed_by_multiview_average():
    clean=torch.ones(2,2,256)
    error=torch.full_like(clean,.1)
    anchors=torch.stack((clean+error,clean+error,clean+error),1)
    values=metrics(anchors,clean)
    torch.testing.assert_close(values[:,:3],values[:,3,None].expand(-1,3))
    torch.testing.assert_close(values[:,3],values[:,4])


def test_opposite_errors_cancel_in_ensemble():
    clean=torch.ones(2,2,256)
    e=torch.full_like(clean,.1)
    anchors=torch.stack((clean+e,clean-e,clean),1)
    values=metrics(anchors,clean)
    torch.testing.assert_close(values[:,3],torch.zeros(2),atol=1e-12,rtol=0)
    assert torch.all(values[:,4] < 0)
