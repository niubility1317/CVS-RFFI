import torch

from experiments.phase1_dura_orthogonal.source_multiview_bands import band_metrics
from experiments.phase1_dura_orthogonal.source_multiview_consistency import metrics


def test_frequency_bands_sum_to_time_domain_metrics():
    generator=torch.Generator().manual_seed(12)
    clean=torch.randn(7,2,256,generator=generator)
    anchors=clean[:,None] + .1*torch.randn(7,3,2,256,generator=generator)
    bands=band_metrics(anchors,clean)
    original=metrics(anchors,clean)
    torch.testing.assert_close(bands[:,:,0].sum(1),torch.ones(7),rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(bands[:,:,1].sum(1),original[:,:3].mean(1),rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(bands[:,:,2].sum(1),original[:,3],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(bands[:,:,3].sum(1),original[:,4],rtol=1e-5,atol=1e-6)


def test_identical_view_errors_cannot_be_averaged_away():
    generator=torch.Generator().manual_seed(13)
    clean=torch.randn(3,2,256,generator=generator)
    noisy=clean + .1*torch.randn(3,2,256,generator=generator)
    anchors=noisy[:,None].repeat(1,3,1,1)
    bands=band_metrics(anchors,clean)
    torch.testing.assert_close(bands[:,:,1],bands[:,:,2],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(bands[:,:,1],bands[:,:,3],rtol=1e-5,atol=1e-6)
