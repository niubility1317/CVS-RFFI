import torch

from experiments.phase1_dura_orthogonal.source_retrieval_geometry import descriptors, retrieval


def test_descriptors_are_phase_and_scale_invariant():
    generator = torch.Generator().manual_seed(51)
    iq = torch.randn((8, 2, 256), generator=generator)
    phase = .73
    rotated = torch.stack((iq[:, 0]*torch.cos(torch.tensor(phase))-iq[:, 1]*torch.sin(torch.tensor(phase)),
                           iq[:, 0]*torch.sin(torch.tensor(phase))+iq[:, 1]*torch.cos(torch.tensor(phase))), dim=1)*2.5
    for a, b in zip(descriptors(iq), descriptors(rotated)):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_retrieval_uses_physical_record_pairs():
    features = torch.eye(8)
    result = retrieval(features, features, torch.zeros(8), torch.ones(8))
    assert result['top1'] == 1 and result['top5'] == 1 and result['mean_reciprocal_rank'] == 1
