from types import SimpleNamespace
import numpy as np
import torch
from experiments.cvs_residual_identity.model import ResidualPhysicalHead
from experiments.pamf_dura_efficiency.soft_rx_reference_probe import original_head_constants


def test_export_matches_actual_original_residual_head_interface_and_scores():
    head=ResidualPhysicalHead(160).eval().requires_grad_(False)
    reader=SimpleNamespace(core=SimpleNamespace(id_backbone=SimpleNamespace(cls_head=head)))
    weight,scale=original_head_constants(reader)
    assert weight.shape==(6,160) and weight.dtype==np.float64 and scale==30.
    x=torch.randn(7,160,generator=torch.Generator().manual_seed(53))
    with torch.no_grad():expected=head.classify(x).tolist()
    features=np.asarray(x.tolist(),np.float64);features/=np.linalg.norm(features,axis=1,keepdims=True)
    np.testing.assert_allclose(scale*features@weight.T,np.asarray(expected),atol=2e-6,rtol=2e-5)
