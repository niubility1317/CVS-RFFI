import pytest
import torch
from experiments.phase1_dura_orthogonal.source_phase_alignment_probe import align_paths, evaluate_batch


def test_phase_and_complex_alignment_correct_known_path_factors():
    torch.manual_seed(3)
    base=torch.complex(torch.randn(2,256),torch.randn(2,256))
    factors=torch.tensor([1+0j,2j,-.5+0j,-3j])
    paths=base[:,None]*factors[None,:,None]
    phase,gain,_,_=align_paths(paths)
    torch.testing.assert_close(phase,base*(1+2+.5+3)/4,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(gain,base,rtol=1e-5,atol=1e-6)


def test_identity_orbit_unchanged_and_invalid_paths_fail():
    x=torch.randn(3,2,256)
    result=evaluate_batch(torch.nn.Identity(),x,x)
    torch.testing.assert_close(result[:,:7],torch.zeros_like(result[:,:7]),atol=1e-12,rtol=0)
    with pytest.raises(ValueError):
        align_paths(torch.zeros(1,4,256,dtype=torch.complex64))
