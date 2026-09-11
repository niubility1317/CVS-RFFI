import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]))
from sat_channel import wiener_phase_noise

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA compatibility regression')
def test_old_cuda_cumsum_fallback_preserves_draws(monkeypatch):
    original=torch.cumsum
    def old_cumsum(x,dim):
        if x.is_cuda:
            raise RuntimeError("cumsum_cuda_kernel does not have a deterministic implementation")
        return original(x,dim=dim)
    monkeypatch.setattr(torch,'cumsum',old_cumsum)
    prior=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        sigma=torch.tensor([.01,.02],device='cuda')
        gen=torch.Generator(device='cuda').manual_seed(71)
        result=wiener_phase_noise(2,32,sigma,'cuda',torch.float32,gen)
        end=gen.get_state()
        refgen=torch.Generator(device='cuda').manual_seed(71)
        inc=torch.randn((2,32),device='cuda',generator=refgen)*sigma[:,None]
        reference=torch.exp(1j*original(inc.cpu(),dim=1).cuda())
        torch.testing.assert_close(result,reference,atol=0,rtol=0)
        assert torch.equal(end,refgen.get_state())
    finally:torch.use_deterministic_algorithms(prior)

def test_unrelated_cumsum_errors_propagate(monkeypatch):
    def failure(*args,**kwargs):raise RuntimeError('unrelated allocation failure')
    monkeypatch.setattr(torch,'cumsum',failure)
    with pytest.raises(RuntimeError,match='unrelated allocation'):
        wiener_phase_noise(1,8,torch.ones(1),'cpu',torch.float32)
