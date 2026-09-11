import sys
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
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

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA compatibility regression')
def test_actual_augmentation_and_release_smoke_old_cuda(monkeypatch,tmp_path):
    import json
    import importlib.util
    import DataAugmentation as aug
    from torch_compat import deterministic_cumsum
    original=torch.cumsum
    def old_cumsum(x,dim):
        if x.is_cuda:raise RuntimeError('cumsum_cuda_kernel does not have a deterministic implementation')
        return original(x,dim=dim)
    monkeypatch.setattr(torch,'cumsum',old_cumsum)
    prior=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        z=torch.ones((2,32),device='cuda',dtype=torch.complex64)
        assert torch.isfinite(aug._apply_phase_noise(z,torch.ones((2,1),device='cuda')*.01)).all()
        torch.testing.assert_close(deterministic_cumsum(z,1),original(z.cpu(),dim=1).cuda(),atol=0,rtol=0)
        path=Path(__file__).parents[1]/'scripts/dispatch_core90_game.py'
        spec=importlib.util.spec_from_file_location('smoke_dispatch',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        config=tmp_path/'config.json'
        config.write_text(json.dumps(dict(game_synthetic=True,game_deterministic=True,game_evidence_version=2,game_no_audit=True,lambda_adv=.35,num_workers=0)),encoding='utf-8')
        module.scratch_checkpoint_smoke({'config_path':str(config)},path.parents[2],tmp_path/'smoke','0')
        report=json.loads((tmp_path/'smoke/result.json').read_text())
        assert report['status']=='PASS' and len(report['stages'])==9
        assert all(row['accepted'] for row in report['stages'])
    finally:torch.use_deterministic_algorithms(prior)
