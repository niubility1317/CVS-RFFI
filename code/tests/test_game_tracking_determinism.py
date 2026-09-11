from types import SimpleNamespace
import torch


def test_explicit_deterministic_runtime_sets_backend_before_seed(monkeypatch):
    from cvsrffi.game_tracking import runtime
    calls=[]
    monkeypatch.setattr(torch,'use_deterministic_algorithms',lambda enabled:calls.append(enabled))
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    monkeypatch.setattr(torch.backends.cudnn,'deterministic',False)
    monkeypatch.setattr(torch.backends.cudnn,'benchmark',True)
    result=runtime.configure_determinism(SimpleNamespace(game_deterministic=True))
    assert calls==[True]
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    assert result['cublas_workspace_config']==':4096:8'


def test_inactive_determinism_does_not_reconfigure_backend():
    from cvsrffi.game_tracking import runtime
    assert runtime.configure_determinism(SimpleNamespace(game_deterministic=False))['requested'] is False
