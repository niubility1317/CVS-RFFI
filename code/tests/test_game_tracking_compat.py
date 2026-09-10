import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.game_tracking import runtime


def test_scaler_on_torch_without_unified_amp_factory(monkeypatch):
    monkeypatch.delattr(torch.amp,'GradScaler')
    scaler=runtime.make_grad_scaler(torch.device('cpu'),False)
    assert not scaler.is_enabled()
    assert scaler.state_dict()=={}
