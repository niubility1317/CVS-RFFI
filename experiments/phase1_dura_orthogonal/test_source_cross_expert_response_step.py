import pytest
import torch
from experiments.phase1_dura_orthogonal.source_cross_expert_response_step import objective


def test_fixed_step_objectives_do_not_change_wave_only_loss():
    wave=torch.tensor(.4);response=torch.tensor(3.)
    torch.testing.assert_close(objective(wave,response,'wave_only'),wave)
    torch.testing.assert_close(objective(wave,response,'wave_plus_response'),torch.tensor(.43))
    with pytest.raises(ValueError):objective(wave,response,'unknown')
