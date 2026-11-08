import pytest
import torch
from experiments.phase1_dura_orthogonal.source_cross_expert_response_stepsize import RATES,objective


def test_fixed_exploratory_rates_and_objective():
    assert RATES==(.00003,.000003)
    wave=torch.tensor(.4);response=torch.tensor(3.)
    torch.testing.assert_close(objective(wave,response,'wave_only'),wave)
    torch.testing.assert_close(objective(wave,response,'wave_plus_response'),torch.tensor(.43))
    with pytest.raises(ValueError):objective(wave,response,'unknown')
