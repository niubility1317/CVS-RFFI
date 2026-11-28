import torch

from experiments.pamf_lora_main.dura_real_gain_lora_full_source import normalize_pair


def test_pair_uses_same_observed_scale_and_unit_observation():
    clean = torch.tensor([[1 + 1j, 2 - 1j]], dtype=torch.complex64)
    observed = 3 * clean
    clean_out, observed_out = normalize_pair(clean, observed)
    torch.testing.assert_close(observed_out, 3 * clean_out)
    torch.testing.assert_close(observed_out.abs().square().mean(-1), torch.ones(1))
