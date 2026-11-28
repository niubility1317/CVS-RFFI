import torch

from experiments.pamf_lora_main.dura_real_gain_wisig_gain_dose import doubled_correction


def test_doubled_correction_keeps_r17_at_zero_and_doubles_only_the_edit():
    prior = torch.tensor([1 + 2j, 3 - 1j])
    edit = torch.tensor([.1 + .2j, -.3 + .4j])
    torch.testing.assert_close(doubled_correction(prior, prior), prior)
    torch.testing.assert_close(doubled_correction(prior, prior + edit), prior + 2 * edit)
