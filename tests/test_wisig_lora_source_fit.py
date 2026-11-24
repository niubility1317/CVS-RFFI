import pytest
import torch

from experiments.pamf_unrolled_restore.wisig_lora_source_fit import perturb, source_iq


def test_source_fit_rejects_a_non_source_cache(tmp_path):
    with pytest.raises(ValueError, match='fixed WiSig source cache'):
        source_iq(tmp_path)


def test_perturbation_only_returns_iq_and_is_reproducible():
    clean = torch.ones(5, 2, 256)
    a = perturb(clean, torch.Generator().manual_seed(4))
    b = perturb(clean, torch.Generator().manual_seed(4))
    assert a.shape == clean.shape
    assert torch.equal(a, b)
    assert not torch.equal(a, clean)
