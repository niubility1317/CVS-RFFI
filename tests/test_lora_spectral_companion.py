import torch

from experiments.pamf_lora_main.tail_pilot_source import MIRROR_RECIPES, RECIPES
from experiments.pamf_unrolled_restore.denoise import spectral_companion


def test_conjugate_pairs_align_mirrored_frequency_on_both_sides():
    values = torch.arange(256, dtype=torch.float32)
    spectrum = torch.complex(values, values + 1)
    mirror = spectral_companion(spectrum, 'mirror')
    adjacent = spectral_companion(spectrum, 'adjacent')
    torch.testing.assert_close(mirror[[1, 39, 127, 129, 183, 255]],
                               spectrum[[255, 217, 129, 127, 73, 1]].conj())
    torch.testing.assert_close(adjacent[[1, 39, 127, 129, 183, 255]],
                               spectrum[[2, 38, 126, 130, 184, 254]].conj())
    assert not mirror[[0, 128]].any() and not adjacent[[0, 128]].any()


def test_mirrored_and_adjacent_models_match_capacity_and_keep_original_checkpoint_shape():
    a = MIRROR_RECIPES['adjacent'].model()
    b = MIRROR_RECIPES['mirror'].model()
    b.load_state_dict(a.state_dict(), strict=True)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    original = RECIPES['tail'].model()
    assert original.coarse.companion is None
    assert sum(p.numel() for p in original.parameters()) == 254351
