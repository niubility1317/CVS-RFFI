from unittest.mock import Mock

import numpy as np
import pytest
import torch

from experiments.pamf_lora_main import mirror_refine_source as source
from experiments.pamf_lora_main.mirror_refine import MirrorRefine, match_dose, protect, spectrum, wave
from experiments.pamf_lora_main.tail_pilot_source import WIDTHS


def test_split_evidence_never_reads_the_chirp_being_updated():
    model = MirrorRefine(WIDTHS)
    gen = torch.Generator().manual_seed(104)
    sampled = torch.randn((1, 1, 256), dtype=torch.complex64, generator=gen)
    observed = sampled.expand(-1, 6, -1).clone()
    quality = torch.linspace(.1, .9, 6).reshape(1, 6, 1)
    direction, confidence = model.evidence(observed, quality)
    assert (confidence[0, 0] > 0).all()
    observed[:, 0] *= 25j
    quality[:, 0] = .01
    changed, changed_confidence = model.evidence(observed, quality)
    torch.testing.assert_close(direction[:, 0], changed[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(confidence[:, 0], changed_confidence[:, 0], atol=0, rtol=0)
    assert not direction[..., [0, 128]].any()


def test_protect_blocks_only_the_supported_direction_and_matches_dose():
    model = MirrorRefine(WIDTHS)
    direction = torch.zeros((1, 1, 4, 256), dtype=torch.complex64)
    direction[..., 0, 127] = 2 - 1j
    original = torch.zeros((1, 1, 256), dtype=torch.complex64)
    original[..., [0, 126, 127, 128, 130]] = 4 + 3j
    fine = wave(original).requires_grad_()
    zeros = torch.zeros_like(fine)
    support = torch.full((1, 1, 4), .75)
    limited = protect(zeros, fine, direction, support)
    expected = original.clone()
    expected[..., 127] *= .25
    torch.testing.assert_close(spectrum(limited), expected)
    limited.abs().square().sum().backward()
    assert fine.grad is not None and fine.grad.isfinite().all()
    other = original.clone()
    other[..., [126, 127]] = torch.tensor([0.5 + 2j, -1.0j])
    protected = wave(other)
    matched = spectrum(match_dose(zeros, fine.detach(), protected, model.band_masks))
    for mask in model.band_masks:
        torch.testing.assert_close(torch.linalg.vector_norm(matched[..., mask], dim=-1),
                                   torch.linalg.vector_norm(other[..., mask], dim=-1))
    torch.testing.assert_close(matched[..., [0, 128]], original[..., [0, 128]])


def test_parent_endpoint_mismatch_is_rejected_before_source_truth(monkeypatch):
    def read(_root, _route, modes):
        rows = {view: np.zeros((len(modes), 1), dtype=int) for view in source.VIEWS}
        if modes == source.MIRROR_MODES:
            rows['clean'][modes.index('mirror'), 0] = 7
        return ['same-source-package'], rows, {}
    scorer = Mock()
    monkeypatch.setattr(source, 'read_frozen', read)
    monkeypatch.setattr(source, 'score_frozen', scorer)
    with pytest.raises(ValueError, match='full/clean/mirror'):
        source.independent_score(dict(run_root='/runs/new'))
    scorer.assert_not_called()
