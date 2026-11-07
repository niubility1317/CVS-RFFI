import pytest
import torch
import json
import numpy as np
import subprocess
import sys

from experiments.phase1_dura_orthogonal.source_instance_contrast import instance_contrast_loss


def test_correct_record_matches_are_preferred():
    first = torch.eye(4, requires_grad=True)
    second = torch.eye(4, requires_grad=True)
    ids = torch.arange(4)
    good = instance_contrast_loss(first, second, ids)
    bad = instance_contrast_loss(first, second.roll(1, 0), ids)
    assert good < bad
    good.backward()
    assert torch.isfinite(first.grad).all() and torch.isfinite(second.grad).all()


def test_multiple_views_of_same_record_are_all_positives():
    first = torch.tensor([[1., 0.], [1., 0.]])
    second = torch.tensor([[1., 0.], [1., 0.]])
    same_record = instance_contrast_loss(first, second, torch.tensor([7, 7]))
    different_records = instance_contrast_loss(first, second, torch.tensor([7, 8]))
    torch.testing.assert_close(same_record, torch.tensor(0.))
    assert different_records > 0


def test_invalid_shapes_or_record_ids_are_rejected():
    with pytest.raises(ValueError):
        instance_contrast_loss(torch.ones(2, 3), torch.ones(3, 3), torch.arange(2))
    with pytest.raises(ValueError):
        instance_contrast_loss(torch.ones(2, 3), torch.ones(2, 3), torch.ones(2))


def test_ground_loader_requires_matching_source_contrast_metadata(tmp_path, monkeypatch):
    from experiments.phase1_dura_orthogonal import ground
    from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral
    from experiments.phase1_dura_orthogonal.noise_gated_dura import GATE_LOG10_LOW, GATE_LOG10_HIGH
    from experiments.phase1_dura_orthogonal.joint_ssl_iq import SourceIQEncoder

    folder = tmp_path/'runs'/'source'
    folder.mkdir(parents=True)
    common = dict(commit='test', epochs=5, architecture='dura_multiscale', objective='wide_instance_contrast',
        DURA_training_reads=1, DURA_frozen=True, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_reads=0, source_fit_size=32768, source_views_per_record=3, source_ssl_feature_weight=.05,
        source_ssl_offdiagonal_weight=None, source_ssl_feature_dim=16, source_ssl_seed=2026110615,
        source_instance_temperature=.1, source_instance_positives='same_physical_record',
        instance_encoder_inference_reads=0, spectral_freq_bins=17, adaptive_filter_taps=9,
        noise_gate_low=GATE_LOG10_LOW, noise_gate_high=GATE_LOG10_HIGH)
    summary = dict(common, status='SOURCE_TRAINED', fit_ids=32768)
    state = dict(common, schema=ground.SCHEMA, fit_ids=np.arange(32768), width=32,
                 model=DuraMultiscaleSpectral(width=32, taps=9).state_dict(),
                 source_ssl_aux_state=SourceIQEncoder().state_dict())
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    torch.save(state, folder/'final.pt')
    monkeypatch.setattr(ground, 'frozen_dura', lambda *a: torch.nn.Identity())
    monkeypatch.setattr(ground.core, 'teacher', lambda *a: (torch.nn.Identity(), None))
    model, _, _ = ground.load_fixed(tmp_path, 'source', 'test')
    assert isinstance(model, DuraMultiscaleSpectral)
    summary['source_instance_temperature'] = .2
    (folder/'source_summary.json').write_text(json.dumps(summary), encoding='utf8')
    with pytest.raises(ValueError, match='instance contrast'):
        ground.load_fixed(tmp_path, 'source', 'test')


@pytest.mark.parametrize('extra', [
    ['--smoke', '--architecture', 'orthogonal'],
    ['--architecture', 'dura_multiscale'],
])
def test_training_rejects_wrong_architecture_or_missing_fixed_source_cache(tmp_path, extra):
    output = tmp_path/'forbidden-output'
    cmd = [sys.executable, '-m', 'experiments.phase1_dura_orthogonal.train_source',
           '--root', str(tmp_path), '--out', str(output), '--commit', 'test',
           '--objective', 'wide_instance_contrast', *extra]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Source instance contrast requires multiscale DURA and fixed source context' in result.stderr
    assert not output.exists()
