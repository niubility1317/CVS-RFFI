import json

import numpy as np
import pytest
import torch

from experiments.pamf_lora_main.w24_online_additive_source import sample_additive


def test_full_block_agc_scales_retained_observation_and_source_together():
    delay = 31
    signal = torch.full((2, 8192 + delay), 4., dtype=torch.complex64)
    signal[0, :delay] = 20.
    main = torch.zeros_like(signal)
    received, source = sample_additive(signal, main, main, delay, torch.tensor([20., 3.]))
    expected_gain = torch.tensor([signal[0].abs().square().mean().rsqrt(), 10**(-3/20)])
    assert received.shape == (2, 8192)
    torch.testing.assert_close(received, signal[:, delay:] * expected_gain[:, None])
    torch.testing.assert_close(received, source, atol=0, rtol=0)


def test_noise_redraw_preserves_proper_input_and_image_covariance():
    signal = torch.zeros((12, 8192), dtype=torch.complex64)
    alpha, beta = 0.75 + .10j, .15 - .12j
    main, image = torch.full_like(signal, alpha), torch.full_like(signal, beta)
    def draw(seed):
        y, s = sample_additive(signal, main, image, 0, torch.zeros(12),
                               generator=torch.Generator().manual_seed(seed))
        assert torch.equal(s, signal)
        return y

    first, second = draw(7), draw(8)
    assert abs(first.abs().square().mean() - (abs(alpha)**2 + abs(beta)**2)) < .01
    assert abs(first.square().mean() - 2*alpha*beta) < .01
    assert abs((first * second.conj()).mean()) < .01


def test_redraw_reuses_clean_identity_and_selected_original_rows():
    from experiments.pamf_lora_main.w24_peer_spectral_direction_source import redraw_batch

    train = torch.arange(60).reshape(5, 3, 4).to(torch.complex64)
    selection = np.array([4, 2, 0])
    expected = train[[4, 2, 0]]
    assert torch.equal(redraw_batch(train, selection, None), expected)

    def sample(rows):
        np.testing.assert_array_equal(rows, selection)
        return expected[:, 1]+1j, expected[:, 2]-1j

    actual = redraw_batch(train, selection, sample)
    assert torch.equal(actual[:, 0], expected[:, 0])
    assert torch.equal(actual[:, 1], expected[:, 1]+1j)
    assert torch.equal(actual[:, 2], expected[:, 2]-1j)


def test_online_fit_rejects_target_handle_and_freeze_requires_draw_proof(tmp_path):
    from experiments.pamf_lora_main.w24_formal import (
        FULL, ONLINE_DIRECTIONS_KIND, RUN_ONLINE_DIRECTIONS_CPU, component_directions_ready, evaluation_modes,
    )
    from experiments.pamf_lora_main.w24_peer_spectral_direction_source import (
        COMPONENT_WEIGHT, DATA_RUN, DRAW_SEED, ISOLATION, ONLINE_BANK_RUN, ONLINE_MODES, ONLINE_RUN,
        PRACTICAL_CACHE_RUN, STEPS, TRAIN_RUN, fit_config,
    )

    root = tmp_path/'runs'/ONLINE_RUN
    roles = dict(project=str(tmp_path), run_root=str(root),
        source_root=str(tmp_path/'releases'/f'{DATA_RUN}-release'/'data'/'source'),
        practical_cache=str(tmp_path/'runs'/PRACTICAL_CACHE_RUN/'source_practical'),
        training_release_root=str(tmp_path/'releases'/f'{TRAIN_RUN}-release'),
        online_operator_bank=str(tmp_path/'runs'/ONLINE_BANK_RUN))
    fit_config(roles)
    with pytest.raises(ValueError, match='source-only'):
        fit_config(dict(roles, native_target_root='disallowed'))
    formal = dict(project=str(tmp_path), run_id=RUN_ONLINE_DIRECTIONS_CPU, third_prior=ONLINE_DIRECTIONS_KIND,
                  run_root=str(tmp_path/'runs'/RUN_ONLINE_DIRECTIONS_CPU), candidate_root=str(root))
    assert not component_directions_ready(formal)
    assert evaluation_modes(FULL, True, third_prior=ONLINE_DIRECTIONS_KIND)[-2:] == ONLINE_MODES[-2:]
    root.mkdir(parents=True)
    (root/'direction.pt').touch()
    metadata = dict(status='SOURCE_FROZEN', steps=STEPS, component_angle_weight=COMPONENT_WEIGHT,
                    online_operator_bank=ONLINE_BANK_RUN, **ISOLATION)
    (root/'fit-complete.json').write_text(json.dumps(metadata), encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid source-only'):
        component_directions_ready(formal)
    metadata['draw_seed'] = DRAW_SEED
    (root/'fit-complete.json').write_text(json.dumps(metadata), encoding='utf-8')
    assert component_directions_ready(formal)
