import numpy as np
import pytest

from experiments.pamf_lora_main.packet_data import CHIRPS, PACKET_LEN
from experiments.pamf_lora_main.w24_native_full import native_full_view, project_delta, validate_predictions


def test_full_view_compensates_delay_without_transmitting_extra_capture_samples(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    import experiments.pamf_lora_main.w24_native_full as native

    fake_package = ModuleType('leo_practical')
    fake_module = ModuleType('leo_practical.channel')
    monkeypatch.setitem(sys.modules, 'leo_practical', fake_package)
    monkeypatch.setitem(sys.modules, 'leo_practical.channel', fake_module)
    fake_package.channel = fake_module

    delay = 24
    source = np.ones(PACKET_LEN, dtype=np.complex64)

    class MockStream:
        def __init__(self, cfg, seed, receiver):
            assert cfg.fractional_delay_half_length == delay

        def process(self, samples, *, preceding_context):
            assert not preceding_context.any()
            assert np.array_equal(samples[:PACKET_LEN], source)
            assert not samples[PACKET_LEN:].any()
            received = np.concatenate([np.zeros(delay), 2 * samples[:-delay]])
            return received, dict(boundary='provided_context', channel_equalization_applied=False,
                                  numerical_filter_latency_samples=delay)

    fake_module.ChannelStream = MockStream
    fake_module.normalize_input = lambda packet: (packet, 1)
    fake_module.stable_seed = lambda *args: 1
    monkeypatch.setattr(native, 'channel', lambda scene: SimpleNamespace(fractional_delay_half_length=delay))
    monkeypatch.setattr(native, 'hardware', lambda scene, session: None)
    output = native_full_view(source, 'practical_high', 'rx', 0, 'source')
    assert np.array_equal(output, source * 2)


def test_projected_dura_edit_preserves_w24_edges_and_unobserved_chirps():
    received = np.ones((2, PACKET_LEN), dtype=np.complex64)
    delta = np.ones((2, len(CHIRPS), 2, 256), dtype=np.float32)
    delta[0] = 0
    corrected = project_delta(received, delta)
    assert corrected[0].tobytes() == received[0].tobytes()
    slots = corrected[1].reshape(-1, 1024)
    assert (slots[:, :24] == 1).all() and (slots[:, -24:] == 1).all()
    assert (slots[[0, 7]] == 1).all() and (slots[list(CHIRPS), 512] != 1).all()


def test_scoring_rejects_missing_matrix_before_loading_truth(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        validate_predictions(tmp_path)


def test_scoring_preserves_existing_class_indices_and_receiver_order(monkeypatch, tmp_path):
    from contextlib import nullcontext
    import zipfile

    from experiments.pamf_lora_main import data
    import experiments.pamf_lora_main.w24_native_full as native

    monkeypatch.setattr(native, 'TARGET', ('test_rx',))
    monkeypatch.setattr(native, 'VIEWS', ('clean', 'practical_high'))
    monkeypatch.setattr(native, 'LIMIT', 3)
    labels = np.array([2, 8, 5], dtype=np.int16)
    expected = labels[data.packet_indices(3, 0)]
    pred = {('test_rx', view, mode): expected for view in native.VIEWS for mode in native.MODES}
    monkeypatch.setattr(native, 'validate_predictions', lambda folder: (pred, {}))
    monkeypatch.setattr(data, 'recording', lambda *args: nullcontext())
    monkeypatch.setattr(data, 'label_ids', lambda capture, limit: labels)
    with zipfile.ZipFile(tmp_path / 'multiple_receiver_test.zip', 'w'):
        pass
    native.score(tmp_path / 'predictions', tmp_path, tmp_path / 'score')
    assert set(native.read(tmp_path / 'score' / 'scores.json')['six_correct'].values()) == {3}
