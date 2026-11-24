import torch

from experiments.pamf_unrolled_restore.wisig_lora_window_mapping import pack, unpack, restore_expanded, restore_guard


def test_round_trip_preserves_each_wisig_sample():
    iq = torch.arange(3 * 2 * 256, dtype=torch.float32).reshape(3, 2, 256)
    six_views = pack(iq)
    assert six_views.shape == (3, 6, 2, 256)
    assert torch.equal(unpack(six_views), iq)


def test_unpack_aligns_the_six_synthetic_views():
    iq = torch.randn(2, 2, 256)
    views = pack(iq).clone()
    views[:, 0] += 6.0
    assert torch.allclose(unpack(views), iq + 1.0, atol=1e-6)


def test_pack_rejects_unexpected_window_layout():
    try:
        pack(torch.zeros(2, 256))
    except ValueError:
        pass
    else:
        raise AssertionError('Expected a single 2-by-256 received WiSig window')


def test_expanded_uses_half_readout_and_aligned_mean():
    iq = torch.randn(2, 2, 256)

    def recovery(views, tracker, models, threshold):
        assert views.shape == (2, 6, 2, 256)
        return torch.stack([views, views + 1, views + 2, views + 3]), 2, None

    result, active = restore_expanded(iq, recovery, None, None, 0.1)
    assert torch.allclose(result, iq + 3, atol=1e-6)
    assert active == 2


def test_guard_half_uses_original_half_position():
    iq = torch.randn(2, 2, 256)

    def recovery(views, tracker, model, gate, threshold):
        return [views, views + 1, views + 2, views + 3], 1

    result, active = restore_guard(iq, recovery, None, None, None, 0.1)
    assert torch.allclose(result, iq + 2, atol=1e-6)
    assert active == 1
