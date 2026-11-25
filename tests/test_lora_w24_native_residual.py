import numpy as np
import pytest

from experiments.pamf_lora_main.w24_native_residual import (
    PACKET_LEN, STRIDE, native_residual, residual_snapshot_type,
)
from leo_practical.channel import receiver_for_session
from experiments.pamf_lora_main.w24_native_residual import native_residual_config


def test_native_residual_retains_all_realized_edges_and_the_legacy_duration_guard():
    packet = np.exp(2j*np.pi*.03*np.arange(PACKET_LEN)).astype(np.complex64)
    config = native_residual_config("practical_mid")
    rx = receiver_for_session(config, 45, "native-residual-test")
    with pytest.raises(ValueError, match="1ms"):
        residual_snapshot_type()(config, 12, rx).process(packet[:STRIDE])
    actual, records = native_residual(packet, "practical_mid", "rx", 12, "test", trace=True)
    again = native_residual(packet, "practical_mid", "rx", 12, "test")
    assert np.array_equal(actual, again)
    assert actual.dtype == np.complex64 and actual.shape == (PACKET_LEN,)
    assert len({row["seed"] for row in records}) == 8
    assert all(row["aperture_context_samples"] == 12 for row in records)
    assert all(row["noise_reference_fs_hz"] == 256000 for row in records)
    assert all(row["numerical_filter_latency_samples"] == 31 for row in records)
    assert all(not row["channel_equalization_applied"] for row in records)
    assert not np.array_equal(actual.reshape(8, STRIDE)[:, :24], packet.reshape(8, STRIDE)[:, :24])
    assert np.array_equal(native_residual(packet, "clean", "rx", 12, "test"), packet)
