import json

import numpy as np
import pytest
import torch

from paper_reproduction.gaskin_tweak_2023.calibration import (
    calibrate_multiple_domains,
    closed_set_predict,
)
from paper_reproduction.gaskin_tweak_2023.official_lora import (
    load_configuration_records,
    read_iq_frames,
    split_record_frames,
)
from paper_reproduction.gaskin_tweak_2023.official_lora_experiment import build_configuration_portability_plan


def _write_record(root, config, device_id, values):
    directory = root / config
    directory.mkdir(parents=True, exist_ok=True)
    data_path = directory / f"IQ_{device_id}.dat"
    np.asarray(values, dtype=np.complex64).tofile(data_path)
    data_path.with_suffix(".sigmf-meta").write_text(
        json.dumps({"_metadata": {"global": {"core:datatype": "cf32"}}}), encoding="utf-8"
    )


def test_official_cf32_reader_keeps_iq_order_and_uses_nonoverlapping_128_sample_frames(tmp_path):
    # Break caught: swapping I/Q or reading the cf32 file as a real-valued stream.
    values = np.arange(256, dtype=np.float32) + 1j * (1000 + np.arange(256, dtype=np.float32))
    _write_record(tmp_path, "Config1", 1, values)

    records = load_configuration_records(tmp_path, "Config1", device_ids=[1])
    frames = read_iq_frames(records[0], frame_indices=[0, 1])

    assert tuple(frames.shape) == (2, 2, 128)
    assert torch.equal(frames[0, 0], torch.arange(128, dtype=torch.float32))
    assert torch.equal(frames[0, 1], 1000 + torch.arange(128, dtype=torch.float32))
    assert torch.equal(frames[1, 0], 128 + torch.arange(128, dtype=torch.float32))


def test_official_cf32_reader_does_not_require_torch_from_numpy_abi_bridge(tmp_path, monkeypatch):
    # Break caught: N607's NumPy 2 / Torch 2.1 pair rejects torch.from_numpy(np.ndarray).
    values = np.arange(128, dtype=np.float32) + 1j * (500 + np.arange(128, dtype=np.float32))
    _write_record(tmp_path, "Config1", 1, values)
    record = load_configuration_records(tmp_path, "Config1", device_ids=[1])[0]

    def reject_numpy_bridge(_array):
        raise TypeError("expected np.ndarray (got numpy.ndarray)")

    monkeypatch.setattr(torch, "from_numpy", reject_numpy_bridge)

    frames = read_iq_frames(record, frame_indices=[0])

    assert tuple(frames.shape) == (1, 2, 128)
    assert torch.equal(frames[0, 0], torch.arange(128, dtype=torch.float32))
    assert torch.equal(frames[0, 1], 500 + torch.arange(128, dtype=torch.float32))


def test_official_lora_split_matches_paper_75_25_split_and_n_is_ten_percent_of_training():
    # Break caught: calibration leaking into the held-out 25% test region.
    split = split_record_frames(total_samples=20_000_000)

    assert split.total_frames == 156_250
    assert split.training == range(0, 117_187)
    assert split.calibration == range(0, 11_718)
    assert split.testing == range(117_187, 156_250)
    assert set(split.calibration).isdisjoint(split.testing)


def test_multiple_configuration_calibration_keeps_one_centroid_radius_pair_per_device_and_domain():
    # Break caught: pooling configurations into one centroid instead of preserving Algorithm 1's repeated calibration.
    features = torch.tensor([[0.0], [2.0], [10.0], [12.0], [100.0], [102.0], [110.0], [112.0]])
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    domains = ["Config1"] * 4 + ["Config2"] * 4

    state = calibrate_multiple_domains(features, labels, domains, samples_per_class=2)

    assert state.labels.tolist() == [0, 1, 0, 1]
    assert state.domains == ("Config1", "Config1", "Config2", "Config2")
    assert state.centroids.squeeze(1).tolist() == [1.0, 11.0, 101.0, 111.0]
    assert state.radii.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert closed_set_predict(torch.tensor([[101.0], [111.0]]), state).tolist() == [0, 1]


def test_configuration_portability_plan_is_config2_source_with_four_single_domain_matrices_and_one_multiple_calibration_result():
    # Break caught: launching a partial grid or an unintended source configuration.
    plan = build_configuration_portability_plan()

    assert plan.source_configuration == "Config2"
    assert plan.device_ids == tuple(range(1, 11))
    assert plan.calibration_configurations == ("Config1", "Config2", "Config3", "Config4")
    assert plan.test_configurations == ("Config1", "Config2", "Config3", "Config4")
    assert {(row.calibration_configuration, row.test_configuration) for row in plan.single_domain_rows} == {
        (calibration, testing)
        for calibration in plan.calibration_configurations
        for testing in plan.test_configurations
    }
    assert plan.group_size == 10
    assert plan.epochs == 100
