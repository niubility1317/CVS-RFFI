from __future__ import annotations

import pytest

from SSDG.train_ssdg import _phase1_tx_partition_view, build_arg_parser


def _dataset():
    labels = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]
    return {
        "tx_list": labels,
        "data": [[f"sample:{label}"] for label in labels],
        "rx_list": ["rx0"],
        "capture_date_list": ["day0"],
    }


def test_phase1_tx_partition_filters_and_contiguously_reindexes_training_view():
    filtered, receipt = _phase1_tx_partition_view(
        _dataset(),
        train_spec="14-10,14-7,20-15,20-19",
        known_validation_spec="6-15",
        proxy_unknown_spec="8-20",
    )

    assert filtered["tx_list"] == ["14-10", "14-7", "20-15", "20-19"]
    assert filtered["data"] == [
        ["sample:14-10"],
        ["sample:14-7"],
        ["sample:20-15"],
        ["sample:20-19"],
    ]
    assert receipt["held_tx_loaded_by_training"] is False
    assert receipt["training_view_contiguous_reindex"] == {
        "0": "14-10",
        "1": "14-7",
        "2": "20-15",
        "3": "20-19",
    }
    assert len(receipt["partition_sha256"]) == 64


@pytest.mark.parametrize(
    "known_validation,proxy_unknown,match",
    [
        ("14-10", "8-20", "roles overlap"),
        ("", "8-20", "requires non-empty"),
        ("not-a-tx", "8-20", "absent from dataset"),
    ],
)
def test_phase1_tx_partition_rejects_invalid_role_manifests(
    known_validation: str,
    proxy_unknown: str,
    match: str,
):
    with pytest.raises(ValueError, match=match):
        _phase1_tx_partition_view(
            _dataset(),
            train_spec="14-10,14-7,20-15,20-19",
            known_validation_spec=known_validation,
            proxy_unknown_spec=proxy_unknown,
        )


def test_phase1_tx_partition_cli_is_explicit():
    args = build_arg_parser().parse_args(
        [
            "--output_dir",
            "unused",
            "--phase1_source_train_tx_ids",
            "14-10,14-7,20-15,20-19",
            "--phase1_source_known_validation_tx_ids",
            "6-15",
            "--phase1_source_proxy_unknown_tx_ids",
            "8-20",
        ]
    )
    assert args.phase1_source_train_tx_ids.endswith("20-19")
    assert args.phase1_source_known_validation_tx_ids == "6-15"
    assert args.phase1_source_proxy_unknown_tx_ids == "8-20"
