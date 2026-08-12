from __future__ import annotations

"""RED contracts for the public source-only CLIC common-receipt exporter."""

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

import evaluate_phase1_clic_postfreeze_pair as PAIR
from test_phase1_clic_postfreeze import (
    SOURCE_TX,
    TRAINING_RUN,
    _checkpoint_fixture,
)


REQUIRED_FIELDS = {
    "arm",
    "fold_index",
    "training_run_root",
    "scene_order",
    "physical_row_count",
    "physical_order_sha256",
    "class_order_sha256",
    "source_split_sha256",
    "common_batch_sequence_sha256",
}
FORBIDDEN_KEY_PARTS = (
    "target",
    "query",
    "truth",
    "role",
)


def _exporter():
    """Resolve the required public API before any expected-rejection context."""

    return getattr(PAIR, "export_clic_common_training_receipt")


def _strict_terminal(paths: Mapping[str, object]) -> dict[str, object]:
    envelope = json.loads(Path(str(paths["terminal"])).read_text(encoding="utf-8"))
    return dict(envelope["strict_core"])


def _assert_source_only_receipt(
    receipt: Mapping[str, object],
    strict: Mapping[str, object],
    *,
    expected_arm: str,
) -> None:
    assert set(REQUIRED_FIELDS).issubset(receipt)
    assert receipt["arm"] == expected_arm
    assert receipt["fold_index"] == 1
    assert receipt["training_run_root"] == TRAINING_RUN
    assert receipt["scene_order"] == list(PAIR.EXPECTED_SCENARIOS)
    assert receipt["physical_row_count"] == strict["physical_order_count"]
    for field in (
        "physical_order_sha256",
        "class_order_sha256",
        "source_split_sha256",
        "common_batch_sequence_sha256",
    ):
        assert receipt[field] == strict[field]
        assert isinstance(receipt[field], str) and len(receipt[field]) == 64
    assert receipt.get("source_only") is True
    key_names = {str(key).lower() for key in receipt}
    assert not any(
        any(part in key_name for part in FORBIDDEN_KEY_PARTS)
        for key_name in key_names
    )
    assert set(receipt).isdisjoint({"target_access", "query_access", "query_truth_access", "query_role_access"})


@pytest.mark.parametrize("arm", ("C", "G"))
def test_export_clic_common_training_receipt_reopens_terminal_and_writes_source_only_projection(
    tmp_path: Path,
    arm: str,
) -> None:
    paths = _checkpoint_fixture(tmp_path, arm=arm, fold=1)
    output = tmp_path / f"{arm}_common_training_receipt.json"
    strict = _strict_terminal(paths)

    _exporter()(
        paths["checkpoint"],
        paths["terminal"],
        output,
        expected_arm=arm,
        fold_index=1,
        training_run_root=TRAINING_RUN,
    )

    receipt = json.loads(output.read_text(encoding="utf-8"))
    _assert_source_only_receipt(receipt, strict, expected_arm=arm)


def test_export_clic_common_training_receipt_refuses_overwrite(tmp_path: Path) -> None:
    paths = _checkpoint_fixture(tmp_path, arm="G", fold=1)
    output = tmp_path / "G_common_training_receipt.json"
    exporter = _exporter()
    exporter(
        paths["checkpoint"],
        paths["terminal"],
        output,
        expected_arm="G",
        fold_index=1,
        training_run_root=TRAINING_RUN,
    )
    before = output.read_bytes()

    with pytest.raises(Exception, match="overwrite|immutable|exists"):
        exporter(
            paths["checkpoint"],
            paths["terminal"],
            output,
            expected_arm="G",
            fold_index=1,
            training_run_root=TRAINING_RUN,
        )
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "tamper",
    (
        "checkpoint_bytes",
        "terminal_bytes",
        "path",
        "arm",
        "fold",
        "physical_count",
        "physical_sha",
        "class_sha",
        "source_split_sha",
        "batch_sha",
    ),
)
def test_export_clic_common_training_receipt_rejects_checkpoint_terminal_path_and_binding_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    paths = _checkpoint_fixture(tmp_path, arm="G", fold=1)
    output = tmp_path / f"G_common_training_receipt_{tamper}.json"
    checkpoint = Path(str(paths["checkpoint"]))
    terminal = Path(str(paths["terminal"]))
    expected_arm = "G"
    fold_index = 1
    training_run_root = TRAINING_RUN

    if tamper == "checkpoint_bytes":
        checkpoint.write_bytes(b"tampered-checkpoint")
    elif tamper == "terminal_bytes":
        envelope = json.loads(terminal.read_text(encoding="utf-8"))
        envelope["strict_core"]["arm"] = "C"
        terminal.write_text(
            json.dumps(envelope, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif tamper == "path":
        checkpoint = tmp_path / "wrong-path" / "final_ssdg.pth"
    elif tamper == "arm":
        expected_arm = "C"
    elif tamper == "fold":
        fold_index = 2
    else:
        envelope = json.loads(terminal.read_text(encoding="utf-8"))
        strict = dict(envelope["strict_core"])
        strict_field = {
            "physical_count": "physical_order_count",
            "physical_sha": "physical_order_sha256",
            "class_sha": "class_order_sha256",
            "source_split_sha": "source_split_sha256",
            "batch_sha": "common_batch_sequence_sha256",
        }[tamper]
        if tamper == "physical_count":
            strict[strict_field] = int(strict[strict_field]) + 1
        else:
            strict[strict_field] = "0" * 64
        envelope["strict_core"] = strict
        terminal.write_text(
            json.dumps(envelope, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    exporter = _exporter()
    with pytest.raises(Exception):
        exporter(
            checkpoint,
            terminal,
            output,
            expected_arm=expected_arm,
            fold_index=fold_index,
            training_run_root=training_run_root,
        )
    assert not output.exists()
