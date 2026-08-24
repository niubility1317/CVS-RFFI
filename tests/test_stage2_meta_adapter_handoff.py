from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.meta_adapter import ResidualMetaAdapter  # noqa: E402
from cvsrffi.stage2_meta_adapter_handoff import (  # noqa: E402
    freeze_da1_reg0_handoff,
)


class _ToyMetaModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.meta_adapter_time = ResidualMetaAdapter(dim=2, rank=1)
        self.cls_head = nn.Linear(2, 2)


def valid_binding() -> dict[str, str]:
    return {
        "checkpoint_id": "ckpt-meta-001",
        "bundle_id": "bundle-meta-001",
        "capsule_id": "capsule-fixed",
        "split_id": "split-fixed",
    }


def test_handoff_contains_no_optimizer_head_or_new_class_update() -> None:
    model = _ToyMetaModel()

    handoff = freeze_da1_reg0_handoff(model, valid_binding())

    assert handoff.state == "DA1_REG0"
    assert handoff.optimizer_state is None
    assert handoff.new_class_support_consumed is False
    assert all("cls_head" not in name for name in handoff.adapted_state)
    assert handoff.checkpoint_id == "ckpt-meta-001"
    assert handoff.bundle_id == "bundle-meta-001"
    assert handoff.capsule_id == "capsule-fixed"
    assert handoff.split_id == "split-fixed"
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.training is False


def test_handoff_rejects_query_truth_or_head_binding_fields() -> None:
    model = _ToyMetaModel()
    binding = valid_binding()
    binding["query_path"] = "should-never-cross-handoff"

    with pytest.raises(ValueError, match="query|truth|allowlist"):
        freeze_da1_reg0_handoff(model, binding)


def test_handoff_serialization_refuses_overwrite(tmp_path: Path) -> None:
    model = _ToyMetaModel()
    handoff = freeze_da1_reg0_handoff(model, valid_binding())
    output = tmp_path / "handoff.json"
    output.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        handoff.write_json(output)
    assert output.read_text(encoding="utf-8") == "keep"
