from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from model_dual_cvsincnet import build_dual_model
from SSDG.train_ssdg import build_arg_parser


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "code" / "configs" / "phase1_adv3b02_nmfdu_gate_v1.json"


def _small_model(**kwargs):
    return build_dual_model(
        num_classes=4,
        num_domains=3,
        model_size="S",
        input_len=256,
        fast_infer_when_no_aux=False,
        **kwargs,
    )


def test_explicit_none_is_tensor_identical_to_legacy_default() -> None:
    """Changing the default path or adding gate parameters to it is a regression."""
    torch.manual_seed(901)
    legacy = _small_model()
    torch.manual_seed(901)
    explicit_none = _small_model(physical_gate_variant="none")

    legacy_state = legacy.state_dict()
    none_state = explicit_none.state_dict()
    assert legacy_state.keys() == none_state.keys()
    for key in legacy_state:
        torch.testing.assert_close(legacy_state[key], none_state[key], rtol=0.0, atol=0.0)
    assert legacy.physical_gate_variant == "none"
    assert explicit_none.physical_gate_variant == "none"
    assert not any("nmfdu" in key for key in legacy_state)


def test_nmfdu_variant_alone_constructs_new_gate_parameters() -> None:
    """Silently accepting the flag without constructing a reachable gate must fail."""
    model = _small_model(physical_gate_variant="nmfdu_v1")
    state_keys = set(model.state_dict())

    assert model.physical_gate_variant == "nmfdu_v1"
    assert model.id_backbone.physical_gate_variant == "nmfdu_v1"
    assert model.id_backbone.nmfdu_gate is not None
    assert any(key.startswith("id_backbone.nmfdu_gate.") for key in state_keys)
    assert not any(key.startswith("dom_backbone.nmfdu_gate.") for key in state_keys)


def test_unknown_physical_gate_variant_fails_closed() -> None:
    with pytest.raises(ValueError, match="physical_gate_variant"):
        _small_model(physical_gate_variant="learned_attention")


def test_cli_and_candidate_config_freeze_the_nmfdu_contract() -> None:
    parser = build_arg_parser()
    defaults = parser.parse_args(["--output_dir", "unused"])
    enabled = parser.parse_args(
        ["--output_dir", "unused", "--physical_gate_variant", "nmfdu_v1"]
    )

    assert defaults.physical_gate_variant == "none"
    assert enabled.physical_gate_variant == "nmfdu_v1"

    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["candidate_id"] == "ADV3B02_NMFDU_GATE_V1_E200"
    assert payload["physical_gate_variant"] == "nmfdu_v1"
    assert payload["branch_names"] == ["raw", "hom", "phase", "pa", "hos"]
    assert payload["epochs"] == 200
    assert payload["stage_boundaries"] == [80, 120, 200]
    assert payload["lambda_sat_cls"] == pytest.approx(0.68)
    assert payload["lambda_sat_cons"] == pytest.approx(0.0)
    assert payload["source_split"] == {"L_s": 0.07, "U_s": 0.63, "V": 0.30}
