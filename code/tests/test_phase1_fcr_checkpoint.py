from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import checkpoint as checkpoint_module  # noqa: E402
from cvsrffi.phase1_fcr_types import FCRConfig  # noqa: E402
from model_dual_cvsincnet import build_dual_model  # noqa: E402


FEATURE_SCHEMA = "ADV3B02:FCR:z_f_id:unit_l2:160:v1"


def _small_model(*, use_fcr: bool, decoder_mode: str | None = None, num_domains: int = 2):
    fcr_config = None
    if use_fcr and decoder_mode is not None:
        fcr_config = FCRConfig(input_len=64, decoder_mode=decoder_mode)
    return build_dual_model(
        num_classes=3,
        num_domains=num_domains,
        model_size="S",
        dataset="wisig",
        input_len=64,
        model_variant="lite_d",
        fast_infer_when_no_aux=False,
        use_fcr=use_fcr,
        fcr_config=fcr_config,
    )


def _save(path: Path, model) -> dict:
    checkpoint_module.save_checkpoint(
        str(path),
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=3,
        args=argparse.Namespace(run_id="unit"),
        split_info={},
        stats={},
    )
    return torch.load(path, map_location="cpu", weights_only=False)


def test_fcr_checkpoint_bundle_is_complete_and_does_not_duplicate_weights(tmp_path: Path) -> None:
    model = _small_model(use_fcr=True)
    payload = _save(tmp_path / "fcr.pth", model)
    bundle = payload["fcr_bundle"]

    assert bundle["bundle_schema"] == "cvs.phase1.adv3b02_fcr.bundle.v1"
    assert bundle["feature_schema"] == FEATURE_SCHEMA
    assert bundle["fcr_config"] == {
        "input_len": 64,
        "content_stride": 4,
        "content_dim": 32,
        "tx_state_dim": 16,
        "channel_dim": 16,
        "receiver_dim": 8,
        "sync_dim": 6,
        "gain_dim": 3,
        "variance_floor": 1e-4,
        "variance_ceiling": 1.0,
        "decoder_mode": "full_physics",
    }
    assert bundle["physical_basis"]["identifier"] == "fixed_response_basis:pa_conjugate_memory4:v1"
    assert bundle["input_normalization"]["version"] == "adv3b02_input_iq:v1"
    assert bundle["fisher_gate"] == {
        "identifier": "FisherIdentifiabilityGate:v1",
        "deterministic": True,
        "trainable_parameters": 0,
        "eps": 1e-8,
    }
    assert bundle["nuisance_schema"] == {
        "order": ["channel", "receiver", "sync", "gain"],
        "dimensions": {"channel": 16, "receiver": 8, "sync": 6, "gain": 3},
        "version": "structured_nuisance:16_8_6_3:v1",
    }
    assert bundle["routing"]["fingerprint_excitation"] == "content.s_hat.detach()"
    assert not any(torch.is_tensor(value) for value in bundle.values())
    assert any(key.startswith("fcr.") for key in payload["model"])


def test_fcr_strict_round_trip_reproduces_single_leo_zf_id(tmp_path: Path) -> None:
    torch.manual_seed(11103)
    source = _small_model(use_fcr=True).eval()
    leo_iq = torch.randn(1, 2, 64)
    with torch.no_grad():
        reference = source(leo_iq, return_aux=True)["z_f_id"].clone()

    path = tmp_path / "fcr.pth"
    _save(path, source)
    restored = _small_model(use_fcr=True).eval()
    checkpoint_module.load_fcr_checkpoint_strict(path, restored, map_location="cpu")
    with torch.no_grad():
        actual = restored(leo_iq, return_aux=True)["z_f_id"]

    torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)


def test_decoder_mode_is_bundled_and_mismatched_evaluation_model_fails_closed(tmp_path: Path) -> None:
    source = _small_model(use_fcr=True, decoder_mode="control")
    payload = _save(tmp_path / "control.pth", source)
    assert payload["fcr_bundle"]["fcr_config"]["decoder_mode"] == "control"

    wrong_mode = _small_model(use_fcr=True, decoder_mode="full_physics")
    with pytest.raises(ValueError, match="fcr_config"):
        checkpoint_module.load_fcr_checkpoint_strict(
            tmp_path / "control.pth",
            wrong_mode,
            map_location="cpu",
        )


def test_legacy_checkpoint_without_bundle_strictly_loads_closed_model(tmp_path: Path) -> None:
    source = _small_model(use_fcr=False)
    payload = _save(tmp_path / "legacy-full.pth", source)
    payload.pop("fcr_bundle", None)
    legacy_path = tmp_path / "legacy.pth"
    torch.save(payload, legacy_path)

    restored = _small_model(use_fcr=False)
    checkpoint_module.load_fcr_checkpoint_strict(legacy_path, restored, map_location="cpu")
    assert restored.state_dict().keys() == source.state_dict().keys()


def test_incompatible_feature_schema_is_rejected_before_model_state_use(tmp_path: Path) -> None:
    source = _small_model(use_fcr=True)
    payload = _save(tmp_path / "valid.pth", source)
    incompatible = copy.deepcopy(payload)
    incompatible["fcr_bundle"]["feature_schema"] = "wrong"
    restored = _small_model(use_fcr=True)
    before = {name: value.clone() for name, value in restored.state_dict().items()}

    with pytest.raises(ValueError, match="feature_schema"):
        checkpoint_module.validate_fcr_bundle_for_model(incompatible, restored)

    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)


def test_trusted_legacy_checkpoint_with_training_metadata_can_warm_start_fcr(
    tmp_path: Path,
) -> None:
    import train

    legacy = _small_model(use_fcr=False)
    path = tmp_path / "legacy-with-namespace.pth"
    torch.save(
        {
            "model": legacy.state_dict(),
            "args": argparse.Namespace(candidate_id="ADV3B02_CORE90_SOFT_E200"),
        },
        path,
    )
    candidate = _small_model(use_fcr=True)

    train.load_init_checkpoint_weights(candidate, str(path), torch.device("cpu"))

    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0.0, atol=0.0)


def test_legacy_adv3_checkpoint_warm_start_preserves_zero_step_identity_logits(
    tmp_path: Path,
) -> None:
    import train

    torch.manual_seed(21001)
    legacy = _small_model(use_fcr=False).eval()
    path = tmp_path / "adv3-legacy.pth"
    torch.save({"model": legacy.state_dict(), "epoch": 200}, path)

    torch.manual_seed(21002)
    candidate = _small_model(use_fcr=True).eval()
    train.load_init_checkpoint_weights(candidate, str(path), torch.device("cpu"))
    iq = torch.randn(3, 2, 64)
    with torch.no_grad():
        legacy_logits = legacy(iq, y_tx=None, return_aux=True)["tx_logits"]
        fcr_logits = candidate(iq, y_tx=None, return_aux=True)["fcr_tx_logits"]
        labels = torch.tensor([0, 1, 2], dtype=torch.long)
        legacy_margin_logits = legacy(iq, y_tx=labels, return_aux=True)["tx_logits"]
        fcr_margin_logits = candidate(iq, y_tx=labels, return_aux=True)["fcr_tx_logits"]

    torch.testing.assert_close(fcr_logits, legacy_logits, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(fcr_margin_logits, legacy_margin_logits, rtol=0.0, atol=3e-6)


def test_locked_v5_warm_start_rejects_wrong_metadata_and_fcr_state(tmp_path: Path) -> None:
    import train

    legacy = _small_model(use_fcr=False)
    candidate = _small_model(use_fcr=True)
    payload = {
        "model": legacy.state_dict(),
        "epoch": 200,
        "final_epoch": 200,
        "candidate_id": "S392002_ADV3B03_MU10_ALPHA20_E200",
        "args": {"seed": 392002},
    }
    path = tmp_path / "locked-v5.pth"
    torch.save(payload, path)
    policy = dict(
        expected_seed=392002,
        expected_epoch=200,
        expected_candidate_id="S392002_ADV3B03_MU10_ALPHA20_E200",
        require_mature_base_complete=True,
    )
    train.load_init_checkpoint_weights(candidate, str(path), torch.device("cpu"), **policy)

    wrong_seed = copy.deepcopy(payload)
    wrong_seed["args"]["seed"] = 392005
    wrong_seed_path = tmp_path / "wrong-seed.pth"
    torch.save(wrong_seed, wrong_seed_path)
    with pytest.raises(ValueError, match="seed"):
        train.load_init_checkpoint_weights(candidate, str(wrong_seed_path), torch.device("cpu"), **policy)

    fcr_payload = copy.deepcopy(payload)
    fcr_payload["model"]["fcr_identity_head.weight"] = candidate.fcr_identity_head.weight.detach().clone()
    fcr_path = tmp_path / "old-fcr.pth"
    torch.save(fcr_payload, fcr_path)
    with pytest.raises(ValueError, match="FCR state"):
        train.load_init_checkpoint_weights(candidate, str(fcr_path), torch.device("cpu"), **policy)


def test_locked_v5_warm_start_rejects_incomplete_mature_base(tmp_path: Path) -> None:
    import train

    legacy = _small_model(use_fcr=False)
    incomplete = dict(legacy.state_dict())
    incomplete.pop("id_backbone.cls_head.head.weight")
    path = tmp_path / "incomplete.pth"
    torch.save(
        {
            "model": incomplete,
            "epoch": 200,
            "candidate_id": "S392002_ADV3B03_MU10_ALPHA20_E200",
            "args": {"seed": 392002},
        },
        path,
    )
    with pytest.raises(RuntimeError, match="mature base"):
        train.load_init_checkpoint_weights(
            _small_model(use_fcr=True),
            str(path),
            torch.device("cpu"),
            expected_seed=392002,
            expected_epoch=200,
            expected_candidate_id="S392002_ADV3B03_MU10_ALPHA20_E200",
            require_mature_base_complete=True,
        )


def test_locked_identity_warm_start_allows_domain_count_change_but_rejects_missing_identity(
    tmp_path: Path,
) -> None:
    import train

    legacy = _small_model(use_fcr=False, num_domains=3)
    payload = {
        "model": legacy.state_dict(),
        "epoch": 200,
        "candidate_id": "S392002_ADV3B03_MU10_ALPHA20_E200",
        "args": {"seed": 392002},
    }
    path = tmp_path / "domain3.pth"
    torch.save(payload, path)
    policy = dict(
        expected_seed=392002,
        expected_epoch=200,
        expected_candidate_id="S392002_ADV3B03_MU10_ALPHA20_E200",
        require_mature_identity_complete=True,
    )
    train.load_init_checkpoint_weights(
        _small_model(use_fcr=True, num_domains=2), str(path), torch.device("cpu"), **policy
    )

    incomplete = copy.deepcopy(payload)
    incomplete["model"].pop("id_backbone.cls_head.head.weight")
    incomplete_path = tmp_path / "missing-identity.pth"
    torch.save(incomplete, incomplete_path)
    with pytest.raises(RuntimeError, match="mature identity"):
        train.load_init_checkpoint_weights(
            _small_model(use_fcr=True, num_domains=2),
            str(incomplete_path),
            torch.device("cpu"),
            **policy,
        )
