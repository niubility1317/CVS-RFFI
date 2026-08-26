from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import build_adv3b02_core90_phase1_bundle as builder
from cvsrffi.phase1_adv3b02_deployment_bundle import (
    CLASS_BINDING_SCHEMA,
    class_handle_binding_sha256,
)
from cvsrffi.phase1_center_lowrank_prototype_bundle import (
    PENDING_OUTER_JOINT_SEAL,
    SCHEMA as COMPONENT_SCHEMA,
)
from cvsrffi.sf_tapft_phase1_binding import _PHASE1_BUNDLE_KEYS


CORE90 = "ADV3B02_CORE90_SOFT_E200"
CORE90_SHA = "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
CURRENT_BINDING_SHA = "a90931dd0266cbd42b1163a61d015d5bfe955d2ab287733d8674b9da92d722d0"
LEGACY_BINDING_SHA = "76735ae6d9b2d7e58f683635ca2644e00fbd27a515246aab9d47488c1ab5111f"
LEGACY_COMPONENT_ROOT = "098badd1e82c05c1029cb02c024fe7d3c433488e8ab22e5c6e2ba0516b8d0055"
LEGACY_COMPONENT_NPZ_SHA = "1ac2424fee2ef804d83d7c8faca8d27c7c0267c0d9d7c8b97af0cf053bfb4ea6"
REAL_TXS = ("14-10", "14-7", "20-15", "20-19", "6-15", "8-20")
REAL_HANDLES = (
    "cls_75aa6d506081240f50cf3b79a0bd91714fa0084a635a472ca63194e57ec1dca2",
    "cls_8b02d99905a8fe579368ac8e37eff51c505aaa89a646eba8892d5d800aa08416",
    "cls_1f33441efa14970113b27483344b7df852a9041984b38d34ce570fafbab6689c",
    "cls_f8dfc2edcccc5344f8e2535a959f13b53a1cddfd6fb22aed6e714de382b58d24",
    "cls_a53ca1280d8fca58e3f4d6d1e9ddabfdab6027a941ee8c3f8c01d9d8ec945725",
    "cls_33bbd16556c6e6305d1b7162f5ea71393afba910a922f9abca5999d5921a2d9d",
)
REAL_D85_COMPONENT = Path(
    r"E:\type10-7\code\snapshots\d81wt\automation_reports\CV-SincNet"
    r"\d85_ground_radius_v2_20260720\artifacts\component"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _d19_binding(checkpoint_sha: str) -> dict[str, object]:
    return {
        "schema": "cvs.phase2.d19_adv3b02_class_binding.v1",
        "checkpoint_sha256": checkpoint_sha,
        "entries": [
            {
                "class_index": index,
                "phase1_tx": tx,
                "registered_class_handle": handle,
            }
            for index, (tx, handle) in enumerate(zip(REAL_TXS, REAL_HANDLES))
        ],
        "evidence": {},
    }


def _component_manifest(checkpoint_sha: str, binding_sha: str) -> dict[str, object]:
    return {
        "schema": COMPONENT_SCHEMA,
        "component_state": PENDING_OUTER_JOINT_SEAL,
        "checkpoint_sha256": checkpoint_sha,
        "class_handle_binding_sha256": binding_sha,
        "pre_sign_content_root_sha256": "1" * 64,
        "generation_config_sha256": "2" * 64,
        "generation_code_sha256": "3" * 64,
        "phase1_stream_sha256": "4" * 64,
        "radius_generation_proof_sha256": "5" * 64,
    }


def _fake_parity(checkpoint_sha: str, runtime_sha: str) -> dict[str, object]:
    return {
        "schema": "cvs.phase1.runtime_checkpoint_parity_receipt.v1",
        "checkpoint_lineage_sha256": checkpoint_sha,
        "runtime_sha256": runtime_sha,
        "parity_status": "PASS",
    }


def _prepare_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, dict[str, object]]:
    checkpoint = tmp_path / "core90.pth"
    torch.save({"candidate_id": CORE90, "args": {"input_len": 256}}, checkpoint)
    checkpoint_sha = _sha256(checkpoint)
    binding_path = tmp_path / "d19.json"
    binding_payload = _d19_binding(checkpoint_sha)
    _write_json(binding_path, binding_payload)
    handles = tuple(
        str(row["registered_class_handle"])
        for row in binding_payload["entries"]  # type: ignore[index]
    )
    binding_sha = class_handle_binding_sha256(handles)
    component_dir = tmp_path / "component"
    component_dir.mkdir()
    manifest = _component_manifest(checkpoint_sha, binding_sha)

    monkeypatch.setattr(
        builder,
        "_load_checkpoint",
        lambda path: torch.load(path, map_location="cpu", weights_only=False),
    )

    def fake_regular_sha(path: Path, *, context: str) -> str:
        if context == "CORE90 checkpoint":
            return CORE90_SHA
        return _sha256(Path(path))

    monkeypatch.setattr(builder, "_sha256_regular", fake_regular_sha)
    manifest["checkpoint_sha256"] = CORE90_SHA
    binding_payload["checkpoint_sha256"] = CORE90_SHA
    _write_json(binding_path, binding_payload)
    monkeypatch.setattr(
        builder,
        "_rebind_legacy_d85_component",
        lambda *_args, **_kwargs: {
            "manifest": dict(manifest),
            "component_dir": str(component_dir),
        },
    )

    monkeypatch.setattr(
        builder,
        "validate_center_lowrank_component",
        lambda *_args, **_kwargs: dict(manifest),
    )
    monkeypatch.setattr(
        builder,
        "load_center_lowrank_component",
        lambda *_args, **_kwargs: SimpleNamespace(
            class_registry=REAL_HANDLES, manifest=dict(manifest)
        ),
    )

    def fake_runtime(
        _checkpoint: object,
        *,
        input_len: int,
        device: torch.device,
        runtime_path: Path,
        parity_seed: int,
        parity_rows: int,
    ) -> tuple[str, dict[str, object], dict[str, object]]:
        assert input_len == 256
        assert str(device) == "cuda:0"
        assert parity_seed == 7281105
        assert parity_rows == 8
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_bytes(b"synthetic-parity-runtime")
        runtime_sha = _sha256(runtime_path)
        return runtime_sha, _fake_parity(checkpoint_sha, runtime_sha), {"status": "PASS"}

    monkeypatch.setattr(builder, "_runtime_and_parity", fake_runtime)

    def fake_bundle(package_root: Path, **kwargs: object) -> dict[str, object]:
        assert json.loads(Path(kwargs["method_lock_path"]).read_text())["method_id"] == CORE90
        assert json.loads(Path(kwargs["class_binding_path"]).read_text())["schema"] == CLASS_BINDING_SCHEMA
        package_root.mkdir(parents=True)
        Path(kwargs["detached_seal_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["detached_seal_path"]).write_text("{}", encoding="utf-8")
        Path(kwargs["signing_request_path"]).write_text("{}", encoding="utf-8")
        return {
            "outer_content_root_sha256": "6" * 64,
            "detached_seal_sha256": _sha256(Path(kwargs["detached_seal_path"])),
            "checkpoint_lineage_sha256": CORE90_SHA,
            "runtime_sha256": json.loads(
                Path(kwargs["parity_receipt_path"]).read_text()
            )["runtime_sha256"],
            "component_pre_sign_content_root_sha256": "1" * 64,
            "class_handle_binding_sha256": binding_sha,
            "parity_receipt_sha256": _sha256(Path(kwargs["parity_receipt_path"])),
            "generation_lock_sha256": _sha256(Path(kwargs["generation_lock_path"])),
            "method_lock_sha256": _sha256(Path(kwargs["method_lock_path"])),
            "generation_config_sha256": "2" * 64,
            "generation_code_sha256": "3" * 64,
        }

    monkeypatch.setattr(builder, "build_unsigned_adv3b02_deployment_bundle", fake_bundle)
    return checkpoint, component_dir, binding_path, manifest


def test_prepare_is_aggregate_only_and_cli_has_no_private_key_surface() -> None:
    allowed = {
        "checkpoint",
        "component_dir",
        "class_binding_source",
        "output_root",
        "device",
        "input_len",
        "parity_seed",
        "parity_rows",
    }
    assert set(inspect.signature(builder.prepare).parameters) == allowed
    forbidden = (
        "dataset",
        "source_path",
        "source_loader",
        "raw_iq",
        "sample_feature",
        "support",
        "query",
        "truth",
        "private",
        "key",
    )
    parser = builder.build_arg_parser()
    option_names = {
        option.lstrip("-").replace("-", "_")
        for action in parser._actions
        for option in action.option_strings
    }
    for subparser in parser._subparsers._group_actions[0].choices.values():
        option_names.update(
            option.lstrip("-").replace("-", "_")
            for action in subparser._actions
            for option in action.option_strings
        )
    assert not any(token in name for name in option_names for token in forbidden)
    assert set(parser._subparsers._group_actions[0].choices) == {"prepare", "finalize"}


def test_prepare_builds_core90_bundle_from_aggregate_only_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, component_dir, binding_path, _manifest = _prepare_fixture(
        tmp_path, monkeypatch
    )
    result = builder.prepare(
        checkpoint=checkpoint,
        component_dir=component_dir,
        class_binding_source=binding_path,
        output_root=tmp_path / "output",
        device="cuda:0",
        input_len=0,
        parity_seed=7281105,
        parity_rows=8,
    )
    assert result["status"] == "AWAITING_EXTERNAL_SIGNATURE"
    assert result["checkpoint_lineage_sha256"] == CORE90_SHA
    assert result["candidate_id"] == CORE90
    assert set((tmp_path / "output").iterdir()) == {
        tmp_path / "output" / "work",
        tmp_path / "output" / "package",
        tmp_path / "output" / "external",
        tmp_path / "output" / "prepare_receipt.json",
    }


def test_prepare_rejects_alternate_checkpoint_before_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "alternate.pth"
    torch.save({"candidate_id": "P1-FULL", "args": {"input_len": 256}}, checkpoint)
    with pytest.raises(builder.Core90BundleError, match="CORE90"):
        builder._load_checkpoint(checkpoint)


def test_checkpoint_candidate_name_cannot_replace_real_core90_sha(tmp_path: Path) -> None:
    checkpoint = tmp_path / "same-name-wrong-bytes.pth"
    torch.save({"candidate_id": CORE90, "args": {"input_len": 256}}, checkpoint)
    assert _sha256(checkpoint) != CORE90_SHA
    with pytest.raises(builder.Core90BundleError, match="immutable CORE90 checkpoint SHA"):
        builder._load_checkpoint(checkpoint)


def test_checkpoint_accepts_real_training_layout_candidate_in_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "real-layout.pth"
    torch.save(
        {
            "candidate_id": None,
            "args": {"candidate_id": CORE90, "input_len": 256},
            "baseline_args": {},
            "model": {},
        },
        checkpoint,
    )
    monkeypatch.setattr(
        builder,
        "_sha256_regular",
        lambda *_args, **_kwargs: CORE90_SHA,
    )

    loaded = builder._load_checkpoint(checkpoint)

    assert loaded["args"]["candidate_id"] == CORE90


@pytest.mark.parametrize(
    "top_candidate,args_candidate",
    [
        (CORE90, "P1-FULL"),
        ("P1-FULL", CORE90),
        (None, "P1-FULL"),
        (None, None),
    ],
)
def test_checkpoint_rejects_conflicting_wrong_or_missing_candidate_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    top_candidate: str | None,
    args_candidate: str | None,
) -> None:
    checkpoint = tmp_path / "bad-identity.pth"
    payload: dict[str, object] = {"args": {"input_len": 256}}
    if top_candidate is not None:
        payload["candidate_id"] = top_candidate
    if args_candidate is not None:
        payload["args"]["candidate_id"] = args_candidate  # type: ignore[index]
    torch.save(payload, checkpoint)
    monkeypatch.setattr(
        builder,
        "_sha256_regular",
        lambda *_args, **_kwargs: CORE90_SHA,
    )

    with pytest.raises(builder.Core90BundleError, match="candidate|conflict"):
        builder._load_checkpoint(checkpoint)


@pytest.mark.parametrize("field", ["phase1_tx", "registered_class_handle"])
def test_d19_binding_rejects_any_real_tx_or_handle_drift(
    tmp_path: Path, field: str
) -> None:
    payload = _d19_binding(CORE90_SHA)
    entries = payload["entries"]
    assert isinstance(entries, list)
    entries[2][field] = "drifted"
    path = tmp_path / "binding.json"
    _write_json(path, payload)
    with pytest.raises(builder.Core90BundleError, match="fixed CORE90 TX/handle mapping"):
        builder._load_d19_class_binding(path, checkpoint_sha256=CORE90_SHA)


def test_real_d85_component_rebind_changes_only_class_registry_and_derived_manifest(
    tmp_path: Path,
) -> None:
    assert REAL_D85_COMPONENT.is_dir()
    source_manifest_path = REAL_D85_COMPONENT / "manifest.json"
    source_npz_path = (
        REAL_D85_COMPONENT
        / "int8_domain_class_center_lowrank_residual_radius_v2.npz"
    )
    source_manifest_bytes = source_manifest_path.read_bytes()
    source_npz_bytes = source_npz_path.read_bytes()
    assert _sha256(source_npz_path) == LEGACY_COMPONENT_NPZ_SHA
    rebound = tmp_path / "rebound_component"

    result = builder._rebind_legacy_d85_component(
        REAL_D85_COMPONENT,
        rebound,
        current_class_handles=REAL_HANDLES,
    )

    assert source_manifest_path.read_bytes() == source_manifest_bytes
    assert source_npz_path.read_bytes() == source_npz_bytes
    rebound_npz = rebound / source_npz_path.name
    assert rebound_npz.read_bytes() != source_npz_bytes
    assert _sha256(rebound_npz) != LEGACY_COMPONENT_NPZ_SHA
    with np.load(source_npz_path, allow_pickle=False) as archive:
        source_arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    with np.load(rebound_npz, allow_pickle=False) as archive:
        rebound_arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    assert set(rebound_arrays) == set(source_arrays)
    for name in source_arrays.keys() - {"class_registry"}:
        assert rebound_arrays[name].dtype == source_arrays[name].dtype
        assert rebound_arrays[name].shape == source_arrays[name].shape
        assert rebound_arrays[name].tobytes() == source_arrays[name].tobytes()
    assert tuple(source_arrays["class_registry"].tolist()) == REAL_TXS
    assert tuple(rebound_arrays["class_registry"].tolist()) == REAL_HANDLES
    before = json.loads(source_manifest_bytes)
    after = json.loads((rebound / "manifest.json").read_text(encoding="utf-8"))
    changed = {key for key in before if before[key] != after[key]}
    assert changed == {
        "class_handle_binding_sha256",
        "component_npz_sha256",
        "pre_sign_content_root_sha256",
        "registry_sha256",
        "resource_audit",
        "serialized_component_bytes",
    }
    resource_changed = {
        key
        for key in before["resource_audit"]
        if before["resource_audit"][key] != after["resource_audit"][key]
    }
    assert resource_changed == {
        "logical_deployment_state_bytes",
        "registry_schema_bytes",
        "serialized_component_bytes",
    }
    assert before["class_handle_binding_sha256"] == LEGACY_BINDING_SHA
    assert before["pre_sign_content_root_sha256"] == LEGACY_COMPONENT_ROOT
    assert after["class_handle_binding_sha256"] == CURRENT_BINDING_SHA
    assert result["component_npz_sha256"] == _sha256(rebound_npz)
    assert result["class_registry"] == list(REAL_HANDLES)
    assert result["current_class_handles"] == list(REAL_HANDLES)


def test_rebound_real_d85_component_satisfies_outer_loader_registry_contract(
    tmp_path: Path,
) -> None:
    from cvsrffi.phase1_adv3b02_deployment_bundle import (
        _validate_component_from_opened,
    )

    rebound = tmp_path / "rebound_component"
    result = builder._rebind_legacy_d85_component(
        REAL_D85_COMPONENT,
        rebound,
        current_class_handles=REAL_HANDLES,
    )
    manifest_raw = (rebound / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    npz_path = rebound / "int8_domain_class_center_lowrank_residual_radius_v2.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    bindings = {
        "checkpoint_lineage_sha256": manifest["checkpoint_sha256"],
        "class_handle_binding_sha256": manifest["class_handle_binding_sha256"],
        "component_pre_sign_content_root_sha256": manifest[
            "pre_sign_content_root_sha256"
        ],
        "generation_config_sha256": manifest["generation_config_sha256"],
        "generation_code_sha256": manifest["generation_code_sha256"],
    }

    component = _validate_component_from_opened(
        manifest=manifest,
        manifest_raw=manifest_raw,
        sha_raw=(rebound / "manifest.sha256").read_bytes(),
        npz_sha256=_sha256(npz_path),
        arrays=arrays,
        bindings=bindings,
    )

    assert list(component.class_registry) == list(REAL_HANDLES)
    assert result["class_registry"] == list(component.class_registry)


def test_rebind_rejects_different_otherwise_valid_v2_component(tmp_path: Path) -> None:
    from cvsrffi.phase1_center_lowrank_prototype_bundle import (
        _pre_sign_content_root,
    )

    source = tmp_path / "other-component"
    source.mkdir()
    for name in (
        "int8_domain_class_center_lowrank_residual_radius_v2.npz",
        "manifest.json",
        "manifest.sha256",
    ):
        (source / name).write_bytes((REAL_D85_COMPONENT / name).read_bytes())
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation_config_sha256"] = "f" * 64
    manifest["pre_sign_content_root_sha256"] = _pre_sign_content_root(
        manifest, LEGACY_COMPONENT_NPZ_SHA
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (source / "manifest.sha256").write_text(
        f"{_sha256(manifest_path)}  manifest.json\n", encoding="ascii"
    )
    with pytest.raises(builder.Core90BundleError, match="known D85"):
        builder._rebind_legacy_d85_component(
            source,
            tmp_path / "rebound",
            current_class_handles=REAL_HANDLES,
        )


@pytest.mark.parametrize("mutation", ["reorder", "duplicate"])
def test_d19_binding_rejects_reordered_or_duplicate_handles(
    tmp_path: Path, mutation: str
) -> None:
    checkpoint_sha = "a" * 64
    payload = _d19_binding(checkpoint_sha)
    entries = payload["entries"]
    assert isinstance(entries, list)
    if mutation == "reorder":
        entries[0], entries[1] = entries[1], entries[0]
    else:
        entries[1]["registered_class_handle"] = entries[0]["registered_class_handle"]
    path = tmp_path / "binding.json"
    _write_json(path, payload)
    with pytest.raises(builder.Core90BundleError, match="order|unique"):
        builder._load_d19_class_binding(path, checkpoint_sha256=checkpoint_sha)


def test_prepare_rejects_component_binding_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, component_dir, binding_path, manifest = _prepare_fixture(
        tmp_path, monkeypatch
    )
    manifest["class_handle_binding_sha256"] = "f" * 64
    with pytest.raises(builder.Core90BundleError, match="binding"):
        builder.prepare(
            checkpoint=checkpoint,
            component_dir=component_dir,
            class_binding_source=binding_path,
            output_root=tmp_path / "output",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "cvs.phase1.center_lowrank_radius_component.v1", "schema"),
        ("component_state", "FORMAL_PHASE2_ELIGIBLE", "state"),
    ],
)
def test_prepare_rejects_wrong_component_schema_or_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    checkpoint, component_dir, binding_path, manifest = _prepare_fixture(
        tmp_path, monkeypatch
    )
    manifest[field] = value
    with pytest.raises(builder.Core90BundleError, match=message):
        builder.prepare(
            checkpoint=checkpoint,
            component_dir=component_dir,
            class_binding_source=binding_path,
            output_root=tmp_path / "output",
        )


def test_prepare_rejects_reused_output_before_input_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(
        builder,
        "_load_checkpoint",
        lambda *_args: (_ for _ in ()).throw(AssertionError("input accessed")),
    )
    with pytest.raises(FileExistsError, match="reuse"):
        builder.prepare(
            checkpoint=tmp_path / "missing.pth",
            component_dir=tmp_path / "missing-component",
            class_binding_source=tmp_path / "missing-binding.json",
            output_root=output,
        )


def test_finalize_loads_formal_bundle_before_writing_exact_sf_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "package"
    package_root.mkdir()
    manifest = {
        "schema": builder.BUNDLE_MANIFEST_SCHEMA,
        "checkpoint_lineage_sha256": "1" * 64,
        "runtime_sha256": "2" * 64,
        "component_pre_sign_content_root_sha256": "3" * 64,
        "class_handle_binding_sha256": "4" * 64,
        "parity_receipt_sha256": "5" * 64,
        "generation_lock_sha256": "6" * 64,
        "method_lock_sha256": "7" * 64,
        "generation_config_sha256": "8" * 64,
        "generation_code_sha256": "9" * 64,
        "outer_content_root_sha256": "a" * 64,
    }
    _write_json(package_root / builder.MANIFEST_RELATIVE_PATH, manifest)
    seal = tmp_path / "deployment.seal.json"
    envelope = tmp_path / "signature.json"
    _write_json(seal, {"outer_content_root_sha256": "a" * 64})
    _write_json(envelope, {"signature_ed25519_hex": "00"})
    binding = tmp_path / "phase1_bundle.json"
    observed: dict[str, object] = {}

    def fake_loader(package: Path, **kwargs: object) -> SimpleNamespace:
        assert not binding.exists()
        observed.update(package=package, kwargs=kwargs)
        return SimpleNamespace(
            formal_phase2_context={
                "formal_phase2_eligible": True,
                "checkpoint_lineage_sha256": "1" * 64,
            },
            method_lock={"method_id": CORE90},
        )

    monkeypatch.setattr(builder, "load_formal_adv3b02_deployment_bundle", fake_loader)
    result = builder.finalize(
        package_root=package_root,
        detached_seal=seal,
        signature_envelope=envelope,
        deployment_binding=binding,
    )
    payload = json.loads(binding.read_text(encoding="utf-8"))
    assert set(payload) == _PHASE1_BUNDLE_KEYS
    assert observed["package"] == package_root.resolve()
    assert observed["kwargs"] == {
        key: value for key, value in payload.items() if key != "package_root"
    }
    assert result["status"] == "FORMAL_PHASE2_ELIGIBLE"


def test_finalize_does_not_publish_mapping_when_formal_loader_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "package"
    package_root.mkdir()
    manifest = {
        "schema": builder.BUNDLE_MANIFEST_SCHEMA,
        **{
            field: "123456789a"[index - 1] * 64
            for index, field in enumerate(builder._MANIFEST_BINDING_FIELDS, 1)
        },
    }
    _write_json(package_root / builder.MANIFEST_RELATIVE_PATH, manifest)
    seal = tmp_path / "seal.json"
    envelope = tmp_path / "envelope.json"
    _write_json(seal, {})
    _write_json(envelope, {})
    binding = tmp_path / "binding.json"
    monkeypatch.setattr(
        builder,
        "load_formal_adv3b02_deployment_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad signature")),
    )
    with pytest.raises(ValueError, match="bad signature"):
        builder.finalize(
            package_root=package_root,
            detached_seal=seal,
            signature_envelope=envelope,
            deployment_binding=binding,
        )
    assert not binding.exists()
