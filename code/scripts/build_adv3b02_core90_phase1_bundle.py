#!/usr/bin/env python
"""Build and finalize the aggregate-only CORE90 Phase1 deployment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
for value in (str(REPO_ROOT), str(CODE_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from cvsrffi.phase1_adv3b02_deployment_bundle import (  # noqa: E402
    BUNDLE_MANIFEST_SCHEMA,
    CLASS_BINDING_SCHEMA,
    MANIFEST_RELATIVE_PATH,
    build_unsigned_adv3b02_deployment_bundle,
    canonical_json_bytes,
    class_handle_binding_sha256,
    load_formal_adv3b02_deployment_bundle,
    sha256_file,
)
from cvsrffi.phase1_center_lowrank_prototype_bundle import (  # noqa: E402
    PENDING_OUTER_JOINT_SEAL,
    SCHEMA as COMPONENT_SCHEMA,
    load_center_lowrank_component,
    validate_center_lowrank_component,
)
from scripts.build_full_ablation_phase1_deployment_bundle import (  # noqa: E402
    _input_len,
    _runtime_and_parity,
)


CORE90_CANDIDATE_ID = "ADV3B02_CORE90_SOFT_E200"
D19_CLASS_BINDING_SCHEMA = "cvs.phase2.d19_adv3b02_class_binding.v1"
PREPARE_RECEIPT_SCHEMA = "cvs.phase1.adv3b02_core90.deployment_prepare_receipt.v1"
FINALIZE_RECEIPT_SCHEMA = "cvs.phase1.adv3b02_core90.deployment_finalize_receipt.v1"
_MANIFEST_BINDING_FIELDS = (
    "checkpoint_lineage_sha256",
    "runtime_sha256",
    "component_pre_sign_content_root_sha256",
    "class_handle_binding_sha256",
    "parity_receipt_sha256",
    "generation_lock_sha256",
    "method_lock_sha256",
    "generation_config_sha256",
    "generation_code_sha256",
    "outer_content_root_sha256",
)


class Core90BundleError(ValueError):
    """Raised when the immutable CORE90 deployment bindings drift."""


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(dict(payload)) + b"\n"
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0),
        0o444,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing CORE90 deployment JSON")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Core90BundleError(f"{context} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Core90BundleError(f"{context} is unreadable") from exc
    if not isinstance(value, Mapping):
        raise Core90BundleError(f"{context} root must be a mapping")
    return dict(value)


def _sha256_regular(path: Path, *, context: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise Core90BundleError(f"{context} must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise Core90BundleError(f"invalid SHA256 for {field}")
    return result


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Core90BundleError("CORE90 checkpoint must be a regular file")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise Core90BundleError("CORE90 checkpoint root must be a mapping")
    if checkpoint.get("candidate_id") != CORE90_CANDIDATE_ID:
        raise Core90BundleError(
            f"checkpoint candidate must be exactly {CORE90_CANDIDATE_ID}"
        )
    return checkpoint


def _load_d19_class_binding(
    path: str | Path, *, checkpoint_sha256: str
) -> tuple[str, ...]:
    payload = _load_json(Path(path), context="D19 class binding")
    if (
        set(payload) != {"schema", "checkpoint_sha256", "entries", "evidence"}
        or payload.get("schema") != D19_CLASS_BINDING_SCHEMA
        or not isinstance(payload.get("entries"), list)
    ):
        raise Core90BundleError("D19 class binding schema drift")
    if str(payload["checkpoint_sha256"]).lower() != checkpoint_sha256:
        raise Core90BundleError("D19 class binding checkpoint mismatch")
    entries = payload["entries"]
    if len(entries) != 6:
        raise Core90BundleError("D19 class binding must contain exactly six classes")
    handles: list[str] = []
    txs: list[str] = []
    for index, entry in enumerate(entries):
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {"class_index", "phase1_tx", "registered_class_handle"}
            or type(entry.get("class_index")) is not int
            or entry.get("class_index") != index
            or not isinstance(entry.get("phase1_tx"), str)
            or not str(entry.get("phase1_tx")).strip()
            or not isinstance(entry.get("registered_class_handle"), str)
            or not str(entry.get("registered_class_handle")).strip()
        ):
            raise Core90BundleError(
                "D19 class binding must preserve contiguous fixed class order"
            )
        txs.append(str(entry["phase1_tx"]))
        handles.append(str(entry["registered_class_handle"]))
    if len(set(txs)) != 6 or len(set(handles)) != 6:
        raise Core90BundleError("D19 class binding TXs and handles must be unique")
    return tuple(handles)


def _load_component_manifest(
    component_dir: Path,
    *,
    checkpoint_sha256: str,
    class_binding_sha256: str,
    class_handles: tuple[str, ...],
) -> dict[str, Any]:
    try:
        manifest = validate_center_lowrank_component(
            component_dir,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_class_handle_binding_sha256=class_binding_sha256,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise Core90BundleError("CORE90 component strict validation failed") from exc
    if manifest.get("schema") != COMPONENT_SCHEMA:
        raise Core90BundleError("CORE90 component schema drift")
    if manifest.get("component_state") != PENDING_OUTER_JOINT_SEAL:
        raise Core90BundleError("CORE90 component state drift")
    if manifest.get("checkpoint_sha256") != checkpoint_sha256:
        raise Core90BundleError("CORE90 component checkpoint binding mismatch")
    if manifest.get("class_handle_binding_sha256") != class_binding_sha256:
        raise Core90BundleError("CORE90 component class binding mismatch")
    component_root = _require_sha256(
        manifest.get("pre_sign_content_root_sha256"),
        field="component_pre_sign_content_root_sha256",
    )
    try:
        component = load_center_lowrank_component(
            component_dir,
            expected_checkpoint_sha256=checkpoint_sha256,
            expected_class_handle_binding_sha256=class_binding_sha256,
            expected_pre_sign_content_root_sha256=component_root,
            allow_pending_outer_joint_seal_development=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise Core90BundleError("CORE90 component strict loader failed") from exc
    if tuple(component.class_registry) != class_handles:
        raise Core90BundleError("CORE90 component ordered class binding mismatch")
    return dict(manifest)


def prepare(
    *,
    checkpoint: str | Path,
    component_dir: str | Path,
    class_binding_source: str | Path,
    output_root: str | Path,
    device: str = "cuda:0",
    input_len: int = 0,
    parity_seed: int = 7281105,
    parity_rows: int = 8,
) -> dict[str, Any]:
    """Prepare an unsigned formal package from immutable aggregate-only inputs."""

    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError("refusing to reuse CORE90 deployment output root")
    checkpoint_path = Path(checkpoint).resolve()
    component_path = Path(component_dir).resolve()
    binding_path = Path(class_binding_source).resolve()
    checkpoint_payload = _load_checkpoint(checkpoint_path)
    checkpoint_sha = _sha256_regular(checkpoint_path, context="CORE90 checkpoint")
    handles = _load_d19_class_binding(
        binding_path, checkpoint_sha256=checkpoint_sha
    )
    binding_sha = class_handle_binding_sha256(handles)
    component = _load_component_manifest(
        component_path,
        checkpoint_sha256=checkpoint_sha,
        class_binding_sha256=binding_sha,
        class_handles=handles,
    )
    resolved_input_len = _input_len(checkpoint_payload, int(input_len))

    work = output / "work"
    locks = work / "locks"
    runtime_path = work / "runtime" / "adv3b02_core90.torchscript.pt"
    runtime_sha, parity, checkpoint_audit = _runtime_and_parity(
        checkpoint_payload,
        input_len=resolved_input_len,
        device=torch.device(str(device)),
        runtime_path=runtime_path,
        parity_seed=int(parity_seed),
        parity_rows=int(parity_rows),
    )
    class_lock_path = locks / "class_binding.json"
    _write_json_new(
        class_lock_path,
        {
            "schema": CLASS_BINDING_SCHEMA,
            "checkpoint_lineage_sha256": checkpoint_sha,
            "class_id_to_handle": [
                {"class_index": index, "class_handle": handle}
                for index, handle in enumerate(handles)
            ],
            "class_handle_binding_sha256": binding_sha,
        },
    )
    parity["checkpoint_lineage_sha256"] = checkpoint_sha
    parity["runtime_sha256"] = runtime_sha
    parity_path = locks / "runtime_checkpoint_parity_receipt.json"
    _write_json_new(parity_path, parity)
    generation_path = locks / "generation_lock.json"
    _write_json_new(
        generation_path,
        {
            "schema": "cvs.phase1.prototype_generation_lock.v1",
            "checkpoint_lineage_sha256": checkpoint_sha,
            "component_pre_sign_content_root_sha256": component[
                "pre_sign_content_root_sha256"
            ],
            "class_handle_binding_sha256": binding_sha,
            "generation_config_sha256": component["generation_config_sha256"],
            "generation_code_sha256": component["generation_code_sha256"],
            "phase1_stream_sha256": component["phase1_stream_sha256"],
            "radius_generation_proof_sha256": component[
                "radius_generation_proof_sha256"
            ],
        },
    )
    method_path = locks / "method_lock.json"
    _write_json_new(
        method_path,
        {
            "schema": "cvs.phase1.adv3b02_method_lock.v1",
            "method_id": CORE90_CANDIDATE_ID,
            "checkpoint_lineage_sha256": checkpoint_sha,
            "runtime_sha256": runtime_sha,
            "component_pre_sign_content_root_sha256": component[
                "pre_sign_content_root_sha256"
            ],
            "class_handle_binding_sha256": binding_sha,
            "parity_receipt_sha256": sha256_file(parity_path),
            "generation_lock_sha256": sha256_file(generation_path),
            "generation_config_sha256": component["generation_config_sha256"],
            "generation_code_sha256": component["generation_code_sha256"],
        },
    )
    package_root = output / "package"
    seal_path = output / "external" / "deployment.seal.json"
    request_path = output / "external" / "signing_request.json"
    bundle = build_unsigned_adv3b02_deployment_bundle(
        package_root,
        torchscript_runtime_path=runtime_path,
        component_dir=component_path,
        class_binding_path=class_lock_path,
        parity_receipt_path=parity_path,
        generation_lock_path=generation_path,
        method_lock_path=method_path,
        detached_seal_path=seal_path,
        signing_request_path=request_path,
    )
    receipt = {
        "schema": PREPARE_RECEIPT_SCHEMA,
        "status": "AWAITING_EXTERNAL_SIGNATURE",
        "candidate_id": CORE90_CANDIDATE_ID,
        "package_root": str(package_root),
        "detached_seal_path": str(seal_path),
        "signing_request_path": str(request_path),
        "checkpoint_lineage_sha256": checkpoint_sha,
        "input_len": resolved_input_len,
        "checkpoint_load_audit": checkpoint_audit,
        **{
            key: value
            for key, value in bundle.items()
            if key
            not in {"manifest_path", "detached_seal_path", "signing_request_path"}
        },
    }
    _write_json_new(output / "prepare_receipt.json", receipt)
    return receipt


def finalize(
    *,
    package_root: str | Path,
    detached_seal: str | Path,
    signature_envelope: str | Path,
    deployment_binding: str | Path,
) -> dict[str, Any]:
    """Run the formal loader, then publish exactly SF-TAPFT's 15-key mapping."""

    binding_path = Path(deployment_binding).resolve()
    if binding_path.exists():
        raise FileExistsError("refusing to reuse CORE90 deployment binding output")
    package = Path(package_root).resolve()
    seal = Path(detached_seal).resolve()
    envelope = Path(signature_envelope).resolve()
    manifest = _load_json(
        package / MANIFEST_RELATIVE_PATH, context="CORE90 deployment manifest"
    )
    if manifest.get("schema") != BUNDLE_MANIFEST_SCHEMA:
        raise Core90BundleError("CORE90 deployment manifest schema drift")
    values = {
        field: _require_sha256(manifest.get(field), field=field)
        for field in _MANIFEST_BINDING_FIELDS
    }
    _load_json(seal, context="CORE90 detached seal")
    _load_json(envelope, context="CORE90 signature envelope")
    mapping = {
        "package_root": str(package),
        "detached_seal_path": str(seal),
        "expected_detached_seal_sha256": _sha256_regular(
            seal, context="CORE90 detached seal"
        ),
        "signature_envelope_path": str(envelope),
        "expected_signature_envelope_sha256": _sha256_regular(
            envelope, context="CORE90 signature envelope"
        ),
        **{f"expected_{field}": value for field, value in values.items()},
    }
    verified = load_formal_adv3b02_deployment_bundle(
        package,
        **{key: value for key, value in mapping.items() if key != "package_root"},
    )
    context = verified.formal_phase2_context
    if context.get("formal_phase2_eligible") is not True:
        raise Core90BundleError("CORE90 bundle did not become formally Phase2 eligible")
    if context.get("checkpoint_lineage_sha256") != values[
        "checkpoint_lineage_sha256"
    ]:
        raise Core90BundleError("formal loader CORE90 checkpoint lineage drift")
    if verified.method_lock.get("method_id") != CORE90_CANDIDATE_ID:
        raise Core90BundleError("formal loader method identity is not CORE90")
    _write_json_new(binding_path, mapping)
    return {
        "schema": FINALIZE_RECEIPT_SCHEMA,
        "status": "FORMAL_PHASE2_ELIGIBLE",
        "deployment_binding": str(binding_path),
        "deployment_binding_sha256": sha256_file(binding_path),
        "checkpoint_lineage_sha256": values["checkpoint_lineage_sha256"],
        "outer_content_root_sha256": values["outer_content_root_sha256"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the aggregate-only CORE90 Phase1 deployment chain."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--checkpoint", required=True)
    prepare_parser.add_argument("--component-dir", required=True)
    prepare_parser.add_argument("--class-binding-source", required=True)
    prepare_parser.add_argument("--output-root", required=True)
    prepare_parser.add_argument("--device", default="cuda:0")
    prepare_parser.add_argument("--input-len", type=int, default=0)
    prepare_parser.add_argument("--parity-seed", type=int, default=7281105)
    prepare_parser.add_argument("--parity-rows", type=int, default=8)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--package-root", required=True)
    finalize_parser.add_argument("--detached-seal", required=True)
    finalize_parser.add_argument("--signature-envelope", required=True)
    finalize_parser.add_argument("--deployment-binding", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    values = vars(args).copy()
    command = values.pop("command")
    result = prepare(**values) if command == "prepare" else finalize(**values)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
