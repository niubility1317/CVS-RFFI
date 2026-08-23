"""Prepare whitelist-only row inputs, smoke adaptation, and predict a row.

Ground-side ``prepare`` binds one support package to an existing
``VALIDATED_ONCE`` row and emits only received IQ plus legal support labels,
and frozen checkpoint class anchors plus class mapping.  ``smoke`` consumes
those exhaustive Phase2 inputs without any query file.  ``run-row`` finishes
support adaptation and freezes the model before it opens a query-only IQ file
and performs independent per-sample prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"
for _path in (str(CODE_ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cvsrffi.stage2_structured_late_block_adaptation import (  # noqa: E402
    Phase2Context,
    StructuredLateBlockConfig,
    adapt_on_target_support,
    predict_query_read_only,
)
from paper_reproduction.cvs_aligned.adv3b02_supervised_da_runner import (  # noqa: E402
    _exact_adv3b02,
)


SUPPORT_KEYS = frozenset({"received_iq", "support_class_indices"})
QUERY_KEYS = frozenset({"received_iq"})
PROTOTYPE_KEYS = frozenset(
    {
        "prototypes",
        "class_ids",
        "checkpoint_sha256",
        "feature_key",
        "decision_scale",
    }
)
ROW_BINDING_KEYS = frozenset(
    {
        "protocol_schema",
        "phase2_data_status",
        "capsule_id",
        "split_id",
        "receiver",
        "method_seed",
        "support_seed",
        "query_seed",
        "new_class_draw_seed",
        "k_shot",
        "scenario",
    }
)
CONTEXT_KEYS = frozenset(
    {
        "protocol_schema",
        "phase2_data_status",
        "capsule_id",
        "split_id",
        "receiver",
        "method_seed",
        "k_shot",
        "scenario",
        "candidate_id",
        "steps",
        "learning_rate",
        "checkpoint_sha256",
        "support_input_count",
    }
)
FORBIDDEN_SUPPORT_MEMBER_TOKENS = (
    "source",
    "clean",
    "query",
    "truth",
    "role",
    "quota",
)
FORBIDDEN_QUERY_MEMBER_TOKENS = (
    "source",
    "clean",
    "truth",
    "label",
    "role",
    "quota",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _ensure_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    if frozenset(value) != expected:
        raise ValueError(f"{name} is not the exhaustive preregistered schema")


def _read_row_binding(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    _require_exact_keys(value, ROW_BINDING_KEYS, "row binding")
    if (
        value["protocol_schema"] != "p2_min_v1"
        or value["phase2_data_status"] != "VALIDATED_ONCE"
        or not str(value["capsule_id"]).strip()
        or not str(value["split_id"]).strip()
        or not str(value["receiver"]).strip()
        or int(value["k_shot"]) < 1
        or not str(value["scenario"]).startswith("leo_")
        or any(
            int(value[name]) < 0
            for name in (
                "method_seed",
                "support_seed",
                "query_seed",
                "new_class_draw_seed",
            )
        )
    ):
        raise ValueError("row binding does not identify a valid frozen Phase2 row")
    return value


def _read_context(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    _require_exact_keys(value, CONTEXT_KEYS, "Phase2 context")
    Phase2Context(
        protocol_schema=str(value["protocol_schema"]),
        phase2_data_status=str(value["phase2_data_status"]),
        capsule_id=str(value["capsule_id"]),
        split_id=str(value["split_id"]),
    ).validate()
    if (
        int(value["k_shot"]) < 1
        or int(value["support_input_count"]) < 1
        or int(value["steps"]) < 1
        or int(value["steps"]) > 40
    ):
        raise ValueError("Phase2 context resource/row bound is invalid")
    return value


def _ordered_class_ids(binding: dict[str, Any], checkpoint_sha256: str) -> tuple[str, ...]:
    if str(binding.get("checkpoint_sha256")) != checkpoint_sha256:
        raise ValueError("class mapping is not bound to the frozen checkpoint")
    entries = sorted(binding.get("entries") or (), key=lambda row: int(row["class_index"]))
    indices = [int(row["class_index"]) for row in entries]
    if indices != list(range(len(entries))):
        raise ValueError("class mapping indices are not contiguous")
    class_ids = tuple(str(row["registered_class_handle"]) for row in entries)
    if len(class_ids) < 2 or len(set(class_ids)) != len(class_ids):
        raise ValueError("class mapping is invalid")
    return class_ids


def _package_member(
    package_manifest: dict[str, Any],
    *,
    role: str,
    scenario: str,
    filename: str,
) -> dict[str, Any]:
    matches = [
        member
        for member in package_manifest.get("members") or ()
        if str(member.get("artifact_role")) == f"{role}:{scenario}"
        and str(member.get("scenario")) == scenario
        and str(member.get("relative_path")) == filename
    ]
    if len(matches) != 1:
        raise ValueError(f"{role} file is not bound to package scenario")
    return dict(matches[0])


def _validate_package_row(
    package_manifest: dict[str, Any],
    row: dict[str, Any],
    class_ids: tuple[str, ...],
) -> None:
    package_classes = tuple(
        str(entry["class_handle"])
        for entry in sorted(
            package_manifest.get("registered_classes") or (),
            key=lambda entry: int(entry["class_index"]),
        )
    )
    if (
        str(package_manifest.get("stage")) != "stage2b"
        or str(package_manifest.get("receiver")) != str(row["receiver"])
        or int(package_manifest.get("seed", -1)) != int(row["method_seed"])
        or int(package_manifest.get("support_pool_max_k", -1)) < int(row["k_shot"])
        or str(row["scenario"])
        not in tuple(package_manifest.get("target_channel_scenarios") or ())
        or package_classes != class_ids
        or any(
            bool(package_manifest.get(name, False))
            for name in (
                "phase2_source_sample_access",
                "phase2_source_cache_access",
                "phase2_source_derived_signal_access",
                "phase2_source_replay",
                "phase2_clean_dataset_reachable",
                "phase2_query_role_oracle_access",
                "phase2_query_true_batch_class_count_access",
                "phase2_query_class_quota_access",
            )
        )
    ):
        raise ValueError("package manifest does not match the frozen row whitelist")


def _validate_authoritative_row_manifest(
    validated_manifest: dict[str, Any],
    package_manifest: dict[str, Any],
    row: dict[str, Any],
) -> None:
    if (
        str(validated_manifest.get("phase2_data_status")) != "VALIDATED_ONCE"
        or str(validated_manifest.get("capsule_id")) != str(row["capsule_id"])
        or str(validated_manifest.get("split_id")) != str(row["split_id"])
        or str(validated_manifest.get("receiver")) != str(row["receiver"])
        or int(validated_manifest.get("method_seed", -1)) != int(row["method_seed"])
        or int(validated_manifest.get("support_seed", -1)) != int(row["support_seed"])
        or int(validated_manifest.get("query_seed", -1)) != int(row["query_seed"])
        or int(validated_manifest.get("k_shot", -1)) != int(row["k_shot"])
        or str(validated_manifest.get("stage_scope")) != "stage2b"
        or str(row["scenario"])
        not in tuple(validated_manifest.get("scenarios") or ())
        or str(validated_manifest.get("package_root_sha256"))
        != str(package_manifest.get("package_root_sha256"))
        or any(
            bool(validated_manifest.get(name, False))
            for name in (
                "phase2_source_sample_access",
                "phase2_source_cache_access",
                "phase2_source_derived_signal_access",
                "phase2_source_replay",
                "phase2_clean_dataset_reachable",
                "phase2_query_role_oracle_access",
                "phase2_query_true_batch_class_count_access",
                "phase2_query_class_quota_access",
                "query_truth_present",
                "query_role_present",
            )
        )
    ):
        raise ValueError(
            "authoritative VALIDATED_ONCE manifest does not bind the support row"
        )


def _load_support_only(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as support:
        if frozenset(support.files) != SUPPORT_KEYS:
            raise ValueError("Phase2 support input is not the exhaustive whitelist")
        received_iq = np.asarray(support["received_iq"], dtype=np.float32)
        labels = np.asarray(support["support_class_indices"], dtype=np.int64)
    if (
        received_iq.ndim != 3
        or received_iq.shape[1] != 2
        or len(received_iq) != len(labels)
        or not np.isfinite(received_iq).all()
    ):
        raise ValueError("support-only received IQ/label alignment is invalid")
    return received_iq, labels


def _load_prototypes(path: Path) -> tuple[np.ndarray, tuple[str, ...], str]:
    with np.load(path, allow_pickle=False) as bundle:
        if frozenset(bundle.files) != PROTOTYPE_KEYS:
            raise ValueError("Phase2 prototype input is not the exhaustive whitelist")
        prototypes = np.asarray(bundle["prototypes"], dtype=np.float32)
        class_ids = tuple(np.asarray(bundle["class_ids"]).astype(str).tolist())
        checkpoint_sha256 = str(np.asarray(bundle["checkpoint_sha256"]).item())
    return prototypes, class_ids, checkpoint_sha256


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).resolve()
    binding_path = Path(args.class_binding).resolve()
    source_support_path = Path(args.support_package).resolve()
    package_manifest = _read_json(Path(args.package_manifest).resolve())
    row = _read_row_binding(Path(args.row_binding).resolve())
    validated_manifest = _read_json(Path(args.validated_row_manifest).resolve())
    output_dir = Path(args.output_dir).resolve()

    model, checkpoint_info = _exact_adv3b02(
        checkpoint_path, device=torch.device("cpu")
    )
    checkpoint_sha256 = str(checkpoint_info["checkpoint_sha256"])
    class_ids = _ordered_class_ids(_read_json(binding_path), checkpoint_sha256)
    _validate_package_row(package_manifest, row, class_ids)
    _validate_authoritative_row_manifest(validated_manifest, package_manifest, row)
    _package_member(
        package_manifest,
        role="support",
        scenario=str(row["scenario"]),
        filename=source_support_path.name,
    )

    with np.load(source_support_path, allow_pickle=False) as source_support:
        member_names = tuple(source_support.files)
        if any(
            token in name.lower()
            for name in member_names
            for token in FORBIDDEN_SUPPORT_MEMBER_TOKENS
        ):
            raise ValueError("support package contains a forbidden member surface")
        required = {
            "support_pool_leo_weak_iq",
            "support_pool_class_indices",
            "support_pool_rank_within_class",
            "support_pool_tokens",
        }
        if not required.issubset(member_names):
            raise ValueError("support package is missing row-binding fields")
        all_iq = np.asarray(
            source_support["support_pool_leo_weak_iq"], dtype=np.float32
        )
        all_labels = np.asarray(
            source_support["support_pool_class_indices"], dtype=np.int64
        )
        all_ranks = np.asarray(
            source_support["support_pool_rank_within_class"], dtype=np.int64
        )
        all_tokens = np.asarray(source_support["support_pool_tokens"]).astype(str)
    if not (len(all_iq) == len(all_labels) == len(all_ranks) == len(all_tokens)):
        raise ValueError("support package physical-row alignment is invalid")
    selected = all_ranks < int(row["k_shot"])
    received_iq = all_iq[selected]
    support_class_indices = all_labels[selected]
    selected_ranks = all_ranks[selected]
    selected_tokens = all_tokens[selected]
    expected_count = len(class_ids) * int(row["k_shot"])
    if (
        received_iq.shape[0] != expected_count
        or received_iq.ndim != 3
        or received_iq.shape[1] != 2
        or not np.isfinite(received_iq).all()
        or len(set(selected_tokens.tolist())) != expected_count
        or any(
            set(selected_ranks[support_class_indices == index].tolist())
            != set(range(int(row["k_shot"])))
            for index in range(len(class_ids))
        )
    ):
        raise ValueError("support package does not match row K-shot selection")

    weight = model.id_backbone.cls_head.head.weight.detach().float().cpu()
    prototypes = F.normalize(weight, dim=1, eps=1.0e-4).numpy().astype(np.float32)
    if prototypes.shape[0] != len(class_ids):
        raise ValueError("checkpoint head and class mapping disagree")
    support_output = output_dir / "support_only.npz"
    prototype_output = output_dir / "frozen_class_prototypes.npz"
    context_output = output_dir / "context.json"
    for path in (support_output, prototype_output, context_output):
        _ensure_absent(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        support_output,
        received_iq=received_iq,
        support_class_indices=support_class_indices,
    )
    np.savez_compressed(
        prototype_output,
        prototypes=prototypes,
        class_ids=np.asarray(class_ids),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        feature_key=np.asarray("feat_joint"),
        decision_scale=np.asarray(
            float(model.id_backbone.cls_head.head.s), dtype=np.float32
        ),
    )
    context = {
        "protocol_schema": str(row["protocol_schema"]),
        "phase2_data_status": str(row["phase2_data_status"]),
        "capsule_id": str(row["capsule_id"]),
        "split_id": str(row["split_id"]),
        "receiver": str(row["receiver"]),
        "method_seed": int(row["method_seed"]),
        "k_shot": int(row["k_shot"]),
        "scenario": str(row["scenario"]),
        "candidate_id": str(args.candidate_id),
        "steps": int(args.steps),
        "learning_rate": float(args.learning_rate),
        "checkpoint_sha256": checkpoint_sha256,
        "support_input_count": int(len(received_iq)),
    }
    _require_exact_keys(context, CONTEXT_KEYS, "emitted Phase2 context")
    _write_json(context_output, context)
    return {
        "status": "PREPARED",
        "checkpoint_load_strict": bool(checkpoint_info["checkpoint_load_strict"]),
        "support_shape": list(received_iq.shape),
        "prototype_shape": list(prototypes.shape),
        "class_count": len(class_ids),
        "support_output": str(support_output),
        "prototype_output": str(prototype_output),
        "context_output": str(context_output),
        "query_input_count": 0,
    }


def prepare_query(args: argparse.Namespace) -> dict[str, Any]:
    query_package_path = Path(args.query_package).resolve()
    package_manifest = _read_json(Path(args.package_manifest).resolve())
    row = _read_row_binding(Path(args.row_binding).resolve())
    _package_member(
        package_manifest,
        role="query",
        scenario=str(row["scenario"]),
        filename=query_package_path.name,
    )
    if (
        str(package_manifest.get("receiver")) != str(row["receiver"])
        or int(package_manifest.get("seed", -1)) != int(row["method_seed"])
    ):
        raise ValueError("query package is not bound to the frozen row")
    with np.load(query_package_path, allow_pickle=False) as source_query:
        member_names = tuple(source_query.files)
        if any(
            token in name.lower()
            for name in member_names
            for token in FORBIDDEN_QUERY_MEMBER_TOKENS
        ):
            raise ValueError("query package contains forbidden feedback")
        if "query_leo_weak_iq" not in member_names:
            raise ValueError("query package is missing received IQ")
        received_iq = np.asarray(source_query["query_leo_weak_iq"], dtype=np.float32)
    if (
        received_iq.ndim != 3
        or received_iq.shape[0] < 1
        or received_iq.shape[1] != 2
        or not np.isfinite(received_iq).all()
    ):
        raise ValueError("query received IQ is invalid")
    output = Path(args.output).resolve()
    _ensure_absent(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, received_iq=received_iq)
    return {"status": "PREPARED", "query_shape": list(received_iq.shape), "output": str(output)}


def _adapt_from_whitelist(
    args: argparse.Namespace,
) -> tuple[
    torch.nn.Module,
    dict[str, Any],
    dict[str, Any],
    Any,
    tuple[str, ...],
]:
    context_value = _read_context(Path(args.context).resolve())
    received_iq, support_class_indices = _load_support_only(
        Path(args.support_only).resolve()
    )
    prototypes, class_ids, prototype_checkpoint_sha256 = _load_prototypes(
        Path(args.frozen_prototypes).resolve()
    )
    if (
        len(received_iq) != int(context_value["support_input_count"])
        or np.any(support_class_indices < 0)
        or np.any(support_class_indices >= len(class_ids))
    ):
        raise ValueError("support-only input does not match preregistered context")
    model, checkpoint_info = _exact_adv3b02(
        Path(args.checkpoint).resolve(), device=torch.device(args.device)
    )
    if (
        str(checkpoint_info["checkpoint_sha256"])
        != prototype_checkpoint_sha256
        or str(context_value["checkpoint_sha256"])
        != prototype_checkpoint_sha256
    ):
        raise ValueError("checkpoint/prototype/context binding mismatch")
    support_labels = tuple(class_ids[int(index)] for index in support_class_indices)
    context = Phase2Context(
        protocol_schema=str(context_value["protocol_schema"]),
        phase2_data_status=str(context_value["phase2_data_status"]),
        capsule_id=str(context_value["capsule_id"]),
        split_id=str(context_value["split_id"]),
    )
    config = StructuredLateBlockConfig(
        candidate_id=str(context_value["candidate_id"]),
        steps=int(context_value["steps"]),
        learning_rate=float(context_value["learning_rate"]),
    )
    audit = adapt_on_target_support(
        model,
        received_iq,
        support_labels,
        prototypes,
        class_ids,
        context=context,
        config=config,
        device=args.device,
    )
    return model, checkpoint_info, context_value, audit, class_ids


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    _, checkpoint_info, context_value, audit, _ = _adapt_from_whitelist(args)
    result = {
        "status": "PASS",
        "checkpoint_load_strict": bool(checkpoint_info["checkpoint_load_strict"]),
        "checkpoint_load_audit": checkpoint_info["checkpoint_load_audit"],
        "support_input_count": int(context_value["support_input_count"]),
        "source_input_count": 0,
        "query_input_count": 0,
        "query_loaded": False,
        "audit": asdict(audit),
    }
    _write_json(Path(args.output).resolve(), result)
    return result


def run_row(args: argparse.Namespace) -> dict[str, Any]:
    model, checkpoint_info, context_value, audit, class_ids = _adapt_from_whitelist(args)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("adaptation did not freeze model before query open")

    query_path = Path(args.query_only).resolve()
    with np.load(query_path, allow_pickle=False) as query_bundle:
        if frozenset(query_bundle.files) != QUERY_KEYS:
            raise ValueError("Phase2 query input is not the exhaustive IQ-only whitelist")
        query_received_iq = np.asarray(query_bundle["received_iq"], dtype=np.float32)
    prototypes, prototype_class_ids, _ = _load_prototypes(
        Path(args.frozen_prototypes).resolve()
    )
    if prototype_class_ids != class_ids:
        raise ValueError("prototype class mapping drift after adaptation")
    prediction = predict_query_read_only(
        model,
        query_received_iq,
        prototypes,
        class_ids,
        context=Phase2Context(
            protocol_schema=str(context_value["protocol_schema"]),
            phase2_data_status=str(context_value["phase2_data_status"]),
            capsule_id=str(context_value["capsule_id"]),
            split_id=str(context_value["split_id"]),
        ),
    )
    rows = [
        {
            "sample_index": index,
            "predicted_class_id": prediction.predicted_class_ids[index],
            "scores": prediction.scores[index].tolist(),
        }
        for index in range(len(prediction.predicted_class_ids))
    ]
    result = {
        "status": "PREDICTIONS_COMPLETE",
        "checkpoint_load_strict": bool(checkpoint_info["checkpoint_load_strict"]),
        "source_input_count": 0,
        "query_input_count": len(rows),
        "query_truth_loaded": False,
        "query_role_loaded": False,
        "query_batch_state_updated": False,
        "audit": asdict(audit),
        "predictions": rows,
    }
    _write_json(Path(args.output).resolve(), result)
    return result


def _add_adaptation_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--support-only", required=True)
    parser.add_argument("--frozen-prototypes", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--checkpoint", required=True)
    prepare_parser.add_argument("--class-binding", required=True)
    prepare_parser.add_argument("--support-package", required=True)
    prepare_parser.add_argument("--package-manifest", required=True)
    prepare_parser.add_argument("--validated-row-manifest", required=True)
    prepare_parser.add_argument("--row-binding", required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--candidate-id", default="TIME_FUSION_V1")
    prepare_parser.add_argument("--steps", type=int, default=1)
    prepare_parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    prepare_parser.set_defaults(handler=prepare)

    query_parser = subparsers.add_parser("prepare-query")
    query_parser.add_argument("--query-package", required=True)
    query_parser.add_argument("--package-manifest", required=True)
    query_parser.add_argument("--row-binding", required=True)
    query_parser.add_argument("--output", required=True)
    query_parser.set_defaults(handler=prepare_query)

    smoke_parser = subparsers.add_parser("smoke")
    _add_adaptation_inputs(smoke_parser)
    smoke_parser.set_defaults(handler=smoke)

    row_parser = subparsers.add_parser("run-row")
    _add_adaptation_inputs(row_parser)
    row_parser.add_argument("--query-only", required=True)
    row_parser.set_defaults(handler=run_row)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
