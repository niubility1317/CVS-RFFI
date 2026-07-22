#!/usr/bin/env python3
"""Audit D7b from sealed LEO_weak enrollment support without opening query."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO = Path(r"E:\type10-7\github_publish\CVS-RFFI-repo")
CODE = REPO / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from cvsrffi.somph_diagnostic_bundle_loader import (  # noqa: E402
    load_verified_somph_predictor_bundle,
)
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS  # noqa: E402
from cvsrffi.stage2_diag_cosine_exploration import (  # noqa: E402
    forward_zid160,
    registered_feature,
)
from cvsrffi.stage2_local_contrastive_boundary import (  # noqa: E402
    LocalBoundaryHead,
    extend_local_contrastive_boundary,
    fit_local_contrastive_boundary,
    score_local_contrastive_boundary,
)
from cvsrffi.stage2_predictor_runtime import (  # noqa: E402
    load_torchscript_backbone_same_fd,
)


ROW = Path(
    r"E:\type10-7\automation_reports\CV-SincNet"
    r"\d4a_single_observation_smoke_20260717_010128\dev_k10_new5_r2"
)
CACHE_ROOT = Path(
    r"E:\type10-7\local_artifacts\d4a_singleobs_cache_rx20_1_seed713101"
)
OUTPUT = Path(
    r"E:\type10-7\automation_reports\CV-SincNet"
    r"\d7b_local_contrastive_boundary_20260717"
    r"\support_only_current_row_audit"
)
FOCUS_TX = {"20-19", "1-18"}
EPS = 1.0e-8


class D7bSupportAuditError(ValueError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def _write_json_new(path: Path, payload: Any) -> str:
    raw = _canonical(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    _readonly(path)
    return hashlib.sha256(raw).hexdigest()


def _write_text_new(path: Path, text: str) -> str:
    raw = text.encode("utf-8")
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    _readonly(path)
    return hashlib.sha256(raw).hexdigest()


def _write_head_new(path: Path, head: LocalBoundaryHead) -> str:
    with path.open("xb") as handle:
        np.savez(
            handle,
            schema=np.asarray(head.schema),
            classes=head.classes.astype(str),
            prototypes=head.prototypes.astype(np.float16),
            rival_indices=head.rival_indices.astype(np.uint16),
            beta=head.beta.astype(np.float16),
            old_class_count=np.asarray(head.old_class_count, dtype=np.int64),
        )
        handle.flush()
        os.fsync(handle.fileno())
    _readonly(path)
    return _sha256_file(path)


def _member(manifest: dict[str, Any], kind: str) -> dict[str, Any]:
    rows = [dict(row) for row in manifest["members"] if row.get("kind") == kind]
    if len(rows) != 1:
        raise D7bSupportAuditError(f"member drift: {kind}")
    return rows[0]


def _load_enrollment(state: str):
    root = ROW / "offline" / "predictor" / state / "enrollment_only"
    seal = ROW / "offline" / "seals" / f"{state}_enrollment.seal.json"
    payloads, manifest, audit = load_verified_somph_predictor_bundle(
        root,
        detached_seal_path=seal,
        expected_seal_sha256=_sha256_file(seal),
    )
    if (
        manifest["profile"] != "enrollment_only"
        or manifest["registration_state"] != state
        or int(manifest["k_shot"]) != 10
        or manifest["phase2_sample_view_policy"]
        != "leo_weak_only_no_clean_access"
        or manifest["phase2_physical_sample_observation_policy"]
        != "single_leo_weak_observation_per_physical_sample"
        or bool(manifest["phase2_additional_leo_channel_state_generation"])
        or not bool(
            manifest[
                "phase2_post_reception_view_from_fixed_received_iq_only"
            ]
        )
        or bool(
            manifest[
                "phase2_post_reception_view_counts_as_additional_physical_sample"
            ]
        )
        or bool(manifest["clean_sample_access"])
        or bool(manifest["clean_derived_signal_access"])
    ):
        raise D7bSupportAuditError("enrollment package protocol drift")
    return root, seal, payloads, manifest, audit


def _extract_support(
    model: torch.nn.Module,
    device: torch.device,
    payloads: dict[str, dict[str, np.ndarray]],
    manifest: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    class_handles = np.asarray(
        [row["class_handle"] for row in manifest["registered_classes"]]
    )
    output: dict[str, dict[str, np.ndarray]] = {}
    all_tokens: set[str] = set()
    all_hashes: set[str] = set()
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        payload = payloads[scenario]
        ranks = np.asarray(payload["support_rank_within_class"], dtype=np.int64)
        indices = np.asarray(payload["support_class_indices"], dtype=np.int64)
        mask = ranks < int(manifest["k_shot"])
        iq = np.asarray(payload["support_leo_weak_iq"], dtype=np.float32)[mask]
        labels = class_handles[indices[mask]].astype(str)
        tokens = np.asarray(payload["support_tokens"]).astype(str)[mask]
        hashes = np.asarray(
            payload["support_post_channel_iq_sha256"]
        ).astype(str)[mask]
        if (
            len(set(tokens.tolist())) != len(tokens)
            or len(set(hashes.tolist())) != len(hashes)
            or all_tokens.intersection(tokens.tolist())
            or all_hashes.intersection(hashes.tolist())
        ):
            raise D7bSupportAuditError(
                "support physical lineage reused within/across scenarios"
            )
        all_tokens.update(tokens.tolist())
        all_hashes.update(hashes.tolist())
        zid = forward_zid160(model, iq, device=device, batch_size=64)
        features = registered_feature(iq, zid)
        output[scenario] = {
            "features": features,
            "labels": labels,
            "tokens": tokens,
            "hashes": hashes,
        }
    return output


def _cache_hash_to_tx() -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        with np.load(
            CACHE_ROOT / f"{scenario}.npz", allow_pickle=False
        ) as archive:
            output[scenario] = dict(
                zip(
                    archive["post_channel_iq_sha256"].astype(str).tolist(),
                    archive["tx_ids"].astype(str).tolist(),
                )
            )
    return output


def _class_to_tx(
    support: dict[str, dict[str, np.ndarray]],
    cache_map: dict[str, dict[str, str]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        row = support[scenario]
        for label, digest in zip(row["labels"], row["hashes"]):
            tx = cache_map[scenario].get(str(digest))
            if tx is None:
                raise D7bSupportAuditError("support hash absent from LEO cache")
            previous = result.setdefault(str(label), tx)
            if previous != tx:
                raise D7bSupportAuditError("class handle to TX mapping drift")
    return result


def _selected_evidence(audit: dict[str, Any]) -> dict[str, Any]:
    selection = audit.get("beta_selection")
    if selection is None:
        selection = audit["beta_selection_for_new_classes"]
    beta = float(selection["selected_beta"])
    rows = [
        row
        for row in selection["candidate_evidence"]
        if float(row["beta"]) == beta
    ]
    if len(rows) != 1:
        raise D7bSupportAuditError("selected beta evidence drift")
    return dict(rows[0])


def _latency_audit(
    head: LocalBoundaryHead,
    sample: np.ndarray,
    *,
    repeats: int = 4000,
) -> dict[str, Any]:
    row = np.asarray(sample, dtype=np.float32).reshape(1, -1)
    prototypes = np.asarray(head.prototypes, dtype=np.float32)

    def identity() -> np.ndarray:
        q = row / np.maximum(np.linalg.norm(row, axis=1, keepdims=True), EPS)
        p = prototypes / np.maximum(
            np.linalg.norm(prototypes, axis=1, keepdims=True), EPS
        )
        return q @ p.T

    def boundary() -> np.ndarray:
        return score_local_contrastive_boundary(row, head)

    for _ in range(100):
        identity()
        boundary()
    identity_rounds = []
    boundary_rounds = []
    for _ in range(7):
        start = time.perf_counter_ns()
        for _ in range(repeats):
            identity()
        identity_rounds.append(
            (time.perf_counter_ns() - start) / repeats / 1000.0
        )
        start = time.perf_counter_ns()
        for _ in range(repeats):
            boundary()
        boundary_rounds.append(
            (time.perf_counter_ns() - start) / repeats / 1000.0
        )
    identity_us = float(np.median(identity_rounds))
    boundary_us = float(np.median(boundary_rounds))
    class_count = head.class_count
    feature_dim = head.feature_dim
    identity_macs = class_count * feature_dim
    boundary_macs = head.estimated_macs_per_query
    identity_numeric_state = 2 * class_count * feature_dim
    return {
        "schema": "cvs.phase2.d7b_host_latency_pareto.v1",
        "input_source": "one_registered_support_representation",
        "accuracy_claim": False,
        "host": "local_windows_cpu_numpy_single_process",
        "repeats_per_round": repeats,
        "rounds": 7,
        "identity_only_median_us_per_sample": identity_us,
        "d7b_median_us_per_sample": boundary_us,
        "latency_delta_us": boundary_us - identity_us,
        "latency_ratio": boundary_us / identity_us,
        "identity_only_macs_per_query": identity_macs,
        "d7b_macs_per_query": boundary_macs,
        "mac_delta": boundary_macs - identity_macs,
        "mac_delta_fraction": (
            (boundary_macs - identity_macs) / identity_macs
        ),
        "identity_only_numeric_state_bytes_fp16": identity_numeric_state,
        "d7b_numeric_state_bytes_fp16_uint16": head.persistent_state_bytes,
        "state_delta_bytes": (
            head.persistent_state_bytes - identity_numeric_state
        ),
        "dense_query_graph_bytes": 0,
        "additional_query_activation_bytes": (
            4 * class_count + 4 * class_count
        ),
        "shared_class_registry_bytes_excluded_from_both_sides": True,
    }


def _state_rows(
    scenario: str,
    state: str,
    head: LocalBoundaryHead,
    class_to_tx: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gram = head.prototypes @ head.prototypes.T
    rival_rows = []
    focus_rows = []
    for class_index, handle in enumerate(head.classes.astype(str).tolist()):
        rival_index = int(head.rival_indices[class_index, 0])
        rival_handle = str(head.classes[rival_index])
        row = {
            "scenario": scenario,
            "registration_state": state,
            "class_handle": handle,
            "transmitter_label": class_to_tx[handle],
            "rival_class_handle": rival_handle,
            "rival_transmitter_label": class_to_tx[rival_handle],
            "prototype_cosine": float(gram[class_index, rival_index]),
            "beta": float(head.beta[class_index]),
            "lifecycle": (
                "old_locked"
                if state == "after" and class_index < head.old_class_count
                else "registered"
            ),
        }
        rival_rows.append(row)
        if row["transmitter_label"] in FOCUS_TX:
            focus_rows.append(dict(row))
    return rival_rows, focus_rows


def main() -> int:
    if OUTPUT.exists():
        raise D7bSupportAuditError("output already exists")
    OUTPUT.mkdir(parents=True)
    loaded = {
        state: _load_enrollment(state) for state in ("before", "after")
    }
    if (
        loaded["before"][3]["feature_runtime_sha256"]
        != loaded["after"][3]["feature_runtime_sha256"]
    ):
        raise D7bSupportAuditError("before/after feature runtime drift")
    runtime_root = loaded["before"][0]
    runtime_manifest = loaded["before"][3]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_torchscript_backbone_same_fd(
        runtime_root,
        _member(runtime_manifest, "feature_runtime"),
        device=device,
    )
    support = {
        state: _extract_support(
            model,
            device,
            loaded[state][2],
            loaded[state][3],
        )
        for state in ("before", "after")
    }
    cache_map = _cache_hash_to_tx()
    class_to_tx = _class_to_tx(support["after"], cache_map)
    before_handles = {
        row["class_handle"]
        for row in loaded["before"][3]["registered_classes"]
    }
    after_handles = {
        row["class_handle"]
        for row in loaded["after"][3]["registered_classes"]
    }
    if set(class_to_tx) != after_handles or not before_handles < after_handles:
        raise D7bSupportAuditError("registration class inventory drift")

    results: dict[str, Any] = {}
    rival_rows: list[dict[str, Any]] = []
    focus_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    state_hashes: dict[str, str] = {}
    latency_rows: list[dict[str, Any]] = []
    old_feature_reexecution_max_abs_diff: dict[str, float] = {}
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        before = support["before"][scenario]
        after = dict(support["after"][scenario])
        before_by_token = {
            str(token): (str(label), str(digest), feature)
            for token, label, digest, feature in zip(
                before["tokens"],
                before["labels"],
                before["hashes"],
                before["features"],
            )
        }
        after_features = np.asarray(after["features"]).copy()
        max_abs_diff = 0.0
        for row_index, (token, label, digest, feature) in enumerate(
            zip(
                after["tokens"],
                after["labels"],
                after["hashes"],
                after["features"],
            )
        ):
            if str(label) not in before_handles:
                continue
            prior = before_by_token.get(str(token))
            if (
                prior is None
                or prior[0] != str(label)
                or prior[1] != str(digest)
            ):
                raise D7bSupportAuditError(
                    "old support token/hash lineage drift"
                )
            max_abs_diff = max(
                max_abs_diff,
                float(np.max(np.abs(prior[2] - feature))),
            )
            # The received IQ hash and class lineage are identical. Reuse the
            # first sealed-runtime execution so old support representation is
            # bitwise identical across registration states.
            after_features[row_index] = prior[2]
        after["features"] = after_features
        old_feature_reexecution_max_abs_diff[scenario] = max_abs_diff
        parent = fit_local_contrastive_boundary(
            before["features"],
            before["labels"],
            physical_sample_ids=before["tokens"].tolist(),
        )
        child = extend_local_contrastive_boundary(
            parent,
            after["features"],
            after["labels"],
            physical_sample_ids=after["tokens"].tolist(),
        )
        old_count = parent.class_count
        if (
            not np.array_equal(child.classes[:old_count], parent.classes)
            or not np.array_equal(
                child.prototypes[:old_count], parent.prototypes
            )
            or not np.array_equal(
                child.rival_indices[:old_count], parent.rival_indices
            )
            or not np.array_equal(child.beta[:old_count], parent.beta)
        ):
            raise D7bSupportAuditError("old D7b state changed after register")
        for token, label, digest, feature in zip(
            after["tokens"],
            after["labels"],
            after["hashes"],
            after["features"],
        ):
            if str(label) not in before_handles:
                continue
            prior = before_by_token.get(str(token))
            if (
                prior is None
                or prior[0] != str(label)
                or prior[1] != str(digest)
                or not np.array_equal(prior[2], feature)
            ):
                raise D7bSupportAuditError(
                    "old support lineage/representation drift"
                )
        parent_evidence = _selected_evidence(parent.support_audit)
        child_evidence = _selected_evidence(child.support_audit)
        results[scenario] = {
            "before": {
                "selected_beta": float(parent.beta[0]),
                "selected_support_evidence": parent_evidence,
                "resource": parent.resource_audit(),
            },
            "after": {
                "selected_new_beta": float(child.beta[-1]),
                "old_beta_vector_sha256": hashlib.sha256(
                    parent.beta.tobytes()
                ).hexdigest(),
                "after_old_beta_vector_sha256": hashlib.sha256(
                    child.beta[:old_count].tobytes()
                ).hexdigest(),
                "selected_support_evidence": child_evidence,
                "resource": child.resource_audit(),
            },
        }
        for state, head, evidence in (
            ("before", parent, parent_evidence),
            ("after", child, child_evidence),
        ):
            rows, focus = _state_rows(
                scenario, state, head, class_to_tx
            )
            rival_rows.extend(rows)
            focus_rows.extend(focus)
            for handle, value in evidence["per_class"].items():
                per_class_rows.append(
                    {
                        "scenario": scenario,
                        "registration_state": state,
                        "class_handle": handle,
                        "transmitter_label": class_to_tx[handle],
                        "lifecycle": value.get(
                            "lifecycle",
                            "registered",
                        ),
                        "support_protocol": evidence["protocol"],
                        "accuracy": float(value["accuracy"]),
                        "mean_margin": float(value["mean_margin"]),
                        "worst_margin": float(value["worst_margin"]),
                    }
                )
        for state, head in (("before", parent), ("after", child)):
            key = f"{scenario}_{state}"
            state_hashes[key] = _write_head_new(
                OUTPUT / f"head_{key}.npz", head
            )
        latency = _latency_audit(child, after["features"][0])
        latency["scenario"] = scenario
        latency["registration_state"] = "after"
        latency_rows.append(latency)

    if {row["transmitter_label"] for row in focus_rows} != FOCUS_TX:
        raise D7bSupportAuditError("focus collision TX coverage incomplete")
    audit = {
        "schema": "cvs.phase2.d7b_support_only_audit.v1",
        "diagnostic_only": True,
        "status": "SUPPORT_ONLY_D7B_LOCKED_NO_QUERY_OPEN",
        "claim_scope": (
            "module_and_support_selection_only_no_independent_performance_claim"
        ),
        "source_row": str(ROW),
        "receiver": loaded["after"][3]["receiver"],
        "seed": int(loaded["after"][3]["seed"]),
        "k_shot": 10,
        "scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
        "opened_package_profiles": [
            "before:enrollment_only",
            "after:enrollment_only",
        ],
        "query_package_opened": False,
        "query_prediction_opened": False,
        "query_score_opened": False,
        "query_truth_opened": False,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "source_sample_access": False,
        "source_derived_signal_access": False,
        "additional_leo_channel_state_generation": False,
        "post_reception_view_counts_as_additional_physical_sample": False,
        "iq_operator_changed": False,
        "representation": "registered_feature_from_fixed_received_leo_weak_iq",
        "beta_candidates": [0.0, 0.05, 0.10, 0.20],
        "rival_count_per_class": 1,
        "class_handle_to_transmitter": class_to_tx,
        "results": results,
        "per_class_support_rows": per_class_rows,
        "rival_rows": rival_rows,
        "focus_collision_rows": focus_rows,
        "latency_pareto": latency_rows,
        "old_feature_reexecution_max_abs_diff": (
            old_feature_reexecution_max_abs_diff
        ),
        "old_feature_state_policy": (
            "identical received-IQ hash reuses first sealed-runtime "
            "representation across before/after"
        ),
        "head_state_sha256": state_hashes,
        "preopen_audit": {
            state: loaded[state][4] for state in ("before", "after")
        },
        "cache_exhaustion_boundary": (
            "current legal pool already consumed as K10 support plus historical "
            "20 query plus fresh 10 query per class/scenario; no query reused"
        ),
    }
    audit_sha = _write_json_new(OUTPUT / "support_audit.json", audit)

    lines = [
        "# D7b注册类局部对比边界support-only审计",
        "",
        "结论：D7b已在当前sealed K=10 LEO_weak enrollment support上完成选择与状态锁定。"
        "本运行只打开before/after的`enrollment_only`包，未打开query、prediction、"
        "score或truth，因此不构成新的独立性能实验。",
        "",
        "协议边界：每个物理样本仅对应一个已接收LEO_weak观测；未生成第二信道状态；"
        "未修改IQ operator；计算representation不增加K；无clean/source访问；"
        "无query角色、类别配额或全局分配。",
        "",
        "## Support选择",
        "",
        "|场景|状态|已注册类数|选择beta|support overall|support floor|worst margin|状态bytes|MAC/query|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in FORMAL_LEO_WEAK_SCENARIOS:
        for state in ("before", "after"):
            row = results[scenario][state]
            evidence = row["selected_support_evidence"]
            resource = row["resource"]
            beta = (
                row["selected_beta"]
                if state == "before"
                else row["selected_new_beta"]
            )
            lines.append(
                f"|`{scenario}`|{state}|{resource['registered_class_count']}|"
                f"{beta:.2f}|{evidence['overall_accuracy']:.4f}|"
                f"{evidence['min_class_accuracy']:.4f}|"
                f"{evidence['worst_margin']:.4f}|"
                f"{resource['persistent_state_bytes']}|"
                f"{resource['estimated_macs_per_query']}|"
            )
    lines.extend(
        [
            "",
            "before的beta用全部旧类物理support leave-two-out选择；after仅对新增类"
            "leave-two-out，旧类prototype/rival/beta全程bitwise锁定，并用旧类support"
            "作新增logit侵入非退化门。",
            "",
            "## 重点碰撞类",
            "",
            "|场景|状态|类|最近rival|prototype cosine|beta|",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in focus_rows:
        lines.append(
            f"|`{row['scenario']}`|{row['registration_state']}|"
            f"`{row['transmitter_label']}`|"
            f"`{row['rival_transmitter_label']}`|"
            f"{row['prototype_cosine']:.4f}|{row['beta']:.2f}|"
        )
    lines.extend(
        [
            "",
            "## 相对identity-only单原型cosine的资源变化",
            "",
            "|场景|MAC增量|MAC增幅|数值状态增量|CPU中位延迟identity/D7b|延迟比|dense query图|",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in latency_rows:
        lines.append(
            f"|`{row['scenario']}`|{row['mac_delta']}|"
            f"{100.0 * row['mac_delta_fraction']:.2f}%|"
            f"{row['state_delta_bytes']}B|"
            f"{row['identity_only_median_us_per_sample']:.2f}/"
            f"{row['d7b_median_us_per_sample']:.2f}us|"
            f"{row['latency_ratio']:.2f}x|0B|"
        )
    lines.extend(
        [
            "",
            "延迟为本机NumPy单进程、以一个support representation作执行输入的"
            "微基准，只用于实现开销比较，不是星上硬件时延结论。类handle注册表为两侧"
            "共享状态，状态增量只计算D7b新增FP16 beta和uint16 rival索引。",
            "",
            "## 判定",
            "",
            "- 训练参数0、适配epoch 0、每类1个rival、无dense query图。",
            "- after旧类prototype/rival/beta与before逐位一致；新增类只注册自身局部边界。",
            "- 当前缓存没有未评分的独立query，故禁止给出accuracy提升、达标或推广结论。",
            "",
        ]
    )
    report_sha = _write_text_new(OUTPUT / "report.md", "\n".join(lines))
    commit = {
        "schema": "cvs.phase2.d7b_support_only_commit.v1",
        "diagnostic_only": True,
        "status": audit["status"],
        "support_audit_sha256": audit_sha,
        "report_sha256": report_sha,
        "head_state_sha256": state_hashes,
        "query_package_opened": False,
        "independent_performance_claim": False,
    }
    commit_sha = _write_json_new(OUTPUT / "COMMIT.json", commit)
    print(
        json.dumps(
            {
                "status": commit["status"],
                "output": str(OUTPUT),
                "focus_rows": focus_rows,
                "latency_pareto": latency_rows,
                "commit_sha256": commit_sha,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
