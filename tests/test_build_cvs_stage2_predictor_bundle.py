from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "code" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import build_cvs_stage2_predictor_bundle as builder  # noqa: E402
from cvsrffi.stage2_ablation_feature_builder import (  # noqa: E402
    _registered_handles,
)


def _arrays():
    labels = ["old-a", "old-b", "new-a", "new-b"]
    roles = ["target_old", "target_old", "target_new", "target_new"]
    tx_ids: list[str] = []
    dataset_roles: list[str] = []
    for label, role in zip(labels, roles):
        tx_ids.extend([label] * 8)
        dataset_roles.extend([role] * 8)
    count = len(tx_ids)
    result = {}
    for scenario_index, scenario in enumerate(builder.FORMAL_LEO_WEAK_SCENARIOS):
        scenario_offset = scenario_index * 100
        result[scenario] = {
            "leo_weak_iq": np.full(
                (count, 2, 8), float(scenario_index + 1), dtype=np.float32
            ),
            "tx_ids": np.asarray(tx_ids),
            "dataset_role": np.asarray(dataset_roles),
            "rx_ids": np.asarray(["20-1"] * count),
            "day_ids": np.asarray(["1"] * count),
            "sig_ids": np.asarray(
                [str(scenario_offset + index) for index in range(count)]
            ),
            "sample_ids": np.asarray(
                [
                    (
                        f"physical|{scenario}|{dataset_roles[index]}|"
                        f"{tx_ids[index]}|20-1|1|{scenario_offset + index}"
                    )
                    for index in range(count)
                ]
            ),
            "overlay_ids": np.asarray(
                [f"overlay|{scenario}|{index}" for index in range(count)]
            ),
            "satellite_seeds": np.asarray([100 + scenario_index] * count, dtype=np.int64),
        }
    return result


def _canonical_manifest_all_arrays():
    labels = ["old-a", "old-b", "new-a", "new-b"]
    role_by_label = {
        "old-a": "target_old",
        "old-b": "target_old",
        "new-a": "target_new",
        "new-b": "target_new",
    }
    query_count_by_class = (
        (5, 4, 4, 4),
        (4, 4, 4, 4),
        (3, 4, 4, 4),
    )
    result = {}
    for scenario_index, scenario in enumerate(builder.FORMAL_LEO_WEAK_SCENARIOS):
        rows = []
        for class_index, label in enumerate(labels):
            role = role_by_label[label]
            for rank in (1, 0):
                rows.append(
                    {
                        "canonical_id": (
                            f"canonical|{scenario}|20-1|{label}|support|{rank}"
                        ),
                        "dataset_role": role,
                        "tx_id": label,
                        "split_role": "support",
                        "split_rank": rank,
                        "value": scenario_index * 1000 + class_index * 100 + rank,
                    }
                )
            for offset in range(query_count_by_class[scenario_index][class_index]):
                rank = 2 + offset
                rows.append(
                    {
                        "canonical_id": (
                            f"canonical|{scenario}|20-1|{label}|query|{rank}"
                        ),
                        "dataset_role": role,
                        "tx_id": label,
                        "split_role": "query",
                        "split_rank": rank,
                        "value": (
                            scenario_index * 1000 + class_index * 100 + rank
                        ),
                    }
                )
        rows.reverse()
        count = len(rows)
        iq = np.empty((count, 2, 8), dtype=np.float32)
        for index, row in enumerate(rows):
            iq[index].fill(float(row["value"]))
        canonical_ids = [str(row["canonical_id"]) for row in rows]
        result[scenario] = {
            "leo_weak_iq": iq,
            "tx_ids": np.asarray([str(row["tx_id"]) for row in rows]),
            "dataset_role": np.asarray(
                [str(row["dataset_role"]) for row in rows]
            ),
            "rx_ids": np.asarray(["20-1"] * count),
            "day_ids": np.asarray([str(scenario_index + 1)] * count),
            "sig_ids": np.asarray(
                [f"sig-{scenario_index}-{index}" for index in range(count)]
            ),
            "sample_ids": np.asarray(canonical_ids),
            "canonical_physical_sample_ids": np.asarray(canonical_ids),
            "split_roles": np.asarray(
                [str(row["split_role"]) for row in rows]
            ),
            "split_ranks": np.asarray(
                [int(row["split_rank"]) for row in rows], dtype=np.int64
            ),
            "overlay_ids": np.asarray(
                [f"overlay|{canonical_id}" for canonical_id in canonical_ids]
            ),
            "satellite_seeds": np.asarray(
                [900 + scenario_index] * count, dtype=np.int64
            ),
        }
    assert [
        int(np.sum(result[scenario]["split_roles"] == "query"))
        for scenario in builder.FORMAL_LEO_WEAK_SCENARIOS
    ] == [17, 16, 15]
    return result


def _args(tmp_path: Path, suffix: str, *, stage: str):
    artifacts = tmp_path / f"artifacts-{suffix}"
    artifacts.mkdir()
    files = {}
    for name in ("candidate", "checkpoint", "adapter", "head"):
        path = artifacts / f"{name}.bin"
        payload = name.encode("ascii")
        if name == "checkpoint":
            payload += b" target_old old-a"
        path.write_bytes(payload)
        files[name] = path
    tta = artifacts / "tta.json"
    tta.write_text(json.dumps({"base_views": 1, "max_views": 5}), encoding="utf-8")
    cache = artifacts / "cache_set.json"
    cache.write_text("{}", encoding="utf-8")
    new_pool = ["new-a", "new-b"]
    draw_seed = 713131 if stage == "stage2c" else 0
    drawn_new = (
        new_pool[
            int(np.random.default_rng(draw_seed).permutation(len(new_pool))[0])
        ]
        if stage == "stage2c"
        else ""
    )
    return SimpleNamespace(
        target_cache_set=cache,
        predictor_out_root=tmp_path / f"predictor-{suffix}",
        scorer_out_root=tmp_path / f"scorer-{suffix}",
        detached_seal_path=None,
        stage=stage,
        receiver="20-1",
        seed=713101,
        support_seed=713111,
        query_seed=713121,
        new_class_draw_seed=draw_seed,
        old_class_labels="old-a,old-b",
        new_class_labels=drawn_new,
        new_class_pool_labels=(
            ",".join(new_pool) if stage == "stage2c" else ""
        ),
        stage2b_reference_new_class_labels="new-a" if stage == "stage2b" else "",
        new_class_count=1 if stage == "stage2c" else 0,
        support_pool_max_k=1,
        query_per_tx=1,
        candidate_lock=files["candidate"],
        checkpoint=files["checkpoint"],
        adapter=files["adapter"],
        head_artifact=files["head"],
        tta_policy_json=tta,
    )


def _manifest_all_args(tmp_path: Path, suffix: str):
    args = _args(tmp_path, suffix, stage="stage2c")
    new_pool = args.new_class_pool_labels.split(",")
    order = np.random.default_rng(args.new_class_draw_seed).permutation(
        len(new_pool)
    )
    args.new_class_labels = ",".join(
        new_pool[int(index)] for index in order
    )
    args.new_class_count = len(new_pool)
    args.support_pool_max_k = 2
    args.query_per_tx = 0
    args.query_policy = "manifest_all"
    return args


def _patch_manifest_all_cache(monkeypatch, arrays):
    calls = []

    def load(_path, *, expected_scope, allowed_roles):
        calls.append(
            {
                "expected_scope": expected_scope,
                "allowed_roles": set(allowed_roles),
            }
        )
        return (
            arrays,
            {
                "schema": "fake-canonical-cache",
                "cache_scope": "stage2_canonical_registered",
            },
            {"status": "PASS", "canonical_split_members_verified": True},
        )

    monkeypatch.setattr(builder, "load_verified_leo_weak_cache_set", load)
    return calls


def _cli_required_args() -> list[str]:
    return [
        "build_cvs_stage2_predictor_bundle.py",
        "--target-cache-set",
        "cache-set.json",
        "--predictor-out-root",
        "predictor",
        "--scorer-out-root",
        "scorer",
        "--stage",
        "stage2b",
        "--receiver",
        "20-1",
        "--seed",
        "1",
        "--old-class-labels",
        "old-a",
        "--support-pool-max-k",
        "1",
        "--query-per-tx",
        "1",
        "--candidate-lock",
        "candidate.json",
        "--checkpoint",
        "checkpoint.bin",
        "--adapter",
        "adapter.bin",
        "--head-artifact",
        "head.bin",
        "--tta-policy-json",
        "tta.json",
    ]


def _bind_formal_phase1_handles(
    tmp_path: Path,
    args: SimpleNamespace,
    monkeypatch,
    *,
    handles: list[str],
) -> Path:
    package_root = tmp_path / f"formal-package-{args.stage}-{args.seed}"
    method_lock = package_root / "locks" / "method_lock.json"
    runtime = (
        package_root
        / "runtime"
        / "adv3b02_runtime.torchscript.pt"
    )
    adapter = (
        package_root
        / "component"
        / "int8_domain_class_center_lowrank_residual_radius_v2.npz"
    )
    for path, payload in (
        (method_lock, b"formal-method-lock"),
        (runtime, b"formal-torchscript-runtime"),
        (adapter, b"formal-component"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    args.candidate_lock = method_lock
    args.checkpoint = runtime
    args.adapter = adapter
    binding = {
        "schema": builder.FORMAL_DEPLOYMENT_BINDING_SCHEMA,
        "package_root": str(package_root.resolve()),
        "detached_seal_path": str(
            (tmp_path / "formal.seal.json").resolve()
        ),
        "detached_seal_sha256": "1" * 64,
        "signature_envelope_path": str(
            (tmp_path / "formal.signature.json").resolve()
        ),
        "signature_envelope_sha256": "2" * 64,
        "checkpoint_lineage_sha256": "3" * 64,
        "runtime_sha256": builder.sha256_file(runtime),
        "component_pre_sign_content_root_sha256": "4" * 64,
        "class_handle_binding_sha256": "5" * 64,
        "parity_receipt_sha256": "6" * 64,
        "generation_lock_sha256": "7" * 64,
        "method_lock_sha256": builder.sha256_file(method_lock),
        "generation_config_sha256": builder.sha256_file(
            args.tta_policy_json
        ),
        "generation_code_sha256": "8" * 64,
        "outer_content_root_sha256": "9" * 64,
        "phase1_completion_receipt_path": str(
            (tmp_path / "completion.json").resolve()
        ),
        "phase1_completion_receipt_sha256": "a" * 64,
        "generation_config_path": str(
            Path(args.tta_policy_json).resolve()
        ),
        "prototype_pt_path": str(Path(args.head_artifact).resolve()),
        "prototype_pt_sha256": builder.sha256_file(
            args.head_artifact
        ),
        "prototype_json_path": str(
            (tmp_path / "prototype.json").resolve()
        ),
        "prototype_json_sha256": "b" * 64,
    }
    binding_path = tmp_path / (
        f"formal-binding-{args.stage}-{args.seed}.json"
    )
    binding_path.write_text(
        json.dumps(binding),
        encoding="utf-8",
    )
    label_path = tmp_path / (
        f"class-label-binding-{args.stage}-{args.seed}.json"
    )
    old_labels = [
        value.strip()
        for value in args.old_class_labels.split(",")
    ]
    label_path.write_text(
        json.dumps(
            {
                "schema": builder.FORMAL_CLASS_LABEL_BINDING_SCHEMA,
                "checkpoint_lineage_sha256": binding[
                    "checkpoint_lineage_sha256"
                ],
                "class_handle_binding_sha256": binding[
                    "class_handle_binding_sha256"
                ],
                "formal_deployment_binding_sha256": builder.sha256_file(
                    binding_path
                ),
                "source_mapping_sha256": "d" * 64,
                "source_checkpoint_sha256": "c" * 64,
                "source_mapping_reused": True,
                "cross_launch_data_identity_required": False,
                "entries": [
                    {
                        "class_index": index,
                        "phase1_tx": label,
                        "registered_class_handle": handles[index],
                    }
                    for index, label in enumerate(old_labels)
                ],
            }
        ),
        encoding="utf-8",
    )
    args.phase1_deployment_binding = binding_path
    args.phase1_class_label_binding = label_path
    monkeypatch.setattr(
        builder,
        "load_formal_adv3b02_deployment_bundle",
        lambda *_args, **_kwargs: SimpleNamespace(
            formal_phase2_context={
                "formal_phase2_eligible": True,
                "outer_signature_verified": True,
            },
            class_binding={
                "class_id_to_handle": [
                    {
                        "class_index": index,
                        "class_handle": handle,
                    }
                    for index, handle in enumerate(handles)
                ]
            },
        ),
    )
    return label_path


def test_support_and_query_seeds_are_independently_effective_and_disjoint() -> None:
    arrays = _arrays()[builder.FORMAL_LEO_WEAK_SCENARIOS[0]]
    common = {
        "receiver": "20-1",
        "support_seed": 101,
        "support_labels": [
            ("target_old", "old-a"),
            ("target_old", "old-b"),
        ],
        "reference_query_labels": [],
        "support_pool_max_k": 2,
        "query_per_tx": 2,
    }
    first = builder._select_support_query(
        arrays, query_seed=201, **common
    )
    second = builder._select_support_query(
        arrays, query_seed=202, **common
    )
    assert first[0].tolist() == second[0].tolist()
    assert first[1].tolist() != second[1].tolist()
    assert set(first[0].tolist()).isdisjoint(first[1].tolist())
    assert set(second[0].tolist()).isdisjoint(second[1].tolist())


def test_stage2c_requires_explicit_draw_seed_and_matching_frozen_pool(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_cache(monkeypatch)
    missing = _args(tmp_path, "missing-draw", stage="stage2c")
    missing.new_class_draw_seed = 0
    with pytest.raises(ValueError, match="new-class pool"):
        builder.build(missing, token_secret=b"d" * 32)

    wrong = _args(tmp_path, "wrong-draw", stage="stage2c")
    wrong.new_class_labels = (
        "new-b" if wrong.new_class_labels == "new-a" else "new-a"
    )
    with pytest.raises(ValueError, match="draw seed"):
        builder.build(wrong, token_secret=b"e" * 32)


def test_stage2c_rejects_partial_pool_even_when_large_enough_to_draw(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_cache(monkeypatch)
    partial = _args(tmp_path, "partial-pool", stage="stage2c")
    partial.new_class_pool_labels = partial.new_class_labels
    with pytest.raises(ValueError, match="canonical complete"):
        builder.build(partial, token_secret=b"p" * 32)


def test_formal_packages_require_explicit_support_and_query_seeds(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_cache(monkeypatch)
    missing = _args(tmp_path, "missing-split-seeds", stage="stage2b")
    missing.support_seed = 0
    with pytest.raises(ValueError, match="must be explicit"):
        builder.build(missing, token_secret=b"f" * 32)


def _patch_cache(monkeypatch):
    arrays = _arrays()
    monkeypatch.setattr(
        builder,
        "load_verified_leo_weak_cache_set",
        lambda *_args, **_kwargs: (
            arrays,
            {"schema": "fake-cache", "cache_scope": "stage2_registered"},
            {"status": "PASS", "external_path": "E:/raw/ManyTx.pkl"},
        ),
    )


def test_formal_phase1_handles_close_predictor_to_feature_cross_chain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_cache(monkeypatch)
    handles = [
        "cls_" + "1" * 64,
        "cls_" + "2" * 64,
    ]
    before = _args(tmp_path, "formal-before", stage="stage2b")
    _bind_formal_phase1_handles(
        tmp_path,
        before,
        monkeypatch,
        handles=handles,
    )
    builder.build(before, token_secret=b"f" * 32)
    before_manifest = json.loads(
        (
            before.predictor_out_root / "package_manifest.json"
        ).read_text(encoding="utf-8")
    )

    after = _args(tmp_path, "formal-after", stage="stage2c")
    after.seed += 1
    _bind_formal_phase1_handles(
        tmp_path,
        after,
        monkeypatch,
        handles=handles,
    )
    builder.build(after, token_secret=b"g" * 32)
    after_manifest = json.loads(
        (
            after.predictor_out_root / "package_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert _registered_handles(before_manifest) == tuple(handles)
    assert _registered_handles(after_manifest)[:2] == tuple(handles)
    assert _registered_handles(after_manifest)[2] not in set(handles)
    for args in (before, after):
        audit = json.loads(
            (
                args.scorer_out_root / "offline_build_audit.json"
            ).read_text(encoding="utf-8")
        )
        assert audit["formal_phase1_class_binding_used"] is True
        assert (
            audit["formal_phase1_class_handle_binding_sha256"]
            == "5" * 64
        )
        assert len(
            audit["phase1_class_label_binding_source_sha256"]
        ) == 64


def test_formal_phase1_label_order_drift_is_rejected_before_cache_open(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path, "formal-order-drift", stage="stage2b")
    handles = [
        "cls_" + "1" * 64,
        "cls_" + "2" * 64,
    ]
    label_path = _bind_formal_phase1_handles(
        tmp_path,
        args,
        monkeypatch,
        handles=handles,
    )
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    payload["entries"][0]["phase1_tx"] = "old-b"
    payload["entries"][1]["phase1_tx"] = "old-a"
    label_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        builder,
        "load_verified_leo_weak_cache_set",
        lambda *_args, **_kwargs: pytest.fail(
            "cache must not open before formal class binding passes"
        ),
    )
    with pytest.raises(
        ValueError,
        match="does not match old-label order",
    ):
        builder.build(args, token_secret=b"h" * 32)


@pytest.mark.parametrize(
    "field",
    [
        "candidate_lock",
        "checkpoint",
        "adapter",
        "head_artifact",
        "tta_policy_json",
    ],
)
def test_formal_phase1_artifact_path_drift_is_rejected_before_cache_open(
    tmp_path: Path,
    monkeypatch,
    field: str,
) -> None:
    args = _args(tmp_path, f"formal-path-{field}", stage="stage2b")
    _bind_formal_phase1_handles(
        tmp_path,
        args,
        monkeypatch,
        handles=["cls_" + "1" * 64, "cls_" + "2" * 64],
    )
    setattr(args, field, tmp_path / f"drift-{field}")
    monkeypatch.setattr(
        builder,
        "load_verified_leo_weak_cache_set",
        lambda *_args, **_kwargs: pytest.fail(
            "cache must not open before formal artifact paths pass"
        ),
    )
    with pytest.raises(ValueError, match=f"path drift for {field}"):
        builder.build(args, token_secret=b"j" * 32)


@pytest.mark.parametrize(
    "field",
    [
        "candidate_lock",
        "checkpoint",
        "head_artifact",
        "tta_policy_json",
    ],
)
def test_formal_phase1_artifact_digest_drift_is_rejected_before_cache_open(
    tmp_path: Path,
    monkeypatch,
    field: str,
) -> None:
    args = _args(tmp_path, f"formal-digest-{field}", stage="stage2b")
    _bind_formal_phase1_handles(
        tmp_path,
        args,
        monkeypatch,
        handles=["cls_" + "1" * 64, "cls_" + "2" * 64],
    )
    Path(getattr(args, field)).write_bytes(b"digest-drift")
    monkeypatch.setattr(
        builder,
        "load_verified_leo_weak_cache_set",
        lambda *_args, **_kwargs: pytest.fail(
            "cache must not open before formal artifact digests pass"
        ),
    )
    with pytest.raises(ValueError, match=f"digest drift for {field}"):
        builder.build(args, token_secret=b"k" * 32)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("checkpoint_lineage_sha256", "checkpoint lineage drift"),
        ("class_handle_binding_sha256", "semantic handle drift"),
        ("formal_deployment_binding_sha256", "deployment digest drift"),
    ],
)
def test_current_class_label_binding_is_atomic_before_cache_open(
    tmp_path: Path,
    monkeypatch,
    field: str,
    message: str,
) -> None:
    args = _args(tmp_path, f"formal-label-{field}", stage="stage2b")
    label_path = _bind_formal_phase1_handles(
        tmp_path,
        args,
        monkeypatch,
        handles=["cls_" + "1" * 64, "cls_" + "2" * 64],
    )
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    payload[field] = "e" * 64
    label_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        builder,
        "load_verified_leo_weak_cache_set",
        lambda *_args, **_kwargs: pytest.fail(
            "cache must not open before current class binding passes"
        ),
    )
    with pytest.raises(ValueError, match=message):
        builder.build(args, token_secret=b"l" * 32)


@pytest.mark.parametrize("mutation", ["extra-entry-key", "missing-row"])
def test_current_class_label_binding_row_schema_is_exact_before_cache_open(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
) -> None:
    args = _args(tmp_path, f"formal-row-{mutation}", stage="stage2b")
    label_path = _bind_formal_phase1_handles(
        tmp_path,
        args,
        monkeypatch,
        handles=["cls_" + "1" * 64, "cls_" + "2" * 64],
    )
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    if mutation == "extra-entry-key":
        payload["entries"][0]["unexpected"] = True
    else:
        payload["entries"].pop()
    label_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        builder,
        "load_verified_leo_weak_cache_set",
        lambda *_args, **_kwargs: pytest.fail(
            "cache must not open before current class rows pass"
        ),
    )
    expected = (
        "row schema drift"
        if mutation == "extra-entry-key"
        else "does not match old-label order"
    )
    with pytest.raises(ValueError, match=expected):
        builder.build(args, token_secret=b"m" * 32)


def test_formal_phase1_authority_is_required_before_cache_open(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path, "formal-authority", stage="stage2b")
    _bind_formal_phase1_handles(
        tmp_path,
        args,
        monkeypatch,
        handles=["cls_" + "1" * 64, "cls_" + "2" * 64],
    )
    monkeypatch.setattr(
        builder,
        "load_formal_adv3b02_deployment_bundle",
        lambda *_args, **_kwargs: SimpleNamespace(
            formal_phase2_context={
                "formal_phase2_eligible": False,
                "outer_signature_verified": True,
            },
            class_binding={},
        ),
    )
    monkeypatch.setattr(
        builder,
        "load_verified_leo_weak_cache_set",
        lambda *_args, **_kwargs: pytest.fail(
            "cache must not open without formal authority"
        ),
    )
    with pytest.raises(ValueError, match="lacks authority"):
        builder.build(args, token_secret=b"n" * 32)


def test_formal_loader_failure_is_before_cache_open(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path, "formal-loader-failure", stage="stage2b")
    _bind_formal_phase1_handles(
        tmp_path,
        args,
        monkeypatch,
        handles=["cls_" + "1" * 64, "cls_" + "2" * 64],
    )

    def reject(*_args, **_kwargs):
        raise ValueError("formal loader rejected")

    monkeypatch.setattr(
        builder,
        "load_formal_adv3b02_deployment_bundle",
        reject,
    )
    monkeypatch.setattr(
        builder,
        "load_verified_leo_weak_cache_set",
        lambda *_args, **_kwargs: pytest.fail(
            "cache must not open after formal loader failure"
        ),
    )
    with pytest.raises(ValueError, match="formal loader rejected"):
        builder.build(args, token_secret=b"o" * 32)


def test_all_deployment_binding_fields_reach_formal_loader(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path, "formal-loader-args", stage="stage2b")
    handles = ["cls_" + "1" * 64, "cls_" + "2" * 64]
    _bind_formal_phase1_handles(
        tmp_path,
        args,
        monkeypatch,
        handles=handles,
    )
    binding = json.loads(
        Path(args.phase1_deployment_binding).read_text(encoding="utf-8")
    )
    captured = {}

    def capture(package_root, **kwargs):
        captured["package_root"] = package_root
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            formal_phase2_context={
                "formal_phase2_eligible": True,
                "outer_signature_verified": True,
            },
            class_binding={
                "class_id_to_handle": [
                    {
                        "class_index": index,
                        "class_handle": handle,
                    }
                    for index, handle in enumerate(handles)
                ]
            },
        )

    monkeypatch.setattr(
        builder,
        "load_formal_adv3b02_deployment_bundle",
        capture,
    )
    actual_handles, _audit = builder._formal_phase1_class_handles(
        args,
        old_labels=["old-a", "old-b"],
    )
    assert actual_handles == handles
    assert captured == {
        "package_root": binding["package_root"],
        "kwargs": {
            "detached_seal_path": binding["detached_seal_path"],
            "expected_detached_seal_sha256": binding[
                "detached_seal_sha256"
            ],
            "signature_envelope_path": binding[
                "signature_envelope_path"
            ],
            "expected_signature_envelope_sha256": binding[
                "signature_envelope_sha256"
            ],
            "expected_checkpoint_lineage_sha256": binding[
                "checkpoint_lineage_sha256"
            ],
            "expected_runtime_sha256": binding["runtime_sha256"],
            "expected_component_pre_sign_content_root_sha256": binding[
                "component_pre_sign_content_root_sha256"
            ],
            "expected_class_handle_binding_sha256": binding[
                "class_handle_binding_sha256"
            ],
            "expected_parity_receipt_sha256": binding[
                "parity_receipt_sha256"
            ],
            "expected_generation_lock_sha256": binding[
                "generation_lock_sha256"
            ],
            "expected_method_lock_sha256": binding[
                "method_lock_sha256"
            ],
            "expected_generation_config_sha256": binding[
                "generation_config_sha256"
            ],
            "expected_generation_code_sha256": binding[
                "generation_code_sha256"
            ],
            "expected_outer_content_root_sha256": binding[
                "outer_content_root_sha256"
            ],
        },
    }


def test_formal_phase1_binding_arguments_are_atomic(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, "formal-atomic", stage="stage2b")
    args.phase1_deployment_binding = tmp_path / "binding.json"
    with pytest.raises(
        ValueError,
        match="must be provided together",
    ):
        builder.build(args, token_secret=b"i" * 32)


def test_stage2c_sealer_physically_separates_truth_and_uses_opaque_tokens(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_cache(monkeypatch)
    args = _args(tmp_path, "c", stage="stage2c")
    result = builder.build(args, token_secret=b"a" * 32)
    assert result["registered_class_count"] == 3
    assert result["support_pool_count"] == 3
    assert result["query_count"] == 3
    predictor_bytes = b"".join(
        path.read_bytes()
        for path in sorted(args.predictor_out_root.iterdir())
        if path.is_file() and path.name not in {"checkpoint.bin", "adapter.bin", "head.bin"}
    )
    assert b"target_old" not in predictor_bytes
    assert b"target_new" not in predictor_bytes
    assert b"old-a" not in predictor_bytes
    assert b"new-a" not in predictor_bytes
    assert not (args.predictor_out_root / "truth_sidecar.json").exists()
    truth = json.loads((args.scorer_out_root / "truth_sidecar.json").read_text(encoding="utf-8"))
    assert {row["evaluation_role"] for row in truth["rows"]} == {
        "target_old",
        "target_new",
    }
    assert all(
        row["true_class_handle"].startswith("cls_") for row in truth["rows"]
    )
    with np.load(
        args.predictor_out_root / "query_leo_clear_weak.npz", allow_pickle=False
    ) as archive:
        tokens = np.asarray(archive["query_tokens"]).astype(str).tolist()
    assert all(token.startswith("qid_") and "|" not in token for token in tokens)


def test_independent_seals_use_unlinkable_query_tokens(tmp_path: Path, monkeypatch) -> None:
    _patch_cache(monkeypatch)
    first = _args(tmp_path, "first", stage="stage2c")
    second = _args(tmp_path, "second", stage="stage2c")
    builder.build(first, token_secret=b"a" * 32)
    builder.build(second, token_secret=b"b" * 32)
    with np.load(first.predictor_out_root / "query_leo_clear_weak.npz") as archive:
        first_tokens = np.asarray(archive["query_tokens"]).astype(str).tolist()
    with np.load(second.predictor_out_root / "query_leo_clear_weak.npz") as archive:
        second_tokens = np.asarray(archive["query_tokens"]).astype(str).tolist()
    assert first_tokens != second_tokens
    for root in (first.predictor_out_root, second.predictor_out_root):
        scenario_token_sets = []
        support_structures = []
        for scenario in builder.FORMAL_LEO_WEAK_SCENARIOS:
            with np.load(root / f"query_{scenario}.npz") as archive:
                query_tokens = np.asarray(archive["query_tokens"]).astype(str).tolist()
                query_manifest = json.loads(str(archive["manifest_json"]))
            with np.load(root / f"support_{scenario}.npz") as archive:
                support_tokens = (
                    np.asarray(archive["support_pool_tokens"]).astype(str).tolist()
                )
                support_structures.append(
                    (
                        np.asarray(archive["support_pool_class_indices"]).tolist(),
                        np.asarray(archive["support_pool_rank_within_class"]).tolist(),
                    )
                )
            scenario_token_sets.append(set(query_tokens) | set(support_tokens))
            assert query_manifest["query_truth_included"] is False
            assert query_manifest["query_role_included"] is False
            assert query_manifest["query_true_batch_class_count_included"] is False
            assert query_manifest["query_class_quota_included"] is False
            assert query_manifest["query_ordering_hint_included"] is False
        assert all(
            scenario_token_sets[left].isdisjoint(scenario_token_sets[right])
            for left in range(len(scenario_token_sets))
            for right in range(left + 1, len(scenario_token_sets))
        )
        assert support_structures[0] == support_structures[1] == support_structures[2]


def test_selected_physical_samples_are_disjoint_within_and_across_scenarios(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_cache(monkeypatch)
    args = _args(tmp_path, "physical-disjoint", stage="stage2c")
    builder.build(args, token_secret=b"p" * 32)
    truth = json.loads(
        (args.scorer_out_root / "truth_sidecar.json").read_text(encoding="utf-8")
    )
    query_ids_by_scenario = {
        scenario: {
            row["physical_sample_id"]
            for row in truth["rows"]
            if row["scenario"] == scenario
        }
        for scenario in builder.FORMAL_LEO_WEAK_SCENARIOS
    }
    assert all(
        query_ids_by_scenario[left].isdisjoint(query_ids_by_scenario[right])
        for left in builder.FORMAL_LEO_WEAK_SCENARIOS
        for right in builder.FORMAL_LEO_WEAK_SCENARIOS
        if left < right
    )
    assert len(truth["rows"]) == 3 * args.query_per_tx * 3
    audit = json.loads(
        (args.scorer_out_root / "offline_build_audit.json").read_text(encoding="utf-8")
    )
    assert audit["same_scenario_support_query_physical_disjointness"] == "PASS"
    assert audit["cross_scenario_selected_physical_disjointness"] == "PASS"


def test_cross_scenario_physical_sample_reuse_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    arrays = _arrays()
    first, second = builder.FORMAL_LEO_WEAK_SCENARIOS[:2]
    arrays[second]["sample_ids"][0] = arrays[first]["sample_ids"][0]
    monkeypatch.setattr(
        builder,
        "load_verified_leo_weak_cache_set",
        lambda *_args, **_kwargs: (
            arrays,
            {"schema": "fake-cache", "cache_scope": "stage2_registered"},
            {"status": "PASS"},
        ),
    )
    args = _args(tmp_path, "physical-reuse", stage="stage2c")
    with pytest.raises(ValueError, match="physical sample reuse across LEO_weak"):
        builder.build(args, token_secret=b"r" * 32)


def test_stage2b_has_old_support_and_role_blind_new_reference_query(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_cache(monkeypatch)
    args = _args(tmp_path, "b", stage="stage2b")
    result = builder.build(args, token_secret=b"c" * 32)
    assert result["registered_class_count"] == 2
    assert result["support_pool_count"] == 2
    assert result["query_count"] == 3
    manifest = json.loads(
        (args.predictor_out_root / "package_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stage"] == "stage2b"
    assert manifest["new_class_count"] == 0
    truth = json.loads(
        (args.scorer_out_root / "truth_sidecar.json").read_text(encoding="utf-8")
    )
    assert sum(row["true_class_handle"] is None for row in truth["rows"]) == 3
    assert {row["scenario"] for row in truth["rows"]} == set(
        builder.FORMAL_LEO_WEAK_SCENARIOS
    )


def test_truth_leak_scan_is_structural_for_npz_numeric_payloads(tmp_path: Path) -> None:
    root = tmp_path / "predictor"
    root.mkdir()
    with (root / "query_leo_clear_weak.npz").open("xb") as handle:
        np.savez(
            handle,
            query_leo_weak_iq=np.frombuffer(b"prefix-old-a-suffix", dtype=np.uint8),
            query_tokens=np.asarray(["qid_" + "1" * 64]),
        )

    builder._reject_predictor_truth_leaks(root, ["old-a"])


@pytest.mark.parametrize("surface", ["member_name", "string_value", "json_file"])
def test_truth_leak_scan_rejects_text_bearing_surfaces(
    tmp_path: Path, surface: str
) -> None:
    root = tmp_path / "predictor"
    root.mkdir()
    if surface == "member_name":
        with (root / "query.npz").open("xb") as handle:
            np.savez(handle, **{"old-a": np.asarray([1], dtype=np.int64)})
    elif surface == "string_value":
        with (root / "query.npz").open("xb") as handle:
            np.savez(handle, query_tokens=np.asarray(["qid_old-a"]))
    else:
        (root / "tta_policy.json").write_text(
            json.dumps({"forbidden": "old-a"}), encoding="utf-8"
        )

    with pytest.raises(ValueError, match="forbidden truth/role token"):
        builder._reject_predictor_truth_leaks(root, ["old-a"])


def test_cli_query_policy_defaults_to_fixed_per_tx(monkeypatch, capsys) -> None:
    captured = {}

    def fake_build(args):
        captured["query_policy"] = args.query_policy
        return {"status": "PASS"}

    monkeypatch.setattr(builder, "build", fake_build)
    monkeypatch.setattr(sys, "argv", _cli_required_args())
    assert builder.main() == 0
    assert captured == {"query_policy": "fixed_per_tx"}
    assert json.loads(capsys.readouterr().out) == {"status": "PASS"}


@pytest.mark.parametrize("query_policy", ["fixed_per_tx", "manifest_all"])
def test_cli_accepts_both_query_policy_choices(
    monkeypatch,
    capsys,
    query_policy: str,
) -> None:
    captured = {}

    def fake_build(args):
        captured["query_policy"] = args.query_policy
        return {"status": "PASS"}

    monkeypatch.setattr(builder, "build", fake_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [*_cli_required_args(), "--query-policy", query_policy],
    )
    assert builder.main() == 0
    assert captured == {"query_policy": query_policy}
    assert json.loads(capsys.readouterr().out) == {"status": "PASS"}


def test_query_policy_enforces_mode_specific_query_per_tx(
    tmp_path: Path,
    monkeypatch,
) -> None:
    arrays = _canonical_manifest_all_arrays()
    _patch_manifest_all_cache(monkeypatch, arrays)
    manifest_all = _manifest_all_args(tmp_path, "manifest-positive-query")
    manifest_all.query_per_tx = 1
    with pytest.raises(ValueError, match="manifest_all.*query_per_tx.*zero"):
        builder.build(manifest_all, token_secret=b"q" * 32)

    fixed = _args(tmp_path, "fixed-zero-query", stage="stage2b")
    fixed.query_policy = "fixed_per_tx"
    fixed.query_per_tx = 0
    with pytest.raises(ValueError, match="fixed_per_tx.*query_per_tx.*positive"):
        builder.build(fixed, token_secret=b"q" * 32)


def test_manifest_all_includes_every_query_and_preserves_truth_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    arrays = _canonical_manifest_all_arrays()
    calls = _patch_manifest_all_cache(monkeypatch, arrays)
    args = _manifest_all_args(tmp_path, "manifest-all")

    result = builder.build(args, token_secret=b"m" * 32)

    expected_counts = dict(
        zip(builder.FORMAL_LEO_WEAK_SCENARIOS, (17, 16, 15))
    )
    assert calls == [
        {
            "expected_scope": "stage2_canonical_registered",
            "allowed_roles": {"target_old", "target_new"},
        }
    ]
    assert result["query_policy"] == "manifest_all"
    assert result["query_count_by_scenario"] == expected_counts
    assert result["support_pool_count"] == 8
    assert not (args.predictor_out_root / "truth_sidecar.json").exists()

    truth = json.loads(
        (args.scorer_out_root / "truth_sidecar.json").read_text(
            encoding="utf-8"
        )
    )
    truth_tokens = [str(row["query_token"]) for row in truth["rows"]]
    assert len(truth_tokens) == 48
    assert Counter(truth_tokens).most_common(1)[0][1] == 1

    all_query_tokens = []
    scenario_token_sets = []
    for scenario_index, scenario in enumerate(
        builder.FORMAL_LEO_WEAK_SCENARIOS
    ):
        source = arrays[scenario]
        source_roles = np.asarray(source["split_roles"]).astype(str)
        canonical_ids = np.asarray(
            source["canonical_physical_sample_ids"]
        ).astype(str)
        expected_query_ids = set(canonical_ids[source_roles == "query"].tolist())
        expected_support_ids = set(
            canonical_ids[source_roles == "support"].tolist()
        )
        assert expected_support_ids.isdisjoint(expected_query_ids)
        observed_truth_rows = [
            row for row in truth["rows"] if row["scenario"] == scenario
        ]
        assert len(observed_truth_rows) == expected_counts[scenario]
        assert {
            str(row["physical_sample_id"]) for row in observed_truth_rows
        } == expected_query_ids

        with np.load(
            args.predictor_out_root / f"query_{scenario}.npz",
            allow_pickle=False,
        ) as archive:
            assert tuple(archive.files) == builder.QUERY_NPZ_MEMBERS
            query_tokens = np.asarray(archive["query_tokens"]).astype(str).tolist()
            assert len(query_tokens) == expected_counts[scenario]
            query_manifest = json.loads(str(archive["manifest_json"]))
            assert query_manifest == {
                "schema": builder.QUERY_SCHEMA,
                "scenario": scenario,
                "query_truth_included": False,
                "query_role_included": False,
                "query_true_batch_class_count_included": False,
                "query_class_quota_included": False,
                "query_ordering_hint_included": False,
                "token_scheme": "hmac_sha256_opaque_v1",
            }
            textual_values = []
            for member in archive.files:
                values = np.asarray(archive[member])
                if values.dtype.kind in {"S", "U"}:
                    textual_values.extend(
                        str(value) for value in values.reshape(-1).tolist()
                    )
            predictor_text = "\n".join([*archive.files, *textual_values])
            forbidden = [
                "target_old",
                "target_new",
                "old-a",
                "old-b",
                "new-a",
                "new-b",
                *canonical_ids.tolist(),
            ]
            assert all(value not in predictor_text for value in forbidden)

        with np.load(
            args.predictor_out_root / f"support_{scenario}.npz",
            allow_pickle=False,
        ) as archive:
            assert tuple(archive.files) == builder.SUPPORT_NPZ_MEMBERS
            assert np.asarray(
                archive["support_pool_class_indices"]
            ).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
            assert np.asarray(
                archive["support_pool_rank_within_class"]
            ).tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
            ordered_labels = [
                "old-a",
                "old-b",
                *args.new_class_labels.split(","),
            ]
            source_class_index = {
                label: index
                for index, label in enumerate(
                    ["old-a", "old-b", "new-a", "new-b"]
                )
            }
            expected_iq_values = [
                float(
                    scenario_index * 1000
                    + source_class_index[label] * 100
                    + rank
                )
                for label in ordered_labels
                for rank in range(2)
            ]
            assert np.asarray(
                archive["support_pool_leo_weak_iq"]
            )[:, 0, 0].tolist() == expected_iq_values
            support_tokens = np.asarray(
                archive["support_pool_tokens"]
            ).astype(str).tolist()

        assert set(query_tokens).isdisjoint(support_tokens)
        scenario_token_sets.append(set(query_tokens) | set(support_tokens))
        all_query_tokens.extend(query_tokens)

    assert all(
        scenario_token_sets[left].isdisjoint(scenario_token_sets[right])
        for left in range(len(scenario_token_sets))
        for right in range(left + 1, len(scenario_token_sets))
    )
    assert set(all_query_tokens) == set(truth_tokens)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-trio",
        "partial-trio",
        "inconsistent-length",
        "invalid-role",
        "noninteger-rank",
        "negative-rank",
        "duplicate-rank",
        "rank-gap",
        "duplicate-canonical-id",
        "missing-support",
        "unregistered-query-truth",
    ],
)
def test_manifest_all_rejects_invalid_canonical_split_arrays(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
) -> None:
    arrays = _canonical_manifest_all_arrays()
    scenario = builder.FORMAL_LEO_WEAK_SCENARIOS[0]
    payload = arrays[scenario]
    support_indices = np.flatnonzero(payload["split_roles"] == "support")
    query_indices = np.flatnonzero(payload["split_roles"] == "query")
    if mutation == "missing-trio":
        for field in (
            "canonical_physical_sample_ids",
            "split_roles",
            "split_ranks",
        ):
            payload.pop(field)
    elif mutation == "partial-trio":
        payload.pop("split_ranks")
    elif mutation == "inconsistent-length":
        payload["split_roles"] = payload["split_roles"][:-1]
    elif mutation == "invalid-role":
        payload["split_roles"][query_indices[0]] = "validation"
    elif mutation == "noninteger-rank":
        payload["split_ranks"] = payload["split_ranks"].astype(np.float64)
    elif mutation == "negative-rank":
        payload["split_ranks"][query_indices[0]] = -1
    elif mutation == "duplicate-rank":
        same_class = support_indices[
            payload["tx_ids"][support_indices] == "old-a"
        ]
        payload["split_ranks"][same_class[0]] = 0
        payload["split_ranks"][same_class[1]] = 0
    elif mutation == "rank-gap":
        same_class = support_indices[
            payload["tx_ids"][support_indices] == "old-a"
        ]
        payload["split_ranks"][same_class[0]] = 0
        payload["split_ranks"][same_class[1]] = 2
    elif mutation == "duplicate-canonical-id":
        payload["canonical_physical_sample_ids"][query_indices[0]] = payload[
            "canonical_physical_sample_ids"
        ][support_indices[0]]
    elif mutation == "missing-support":
        payload["split_roles"][support_indices[0]] = "query"
    elif mutation == "unregistered-query-truth":
        old_query_indices = query_indices[
            payload["dataset_role"][query_indices] == "target_old"
        ]
        payload["tx_ids"][old_query_indices[0]] = "old-unregistered"
    else:  # pragma: no cover - the parameter list is exhaustive.
        raise AssertionError(mutation)

    _patch_manifest_all_cache(monkeypatch, arrays)
    args = _manifest_all_args(tmp_path, f"invalid-{mutation}")
    with pytest.raises(ValueError, match="canonical|split|rank|support|registered"):
        builder.build(args, token_secret=b"v" * 32)


def test_manifest_all_rejects_cross_scene_canonical_id_reuse(
    tmp_path: Path,
    monkeypatch,
) -> None:
    arrays = _canonical_manifest_all_arrays()
    first, second = builder.FORMAL_LEO_WEAK_SCENARIOS[:2]
    first_query = int(np.flatnonzero(arrays[first]["split_roles"] == "query")[0])
    second_query = int(np.flatnonzero(arrays[second]["split_roles"] == "query")[0])
    arrays[second]["canonical_physical_sample_ids"][second_query] = arrays[first][
        "canonical_physical_sample_ids"
    ][first_query]
    _patch_manifest_all_cache(monkeypatch, arrays)
    args = _manifest_all_args(tmp_path, "cross-scene-canonical-reuse")
    with pytest.raises(ValueError, match="canonical.*across.*scenario"):
        builder.build(args, token_secret=b"x" * 32)


def test_legacy_default_result_and_artifact_schema_remain_unchanged(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_cache(monkeypatch)
    args = _args(tmp_path, "legacy-schema", stage="stage2c")
    result = builder.build(args, token_secret=b"l" * 32)
    assert set(result) == {
        "stage",
        "predictor_package",
        "predictor_package_root_sha256",
        "predictor_package_seal",
        "predictor_package_seal_sha256",
        "scoring_manifest",
        "scoring_manifest_sha256",
        "registered_class_count",
        "support_pool_count",
        "query_count",
        "support_seed",
        "query_seed",
        "new_class_draw_seed",
    }
    manifest = json.loads(
        (args.predictor_out_root / "package_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "query_policy" not in manifest
    for scenario in builder.FORMAL_LEO_WEAK_SCENARIOS:
        with np.load(
            args.predictor_out_root / f"query_{scenario}.npz",
            allow_pickle=False,
        ) as archive:
            assert tuple(archive.files) == builder.QUERY_NPZ_MEMBERS
        with np.load(
            args.predictor_out_root / f"support_{scenario}.npz",
            allow_pickle=False,
        ) as archive:
            assert tuple(archive.files) == builder.SUPPORT_NPZ_MEMBERS
