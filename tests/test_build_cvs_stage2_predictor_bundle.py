from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "code" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import build_cvs_stage2_predictor_bundle as builder  # noqa: E402


def _arrays():
    labels = ["old-a", "old-b", "new-a", "new-b"]
    roles = ["target_old", "target_old", "target_new", "target_new"]
    tx_ids: list[str] = []
    dataset_roles: list[str] = []
    for label, role in zip(labels, roles):
        tx_ids.extend([label, label])
        dataset_roles.extend([role, role])
    count = len(tx_ids)
    result = {}
    for scenario_index, scenario in enumerate(builder.FORMAL_LEO_WEAK_SCENARIOS):
        result[scenario] = {
            "leo_weak_iq": np.full(
                (count, 2, 8), float(scenario_index + 1), dtype=np.float32
            ),
            "tx_ids": np.asarray(tx_ids),
            "dataset_role": np.asarray(dataset_roles),
            "rx_ids": np.asarray(["20-1"] * count),
            "day_ids": np.asarray(["1"] * count),
            "sig_ids": np.asarray([str(index) for index in range(count)]),
            "sample_ids": np.asarray(
                [
                    f"{dataset_roles[index]}|{tx_ids[index]}|20-1|1|1|{index}"
                    for index in range(count)
                ]
            ),
            "overlay_ids": np.asarray(
                [f"overlay|{scenario}|{index}" for index in range(count)]
            ),
            "satellite_seeds": np.asarray([100 + scenario_index] * count, dtype=np.int64),
        }
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
    return SimpleNamespace(
        target_cache_set=cache,
        predictor_out_root=tmp_path / f"predictor-{suffix}",
        scorer_out_root=tmp_path / f"scorer-{suffix}",
        detached_seal_path=None,
        stage=stage,
        receiver="20-1",
        seed=713101,
        old_class_labels="old-a,old-b",
        new_class_labels="new-a" if stage == "stage2c" else "",
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


def test_truth_leak_scan_ignores_short_label_bytes_inside_numeric_iq(
    tmp_path: Path,
) -> None:
    root = tmp_path / "predictor"
    root.mkdir()
    numeric_iq = np.frombuffer(b"1-8\x00", dtype=np.float32).copy()
    with (root / "query_leo_clear_weak.npz").open("xb") as handle:
        np.savez(
            handle,
            query_leo_weak_iq=numeric_iq,
            query_tokens=np.asarray(["qid_" + "1" * 64]),
        )
    assert b"1-8" in (root / "query_leo_clear_weak.npz").read_bytes()

    builder._reject_predictor_truth_leaks(root, ["1-8", "target_old"])


def test_truth_leak_scan_rejects_label_in_npz_text_member(tmp_path: Path) -> None:
    root = tmp_path / "predictor"
    root.mkdir()
    with (root / "query_leo_clear_weak.npz").open("xb") as handle:
        np.savez(
            handle,
            query_leo_weak_iq=np.zeros((1, 2, 8), dtype=np.float32),
            query_tokens=np.asarray(["qid_target_old|14-10"]),
        )

    with pytest.raises(
        ValueError,
        match=r"query_leo_clear_weak\.npz:query_tokens",
    ):
        builder._reject_predictor_truth_leaks(root, ["target_old", "14-10"])


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
        scenario_tokens = []
        for scenario in builder.FORMAL_LEO_WEAK_SCENARIOS:
            with np.load(root / f"query_{scenario}.npz") as archive:
                scenario_tokens.append(np.asarray(archive["query_tokens"]).astype(str).tolist())
        assert scenario_tokens[0] == scenario_tokens[1] == scenario_tokens[2]


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
    assert sum(row["true_class_handle"] is None for row in truth["rows"]) == 1
