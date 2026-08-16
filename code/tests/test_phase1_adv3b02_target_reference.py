from __future__ import annotations

"""Task 2 contracts for ADV3B02 source sealing and blind target prediction.

The train-data configuration is never caller-authored. It is first derived
from one ADV final checkpoint, its same-directory completion receipt, and one
matching source-only CLIC clean-v4 authority. The separate blind-prediction
fixture contains received IQ and opaque lineage only: it deliberately does
not create a target truth sidecar, known-test configuration, ADV reference,
or target identity fields.
"""

import copy
import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = CODE_ROOT / "evaluate_phase1_adv3b02_target_leo.py"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi import phase1_clic_target_leo as TARGET
from cvsrffi.leo_weak_cache import FORMAL_LEO_WEAK_SCENARIOS, sha256_file


ADV_RUN_ID = "phase1_adv3b02_clic6_20260816_v2"
ADV_CANDIDATE = "F1_ADV3B02_CLIC"
CLIC_ARM = "C"
CLIC_CANDIDATE = "F1C_CLIC12"
SOURCE_LOCAL4 = ("20-15", "20-19", "6-15", "8-20")
HELD_TX = "14-7"
PROXY_TX = "14-10"
SOURCE_RECEIVER_INDICES = tuple(str(index) for index in range(7))
SOURCE_DAY_INDICES = tuple(str(index) for index in range(2))
SOURCE_RECEIVER_IDS = tuple(f"src-rx-{index}" for index in range(7))
SOURCE_DAY_IDS = ("2021_03_01", "2021_03_08")
FROZEN_WISIG_SHA256 = (
    "2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f"
)
ROWS_PER_SCENE = 1040
TOTAL_ROWS = ROWS_PER_SCENE * len(FORMAL_LEO_WEAK_SCENARIOS)


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(TARGET.canonical_json_bytes(payload) + b"\n")
    return path


def _load_task2_module():
    """Load the owned evaluator only after producing a clear RED failure.

    Break caught: a missing Task2 API is silently skipped or converted into an
    unrelated fixture/collection failure.
    """

    if not EVALUATOR_PATH.is_file():
        pytest.fail(
            "Task2 RED: evaluate_phase1_adv3b02_target_leo.py is absent; "
            "seal_adv3b02_train_data_config and "
            "publish_adv3b02_target_prediction APIs have not been implemented"
        )
    spec = importlib.util.spec_from_file_location("_adv3b02_target_task2", EVALUATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_split_receipt(
    *, wisig_sha256: str = FROZEN_WISIG_SHA256
) -> dict[str, Any]:
    """Build the public aggregate source-split receipt shape, not sample rows."""

    receipt: dict[str, Any] = {
        "schema": "cvs.phase1.source_split_receipt.v1",
        "seed": 392002,
        "split_mode": "tx_rx_day_1_6_3",
        "wisig_pkl_sha256": wisig_sha256,
        "source_days": list(SOURCE_DAY_INDICES),
        "target_days": ["2"],
        "source_receivers": list(SOURCE_RECEIVER_INDICES),
        "target_receivers": ["7"],
        "source_target_receiver_overlap_count": 0,
        "labeled_indices_sha256": _sha_text("adv-labeled-indices"),
        "unlabeled_indices_sha256": _sha_text("adv-unlabeled-indices"),
        "source_validation_indices_sha256": _sha_text("adv-validation-indices"),
        "labeled_size": 3920,
        "unlabeled_size": 35280,
        "source_validation_size": 16800,
        "source_pool_size": 56000,
        "requested_labeled_ratio": 0.07,
        "requested_unlabeled_ratio": 0.63,
        "requested_source_val_ratio": 0.30,
        "requested_rho_label": 0.10,
        "realized_rho_label": 0.10,
        "realized_source_val_fraction": 0.30,
        "realized_rho_tolerance": 0.002,
        "realized_source_val_tolerance": 0.002,
        "realized_rho_within_tolerance": True,
        "realized_source_val_within_tolerance": True,
    }
    receipt["split_manifest_sha256"] = TARGET.canonical_sha256(receipt)
    return receipt


def _tx_partition_receipt() -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": "cvs.phase1.tx_partition_receipt.v1",
        "enabled": True,
        "source_known_train_tx": list(SOURCE_LOCAL4),
        "source_known_validation_tx": [HELD_TX],
        "source_proxy_unknown_tx": [PROXY_TX],
        "dataset_tx_order": [*SOURCE_LOCAL4, HELD_TX, PROXY_TX],
        "dataset_tx_count": 6,
        "training_tx_count": 4,
        "allow_empty_proxy_unknown": False,
        "training_view_contiguous_reindex": {
            str(index): tx_id for index, tx_id in enumerate(SOURCE_LOCAL4)
        },
        "held_tx_loaded_by_training": False,
    }
    receipt["partition_sha256"] = TARGET.canonical_sha256(receipt)
    return receipt


def _matching_clic_normalized_train_config(
    *, wisig_sha256: str = FROZEN_WISIG_SHA256, input_len: int = 4
) -> dict[str, Any]:
    """The semantic C/G comparison surface expected from clean-v4 authority."""

    return {
        "dataset_provenance": {
            "dataset_schema": "WiSig",
            "wisig_pkl_sha256": wisig_sha256,
        },
        "source_train_tx_ids": list(SOURCE_LOCAL4),
        "source_validation_tx_ids": [HELD_TX],
        "source_proxy_tx_ids": [PROXY_TX],
        "source_receiver_ids": list(SOURCE_RECEIVER_IDS),
        "source_day_ids": list(SOURCE_DAY_IDS),
        "split_mode": "tx_rx_day_1_6_3",
        "role_construction": {
            "split_mode": "tx_rx_day_1_6_3",
            "labeled_ratio": 0.07,
            "unlabeled_ratio": 0.63,
            "source_val_ratio": 0.30,
        },
        "physical_row_selection": {
            "selection_policy": "pre_registered_tx_rx_day_eq_split_by_sig_i",
            "group_axes": ["tx_id", "rx_id", "day_id", "eq_id"],
        },
        "preprocessing": {"input_len": input_len, "iq_dtype": "float32"},
        "single_leo_training_scenes": list(FORMAL_LEO_WEAK_SCENARIOS),
    }


def _physical_key(
    tx_id: str, rx_id: str, day_id: str, eq_id: str, sig_id: str
) -> str:
    return "\x1f".join((tx_id, rx_id, day_id, eq_id, sig_id))


def _clean_l_rows() -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    tx_ids: list[str] = []
    rx_ids: list[str] = []
    day_ids: list[str] = []
    eq_ids: list[str] = []
    sig_ids: list[str] = []
    for tx_id in SOURCE_LOCAL4:
        for rx_id in SOURCE_RECEIVER_IDS:
            for repeat in range(140):
                tx_ids.append(tx_id)
                rx_ids.append(rx_id)
                day_ids.append(SOURCE_DAY_IDS[repeat % len(SOURCE_DAY_IDS)])
                eq_ids.append("eq-source")
                sig_ids.append(f"l-{repeat:03d}")
    assert len(tx_ids) == 3920
    return tx_ids, rx_ids, day_ids, eq_ids, sig_ids


def _clean_v_rows() -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    tx_ids: list[str] = []
    rx_ids: list[str] = []
    day_ids: list[str] = []
    eq_ids: list[str] = []
    sig_ids: list[str] = []
    for tx_id in SOURCE_LOCAL4:
        for rx_id in SOURCE_RECEIVER_IDS:
            for day_id in SOURCE_DAY_IDS:
                for repeat in range(300):
                    tx_ids.append(tx_id)
                    rx_ids.append(rx_id)
                    day_ids.append(day_id)
                    eq_ids.append("eq-source")
                    sig_ids.append(f"v-{repeat:03d}")
    assert len(tx_ids) == 16800
    return tx_ids, rx_ids, day_ids, eq_ids, sig_ids


def _clean_proxy_rows() -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    tx_ids = [PROXY_TX for _ in range(400)]
    rx_ids = [
        SOURCE_RECEIVER_IDS[index % len(SOURCE_RECEIVER_IDS)] for index in range(400)
    ]
    day_ids = [SOURCE_DAY_IDS[index % len(SOURCE_DAY_IDS)] for index in range(400)]
    eq_ids = ["eq-proxy" for _ in range(400)]
    sig_ids = [f"p-{index:03d}" for index in range(400)]
    return tx_ids, rx_ids, day_ids, eq_ids, sig_ids


def _write_clean_v4_authority(
    path: Path,
    *,
    wisig_sha256: str = FROZEN_WISIG_SHA256,
    manifest_overrides: dict[str, Any] | None = None,
) -> Path:
    """Write a complete clean-v4 metadata authority with inert forbidden members.

    The inert feature members make a metadata-only read observable: a sealer
    that touches any of them will fail the guarded RED test below.
    """

    l_tx, l_rx, l_day, l_eq, l_sig = _clean_l_rows()
    v_tx, v_rx, v_day, v_eq, v_sig = _clean_v_rows()
    p_tx, p_rx, p_day, p_eq, p_sig = _clean_proxy_rows()
    v_keys = [
        _physical_key(tx_id, rx_id, day_id, eq_id, sig_id)
        for tx_id, rx_id, day_id, eq_id, sig_id in zip(
            v_tx, v_rx, v_day, v_eq, v_sig, strict=True
        )
    ]
    source_split = _source_split_receipt(wisig_sha256=wisig_sha256)
    partition = _tx_partition_receipt()
    manifest: dict[str, Any] = {
        "schema": "cvs.phase1.clic_lv_export.v1",
        "method": "P1_CLIC",
        "source_only": True,
        "candidate_id": CLIC_CANDIDATE,
        "run_id": "phase1_clic12_20260812_v5",
        "training_run_contract": "phase1_clic12_20260812_v5",
        "clic_enabled": False,
        "unlabeled_loader_constructed": False,
        "unlabeled_forward_rows": 0,
        "source_tx_ids": list(SOURCE_LOCAL4),
        "known_validation_tx_ids": [HELD_TX],
        "proxy_unknown_tx_ids": [PROXY_TX],
        "source_receiver_ids": list(SOURCE_RECEIVER_IDS),
        "source_receiver_ids_sha256": TARGET.canonical_sha256(
            list(SOURCE_RECEIVER_IDS)
        ),
        "source_day_ids": list(SOURCE_DAY_IDS),
        "source_day_ids_sha256": TARGET.canonical_sha256(list(SOURCE_DAY_IDS)),
        "wisig_pkl_sha256": wisig_sha256,
        "source_split_receipt": source_split,
        "source_split_receipt_sha256": TARGET.canonical_sha256(source_split),
        "tx_partition_receipt": partition,
        "tx_partition_receipt_sha256": TARGET.canonical_sha256(partition),
        # This must match ADV's same split receipt; it is an index receipt,
        # not a feature/physical row read.
        "source_validation_indices_sha256": source_split[
            "source_validation_indices_sha256"
        ],
        "source_validation_physical_order_sha256": TARGET.canonical_sha256(v_keys),
        "labeled_validation_physical_disjoint": True,
        "labeled_validation_proxy_physical_disjoint": True,
        "labeled_row_count": len(l_tx),
        "source_validation_row_count": len(v_tx),
        "proxy_row_count": len(p_tx),
        "source_checkpoint_sha256": _sha_text("matching-clic-C-checkpoint"),
        "terminal_receipt_sha256": _sha_text("matching-clic-C-terminal"),
        "clean_source_runtime_access": False,
        "query_fit_access": False,
    }
    if manifest_overrides:
        manifest.update(copy.deepcopy(manifest_overrides))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        dataset_role=np.asarray(
            ["labeled_fit"] * len(l_tx)
            + ["source_validation_known"] * len(v_tx)
            + ["proxy_unknown"] * len(p_tx),
            dtype=str,
        ),
        tx_ids=np.asarray(l_tx + v_tx + p_tx, dtype=str),
        rx_ids=np.asarray(l_rx + v_rx + p_rx, dtype=str),
        day_ids=np.asarray(l_day + v_day + p_day, dtype=str),
        eq_ids=np.asarray(l_eq + v_eq + p_eq, dtype=str),
        sig_ids=np.asarray(l_sig + v_sig + p_sig, dtype=str),
        manifest_json=np.asarray(
            json.dumps(manifest, ensure_ascii=True, sort_keys=True)
        ),
        z_id=np.asarray([[13.0]], dtype=np.float32),
        features=np.asarray([[17.0]], dtype=np.float32),
        tx_logits=np.asarray([[19.0]], dtype=np.float32),
    )
    return path


def _write_wisig_authority(
    path: Path,
    *,
    source_receiver_ids: tuple[str, ...] = SOURCE_RECEIVER_IDS,
    source_day_ids: tuple[str, ...] = SOURCE_DAY_IDS,
) -> Path:
    """Write a real pickle whose physical axes can be independently reopened."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": [],
        "tx_list": [*SOURCE_LOCAL4, HELD_TX, PROXY_TX],
        "rx_list": [*source_receiver_ids, "target-rx-7"],
        "capture_date_list": [*source_day_ids, "2021_03_15"],
        "equalized_list": [1],
    }
    path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    return path


def _expected_physical_axis_binding(paths: dict[str, Path]) -> dict[str, Any]:
    receiver_mapping = [
        {"index": index, "physical_id": physical_id}
        for index, physical_id in zip(
            SOURCE_RECEIVER_INDICES, SOURCE_RECEIVER_IDS, strict=True
        )
    ]
    day_mapping = [
        {"index": index, "physical_id": physical_id}
        for index, physical_id in zip(SOURCE_DAY_INDICES, SOURCE_DAY_IDS, strict=True)
    ]
    binding: dict[str, Any] = {
        "schema": "cvs.phase1.wisig_source_physical_axis_binding.v1",
        "wisig_pkl_path": str(paths["wisig"].resolve()),
        "wisig_pkl_sha256": sha256_file(paths["wisig"]),
        "source_receiver_indices": list(SOURCE_RECEIVER_INDICES),
        "source_receiver_indices_sha256": TARGET.canonical_sha256(
            list(SOURCE_RECEIVER_INDICES)
        ),
        "source_receiver_ids": list(SOURCE_RECEIVER_IDS),
        "source_receiver_ids_sha256": TARGET.canonical_sha256(
            list(SOURCE_RECEIVER_IDS)
        ),
        "source_receiver_index_to_physical": receiver_mapping,
        "source_receiver_index_to_physical_sha256": TARGET.canonical_sha256(
            receiver_mapping
        ),
        "source_day_indices": list(SOURCE_DAY_INDICES),
        "source_day_indices_sha256": TARGET.canonical_sha256(
            list(SOURCE_DAY_INDICES)
        ),
        "source_day_ids": list(SOURCE_DAY_IDS),
        "source_day_ids_sha256": TARGET.canonical_sha256(list(SOURCE_DAY_IDS)),
        "source_day_index_to_physical": day_mapping,
        "source_day_index_to_physical_sha256": TARGET.canonical_sha256(day_mapping),
    }
    return binding


def _write_training_authorities(
    root: Path,
    *,
    dataset_source_receiver_ids: tuple[str, ...] = SOURCE_RECEIVER_IDS,
    dataset_source_day_ids: tuple[str, ...] = SOURCE_DAY_IDS,
    input_len: int = 4,
) -> dict[str, Path]:
    """Produce the only legal file inputs to the Task2 train-data sealer."""

    import torch

    wisig = _write_wisig_authority(
        root / "Dataset_WigSig" / "ManySig.pkl",
        source_receiver_ids=dataset_source_receiver_ids,
        source_day_ids=dataset_source_day_ids,
    )
    wisig_sha256 = sha256_file(wisig)
    source_split = _source_split_receipt(wisig_sha256=wisig_sha256)
    partition = _tx_partition_receipt()
    checkpoint = root / "runs" / ADV_RUN_ID / ADV_CANDIDATE / "final_ssdg.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint_payload = {
        "checkpoint_schema": "ssdg_phase1_training_state_v2",
        "checkpoint_role": "training_final_only",
        "checkpoint_selection": "final_only",
        "run_id": ADV_RUN_ID,
        "candidate_id": ADV_CANDIDATE,
        "final_epoch": 200,
        "args": {
            "run_id": ADV_RUN_ID,
            "candidate_id": ADV_CANDIDATE,
            "base_candidate": "ADV3B02_CORE90_SOFT_E200_CLIC_EQ_RHO07_FINAL",
            "dataset": "wisig",
            "wisig_pkl": str(wisig.resolve()),
            "wisig_pkl_sha256": wisig_sha256,
            "wisig_out_len": input_len,
            "num_classes": 4,
            "model_size": "S",
            "split_mode": "tx_rx_day_1_6_3",
            "labeled_ratio": 0.07,
            "unlabeled_ratio": 0.63,
            "source_val_ratio": 0.30,
            "seed": 392002,
            "checkpoint_selection": "final_only",
            "phase1_source_train_tx_ids": ",".join(SOURCE_LOCAL4),
            "phase1_source_known_validation_tx_ids": HELD_TX,
            "phase1_source_proxy_unknown_tx_ids": PROXY_TX,
            "wisig_train_rxs": ",".join(SOURCE_RECEIVER_INDICES),
            "wisig_train_days": ",".join(SOURCE_DAY_INDICES),
            "sat_train_scenarios": ",".join(FORMAL_LEO_WEAK_SCENARIOS),
            "eval_sat_scenarios": ",".join(FORMAL_LEO_WEAK_SCENARIOS),
        },
        "model": {},
        "split_info": {
            "mode": "tx_rx_day_1_6_3",
            "source_split_receipt": copy.deepcopy(source_split),
            "tx_partition_receipt": copy.deepcopy(partition),
            "class_id_to_tx": list(SOURCE_LOCAL4),
        },
    }
    torch.save(checkpoint_payload, checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    completion = _write_json(
        checkpoint.parent / "phase1_training_completion_receipt.json",
        {
            "schema": "cvs.phase1.training_completion_receipt.v1",
            "run_id": ADV_RUN_ID,
            "row_key": "",
            "ablation_id": "",
            "train_seed": 392002,
            "wisig_pkl_sha256": wisig_sha256,
            "source_split_receipt": copy.deepcopy(source_split),
            "selected_checkpoint_sha256": checkpoint_sha,
            "terminal_status": "NON_PROMOTABLE_P0_DISABLED",
            "exit_code": 8,
            "phase1_training_complete": False,
            "technical_only": False,
            "formal_performance_claim": False,
            "claim": "PHASE1_SOURCE_ONLY_TRAINING_RECEIPT",
        },
    )
    clean = _write_clean_v4_authority(
        root
        / "runs"
        / "phase1_clic_postfreeze_20260812_v4"
        / CLIC_CANDIDATE
        / "source_clean_proxy.npz",
        wisig_sha256=wisig_sha256,
    )
    return {
        "checkpoint": checkpoint,
        "completion": completion,
        "clean": clean,
        "wisig": wisig,
    }


def _write_iq_only_package(root: Path) -> Path:
    """Build a full 3,120-row predictor-safe package with no truth fields."""

    package = root / "iq_only_package"
    package.mkdir(parents=True)
    scenes = np.repeat(
        np.asarray(FORMAL_LEO_WEAK_SCENARIOS, dtype="U32"), ROWS_PER_SCENE
    )
    received_iq = np.arange(TOTAL_ROWS * 2 * 4, dtype=np.float32).reshape(
        TOTAL_ROWS, 2, 4
    )
    iq_hashes = np.asarray(
        [
            hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
            for row in received_iq
        ],
        dtype="U64",
    )
    lineage_sha = _sha_text("task2-iq-only-lineage")
    opaque_tokens = np.asarray(
        [
            TARGET.opaque_token(
                lineage_sha256=lineage_sha,
                scene=str(scenes[index]),
                ordinal=index,
                received_iq_sha256=str(iq_hashes[index]),
            )
            for index in range(TOTAL_ROWS)
        ],
        dtype="U64",
    )
    data_path = package / "received_iq.npz"
    np.savez(
        data_path,
        received_iq=received_iq,
        opaque_tokens=opaque_tokens,
        scenes=scenes,
        received_iq_sha256=iq_hashes,
    )
    row_count_by_scene = {
        scene: int(np.sum(scenes == scene)) for scene in FORMAL_LEO_WEAK_SCENARIOS
    }
    manifest_base: dict[str, Any] = {
        "schema": "cvs.phase1.clic_target_iq_only_package.v1",
        "capsule_id": "task2-capsule",
        "split_id": "task2-split",
        "protocol_schema": "p2_min_v1",
        "query_truth_included": False,
        "query_role_included": False,
        "single_leo_observation": True,
        "scenes": list(FORMAL_LEO_WEAK_SCENARIOS),
        "scene_physical_id_sha256": {
            scene: _sha_text(f"physical-{scene}")
            for scene in FORMAL_LEO_WEAK_SCENARIOS
        },
        "scene_physical_id_pairwise_disjoint": True,
        "physical_sample_scenario_assignment_sha256": _sha_text(
            "scene-assignment"
        ),
        "scene_seed_assignment_sha256": _sha_text("seed-assignment"),
        "target_receiver_set_sha256": _sha_text("target-receivers"),
        "target_registered_tx_set_sha256": _sha_text("target-registered-set"),
        "target_unknown_tx_set_sha256": _sha_text("target-unknown-set"),
        "target_day_set_sha256": _sha_text("target-days"),
        "merged_physical_sample_ids_sha256": _sha_text("merged-physical-ids"),
        "received_iq_sha256_root": TARGET.canonical_sha256(iq_hashes.tolist()),
        "received_iq_data_sha256": sha256_file(data_path),
        "row_count": TOTAL_ROWS,
        "row_count_by_scene": row_count_by_scene,
        "opaque_token_sha256": TARGET.canonical_sha256(opaque_tokens.tolist()),
        "lineage_sha256": lineage_sha,
        "validator_receipt_sha256": _sha_text("validator-receipt"),
        "cache_set_manifest_sha256": _sha_text("cache-set-manifest"),
        "known_test_config_raw_sha256": _sha_text("known-config-raw-not-opened"),
        "known_test_config_normalized_sha256": _sha_text(
            "known-config-normalized-not-opened"
        ),
    }
    manifest = dict(manifest_base, package_sha256=TARGET.canonical_sha256(manifest_base))
    _write_json(package / "manifest.json", manifest)
    return package


def _seal_train_config(adv: Any, paths: dict[str, Path], output: Path) -> Path:
    return adv.seal_adv3b02_train_data_config(
        checkpoint_path=paths["checkpoint"],
        completion_receipt_path=paths["completion"],
        clean_v4_npz_path=paths["clean"],
        output_path=output,
    )


def _write_sealed_inputs(root: Path, adv: Any) -> dict[str, Path]:
    paths = _write_training_authorities(root)
    train_config = _seal_train_config(
        adv,
        paths,
        paths["checkpoint"].parent / "phase1_adv3b02_train_data_config.json",
    )
    return {
        **paths,
        "train_config": train_config,
        "package": _write_iq_only_package(root),
    }


def _install_real_ssdg_model_state(
    paths: dict[str, Path], *, input_len: int
) -> None:
    """Build the state through the same SSDG training architecture entrypoints."""

    import torch
    from SSDG import train_ssdg

    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=False
    )
    parser = train_ssdg.build_arg_parser()
    parsed = parser.parse_args(
        ["--output_dir", str(paths["checkpoint"].parent / "model-build")]
    )
    for key, value in checkpoint["args"].items():
        setattr(parsed, key, value)
    model_args = train_ssdg.merge_checkpoint_args(
        checkpoint,
        parsed,
        input_len=input_len,
        num_domains=2,
    )
    model_args = train_ssdg._apply_model_cli_args(model_args, parsed)
    model = train_ssdg.build_baseline_model(model_args, torch.device("cpu"))
    checkpoint["model"] = model.state_dict()
    torch.save(checkpoint, paths["checkpoint"])

    completion = json.loads(paths["completion"].read_text(encoding="utf-8"))
    completion["selected_checkpoint_sha256"] = sha256_file(paths["checkpoint"])
    _write_json(paths["completion"], completion)


class _FakeADVRuntime:
    """A narrow model-boundary double; package and seal logic stay real."""

    def __init__(self, *, mutate_after_first_forward: Path | None = None) -> None:
        self.source_class_order = list(SOURCE_LOCAL4)
        self.source_class_order_sha256 = TARGET.canonical_sha256(
            list(SOURCE_LOCAL4)
        )
        self.forward_inputs: list[np.ndarray] = []
        self.forward_scenes: list[str] = []
        self._mutation_path = mutate_after_first_forward

    def forward_once(self, received_iq: Any, *, scene: str) -> dict[str, Any]:
        values = np.asarray(received_iq, dtype=np.float32)
        self.forward_inputs.append(np.array(values, copy=True))
        self.forward_scenes.append(scene)
        if len(self.forward_inputs) == 1 and self._mutation_path is not None:
            self._mutation_path.write_bytes(b"post-verify package mutation")
        return {
            "tx_logits": np.asarray([0.1, 0.2, 0.9, 0.3], dtype=np.float32)
        }


def _publish(adv: Any, paths: dict[str, Path], output: Path) -> Path:
    return adv.publish_adv3b02_target_prediction(
        checkpoint_path=paths["checkpoint"],
        completion_receipt_path=paths["completion"],
        train_config_manifest_path=paths["train_config"],
        iq_only_package_path=paths["package"],
        output_path=output,
    )


def test_task2_seals_train_config_only_from_checkpoint_completion_and_clean_v4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: caller-authored config, feature reads, or weak source binding."""

    adv = _load_task2_module()
    paths = _write_training_authorities(tmp_path)
    forbidden_members = {"z_id", "features", "tx_logits"}
    forbidden_reads: list[str] = []
    original_getitem = np.lib.npyio.NpzFile.__getitem__

    def metadata_only_getitem(archive: Any, key: str) -> Any:
        if str(key) in forbidden_members:
            forbidden_reads.append(str(key))
            raise AssertionError(f"clean-v4 feature member was read: {key}")
        return original_getitem(archive, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", metadata_only_getitem)
    output = tmp_path / "phase1_adv3b02_train_data_config.json"
    result = _seal_train_config(adv, paths, output)

    assert result == output.resolve()
    assert forbidden_reads == []
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "cvs.phase1.adv3b02_train_data_config.v1"
    assert payload["immutable"] is True
    assert payload["checkpoint_sha256"] == sha256_file(paths["checkpoint"])
    assert payload["completion_receipt_sha256"] == sha256_file(paths["completion"])
    assert payload["clean_v4_npz_sha256"] == sha256_file(paths["clean"])
    assert payload["wisig_pkl_path"] == str(paths["wisig"].resolve())
    assert payload["wisig_pkl_sha256"] == sha256_file(paths["wisig"])
    assert payload["fold_index"] == 1
    assert payload["clic_clean_arm"] == CLIC_ARM
    assert payload["source_class_order"] == list(SOURCE_LOCAL4)
    assert payload["source_class_order_sha256"] == TARGET.canonical_sha256(
        list(SOURCE_LOCAL4)
    )
    assert payload["baseline_terminal_status"] == "NON_PROMOTABLE_P0_DISABLED"
    assert payload["baseline_exit_code"] == 8
    assert payload["baseline_promotion_ready"] is False
    assert payload["formal_performance_claim"] is False
    expected_binding = _expected_physical_axis_binding(paths)
    assert payload["physical_axis_binding"] == expected_binding
    assert payload["physical_axis_binding_sha256"] == TARGET.canonical_sha256(
        expected_binding
    )
    expected = TARGET.normalize_train_data_config(
        _matching_clic_normalized_train_config(
            wisig_sha256=sha256_file(paths["wisig"])
        )
    )
    assert TARGET.normalize_train_data_config(payload["normalized"]) == expected
    assert payload["normalized_sha256"] == TARGET.canonical_sha256(expected)
    assert TARGET.normalize_train_data_config(
        _matching_clic_normalized_train_config(
            wisig_sha256=sha256_file(paths["wisig"])
        )
    ) == expected
    assert "source_split_receipt" not in payload
    assert "tx_partition_receipt" not in payload

    with pytest.raises(Exception, match="immutable|already exists|overwrite"):
        _seal_train_config(adv, paths, output)


def test_task2_sealer_accepts_empty_optional_top_level_wisig_shas(
    tmp_path: Path,
) -> None:
    """Real ADV artifacts bind WiSig through the self-sealed split receipt."""

    import torch

    adv = _load_task2_module()
    paths = _write_training_authorities(tmp_path)
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=False
    )
    checkpoint["args"]["wisig_pkl_sha256"] = ""
    torch.save(checkpoint, paths["checkpoint"])
    completion = json.loads(paths["completion"].read_text(encoding="utf-8"))
    completion["wisig_pkl_sha256"] = ""
    completion["selected_checkpoint_sha256"] = sha256_file(paths["checkpoint"])
    _write_json(paths["completion"], completion)
    output = tmp_path / "real-artifact-style.train_config.json"

    assert _seal_train_config(adv, paths, output) == output.resolve()
    sealed = json.loads(output.read_text(encoding="utf-8"))
    assert sealed["wisig_pkl_sha256"] == sha256_file(paths["wisig"])
    assert sealed["normalized"]["dataset_provenance"]["wisig_pkl_sha256"] == sha256_file(
        paths["wisig"]
    )


@pytest.mark.parametrize(
    "drift",
    ("checkpoint-top-level", "completion-top-level", "nested-receipt"),
)
def test_task2_sealer_rejects_nonempty_or_nested_wisig_sha_drift(
    tmp_path: Path, drift: str
) -> None:
    """Optional copies may be empty, but nonempty copies and the receipt must agree."""

    import torch

    adv = _load_task2_module()
    paths = _write_training_authorities(tmp_path)
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=False
    )
    completion = json.loads(paths["completion"].read_text(encoding="utf-8"))
    bad_sha = "0" * 64
    if drift == "checkpoint-top-level":
        checkpoint["args"]["wisig_pkl_sha256"] = bad_sha
    elif drift == "completion-top-level":
        completion["wisig_pkl_sha256"] = bad_sha
    else:
        checkpoint["args"]["wisig_pkl_sha256"] = ""
        completion["wisig_pkl_sha256"] = ""
        for receipt in (
            checkpoint["split_info"]["source_split_receipt"],
            completion["source_split_receipt"],
        ):
            receipt["wisig_pkl_sha256"] = bad_sha
            receipt.pop("split_manifest_sha256")
            receipt["split_manifest_sha256"] = TARGET.canonical_sha256(receipt)
    torch.save(checkpoint, paths["checkpoint"])
    completion["selected_checkpoint_sha256"] = sha256_file(paths["checkpoint"])
    _write_json(paths["completion"], completion)
    output = tmp_path / f"{drift}.train_config.json"

    with pytest.raises(Exception, match="WiSig|wisig|SHA|drift"):
        _seal_train_config(adv, paths, output)
    assert not output.exists()


@pytest.mark.parametrize("axis", ("receiver", "day"))
def test_task2_sealer_rejects_same_count_wrong_wisig_physical_axis(
    tmp_path: Path, axis: str
) -> None:
    """Break caught: index counts pass while WiSig and clean physical labels differ."""

    adv = _load_task2_module()
    receiver_ids = SOURCE_RECEIVER_IDS
    day_ids = SOURCE_DAY_IDS
    if axis == "receiver":
        receiver_ids = (*SOURCE_RECEIVER_IDS[:-1], "foreign-source-rx")
    else:
        day_ids = (SOURCE_DAY_IDS[0], "2099_12_31")
    paths = _write_training_authorities(
        tmp_path,
        dataset_source_receiver_ids=receiver_ids,
        dataset_source_day_ids=day_ids,
    )
    output = tmp_path / f"wrong-{axis}-physical-axis.json"

    with pytest.raises(Exception, match="WiSig|physical|axis|receiver|day|drift"):
        _seal_train_config(adv, paths, output)
    assert not output.exists()


def test_task2_sealer_rejects_wisig_dataset_toctou_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: the derived WiSig authority changes during metadata reopen."""

    adv = _load_task2_module()
    paths = _write_training_authorities(tmp_path)
    import dataset_wisig

    original_loader = dataset_wisig.load_wisig_compact_pkl

    def mutate_after_reopen(path: str) -> dict[str, Any]:
        payload = original_loader(path)
        Path(path).write_bytes(b"WiSig TOCTOU mutation")
        return payload

    monkeypatch.setattr(dataset_wisig, "load_wisig_compact_pkl", mutate_after_reopen)
    output = tmp_path / "wisig-toctou.json"

    with pytest.raises(Exception, match="WiSig|changed|SHA|sha|TOCTOU"):
        _seal_train_config(adv, paths, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("terminal_status", "COMPLETE"),
        ("exit_code", 0),
        ("exit_code", 7),
        ("exit_code", 9),
        ("phase1_training_complete", True),
        ("technical_only", True),
        ("formal_performance_claim", True),
        ("claim", "NO_PERFORMANCE_RESULT"),
    ),
    ids=(
        "complete-status",
        "zero-exit",
        "other-nonzero-exit-low",
        "other-nonzero-exit-high",
        "training-complete-true",
        "technical-only-true",
        "performance-claim-true",
        "claim-drift",
    ),
)
def test_task2_rejects_non_exact_baseline_terminal_tuple(
    tmp_path: Path, field: str, value: Any
) -> None:
    """Break caught: arbitrary failure/success states are treated as the frozen baseline."""

    adv = _load_task2_module()
    paths = _write_training_authorities(tmp_path)
    receipt = json.loads(paths["completion"].read_text(encoding="utf-8"))
    receipt[field] = value
    _write_json(paths["completion"], receipt)
    output = tmp_path / f"terminal-drift-{field}-{value}.json"

    with pytest.raises(
        Exception, match="baseline|terminal|completion|status|exit|claim|tuple|drift"
    ):
        _seal_train_config(adv, paths, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "missing_field",
    (
        "terminal_status",
        "exit_code",
        "phase1_training_complete",
        "technical_only",
        "formal_performance_claim",
        "claim",
    ),
)
def test_task2_rejects_missing_baseline_terminal_tuple_field(
    tmp_path: Path, missing_field: str
) -> None:
    """Break caught: an incomplete baseline gate receipt is silently accepted."""

    adv = _load_task2_module()
    paths = _write_training_authorities(tmp_path)
    receipt = json.loads(paths["completion"].read_text(encoding="utf-8"))
    receipt.pop(missing_field)
    _write_json(paths["completion"], receipt)
    output = tmp_path / f"terminal-missing-{missing_field}.json"

    with pytest.raises(
        Exception, match="baseline|terminal|completion|status|exit|claim|tuple|missing"
    ):
        _seal_train_config(adv, paths, output)
    assert not output.exists()


@pytest.mark.parametrize("drift", ("checkpoint", "completion", "clean_v4"))
def test_task2_train_config_sealer_rejects_cross_authority_drift_before_output(
    tmp_path: Path, drift: str
) -> None:
    """Break caught: fold/roles/ratios/axis receipts can drift across authorities."""

    adv = _load_task2_module()
    paths = _write_training_authorities(tmp_path)
    output = tmp_path / f"{drift}.train_config.json"
    if drift == "checkpoint":
        import torch

        payload = torch.load(
            paths["checkpoint"], map_location="cpu", weights_only=False
        )
        payload["args"]["labeled_ratio"] = 0.10
        torch.save(payload, paths["checkpoint"])
    elif drift == "completion":
        payload = json.loads(paths["completion"].read_text(encoding="utf-8"))
        payload["source_split_receipt"]["source_receivers"] = ["0", "1"]
        _write_json(paths["completion"], payload)
    else:
        _write_clean_v4_authority(
            paths["clean"],
            manifest_overrides={
                "tx_partition_receipt": {
                    **_tx_partition_receipt(),
                    "source_proxy_unknown_tx": [HELD_TX],
                }
            },
        )

    with pytest.raises(Exception, match="drift|split|ratio|role|axis|receipt|SHA|source"):
        _seal_train_config(adv, paths, output)
    assert not output.exists()


@pytest.mark.parametrize("mutated_input", ("checkpoint", "completion", "clean_v4"))
def test_task2_train_config_sealer_rejects_toctou_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutated_input: str
) -> None:
    """Break caught: an authority byte swap during reopen leaves a sealed config."""

    adv = _load_task2_module()
    paths = _write_training_authorities(tmp_path)
    original = adv._read_verified_clean_v4_metadata_only

    def mutate_after_clean_reopen(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        target = {
            "checkpoint": paths["checkpoint"],
            "completion": paths["completion"],
            "clean_v4": paths["clean"],
        }[mutated_input]
        target.write_bytes(b"TOCTOU mutation")
        return result

    monkeypatch.setattr(
        adv, "_read_verified_clean_v4_metadata_only", mutate_after_clean_reopen
    )
    output = tmp_path / f"{mutated_input}.toctou.json"
    with pytest.raises(Exception, match="changed|SHA|sha|TOCTOU|input"):
        _seal_train_config(adv, paths, output)
    assert not output.exists()


def test_task2_blind_prediction_seals_all_rows_once_without_target_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: the publisher skips/retries rows or leaks target-side identity."""

    adv = _load_task2_module()
    paths = _write_sealed_inputs(tmp_path, adv)
    runtime = _FakeADVRuntime()
    loader_calls: list[tuple[Path, Path, Path]] = []

    def forbidden_source_reopen(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("blind publisher reopened a source authority")

    monkeypatch.setattr(
        adv, "_read_verified_wisig_physical_axes", forbidden_source_reopen
    )
    monkeypatch.setattr(adv, "_validate_clean_against_adv", forbidden_source_reopen)
    monkeypatch.setattr(
        adv._wisig, "load_wisig_compact_pkl", forbidden_source_reopen
    )

    def load_runtime(
        *,
        checkpoint_path: Path,
        completion_receipt_path: Path,
        train_config_manifest_path: Path,
    ):
        loader_calls.append(
            (checkpoint_path, completion_receipt_path, train_config_manifest_path)
        )
        return runtime

    monkeypatch.setattr(adv, "load_verified_adv3b02_runtime", load_runtime)
    output = tmp_path / "prediction.json"
    result = _publish(adv, paths, output)

    assert result == output.resolve()
    assert loader_calls == [
        (
            paths["checkpoint"].resolve(),
            paths["completion"].resolve(),
            paths["train_config"].resolve(),
        )
    ]
    assert len(runtime.forward_inputs) == TOTAL_ROWS
    assert runtime.forward_scenes == list(
        np.repeat(np.asarray(FORMAL_LEO_WEAK_SCENARIOS), ROWS_PER_SCENE)
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "cvs.phase1.adv3b02_target_prediction.v1"
    assert payload["sealed"] is True
    assert payload["truth_sidecar_opened"] is False
    assert payload["row_count"] == TOTAL_ROWS
    assert payload["forward_count"] == TOTAL_ROWS
    assert payload["source_class_order"] == list(SOURCE_LOCAL4)
    assert payload["source_class_order_sha256"] == TARGET.canonical_sha256(
        list(SOURCE_LOCAL4)
    )
    assert payload["baseline_terminal_status"] == "NON_PROMOTABLE_P0_DISABLED"
    assert payload["baseline_exit_code"] == 8
    assert payload["baseline_promotion_ready"] is False
    assert payload["formal_performance_claim"] is False
    sealed_train_config = json.loads(paths["train_config"].read_text(encoding="utf-8"))
    assert payload["checkpoint_sha256"] == sha256_file(paths["checkpoint"])
    assert payload["completion_receipt_sha256"] == sha256_file(paths["completion"])
    assert payload["train_config_manifest_sha256"] == sha256_file(paths["train_config"])
    assert payload["train_config_normalized_sha256"] == sealed_train_config[
        "normalized_sha256"
    ]
    assert payload["train_config_physical_axis_binding_sha256"] == (
        sealed_train_config["physical_axis_binding_sha256"]
    )
    assert payload["package_manifest_sha256"] == sha256_file(
        paths["package"] / "manifest.json"
    )
    assert payload["received_iq_data_sha256"] == sha256_file(
        paths["package"] / "received_iq.npz"
    )
    assert payload["package_sha256"] == json.loads(
        (paths["package"] / "manifest.json").read_text(encoding="utf-8")
    )["package_sha256"]
    assert payload["input_artifact_sha256"] == {
        "checkpoint": payload["checkpoint_sha256"],
        "completion_receipt": payload["completion_receipt_sha256"],
        "train_config_manifest": payload["train_config_manifest_sha256"],
        "iq_only_package_manifest": payload["package_manifest_sha256"],
        "received_iq_data": payload["received_iq_data_sha256"],
    }
    assert payload["target_fit_rows"] == 0
    assert payload["target_update_rows"] == 0
    assert payload["target_retry_count"] == 0
    assert payload["target_selection_count"] == 0
    assert payload["target_selection_feedback"] is False
    assert len(payload["rows"]) == TOTAL_ROWS
    assert all(
        set(row)
        == {"opaque_token", "scene", "received_iq_sha256", "predicted_index"}
        for row in payload["rows"]
    )
    assert {int(row["predicted_index"]) for row in payload["rows"]} == {2}
    assert all(
        not {"truth", "role", "tx_id", "rx_id", "day_id"} & set(row)
        for row in payload["rows"]
    )
    raw = output.read_text(encoding="utf-8").lower()
    for forbidden in (
        "known_test_config_manifest_path",
        "adv3b02_target_known_reference",
        "target-rx",
    ):
        assert forbidden not in raw

    with pytest.raises(Exception, match="immutable|already exists|overwrite"):
        _publish(adv, paths, output)


def test_task2_publisher_rejects_recanonicalized_physical_axis_binding_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: publisher trusts rehashed binding that contradicts normalized IDs."""

    adv = _load_task2_module()
    paths = _write_sealed_inputs(tmp_path, adv)
    config = json.loads(paths["train_config"].read_text(encoding="utf-8"))
    binding = _expected_physical_axis_binding(paths)
    binding["source_receiver_index_to_physical"][0]["physical_id"] = (
        "foreign-source-rx"
    )
    binding["source_receiver_index_to_physical_sha256"] = TARGET.canonical_sha256(
        binding["source_receiver_index_to_physical"]
    )
    config["wisig_pkl_path"] = str(paths["wisig"].resolve())
    config["wisig_pkl_sha256"] = sha256_file(paths["wisig"])
    config["physical_axis_binding"] = binding
    config["physical_axis_binding_sha256"] = TARGET.canonical_sha256(binding)
    _write_json(paths["train_config"], config)

    runtime = _FakeADVRuntime()
    monkeypatch.setattr(
        adv,
        "load_verified_adv3b02_runtime",
        lambda **_kwargs: runtime,
    )
    output = tmp_path / "binding-drift-prediction.json"

    with pytest.raises(Exception, match="physical|axis|binding|receiver|normalized|drift"):
        _publish(adv, paths, output)
    assert runtime.forward_inputs == []
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation", ("checkpoint", "completion", "train_config", "package")
)
def test_task2_rejects_any_prediction_input_mutation_before_row_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Break caught: a byte-swapped required input reaches even one target forward."""

    adv = _load_task2_module()
    paths = _write_sealed_inputs(tmp_path, adv)
    runtime = _FakeADVRuntime()
    monkeypatch.setattr(
        adv,
        "load_verified_adv3b02_runtime",
        lambda **_kwargs: runtime,
    )
    if mutation == "package":
        (paths["package"] / "received_iq.npz").write_bytes(b"mutated package bytes")
    else:
        paths[mutation].write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        Exception, match="SHA|sha|sealed|checkpoint|completion|config|package|invalid"
    ):
        _publish(adv, paths, tmp_path / f"{mutation}.json")
    assert runtime.forward_inputs == []


def test_task2_rejects_post_verify_mutation_without_sealing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: TOCTOU mutation after input verification still yields evidence."""

    adv = _load_task2_module()
    paths = _write_sealed_inputs(tmp_path, adv)
    runtime = _FakeADVRuntime(
        mutate_after_first_forward=paths["package"] / "received_iq.npz"
    )
    monkeypatch.setattr(
        adv,
        "load_verified_adv3b02_runtime",
        lambda **_kwargs: runtime,
    )
    output = tmp_path / "must_not_seal.json"

    with pytest.raises(Exception, match="changed|SHA|sha|TOCTOU|package"):
        _publish(adv, paths, output)
    # A fail-closed publisher may detect the swap immediately before the next
    # row or at the final sealing recheck; either way no mutable output may be
    # emitted.
    assert len(runtime.forward_inputs) >= 1
    assert not output.exists()


def test_task2_safe_torch_bridge_avoids_legacy_numpy_entrypoints() -> None:
    """Break caught: a real forward crosses Torch/NumPy through forbidden APIs."""

    adv = _load_task2_module()
    import torch

    original_from_numpy = torch.from_numpy
    original_tensor_numpy = torch.Tensor.numpy

    def forbidden(*_args: Any, **_kwargs: Any):
        raise AssertionError("legacy Torch/NumPy bridge was used")

    try:
        torch.from_numpy = forbidden
        torch.Tensor.numpy = forbidden
        result = adv.safe_received_iq_tensor(
            np.zeros((2, 4), dtype=np.float32), input_len=4
        )
    finally:
        torch.from_numpy = original_from_numpy
        torch.Tensor.numpy = original_tensor_numpy
    assert tuple(result.shape) == (1, 2, 4)
    assert result.dtype == torch.float32


def test_task2_real_ssdg_runtime_reconstructs_and_forwards_without_query_inputs(
    tmp_path: Path,
) -> None:
    """Break caught: the strict loader passes doubles but not a real SSDG state."""

    adv = _load_task2_module()
    input_len = 256
    paths = _write_training_authorities(tmp_path, input_len=input_len)
    _install_real_ssdg_model_state(paths, input_len=input_len)
    train_config = _seal_train_config(
        adv,
        paths,
        paths["checkpoint"].parent / "phase1_adv3b02_train_data_config.json",
    )

    runtime = adv.load_verified_adv3b02_runtime(
        checkpoint_path=paths["checkpoint"],
        completion_receipt_path=paths["completion"],
        train_config_manifest_path=train_config,
    )
    output = runtime.forward_once(
        np.zeros((2, input_len), dtype=np.float32),
        scene=FORMAL_LEO_WEAK_SCENARIOS[0],
    )

    assert runtime.source_class_order == list(SOURCE_LOCAL4)
    assert set(output) == {"tx_logits"}
    logits = np.asarray(output["tx_logits"], dtype=np.float64)
    assert logits.shape == (4,)
    assert np.isfinite(logits).all()
    assert not any(
        marker in path.name.lower()
        for path in tmp_path.rglob("*")
        for marker in ("truth", "known", "reference", "query")
    )


def test_task2_cli_keeps_sealer_source_only_and_prediction_blind() -> None:
    """Break caught: truth/config/reference paths become CLI inputs."""

    adv = _load_task2_module()
    parser = adv.build_arg_parser()
    sealed = parser.parse_args(
        [
            "--seal-train-data-config",
            "--checkpoint",
            "final_ssdg.pth",
            "--completion-receipt-json",
            "phase1_training_completion_receipt.json",
            "--clean-v4-npz",
            "source_clean_proxy.npz",
            "--output",
            "train_config.json",
        ]
    )
    assert sealed.seal_train_data_config is True
    assert sealed.publish_target_prediction is False
    prediction = parser.parse_args(
        [
            "--publish-target-prediction",
            "--checkpoint",
            "final_ssdg.pth",
            "--completion-receipt-json",
            "phase1_training_completion_receipt.json",
            "--train-config-manifest",
            "train_config.json",
            "--iq-only-package",
            "package",
            "--output",
            "prediction.json",
        ]
    )
    assert prediction.publish_target_prediction is True
    assert {
        key
        for key in vars(prediction)
        if any(
            word in key.lower()
            for word in ("truth", "known_test", "reference", "metric")
        )
    } == set()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--publish-target-prediction", "--truth-sidecar", "truth.json"]
        )
