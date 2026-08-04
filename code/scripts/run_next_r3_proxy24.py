#!/usr/bin/env python3
"""Run the frozen NEXT-R3 24-row proxy with a truth-free prediction phase.

This entry is intentionally a thin, immutable runner around the already
frozen NEXT-R3 matrix/runtime/artifact/scorer modules.  It accepts external
received-IQ, Phase-1 cell *metadata*, checkpoint, tap and D106 RDCE-wire
files.  Every pre-ReLU/canonical row consumed by prediction is generated from
the passed received-IQ through the pinned checkpoint bridge; no externally
stored feature tensor is accepted as a prediction input.  No file is
overwritten.  A missing or incomplete real archive is reported as
``MISSING_REAL_INPUT_ARTIFACTS``; the runner never substitutes synthetic
cells or a historical best result.

``predict`` never opens a truth catalog.  ``score`` validates the complete
prediction, manifest and completion receipts first and only then opens the
opaque truth mapping.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_ROOT.parent
for candidate in (str(SCRIPT_ROOT), str(CODE_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from run_d106_rcmr_g0_one_shot import _predecessor_locks  # noqa: E402
from cvsrffi import stage2_d106_rdce_asset as d106_asset  # noqa: E402
from cvsrffi import stage2_d106_rdce_runtime as d106_runtime  # noqa: E402
from cvsrffi import stage2_next_r3_artifact as artifact  # noqa: E402
from cvsrffi import stage2_next_r3_matrix as matrix  # noqa: E402
from cvsrffi import stage2_next_r3_rdce_tsl_runtime as runtime  # noqa: E402
from cvsrffi import stage2_next_r3_score as scorer  # noqa: E402
from cvsrffi import stage2_next_r3_tsl160 as tsl  # noqa: E402
from cvsrffi import stage2_zid_student_t_qknn as qknn  # noqa: E402
from cvsrffi.stage2_lpo_rc_qknn import TypedValidatedOnceP2SplitHandle  # noqa: E402


RUNNER_SCHEMA = "cvs.stage2.next_r3.proxy24.runner.v1"
COMPLETION_SCHEMA = "cvs.stage2.next_r3.proxy24.completion.v1"
MANIFEST_SCHEMA = "cvs.stage2.next_r3.proxy24.manifest.v1"
RESOURCE_SCHEMA = "cvs.stage2.next_r3.proxy24.resource.v1"
SMOKE_SCHEMA = "cvs.stage2.next_r3.proxy24.real_checkpoint_smoke.v1"
BRIDGE_FEATURE_SCHEMA = "cvs.stage2.next_r3.proxy24.bridge_feature_binding.v1"
PREDICTOR_PACKAGE_SCHEMA = "cvs.stage2.next_r3.proxy24.predictor_package.v2"
PREPARE_SCHEMA = "cvs.stage2.next_r3.proxy24.prepare.v2"
QUERY_ORDER_RULE = "sha256_outer_key_physical_id_v1"
MISSING_PREFIX = "MISSING_REAL_INPUT_ARTIFACTS"

RECEIVED_MEMBERS = (
    "received_iq",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
    "observation_ids",
)
CELL_MEMBERS = ("zid160", "receiver_ids", "class_ids", "physical_ids")
CELL_TAP_MEMBERS = (
    "pre_relu",
    "z_dom",
    "tx_labels",
    "receiver_ids",
    "day_ids",
    "physical_ids",
    "scenario_names",
    "observation_ids",
)
ROW_COUNT = 588
PHYSICAL_PER_CELL = 14


class NextR3Proxy24Error(ValueError):
    """The frozen NEXT-R3 runner closure did not hold."""


class MissingRealInputArtifacts(NextR3Proxy24Error):
    """Required real assets or Phase-1 per-physical cells are unavailable."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _plain(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha(path.read_bytes())


def _require_sha(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or text != text.lower() or any(char not in "0123456789abcdef" for char in text):
        raise NextR3Proxy24Error(f"{name} must be a lowercase SHA256")
    return text


def _write_json_new(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise NextR3Proxy24Error(f"output overwrite refused: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(_plain(value), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _new_root(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if path != resolved or not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise NextR3Proxy24Error("run root must be a new absolute child of an existing directory")
    path.mkdir()
    (path / "rows").mkdir()
    return path


def _require_file(path: Path | None, expected_sha256: str | None, name: str) -> bytes:
    if path is None or expected_sha256 is None:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: {name} path/SHA256 is required")
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: {name} is not a regular file")
    expected = _require_sha(expected_sha256, f"{name} SHA256")
    payload = path.read_bytes()
    if _sha(payload) != expected:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: {name} SHA256 mismatch")
    return payload


def _load_npz(
    payload: bytes,
    expected_members: Sequence[str],
    name: str,
    *,
    selected_members: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
            if tuple(loaded.files) != tuple(expected_members):
                raise MissingRealInputArtifacts(
                    f"{MISSING_PREFIX}: {name} members must be {tuple(expected_members)}"
                )
            selected = tuple(expected_members if selected_members is None else selected_members)
            if any(member not in expected_members for member in selected):
                raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: {name} selected-member drift")
            arrays = {member: np.asarray(loaded[member]) for member in selected}
    except MissingRealInputArtifacts:
        raise
    except Exception as error:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: {name} is not a no-pickle NPZ") from error
    if any(array.dtype.hasobject for array in arrays.values()):
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: {name} contains an object array")
    return arrays


def _strings(value: np.ndarray, *, name: str, count: int, unique: bool = False) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != count or array.dtype.kind not in "US":
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: {name} must be a string[{count}] array")
    result = tuple(str(item) for item in array.tolist())
    if any(not item for item in result) or (unique and len(set(result)) != len(result)):
        suffix = "/duplicate IDs" if unique else ""
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: {name} contains blank{suffix}")
    return result


@dataclass(frozen=True, slots=True)
class SourceRows:
    received_iq: np.ndarray
    receiver_ids: tuple[str, ...]
    day_ids: tuple[str, ...]
    physical_ids: tuple[str, ...]
    scenario_names: tuple[str, ...]
    observation_ids: tuple[str, ...]
    receiver_registry: tuple[str, ...]
    received_iq_sha256: str


@dataclass(frozen=True, slots=True)
class CellRows:
    receiver_ids: tuple[str, ...]
    class_ids: tuple[str, ...]
    physical_ids: tuple[str, ...]


def _load_real_rows(args: argparse.Namespace) -> tuple[SourceRows, CellRows, Mapping[str, Any]]:
    received_bytes = _require_file(args.received_iq, args.received_iq_sha256, "received-IQ archive")
    cell_bytes = _require_file(args.phase1_cells, args.phase1_cells_sha256, "Phase-1 cell archive")
    received = _load_npz(received_bytes, RECEIVED_MEMBERS, "received-IQ archive")
    received_iq = np.asarray(received["received_iq"])
    if received_iq.dtype != np.dtype("<f4") or received_iq.ndim != 3 or received_iq.shape[0] != ROW_COUNT or received_iq.shape[1] != 2:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: received_iq must be little-endian float32 [588,2,T]")
    receiver_ids = _strings(received["receiver_ids"], name="received.receiver_ids", count=ROW_COUNT)
    day_ids = _strings(received["day_ids"], name="received.day_ids", count=ROW_COUNT)
    physical_ids = _strings(received["physical_ids"], name="received.physical_ids", count=ROW_COUNT, unique=True)
    scenario_names = _strings(received["scenario_names"], name="received.scenario_names", count=ROW_COUNT)
    observation_ids = _strings(received["observation_ids"], name="received.observation_ids", count=ROW_COUNT, unique=True)

    # External Phase-1 cell archives may contribute only sealed identity/class
    # metadata.  Their precomputed ``zid160``/``pre_relu`` arrays are never
    # materialized or used by prediction; bridge output below supplies every
    # consumed feature row.
    try:
        cell_arrays = _load_npz(
            cell_bytes,
            CELL_MEMBERS,
            "Phase-1 cell archive",
            selected_members=("receiver_ids", "class_ids", "physical_ids"),
        )
        cell_receiver = _strings(cell_arrays["receiver_ids"], name="cells.receiver_ids", count=ROW_COUNT)
        cell_class = _strings(cell_arrays["class_ids"], name="cells.class_ids", count=ROW_COUNT)
        cell_physical = _strings(cell_arrays["physical_ids"], name="cells.physical_ids", count=ROW_COUNT, unique=True)
    except MissingRealInputArtifacts:
        try:
            tap = _load_npz(
                cell_bytes,
                CELL_TAP_MEMBERS,
                "Phase-1 cell archive",
                selected_members=("receiver_ids", "tx_labels", "physical_ids"),
            )
            cell_receiver = _strings(tap["receiver_ids"], name="cells.receiver_ids", count=ROW_COUNT)
            cell_class = _strings(tap["tx_labels"], name="cells.class_ids", count=ROW_COUNT)
            cell_physical = _strings(tap["physical_ids"], name="cells.physical_ids", count=ROW_COUNT, unique=True)
        except MissingRealInputArtifacts as error:
            raise MissingRealInputArtifacts(
                f"{MISSING_PREFIX}: complete Phase-1 per-physical normalized-ReLU cells are required"
            ) from error
    if len(cell_physical) != ROW_COUNT or cell_receiver != receiver_ids or cell_physical != physical_ids:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: Phase-1 cell metadata must align with received-IQ physical rows")
    if any(len([1 for r, c in zip(cell_receiver, cell_class, strict=True) if r == receiver and c == class_id]) != PHYSICAL_PER_CELL for receiver in sorted(set(cell_receiver)) for class_id in sorted(set(cell_class))):
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: Phase-1 cells lack a complete per-physical receiver/class cell")
    if len(set(receiver_ids)) != 7 or len(set(cell_class)) != 6:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: Phase-1 cell metadata must be a complete 7x6 grid")
    rows = SourceRows(
        received_iq=np.ascontiguousarray(received_iq, dtype=np.float32),
        receiver_ids=receiver_ids,
        day_ids=day_ids,
        physical_ids=physical_ids,
        scenario_names=scenario_names,
        observation_ids=observation_ids,
        receiver_registry=tuple(sorted(set(receiver_ids))),
        received_iq_sha256=_sha(received_bytes),
    )
    cells = CellRows(
        receiver_ids=cell_receiver,
        class_ids=cell_class,
        physical_ids=cell_physical,
    )
    return rows, cells, {"received_iq_sha256": _sha(received_bytes), "phase1_cells_sha256": _sha(cell_bytes)}


def _load_received_rows(args: argparse.Namespace) -> SourceRows:
    """Load only the sealed received-IQ capsule for the predictor phase."""

    received_bytes = _require_file(args.received_iq, args.received_iq_sha256, "received-IQ archive")
    received = _load_npz(received_bytes, RECEIVED_MEMBERS, "received-IQ archive")
    received_iq = np.asarray(received["received_iq"])
    if (
        received_iq.dtype != np.dtype("<f4")
        or received_iq.ndim != 3
        or received_iq.shape[0] != ROW_COUNT
        or received_iq.shape[1] != 2
        or not np.isfinite(received_iq).all()
    ):
        raise MissingRealInputArtifacts(
            f"{MISSING_PREFIX}: received_iq must be finite little-endian float32 [588,2,T]"
        )
    physical_ids = _strings(
        received["physical_ids"], name="received.physical_ids", count=ROW_COUNT, unique=True
    )
    return SourceRows(
        received_iq=np.ascontiguousarray(received_iq, dtype=np.float32),
        receiver_ids=_strings(received["receiver_ids"], name="received.receiver_ids", count=ROW_COUNT),
        day_ids=_strings(received["day_ids"], name="received.day_ids", count=ROW_COUNT),
        physical_ids=physical_ids,
        scenario_names=_strings(received["scenario_names"], name="received.scenario_names", count=ROW_COUNT),
        observation_ids=_strings(
            received["observation_ids"], name="received.observation_ids", count=ROW_COUNT, unique=True
        ),
        receiver_registry=tuple(
            sorted(
                set(
                    _strings(received["receiver_ids"], name="received.receiver_ids", count=ROW_COUNT)
                )
            )
        ),
        received_iq_sha256=_sha(received_bytes),
    )


def _string_tuple(value: Any, *, name: str, count: int, unique: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise NextR3Proxy24Error(f"truth-free split {name} must contain {count} strings")
    result = tuple(str(item) for item in value)
    if any(type(item) is not str or not item for item in value) or (unique and len(set(result)) != len(result)):
        raise NextR3Proxy24Error(f"truth-free split {name} contains invalid IDs")
    return result


def _require_split_id_pairs(
    value: Mapping[str, Any],
    *,
    physical_name: str,
    observation_name: str,
    count: int,
    physical_to_observation: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    physical_ids = _string_tuple(value.get(physical_name), name=physical_name, count=count)
    observation_ids = _string_tuple(value.get(observation_name), name=observation_name, count=count)
    if any(physical_to_observation.get(physical_id) != observation_id for physical_id, observation_id in zip(physical_ids, observation_ids, strict=True)):
        raise NextR3Proxy24Error("truth-free split physical/observation identity drift")
    return physical_ids, observation_ids


class BridgeFeatureCache:
    """One received-IQ/checkpoint-derived cache shared by all 24 rows."""

    def __init__(self, bridge: Any, rows: SourceRows, checkpoint_sha256: str) -> None:
        if _require_sha(getattr(bridge, "checkpoint_sha256", None), "bridge checkpoint SHA256") != checkpoint_sha256:
            raise NextR3Proxy24Error("checkpoint bridge SHA256 drift")
        bridge_rows = getattr(bridge, "rows", None)
        if (
            bridge_rows is not rows
            and (
                tuple(getattr(bridge_rows, "physical_ids", ())) != rows.physical_ids
                or tuple(getattr(bridge_rows, "observation_ids", ())) != rows.observation_ids
                or not np.array_equal(np.asarray(getattr(bridge_rows, "received_iq", ())), rows.received_iq)
            )
        ):
            raise NextR3Proxy24Error("received-IQ/checkpoint bridge input binding drift")
        self._bridge = bridge
        self._rows = rows
        self._checkpoint_sha256 = checkpoint_sha256
        self._index = {physical_id: index for index, physical_id in enumerate(rows.physical_ids)}
        self._values: dict[str, np.ndarray] = {}

    def take(self, physical_ids: Sequence[str]) -> np.ndarray:
        ids = tuple(str(item) for item in physical_ids)
        if not ids or len(set(ids)) != len(ids) or any(item not in self._index for item in ids):
            raise NextR3Proxy24Error("bridge feature request physical-ID drift")
        missing = tuple(item for item in ids if item not in self._values)
        for start in range(0, len(missing), 128):
            batch_ids = missing[start : start + 128]
            indices = tuple(self._index[item] for item in batch_ids)
            try:
                _logits, features = self._bridge.forward_indices(indices)
            except Exception as error:
                raise NextR3Proxy24Error("received-IQ checkpoint bridge forward failed") from error
            value = np.ascontiguousarray(np.asarray(features), dtype=np.float32)
            if value.shape != (len(batch_ids), tsl.Z_DIM) or not np.isfinite(value).all():
                raise NextR3Proxy24Error("received-IQ checkpoint bridge feature contract drift")
            for physical_id, row in zip(batch_ids, value, strict=True):
                frozen = np.array(row, dtype=np.float32, copy=True, order="C")
                frozen.setflags(write=False)
                self._values[physical_id] = frozen
        return np.ascontiguousarray(np.stack(tuple(self._values[item] for item in ids)), dtype=np.float32)

    def receipt(self, physical_ids: Sequence[str]) -> Mapping[str, Any]:
        raw = self.take(physical_ids)
        canonical = tsl.canonical_d106_relu_zid160(raw)
        return {
            "schema": BRIDGE_FEATURE_SCHEMA,
            "feature_source": "received_iq_checkpoint_bridge",
            "external_feature_archive_consumed": False,
            "checkpoint_sha256": self._checkpoint_sha256,
            "received_iq_sha256": self._rows.received_iq_sha256,
            "physical_ids_sha256": _sha(_canonical(list(physical_ids))),
            "bridge_z160_receipt": _array_receipt(raw),
            "canonical_zid160_receipt": _array_receipt(canonical),
        }


def _build_phase1_cells(
    cells: CellRows, feature_cache: BridgeFeatureCache
) -> tuple[tsl.TSL160Phase1Cell, ...]:
    by_key: dict[tuple[str, str], list[int]] = {}
    for index, key in enumerate(zip(cells.receiver_ids, cells.class_ids, strict=True)):
        by_key.setdefault(key, []).append(index)
    if len(by_key) != 42 or any(len(indices) != PHYSICAL_PER_CELL for indices in by_key.values()):
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: full 7x6 Phase-1 per-physical cells are required")
    return tuple(
        tsl.TSL160Phase1Cell(
            receiver_id=receiver,
            class_handle=class_id,
            physical_ids=tuple(cells.physical_ids[index] for index in indices),
            zid160=feature_cache.take(tuple(cells.physical_ids[index] for index in indices)),
        )
        for (receiver, class_id), indices in sorted(by_key.items())
    )


@dataclass(frozen=True, slots=True)
class PreparedPredictorRow:
    row_id: str
    support_physical_ids: tuple[str, ...]
    support_observation_ids: tuple[str, ...]
    support_labels: tuple[str, ...]
    query_physical_ids: tuple[str, ...]
    query_observation_ids: tuple[str, ...]
    prior_key: str

    def support_for(self, class_id: str) -> tuple[str, ...]:
        return tuple(
            physical_id
            for physical_id, label in zip(self.support_physical_ids, self.support_labels, strict=True)
            if label == class_id
        )


@dataclass(frozen=True, slots=True)
class PreparedPredictorPackage:
    class_registry: tuple[str, ...]
    rows_by_id: Mapping[str, PreparedPredictorRow]
    priors_by_key: Mapping[str, tsl.TSL160Phase1Prior]
    pair_bindings: Mapping[str, Mapping[str, Any]]
    received_iq_sha256: str
    checkpoint_sha256: str
    phase1_cells_sha256: str
    capsule_id: str
    split_id: str
    validator_receipt_sha256: str
    sha256: str

    def row(self, row_id: str) -> PreparedPredictorRow:
        try:
            return self.rows_by_id[row_id]
        except KeyError as error:
            raise NextR3Proxy24Error("predictor package row is missing") from error

    def prior(self, key: str) -> tsl.TSL160Phase1Prior:
        try:
            return self.priors_by_key[key]
        except KeyError as error:
            raise NextR3Proxy24Error("predictor package prior is missing") from error


def _outer_key(held_receiver: str, held_class: str) -> str:
    return _sha(f"NEXT-R3-PROXY24|{held_receiver}|{held_class}".encode("utf-8"))[:24]


def _common_query_order(
    held_receiver: str, held_class: str, physical_ids: Sequence[str]
) -> tuple[str, ...]:
    key = _outer_key(held_receiver, held_class)
    return tuple(
        sorted(
            (str(item) for item in physical_ids),
            key=lambda item: _sha(f"{QUERY_ORDER_RULE}|{key}|{item}".encode("utf-8")),
        )
    )


def _metadata_cell_indices(cells: CellRows) -> Mapping[tuple[str, str], tuple[int, ...]]:
    result: dict[tuple[str, str], list[int]] = {}
    for index, key in enumerate(zip(cells.receiver_ids, cells.class_ids, strict=True)):
        result.setdefault(key, []).append(index)
    if len(result) != 42 or any(len(indices) != PHYSICAL_PER_CELL for indices in result.values()):
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: Phase-1 metadata grid is incomplete")
    return {
        key: tuple(
            sorted(
                indices,
                key=lambda index: _sha(
                    f"NEXT-R3|{key[0]}|{key[1]}|{cells.physical_ids[index]}".encode("utf-8")
                ),
            )
        )
        for key, indices in result.items()
    }


def _reject_predictor_package_forbidden_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lower = str(key).lower()
            if (
                "truth" in lower
                or lower in {"tx_labels", "class_ids", "query_labels"}
                or "role" in lower
                or lower.startswith("reg0_query")
                or lower.startswith("reg1_query")
                or "query_class" in lower
            ):
                raise NextR3Proxy24Error("predictor package contains forbidden truth/role/query-label fields")
            _reject_predictor_package_forbidden_fields(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_predictor_package_forbidden_fields(item)


def _ordered_physical_root(values: Sequence[str]) -> str:
    return _sha("\n".join(str(item) for item in values).encode("utf-8"))


def _pair_compact_binding(
    binding: Mapping[str, Any], *, prior_key: str
) -> Mapping[str, Any]:
    fields = (
        "k1_row_id",
        "k5_row_id",
        "phase1_fit_physical_root_sha256",
        "support_k1_physical_root_sha256",
        "support_k5_physical_root_sha256",
        "query_physical_root_sha256",
        "k1_is_exact_k5_prefix",
        "binding_sha256",
    )
    return {"prior_key": prior_key, **{field: binding[field] for field in fields}}


def _build_prepare_package(
    rows: SourceRows,
    cells: CellRows,
    source_meta: Mapping[str, Any],
    args: argparse.Namespace,
    feature_cache: BridgeFeatureCache,
    checkpoint_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, str], Mapping[str, Any]]:
    classes = tuple(sorted(set(cells.class_ids)))
    if len(classes) != matrix.CLASS_COUNT:
        raise NextR3Proxy24Error("prepare Phase-1 class registry drift")
    plan = matrix.build_next_r3_proxy24_plan(classes)
    matrix.validate_next_r3_proxy24_plan(plan)
    cells_typed = _build_phase1_cells(cells, feature_cache)
    cell_indices = _metadata_cell_indices(cells)
    observation_by_physical = dict(zip(rows.physical_ids, rows.observation_ids, strict=True))
    class_by_physical = dict(zip(cells.physical_ids, cells.class_ids, strict=True))
    priors: dict[str, Mapping[str, Any]] = {}
    prior_objects: dict[str, tsl.TSL160Phase1Prior] = {}
    pair_binding_values: dict[str, Mapping[str, Any]] = {}
    package_rows: list[Mapping[str, Any]] = []
    prepared_by_id: dict[str, Mapping[str, Any]] = {}
    representation_rule_sha = _sha(tsl.REPRESENTATION_RULE.encode("utf-8"))
    for planned in plan["rows"]:
        held_receiver = str(planned["held_receiver"])
        held_class = str(planned["held_class"])
        active_k = int(planned["active_k"])
        retained = tuple(str(item) for item in planned["retained_classes"])
        all_classes = tuple(str(item) for item in planned["all_registered_classes"])
        prior_key = _outer_key(held_receiver, held_class)
        if prior_key not in prior_objects:
            eligible = tuple(
                cell
                for cell in cells_typed
                if cell.receiver_id != held_receiver and cell.class_handle != held_class
            )
            phase1_root = tsl.phase1_physical_id_root(eligible)
            binding = tsl.TSL160RuntimeBinding(
                outer_fold_id=f"r3/{held_receiver}/{held_class}|classes={','.join(retained)}",
                checkpoint_sha256=checkpoint_sha256,
                representation_rule_sha256=representation_rule_sha,
                phase1_physical_id_root_sha256=phase1_root,
                phase1_seal_sha256=_require_sha(args.phase1_cells_sha256, "Phase-1 cell SHA256"),
            )
            prior_build = _build_prior_from_cells(
                cells_typed,
                held_receiver=held_receiver,
                held_class=held_class,
                binding=binding,
            )
            prior_objects[prior_key] = prior_build.prior
            wire = tsl.serialize_tsl160_prior(prior_build.prior)
            priors[prior_key] = {
                "prior_key": prior_key,
                "prior_wire_json": wire.decode("ascii"),
                "prior_sha256": prior_build.prior.prior_sha256,
            }
        support_classes = retained + (held_class,)
        support_ids = tuple(
            cells.physical_ids[index]
            for class_id in support_classes
            for index in cell_indices[(held_receiver, class_id)][:active_k]
        )
        support_labels = tuple(
            class_id for class_id in support_classes for _ in range(active_k)
        )
        unordered_query = tuple(
            cells.physical_ids[index]
            for class_id in all_classes
            for index in cell_indices[(held_receiver, class_id)][matrix.MAX_SUPPORT_K:]
        )
        query_ids = _common_query_order(held_receiver, held_class, unordered_query)
        prepared = {
            "row_id": str(planned["row_id"]),
            "support_physical_ids": list(support_ids),
            "support_observation_ids": [observation_by_physical[item] for item in support_ids],
            "support_labels": list(support_labels),
            "query_physical_ids": list(query_ids),
            "query_observation_ids": [observation_by_physical[item] for item in query_ids],
            "prior_key": prior_key,
        }
        package_rows.append(prepared)
        prepared_by_id[str(planned["row_id"])] = prepared
    for planned in plan["rows"]:
        if int(planned["active_k"]) != 1:
            continue
        held_receiver = str(planned["held_receiver"])
        held_class = str(planned["held_class"])
        all_classes = tuple(str(item) for item in planned["all_registered_classes"])
        row_k1 = matrix.outer_key_from_mapping(planned)
        row_k5 = matrix.outer_key_from_mapping(
            next(
                item
                for item in plan["rows"]
                if item["held_receiver"] == held_receiver
                and item["held_class"] == held_class
                and item["active_k"] == 5
            )
        )
        prior_key = _outer_key(held_receiver, held_class)
        prior = prior_objects[prior_key]
        phase1_ids = tuple(
            physical_id
            for cell in cells_typed
            if cell.receiver_id != held_receiver and cell.class_handle != held_class
            for physical_id in cell.physical_ids
        )
        k1 = prepared_by_id[row_k1.row_id]
        k5 = prepared_by_id[row_k5.row_id]
        query_ids = tuple(str(item) for item in k1["query_physical_ids"])
        opaque_query_buckets = {
            class_id: query_ids[index * matrix.QUERY_PER_CLASS : (index + 1) * matrix.QUERY_PER_CLASS]
            for index, class_id in enumerate(all_classes)
        }
        binding = matrix.bind_next_r3_physical_ids(
            row_k1=row_k1,
            row_k5=row_k5,
            loco_fold_receipt={
                "held_receiver": held_receiver,
                "held_class": held_class,
                "phase1_fit_count": len(phase1_ids),
                "phase1_fit_physical_root_sha256": _ordered_physical_root(phase1_ids),
            },
            phase1_fit_ids=phase1_ids,
            k1_support_ids_by_class={
                class_id: tuple(
                    item
                    for item, label in zip(k1["support_physical_ids"], k1["support_labels"], strict=True)
                    if label == class_id
                )
                for class_id in all_classes
            },
            k5_support_ids_by_class={
                class_id: tuple(
                    item
                    for item, label in zip(k5["support_physical_ids"], k5["support_labels"], strict=True)
                    if label == class_id
                )
                for class_id in all_classes
            },
            query_ids_by_class=opaque_query_buckets,
        )
        # TSL seals the labeled Phase-1 cell records with its canonical-record
        # root, while the matrix pair receipt seals the ordered physical-ID
        # sequence.  They are deliberately distinct representations of the
        # same prepare-only Phase-1 input and must not be conflated in predict.
        pair_binding_values[prior_key] = _pair_compact_binding(binding, prior_key=prior_key)
    package = {
        "schema": PREDICTOR_PACKAGE_SCHEMA,
        "protocol_schema": "p2_min_v1",
        "received_iq_sha256": rows.received_iq_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "phase1_cells_sha256": source_meta["phase1_cells_sha256"],
        "capsule_id": _require_sha(args.capsule_id, "capsule ID"),
        "split_id": _require_sha(args.split_id, "split ID"),
        "validator_receipt_sha256": _require_sha(
            args.validator_receipt_sha256, "validator receipt SHA256"
        ),
        "query_order_rule": QUERY_ORDER_RULE,
        "rows": package_rows,
        "priors": [priors[key] for key in sorted(priors)],
        "pair_bindings": [pair_binding_values[key] for key in sorted(pair_binding_values)],
    }
    _reject_predictor_package_forbidden_fields(package)
    truth = {
        query_id: class_by_physical[query_id]
        for prepared in package_rows
        for query_id in prepared["query_physical_ids"]
    }
    bridge_receipt = feature_cache.receipt(rows.physical_ids)
    return package, truth, {"plan": plan, "bridge_receipt": bridge_receipt}


def _write_prepare_artifacts(
    output_dir: Path,
    *,
    package: Mapping[str, Any],
    truth: Mapping[str, str],
    receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    root = _new_root(output_dir)
    package_path = root / "predictor_package.json"
    truth_path = root / "truth.json"
    receipt_path = root / "prepare_receipt.json"
    _write_json_new(package_path, package)
    _write_json_new(truth_path, truth)
    receipt_document = {
        "schema": PREPARE_SCHEMA,
        "package_sha256": _sha_file(package_path),
        "truth_sha256": _sha_file(truth_path),
        "truth_in_predictor_package": False,
        "package_has_two_query_lists": False,
        "package_has_full_class_ids": False,
        **_plain(receipt),
    }
    _write_json_new(receipt_path, receipt_document)
    return {
        "output_dir": str(root),
        "package": str(package_path),
        "package_sha256": _sha_file(package_path),
        "truth": str(truth_path),
        "truth_sha256": _sha_file(truth_path),
        "receipt": str(receipt_path),
        "receipt_sha256": _sha_file(receipt_path),
    }


def run_prepare(args: argparse.Namespace) -> Mapping[str, Any]:
    rows, cells, source_meta = _load_real_rows(args)
    checkpoint_sha = _require_sha(args.checkpoint_sha256, "checkpoint SHA256")
    _require_file(args.checkpoint, checkpoint_sha, "checkpoint")
    args.capsule_id = _require_sha(args.capsule_id, "capsule ID")
    args.split_id = _require_sha(args.split_id, "split ID")
    args.validator_receipt_sha256 = _require_sha(
        args.validator_receipt_sha256, "validator receipt SHA256"
    )
    args.phase1_cells_sha256 = _require_sha(args.phase1_cells_sha256, "Phase-1 cell SHA256")
    bridge = _load_checkpoint_bridge(args, rows, checkpoint_sha)
    feature_cache = BridgeFeatureCache(bridge, rows, checkpoint_sha)
    package, truth, evidence = _build_prepare_package(
        rows, cells, source_meta, args, feature_cache, checkpoint_sha
    )
    return _write_prepare_artifacts(
        args.output_dir,
        package=package,
        truth=truth,
        receipt={
            "received_iq_sha256": rows.received_iq_sha256,
            "checkpoint_sha256": checkpoint_sha,
            "phase1_cells_sha256": source_meta["phase1_cells_sha256"],
            "matrix_sha256": evidence["plan"]["matrix_sha256"],
            "bridge_feature_binding": evidence["bridge_receipt"],
        },
    )


def _load_predictor_package(
    args: argparse.Namespace, rows: SourceRows, checkpoint_sha256: str
) -> tuple[PreparedPredictorPackage, Mapping[str, Any]]:
    payload = _require_file(
        getattr(args, "package", None),
        getattr(args, "package_sha256", None),
        "predictor package",
    )
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR3Proxy24Error("predictor package must be UTF-8 JSON") from error
    _reject_predictor_package_forbidden_fields(document)
    expected_top = {
        "schema",
        "protocol_schema",
        "received_iq_sha256",
        "checkpoint_sha256",
        "phase1_cells_sha256",
        "capsule_id",
        "split_id",
        "validator_receipt_sha256",
        "query_order_rule",
        "rows",
        "priors",
        "pair_bindings",
    }
    if type(document) is not dict or set(document) != expected_top:
        raise NextR3Proxy24Error("predictor package schema drift")
    if (
        document["schema"] != PREDICTOR_PACKAGE_SCHEMA
        or document["protocol_schema"] != "p2_min_v1"
        or document["received_iq_sha256"] != rows.received_iq_sha256
        or document["checkpoint_sha256"] != checkpoint_sha256
        or document["query_order_rule"] != QUERY_ORDER_RULE
    ):
        raise NextR3Proxy24Error("predictor package provenance/query-order drift")
    phase1_cells_sha = _require_sha(document["phase1_cells_sha256"], "package Phase-1 cell SHA256")
    capsule_id = _require_sha(document["capsule_id"], "package capsule ID")
    split_id = _require_sha(document["split_id"], "package split ID")
    validator_sha = _require_sha(
        document["validator_receipt_sha256"], "package validator receipt SHA256"
    )
    if not isinstance(document["rows"], list) or len(document["rows"]) != matrix.ROW_COUNT:
        raise NextR3Proxy24Error("predictor package must seal all 24 rows")
    row_keys = {
        "row_id",
        "support_physical_ids",
        "support_observation_ids",
        "support_labels",
        "query_physical_ids",
        "query_observation_ids",
        "prior_key",
    }
    support_label_values: list[str] = []
    for raw in document["rows"]:
        if type(raw) is not dict or set(raw) != row_keys:
            raise NextR3Proxy24Error("predictor package row contains forbidden role/query fields")
        labels = raw.get("support_labels")
        if not isinstance(labels, list):
            raise NextR3Proxy24Error("predictor package support labels are malformed")
        support_label_values.extend(labels)
    classes = tuple(sorted(set(str(item) for item in support_label_values)))
    if len(classes) != matrix.CLASS_COUNT or any(type(item) is not str or not item for item in support_label_values):
        raise NextR3Proxy24Error("predictor package support-only class registry drift")
    plan = matrix.build_next_r3_proxy24_plan(classes)
    matrix.validate_next_r3_proxy24_plan(plan)
    plan_by_id = {str(item["row_id"]): item for item in plan["rows"]}
    physical_to_observation = dict(zip(rows.physical_ids, rows.observation_ids, strict=True))
    physical_to_receiver = dict(zip(rows.physical_ids, rows.receiver_ids, strict=True))
    prepared: dict[str, PreparedPredictorRow] = {}
    for raw in document["rows"]:
        row_id = raw.get("row_id")
        if type(row_id) is not str or row_id not in plan_by_id or row_id in prepared:
            raise NextR3Proxy24Error("predictor package row identity drift")
        planned = plan_by_id[row_id]
        active_k = int(planned["active_k"])
        support_ids, support_observations = _require_split_id_pairs(
            raw,
            physical_name="support_physical_ids",
            observation_name="support_observation_ids",
            count=matrix.CLASS_COUNT * active_k,
            physical_to_observation=physical_to_observation,
        )
        labels = _string_tuple(
            raw["support_labels"], name="support_labels", count=len(support_ids), unique=False
        )
        expected_labels = tuple(
            class_id
            for class_id in tuple(planned["retained_classes"]) + (str(planned["held_class"]),)
            for _ in range(active_k)
        )
        if labels != expected_labels:
            raise NextR3Proxy24Error("predictor package permits only class-major K-shot support labels")
        query_ids, query_observations = _require_split_id_pairs(
            raw,
            physical_name="query_physical_ids",
            observation_name="query_observation_ids",
            count=matrix.CLASS_COUNT * matrix.QUERY_PER_CLASS,
            physical_to_observation=physical_to_observation,
        )
        if (
            set(support_ids) & set(query_ids)
            or any(physical_to_receiver[item] != planned["held_receiver"] for item in support_ids + query_ids)
            or query_ids
            != _common_query_order(
                str(planned["held_receiver"]), str(planned["held_class"]), query_ids
            )
        ):
            raise NextR3Proxy24Error("predictor package common-query identity/order drift")
        prior_key = raw.get("prior_key")
        if type(prior_key) is not str or prior_key != _outer_key(
            str(planned["held_receiver"]), str(planned["held_class"])
        ):
            raise NextR3Proxy24Error("predictor package prior-key drift")
        prepared[row_id] = PreparedPredictorRow(
            row_id=row_id,
            support_physical_ids=support_ids,
            support_observation_ids=support_observations,
            support_labels=labels,
            query_physical_ids=query_ids,
            query_observation_ids=query_observations,
            prior_key=prior_key,
        )
    if set(prepared) != set(plan_by_id):
        raise NextR3Proxy24Error("predictor package lacks a matrix row")
    prior_keys = {"prior_key", "prior_wire_json", "prior_sha256"}
    if not isinstance(document["priors"], list) or len(document["priors"]) != 12:
        raise NextR3Proxy24Error("predictor package prior coverage drift")
    priors: dict[str, tsl.TSL160Phase1Prior] = {}
    for raw in document["priors"]:
        if type(raw) is not dict or set(raw) != prior_keys:
            raise NextR3Proxy24Error("predictor package prior schema drift")
        key = raw.get("prior_key")
        wire_text = raw.get("prior_wire_json")
        if type(key) is not str or type(wire_text) is not str or key in priors:
            raise NextR3Proxy24Error("predictor package prior identity drift")
        try:
            prior = tsl.deserialize_tsl160_prior(wire_text.encode("ascii"))
        except Exception as error:
            raise NextR3Proxy24Error("predictor package sealed prior wire drift") from error
        if raw.get("prior_sha256") != prior.prior_sha256:
            raise NextR3Proxy24Error("predictor package sealed prior hash drift")
        priors[key] = prior
    pair_keys = {
        "prior_key",
        "k1_row_id",
        "k5_row_id",
        "phase1_fit_physical_root_sha256",
        "support_k1_physical_root_sha256",
        "support_k5_physical_root_sha256",
        "query_physical_root_sha256",
        "k1_is_exact_k5_prefix",
        "binding_sha256",
    }
    if not isinstance(document["pair_bindings"], list) or len(document["pair_bindings"]) != 12:
        raise NextR3Proxy24Error("predictor package K-pair binding coverage drift")
    pair_bindings: dict[str, Mapping[str, Any]] = {}
    for raw in document["pair_bindings"]:
        if type(raw) is not dict or set(raw) != pair_keys:
            raise NextR3Proxy24Error("predictor package K-pair binding schema drift")
        key = raw.get("prior_key")
        if type(key) is not str or key in pair_bindings:
            raise NextR3Proxy24Error("predictor package K-pair binding identity drift")
        for field in (
            "phase1_fit_physical_root_sha256",
            "support_k1_physical_root_sha256",
            "support_k5_physical_root_sha256",
            "query_physical_root_sha256",
            "binding_sha256",
        ):
            _require_sha(raw[field], f"predictor package {field}")
        if raw["k1_is_exact_k5_prefix"] is not True:
            raise NextR3Proxy24Error("predictor package K-pair prefix flag drift")
        pair_bindings[key] = dict(raw)
    for planned in plan["rows"]:
        if int(planned["active_k"]) != 1:
            continue
        held_receiver = str(planned["held_receiver"])
        held_class = str(planned["held_class"])
        key = _outer_key(held_receiver, held_class)
        row_k1 = prepared[str(planned["row_id"])]
        row_k5 = next(
            prepared[str(item["row_id"])]
            for item in plan["rows"]
            if item["held_receiver"] == held_receiver
            and item["held_class"] == held_class
            and item["active_k"] == 5
        )
        prior = priors.get(key)
        binding = pair_bindings.get(key)
        expected_outer = f"r3/{held_receiver}/{held_class}|classes={','.join(planned['retained_classes'])}"
        if (
            prior is None
            or binding is None
            or prior.binding.outer_fold_id != expected_outer
            or prior.binding.checkpoint_sha256 != checkpoint_sha256
            or prior.binding.phase1_seal_sha256 != phase1_cells_sha
            or binding["k1_row_id"] != row_k1.row_id
            or binding["k5_row_id"] != row_k5.row_id
            or binding["support_k1_physical_root_sha256"]
            != _ordered_physical_root(
                tuple(
                    physical_id
                    for class_id in classes
                    for physical_id in row_k1.support_for(class_id)
                )
            )
            or binding["support_k5_physical_root_sha256"]
            != _ordered_physical_root(
                tuple(
                    physical_id
                    for class_id in classes
                    for physical_id in row_k5.support_for(class_id)
                )
            )
            or binding["query_physical_root_sha256"]
            != _ordered_physical_root(row_k1.query_physical_ids)
            or row_k1.query_physical_ids != row_k5.query_physical_ids
            or row_k1.query_observation_ids != row_k5.query_observation_ids
            or any(
                row_k1.support_for(class_id) != row_k5.support_for(class_id)[:1]
                for class_id in classes
            )
        ):
            raise NextR3Proxy24Error("predictor package sealed prior/K-pair/common-query drift")
    return (
        PreparedPredictorPackage(
            class_registry=classes,
            rows_by_id=prepared,
            priors_by_key=priors,
            pair_bindings=pair_bindings,
            received_iq_sha256=rows.received_iq_sha256,
            checkpoint_sha256=checkpoint_sha256,
            phase1_cells_sha256=phase1_cells_sha,
            capsule_id=capsule_id,
            split_id=split_id,
            validator_receipt_sha256=validator_sha,
            sha256=_sha(payload),
        ),
        plan,
    )


def _read_d106_asset(
    path: Path,
    expected_sha256: str,
    checkpoint_sha256: str,
    *,
    tap_sha256: str,
    tap_receipt_sha256: str,
) -> d106_asset.D106RDCEAsset:
    payload = _require_file(path, expected_sha256, "D106 RDCE asset wire")
    try:
        if not payload.startswith(d106_asset.WIRE_MAGIC):
            raise ValueError("wire magic")
        offset = len(d106_asset.WIRE_MAGIC)
        header_size = struct.unpack(">I", payload[offset : offset + 4])[0]
        offset += 4
        header = json.loads(payload[offset : offset + header_size].decode("utf-8"))
        asset_header = header["asset"]
        lineage_fields = {name: asset_header[name] for name in (
            "checkpoint_sha256", "runtime_sha256", "method_lock_sha256", "split_id", "tap_sha256", "construction_code_sha256", "content_root_sha256", "source_receipt_sha256", "tap_receipt_sha256", "tap_authority_sha256"
        )}
        lineage = d106_asset.D106RDCEAssetLineage(**lineage_fields)
        asset = d106_asset.deserialize_d106_rdce_asset(
            payload,
            expected_wire_sha256=expected_sha256,
            expected_lineage=lineage,
        )
    except Exception as error:
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: D106 RDCE asset wire cannot be loaded") from error
    if (
        asset.checkpoint_sha256 != _require_sha(checkpoint_sha256, "checkpoint SHA256")
        or asset.tap_sha256 != _require_sha(tap_sha256, "D106 tap archive SHA256")
        or asset.tap_receipt_sha256 != _require_sha(tap_receipt_sha256, "D106 tap receipt SHA256")
        or not asset.is_formal_deployable
    ):
        raise MissingRealInputArtifacts(f"{MISSING_PREFIX}: D106 RDCE asset/checkpoint binding is not formal")
    return asset


def _load_checkpoint_bridge(
    args: argparse.Namespace, rows: SourceRows, checkpoint_sha256: str
) -> Any:
    """Build the existing real bridge from received-IQ and checkpoint only."""

    if args.received_iq_receipt is None or args.received_iq_receipt_sha256 is None:
        raise MissingRealInputArtifacts(
            f"{MISSING_PREFIX}: received-IQ legality receipt path/SHA256 is required for the real bridge"
        )
    _require_file(args.received_iq_receipt, args.received_iq_receipt_sha256, "received-IQ receipt")
    try:
        from cvsrffi import stage2_next_r1_real as real_bridge
        from cvsrffi.stage2_d105_phase1_bundle import (
            build_d105_exact_model_from_checkpoint,
            load_d105_exact_sha_bound_checkpoint,
        )
        checkpoint, _load_receipt = load_d105_exact_sha_bound_checkpoint(
            args.checkpoint, checkpoint_sha256
        )
        model, _build_receipt = build_d105_exact_model_from_checkpoint(
            checkpoint,
            input_len=int(rows.received_iq.shape[2]),
            device=args.device,
        )
        # ``NextR1RealModelBridge.forward_indices`` consumes only
        # ``rows.received_iq``.  Supplying the received-only row object keeps
        # source labels/features outside the prediction process altogether.
        return real_bridge.NextR1RealModelBridge(model, rows, checkpoint_sha256, args.device)
    except MissingRealInputArtifacts:
        raise
    except Exception as error:
        raise NextR3Proxy24Error("real checkpoint/model bridge load failed") from error


def _lock_for_k(phase1_metadata_sha256: str, active_k: int) -> qknn.Phase1ZIDStudentTLock:
    # The predecessor lock is imported from the frozen D106/D129 implementation;
    # this runner does not choose or tune any qKNN/rho/nu parameter.
    locks = _predecessor_locks(
        SimpleNamespace(receipt={"phase1_cell_metadata_sha256": phase1_metadata_sha256})
    )
    by_k = {lock.active_k: lock for lock in locks}
    if active_k not in by_k:
        raise NextR3Proxy24Error(f"frozen predecessor lock missing K{active_k}")
    return by_k[active_k]


def _array_receipt(value: np.ndarray) -> Mapping[str, Any]:
    array = np.ascontiguousarray(value)
    return {"dtype": array.dtype.str, "shape": list(array.shape), "sha256": _sha(array.tobytes(order="C"))}


def _write_row_authority(
    path: Path,
    *,
    support: np.ndarray,
    labels: Sequence[str],
    support_ids: Sequence[str],
    query_ids: Sequence[str],
    classes: Sequence[str],
    lock: qknn.Phase1ZIDStudentTLock,
    args: argparse.Namespace,
    row_id: str,
    active_k: int,
) -> d106_runtime._D106RDCERowAuthority:
    canonical_support = tsl.canonical_d106_relu_zid160(support)
    bank = qknn.build_typed_zid_support_bank(canonical_support, labels, classes, config=lock)
    support_ids = tuple(str(value) for value in support_ids)
    query_ids = tuple(str(value) for value in query_ids)
    validator_sha = _require_sha(args.validator_receipt_sha256, "validator receipt SHA256")
    split = TypedValidatedOnceP2SplitHandle(
        capsule_id=_require_sha(args.capsule_id, "capsule ID"),
        split_id=_require_sha(args.split_id, "split ID"),
        validator_receipt_sha256=validator_sha,
        support_physical_root_sha256=d106_runtime._physical_root(support_ids),
        query_physical_root_sha256=d106_runtime._physical_root(query_ids),
        support_query_disjoint=True,
    )
    document = {
        "schema": d106_runtime.ROW_AUTHORITY_SCHEMA,
        "capsule_id": split.capsule_id,
        "split_id": split.split_id,
        "validator_receipt_sha256": split.validator_receipt_sha256,
        "row_id": row_id,
        "seed": int(args.seed),
        "active_k": active_k,
        "registered_classes": list(classes),
        "support_z_id_receipt": _array_receipt(canonical_support),
        "support_labels_receipt": _array_receipt(np.asarray(labels, dtype="<U64")),
        "support_physical_ids_receipt": _array_receipt(np.asarray(support_ids, dtype="<U128")),
        "ordered_support_physical_ids_sha256": d106_runtime._ordered_physical_root(support_ids),
        "qknn_bank_sha256": bank.bank_receipt_sha256,
        "support_physical_root_sha256": split.support_physical_root_sha256,
        "query_physical_root_sha256": split.query_physical_root_sha256,
        "protocol_schema": split.protocol_schema,
        "phase2_data_status": split.phase2_data_status,
        "support_query_disjoint": True,
    }
    _write_json_new(path, document)
    return d106_runtime.load_d106_rdce_row_authority(path, expected_authority_sha256=_sha_file(path))


def _checkpoint_smoke(bridge: Any, indices: Sequence[int], checkpoint_sha256: str) -> Mapping[str, Any]:
    if not indices:
        raise NextR3Proxy24Error("real checkpoint smoke needs at least one physical row")
    try:
        first = bridge.forward_indices(tuple(indices))
        second = bridge.forward_indices(tuple(indices))
        first_z = np.asarray(first[1], dtype=np.float32)
        second_z = np.asarray(second[1], dtype=np.float32)
        if first_z.shape != second_z.shape or not np.array_equal(first_z, second_z) or not np.isfinite(first_z).all():
            raise ValueError("non-repeatable or non-finite checkpoint output")
    except Exception as error:
        raise NextR3Proxy24Error("real checkpoint no-truth smoke failed") from error
    return {"schema": SMOKE_SCHEMA, "checkpoint_sha256": checkpoint_sha256, "sample_count": len(indices), "canonical_repeat_exact": True, "truth_loaded": False, "query_truth_access": False}


def _execute_prepared_fold(
    rows: SourceRows,
    plan_row: Mapping[str, Any],
    package: PreparedPredictorPackage,
    *,
    args: argparse.Namespace,
    asset: d106_asset.D106RDCEAsset,
    feature_cache: BridgeFeatureCache,
) -> tuple[runtime.NextR3RuntimeResult, Mapping[str, Any]]:
    """Predict one row using only the sealed prior/support/common-query package."""

    held_receiver = str(plan_row["held_receiver"])
    held_class = str(plan_row["held_class"])
    active_k = int(plan_row["active_k"])
    all_classes = tuple(str(value) for value in plan_row["all_registered_classes"])
    retained = tuple(str(value) for value in plan_row["retained_classes"])
    prepared = package.row(str(plan_row["row_id"]))
    prior = package.prior(prepared.prior_key)
    expected_outer = f"r3/{held_receiver}/{held_class}|classes={','.join(retained)}"
    if (
        prior.binding.outer_fold_id != expected_outer
        or prior.binding.checkpoint_sha256 != package.checkpoint_sha256
        or prior.binding.phase1_seal_sha256 != package.phase1_cells_sha256
        or prepared.query_physical_ids
        != _common_query_order(held_receiver, held_class, prepared.query_physical_ids)
    ):
        raise NextR3Proxy24Error("prepared prior/common-query runtime binding drift")
    reg1_support_ids = prepared.support_physical_ids
    reg1_support_labels = prepared.support_labels
    reg0_support_ids = tuple(
        physical_id
        for physical_id, label in zip(reg1_support_ids, reg1_support_labels, strict=True)
        if label in retained
    )
    reg0_support_labels = tuple(label for label in reg1_support_labels if label in retained)
    raw_reg0_support = feature_cache.take(reg0_support_ids)
    raw_reg1_support = feature_cache.take(reg1_support_ids)
    raw_query = feature_cache.take(prepared.query_physical_ids)
    bridge_binding = runtime.NextR3RDCEBridgeBinding(
        checkpoint_sha256=package.checkpoint_sha256,
        capsule_id=package.capsule_id,
        split_id=package.split_id,
        row_id=str(plan_row["row_id"]),
        seed=int(args.seed),
        received_iq_root_sha256=rows.received_iq_sha256,
        tap_sha256=_require_sha(args.d106_tap_archive_sha256, "D106 tap archive SHA256"),
        representation_rule_sha256=_sha(tsl.REPRESENTATION_RULE.encode("utf-8")),
        phase1_physical_id_root_sha256=prior.binding.phase1_physical_id_root_sha256,
        phase1_seal_sha256=package.phase1_cells_sha256,
        outer_fold_id=expected_outer,
    )
    reg0 = runtime.NextR3RegistrationInput(
        registration_state="REG0",
        received_iq_root_sha256=rows.received_iq_sha256,
        support_pre_relu160=raw_reg0_support,
        query_pre_relu160=raw_query,
        support_labels=reg0_support_labels,
        registered_classes=retained,
        support_physical_ids=reg0_support_ids,
        query_physical_ids=prepared.query_physical_ids,
    )
    reg1 = runtime.NextR3RegistrationInput(
        registration_state="REG1",
        received_iq_root_sha256=rows.received_iq_sha256,
        support_pre_relu160=raw_reg1_support,
        query_pre_relu160=raw_query,
        support_labels=reg1_support_labels,
        registered_classes=all_classes,
        support_physical_ids=reg1_support_ids,
        query_physical_ids=prepared.query_physical_ids,
    )
    lock = _lock_for_k(package.phase1_cells_sha256, active_k)
    authority_path = args._run_root / "rows" / f"{plan_row['row_id']}.row_authority.json"
    authority = _write_row_authority(
        authority_path,
        support=raw_reg0_support,
        labels=reg0.support_labels,
        support_ids=reg0.support_physical_ids,
        query_ids=reg0.query_physical_ids,
        classes=retained,
        lock=lock,
        args=args,
        row_id=str(plan_row["row_id"]),
        active_k=active_k,
    )
    support_rows = d106_runtime.D106RDCESupportRows(
        support_z_id=tsl.canonical_d106_relu_zid160(raw_reg0_support),
        support_labels=np.asarray(reg0.support_labels, dtype="<U64"),
        support_physical_ids=np.asarray(reg0.support_physical_ids, dtype="<U128"),
        qknn_bank=qknn.build_typed_zid_support_bank(
            tsl.canonical_d106_relu_zid160(raw_reg0_support),
            reg0.support_labels,
            retained,
            config=lock,
        ),
        split_handle=TypedValidatedOnceP2SplitHandle(
            capsule_id=package.capsule_id,
            split_id=package.split_id,
            validator_receipt_sha256=package.validator_receipt_sha256,
            support_physical_root_sha256=d106_runtime._physical_root(reg0.support_physical_ids),
            query_physical_root_sha256=d106_runtime._physical_root(reg0.query_physical_ids),
            support_query_disjoint=True,
        ),
        row_id=str(plan_row["row_id"]),
        seed=int(args.seed),
    )
    state = d106_runtime.fit_d106_rdce_runtime(asset, support_rows, row_authority=authority)
    result = runtime.execute_next_r3_four_state(
        bridge=bridge_binding,
        da1_reg0_state=state,
        reg0=reg0,
        reg1=reg1,
        qknn_lock=lock,
        tsl_prior=prior,
        tsl_runtime_binding=prior.binding,
    )
    return result, {
        "prior_receipt": {"prior_sha256": prior.prior_sha256, "sealed_in_prepare": True},
        "pair_binding": package.pair_bindings[prepared.prior_key],
        "input_feature_binding": {
            "reg0_support": feature_cache.receipt(reg0_support_ids),
            "reg1_support": feature_cache.receipt(reg1_support_ids),
            "common_query": feature_cache.receipt(prepared.query_physical_ids),
            "reg0_reg1_common_query_byte_identical": True,
        },
        "smoke_receipt": {"truth_loaded": False},
    }


def _build_prior_from_cells(cells: Sequence[tsl.TSL160Phase1Cell], *, held_receiver: str, held_class: str, binding: tsl.TSL160RuntimeBinding) -> tsl.TSL160PriorBuild:
    eligible = tuple(cell for cell in cells if cell.receiver_id != held_receiver and cell.class_handle != held_class)
    active_classes = tuple(sorted({cell.class_handle for cell in eligible}))
    folds: list[tsl.TSL160PhysicalLOOFold] = []
    for cell in eligible:
        for index, pid in enumerate(cell.physical_ids):
            support_rows = [row for source in eligible for row_pid, row in zip(source.physical_ids, source.zid160, strict=True) if row_pid != pid]
            support_labels = [source.class_handle for source in eligible for row_pid in source.physical_ids if row_pid != pid]
            support_ids = [row_pid for source in eligible for row_pid in source.physical_ids if row_pid != pid]
            folds.append(tsl.TSL160PhysicalLOOFold(fold_id=f"{cell.receiver_id}/{cell.class_handle}/{index:02d}", receiver_id=cell.receiver_id, class_handle=cell.class_handle, registered_classes=active_classes, support_zid160=np.stack(support_rows).astype(np.float32), support_labels=tuple(support_labels), support_physical_ids=tuple(support_ids), validation_zid160=cell.zid160[index:index+1], validation_labels=(cell.class_handle,), validation_physical_ids=(pid,)))
    return tsl.build_tsl160_phase1_prior(cells, folds, binding=binding, held_receiver=held_receiver, held_class=held_class)


def _save_prediction_row(
    root: Path,
    index: int,
    result: runtime.NextR3RuntimeResult,
    input_feature_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    row_id = result.bridge.row_id
    stem = f"{index:03d}_{_sha(row_id.encode('utf-8'))[:16]}"
    row_path = root / "rows" / f"{stem}.json"
    payload = artifact.build_next_r3_prediction_artifact
    del payload  # keep the row receipt intentionally derived below
    row_document = {
        "schema": RUNNER_SCHEMA,
        "row_id": row_id,
        "runtime_receipt": _plain(result.runtime_receipt),
        "resource_receipt": _plain(result.resource_receipt),
        "input_feature_binding": _plain(input_feature_binding),
        "smoke_receipt": {"truth_loaded": False},
    }
    _write_json_new(row_path, row_document)
    return {"row_id": row_id, "path": row_path.relative_to(root).as_posix(), "sha256": _sha_file(row_path)}


def run_predict(args: argparse.Namespace) -> Mapping[str, Any]:
    # Input validation precedes run-root creation, so a missing sealed package
    # leaves no misleading partial run or smoke marker behind.  This phase has
    # no Phase-1 cell archive or truth/split input: those stay in ``prepare``.
    rows = _load_received_rows(args)
    checkpoint_sha = _require_sha(args.checkpoint_sha256, "checkpoint SHA256")
    _require_file(args.checkpoint, checkpoint_sha, "checkpoint")
    tap_sha = _require_sha(args.d106_tap_archive_sha256, "D106 tap archive SHA256")
    tap_receipt_sha = _require_sha(args.d106_tap_receipt_sha256, "D106 tap receipt SHA256")
    _require_file(args.d106_tap_archive, tap_sha, "D106 tap archive")
    _require_file(args.d106_tap_receipt, tap_receipt_sha, "D106 tap receipt")
    package, plan = _load_predictor_package(args, rows, checkpoint_sha)
    args.capsule_id = package.capsule_id
    args.split_id = package.split_id
    args.validator_receipt_sha256 = package.validator_receipt_sha256
    args.phase1_cells_sha256 = package.phase1_cells_sha256
    asset_wire = _read_d106_asset(
        args.d106_rdce_wire,
        args.d106_rdce_wire_sha256,
        checkpoint_sha,
        tap_sha256=tap_sha,
        tap_receipt_sha256=tap_receipt_sha,
    )
    # The bridge consumes only received IQ and the pinned checkpoint.  The
    # predictor reconstructs no Phase-1 feature/prior data itself.
    bridge = _load_checkpoint_bridge(args, rows, checkpoint_sha)
    feature_cache = BridgeFeatureCache(bridge, rows, checkpoint_sha)
    first_prepared = package.row(str(plan["rows"][0]["row_id"]))
    index_by_physical = {physical_id: index for index, physical_id in enumerate(rows.physical_ids)}
    first_indices = tuple(index_by_physical[item] for item in first_prepared.support_physical_ids[:2])
    smoke = _checkpoint_smoke(bridge, first_indices, checkpoint_sha)
    root = _new_root(args.run_root)
    args._run_root = root
    _write_json_new(root / "plan.json", plan)
    _write_json_new(root / "preregistration.json", {"schema": RUNNER_SCHEMA, "run_id": args.run_id, "candidate_id": matrix.CANDIDATE_ID, "matrix_sha256": plan["matrix_sha256"], "row_count": matrix.ROW_COUNT, "state_prediction_count": matrix.STATE_PREDICTION_COUNT, "truth_loaded": False, "checkpoint_sha256": checkpoint_sha, "received_iq_sha256": rows.received_iq_sha256, "phase1_cells_sha256": package.phase1_cells_sha256, "predictor_package_sha256": package.sha256, "predictor_package_truth_free": True, "external_feature_archive_consumed": False, "output_overwrite": False})
    _write_json_new(root / "bridge_feature_binding.json", feature_cache.receipt(rows.physical_ids))
    _write_json_new(root / "smoke.json", smoke)
    selected_rows = tuple(plan["rows"][:1]) if args.smoke_one_row_no_truth else tuple(plan["rows"])
    runtime_results: dict[str, runtime.NextR3RuntimeResult] = {}
    resources: list[Mapping[str, Any]] = []
    row_receipts: list[Mapping[str, Any]] = []
    for index, planned in enumerate(selected_rows):
        result, receipts = _execute_prepared_fold(
            rows,
            planned,
            package,
            args=args,
            asset=asset_wire,
            feature_cache=feature_cache,
        )
        runtime_results[str(planned["row_id"])] = result
        resources.append({"row_id": planned["row_id"], "resource_receipt": _plain(result.resource_receipt), "prior_receipt": receipts["prior_receipt"]})
        row_file = _save_prediction_row(root, index, result, receipts["input_feature_binding"])
        row_receipts.append({"row_id": planned["row_id"], "runtime_receipt_sha256": _sha(_canonical(result.runtime_receipt)), "resource_receipt_sha256": _sha(_canonical(result.resource_receipt)), "input_feature_binding_sha256": _sha(_canonical(receipts["input_feature_binding"])), "row_file_sha256": row_file["sha256"]})
    if args.smoke_one_row_no_truth:
        completion = {"schema": COMPLETION_SCHEMA, "status": "SMOKE_NO_TRUTH", "run_id": args.run_id, "row_count": 1, "truth_loaded": False, "prediction_complete": False, "smoke_receipt_sha256": _sha_file(root / "smoke.json")}
        _write_json_new(root / "completion.json", completion)
        return completion
    prediction = artifact.build_next_r3_prediction_artifact(plan, runtime_results)
    _write_json_new(root / "prediction.json", prediction)
    _write_json_new(root / "resource.json", {"schema": RESOURCE_SCHEMA, "row_count": len(resources), "truth_loaded": False, "rows": resources})
    manifest = {"schema": MANIFEST_SCHEMA, "candidate_id": matrix.CANDIDATE_ID, "matrix_sha256": plan["matrix_sha256"], "row_count": matrix.ROW_COUNT, "state_prediction_count": matrix.STATE_PREDICTION_COUNT, "arm_prediction_count": matrix.ARM_PREDICTION_COUNT, "rows": row_receipts, "all_rows_sealed": True, "sealed_before_scoring": True, "truth_loaded": False}
    manifest["manifest_sha256"] = _sha(_canonical(manifest))
    _write_json_new(root / "manifest.json", manifest)
    completion = {"schema": COMPLETION_SCHEMA, "status": "ARTIFACTS_COMPLETE_NOT_SCORED", "run_id": args.run_id, "row_count": matrix.ROW_COUNT, "state_prediction_count": matrix.STATE_PREDICTION_COUNT, "truth_loaded": False, "prediction_sha256": _sha_file(root / "prediction.json"), "manifest_sha256": _sha_file(root / "manifest.json"), "resource_sha256": _sha_file(root / "resource.json"), "plan_sha256": _sha_file(root / "plan.json"), "smoke_receipt_sha256": _sha_file(root / "smoke.json")}
    _write_json_new(root / "completion.json", completion)
    return completion


def run_score(args: argparse.Namespace) -> Mapping[str, Any]:
    root = args.run_root.resolve(strict=True)
    required = ("plan.json", "prediction.json", "manifest.json", "resource.json", "completion.json", "smoke.json")
    if any(not (root / name).is_file() for name in required):
        raise NextR3Proxy24Error("score requires a complete sealed 24-row prediction set")
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "ARTIFACTS_COMPLETE_NOT_SCORED" or completion.get("row_count") != matrix.ROW_COUNT or completion.get("truth_loaded") is not False:
        raise NextR3Proxy24Error("score refused incomplete prediction closure")
    expected = {"prediction_sha256": _sha_file(root / "prediction.json"), "manifest_sha256": _sha_file(root / "manifest.json"), "resource_sha256": _sha_file(root / "resource.json"), "plan_sha256": _sha_file(root / "plan.json")}
    if any(completion.get(name) != value for name, value in expected.items()):
        raise NextR3Proxy24Error("score refused hash-mismatched prediction closure")
    plan = matrix.validate_next_r3_proxy24_plan(json.loads((root / "plan.json").read_text(encoding="utf-8")))
    prediction = json.loads((root / "prediction.json").read_text(encoding="utf-8"))
    scorer._validate_prediction(prediction, plan)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest_sha = manifest.pop("manifest_sha256", None)
    if manifest_sha != _sha(_canonical(manifest)) or manifest.get("all_rows_sealed") is not True or manifest.get("sealed_before_scoring") is not True or manifest.get("row_count") != matrix.ROW_COUNT:
        raise NextR3Proxy24Error("score refused invalid sealed manifest")
    output = args.output.resolve(strict=False)
    if not output.is_absolute() or output.exists() or not output.parent.is_dir():
        raise NextR3Proxy24Error("score output must be a new absolute file")
    # Truth is opened only after all prediction-side closure checks above.
    truth_bytes = _require_file(args.truth, args.truth_sha256, "truth catalog")
    try:
        truth = json.loads(truth_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NextR3Proxy24Error("truth catalog must be UTF-8 JSON mapping") from error
    if not isinstance(truth, Mapping):
        raise NextR3Proxy24Error("truth catalog must be a mapping")
    result = dict(scorer.score_next_r3_proxy24(prediction=prediction, plan=plan, truth_by_query_id=truth))
    result["prediction_sha256"] = expected["prediction_sha256"]
    result["truth_sha256"] = _sha_file(args.truth)
    _write_json_new(output, result)
    return {"score_sha256": _sha_file(output), "truth_opened_after_complete_prediction": True, "row_count": matrix.ROW_COUNT}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--received-iq", required=True, type=Path)
    prepare.add_argument("--received-iq-sha256", required=True)
    prepare.add_argument("--received-iq-receipt", required=True, type=Path)
    prepare.add_argument("--received-iq-receipt-sha256", required=True)
    prepare.add_argument("--phase1-cells", required=True, type=Path)
    prepare.add_argument("--phase1-cells-sha256", required=True)
    prepare.add_argument("--checkpoint", required=True, type=Path)
    prepare.add_argument("--checkpoint-sha256", required=True)
    prepare.add_argument("--capsule-id", required=True)
    prepare.add_argument("--split-id", required=True)
    prepare.add_argument("--validator-receipt-sha256", required=True)
    prepare.add_argument("--device", default="cuda:0")
    prepare.set_defaults(func=run_prepare)
    predict = commands.add_parser("predict")
    predict.add_argument("--run-id", required=True)
    predict.add_argument("--run-root", required=True, type=Path)
    predict.add_argument("--received-iq", required=True, type=Path)
    predict.add_argument("--received-iq-sha256", required=True)
    predict.add_argument("--received-iq-receipt", required=True, type=Path)
    predict.add_argument("--received-iq-receipt-sha256", required=True)
    predict.add_argument("--package", required=True, type=Path)
    predict.add_argument("--package-sha256", required=True)
    predict.add_argument("--checkpoint", required=True, type=Path)
    predict.add_argument("--checkpoint-sha256", required=True)
    predict.add_argument("--d106-tap-archive", required=True, type=Path)
    predict.add_argument("--d106-tap-archive-sha256", required=True)
    predict.add_argument("--d106-tap-receipt", required=True, type=Path)
    predict.add_argument("--d106-tap-receipt-sha256", required=True)
    predict.add_argument("--d106-rdce-wire", required=True, type=Path)
    predict.add_argument("--d106-rdce-wire-sha256", required=True)
    predict.add_argument("--seed", type=int, default=104713)
    predict.add_argument("--device", default="cuda:0")
    predict.add_argument("--smoke-one-row-no-truth", action="store_true")
    predict.set_defaults(func=run_predict)
    score = commands.add_parser("score")
    score.add_argument("--run-root", required=True, type=Path)
    score.add_argument("--truth", required=True, type=Path)
    score.add_argument("--truth-sha256", required=True)
    score.add_argument("--output", required=True, type=Path)
    score.set_defaults(func=run_score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if getattr(args, "command", None) in {"prepare", "predict"}:
        _require_file(args.received_iq_receipt, args.received_iq_receipt_sha256, "received-IQ receipt")
    if getattr(args, "command", None) == "predict":
        if args.seed < 0:
            raise NextR3Proxy24Error("seed must be non-negative")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
