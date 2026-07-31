"""Immutable, truth-separated local Target25 runner for frozen D105.

This module intentionally has no dataset loader and no N607 coupling.  A
caller supplies already-authorized, truth-free row contexts to ``predictor``;
the runner freezes the 25-job plan, seals immutable prediction artifacts, and
only then permits an independent truth-side callable to read labels.

The contract is deliberately narrow:

* one fixed seed (713102), five receivers and five frozen K/new-count slices;
* each outer job has three mutually disjoint LEO scenarios and four arms;
* one row artifact contains 12 scenario-arm before/after pairs (24 state surfaces);
* no output path can be reused or overwritten; and
* two distinct zero-prediction failures with the same normalized fingerprint
  stop later dispatch without inspecting any performance value.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from types import MappingProxyType
from typing import Any

from .stage2_d105_four_arm import ARMS


SCHEMA = "cvs.phase2.d105.target25_runner.v1"
PREDICTION_SCHEMA = SCHEMA + ".prediction"
PLAN_MANIFEST_SCHEMA = SCHEMA + ".plan_manifest"
PREDICTION_MANIFEST_SCHEMA = SCHEMA + ".prediction_manifest"
PARTIAL_MANIFEST_SCHEMA = SCHEMA + ".partial_prediction_manifest"
TRUTH_MANIFEST_SCHEMA = SCHEMA + ".truth_side_manifest"
TRUTH_CATALOG_SCHEMA = SCHEMA + ".truth_catalog"
TRUTH_OPEN_SCHEMA = SCHEMA + ".truth_first_open"
SCORE_SCHEMA = SCHEMA + ".score"
SCORE_MANIFEST_SCHEMA = SCHEMA + ".score_manifest"

PROTOCOL_SCHEMA = "p2_min_v1"
TARGET25_SEED = 713102
FORMAL_CLAIM_SCOPE = "FORMAL_CONFIRMATION"
DEVELOPMENT_CLAIM_SCOPE = "DEVELOPMENT_SCREEN_ONLY_NON_PROMOTABLE"
CLAIM_SCOPES = (FORMAL_CLAIM_SCOPE, DEVELOPMENT_CLAIM_SCOPE)
LEO_SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
TARGET25_SLICES = ((10, 5), (10, 10), (10, 20), (5, 20), (1, 20))
OUTER_ROW_COUNT = 25
SCENARIO_ROW_COUNT = 75
# 300 causal arm pairs, each with separately sealed S_B and S_C prediction
# surfaces.  Both counts are formal coverage gates and must never be conflated.
SCENARIO_ARM_PAIR_COUNT = 300
STATE_PREDICTION_SURFACE_COUNT = 600
SCENARIO_ARM_COUNT = SCENARIO_ARM_PAIR_COUNT


class D105Target25RunnerError(ValueError):
    """Raised when Target25 plan, prediction, or scoring closure drifts."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the canonical JSON SHA256 used by all D105 runner receipts."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise D105Target25RunnerError(f"{name} must be a lowercase SHA256")
    return text


def _require_text(value: Any, name: str) -> str:
    text = str(value)
    if not text or text.strip() != text:
        raise D105Target25RunnerError(f"{name} must be non-empty trimmed text")
    return text


def _unique_texts(values: Sequence[Any], name: str) -> tuple[str, ...]:
    result = tuple(_require_text(value, name) for value in values)
    if not result or len(set(result)) != len(result):
        raise D105Target25RunnerError(f"{name} must contain unique non-empty values")
    return result


def _physical_root(values: Sequence[str]) -> str:
    return canonical_sha256(sorted(values))


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    """Write, sync, and publish one read-only JSON file exactly once."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise D105Target25RunnerError(f"unsafe immutable-output parent: {path.parent}")
    payload = _canonical_bytes(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    # ``S_IREAD`` clears every write bit on POSIX and maps to Windows' readonly
    # file attribute.  Readers below reject a file if any write bit reappears.
    os.chmod(path, stat.S_IREAD)
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise D105Target25RunnerError(f"immutable output remains writable: {path}")
    return hashlib.sha256(payload).hexdigest()


def _write_bytes_new_readonly(path: Path, payload: bytes) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise D105Target25RunnerError(f"unsafe immutable-output parent: {path.parent}")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, stat.S_IREAD)
    if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise D105Target25RunnerError(f"immutable output remains writable: {path}")
    return hashlib.sha256(payload).hexdigest()


def _read_json_regular(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise D105Target25RunnerError(f"expected regular immutable JSON file: {path}")
    mode = path.stat().st_mode
    if not stat.S_ISREG(mode):
        raise D105Target25RunnerError(f"non-regular immutable JSON file: {path}")
    if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise D105Target25RunnerError(f"immutable JSON file is writable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise D105Target25RunnerError(f"expected JSON object: {path}")
    return value


def _relative_file(root: Path, value: str, name: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise D105Target25RunnerError(f"{name} must be a relative child path")
    candidate = (root / path).resolve(strict=True)
    root_resolved = root.resolve(strict=True)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise D105Target25RunnerError(f"{name} escapes its immutable root") from error
    return candidate


@dataclass(frozen=True, slots=True)
class D105Target25StatePlan:
    """One independently authorized Stage2-B or Stage2-C state surface."""

    stage: str
    capsule_id: str
    split_id: str
    authority_receipt_sha256: str
    authority_envelope_sha256: str
    data_feature_runtime_sha256: str
    data_materialization_lock_sha256: str
    d105_candidate_runtime_manifest_sha256: str
    d105_candidate_method_lock_sha256: str
    support_physical_ids: tuple[str, ...]
    query_physical_ids: tuple[str, ...]
    registered_classes: tuple[str, ...]
    old_classes: tuple[str, ...]
    new_classes: tuple[str, ...]
    prediction_context_sha256: str

    def __post_init__(self) -> None:
        if self.stage not in ("S_B", "S_C"):
            raise D105Target25RunnerError("state stage must be S_B or S_C")
        object.__setattr__(self, "capsule_id", _require_sha256(self.capsule_id, "capsule_id"))
        object.__setattr__(self, "split_id", _require_sha256(self.split_id, "split_id"))
        object.__setattr__(
            self,
            "authority_receipt_sha256",
            _require_sha256(self.authority_receipt_sha256, "authority_receipt_sha256"),
        )
        object.__setattr__(
            self,
            "authority_envelope_sha256",
            _require_sha256(
                self.authority_envelope_sha256, "authority_envelope_sha256"
            ),
        )
        for name in (
            "data_feature_runtime_sha256",
            "data_materialization_lock_sha256",
            "d105_candidate_runtime_manifest_sha256",
            "d105_candidate_method_lock_sha256",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), name)
            )
        support = _unique_texts(self.support_physical_ids, "support physical IDs")
        query = _unique_texts(self.query_physical_ids, "query physical IDs")
        registered = _unique_texts(self.registered_classes, "registered classes")
        old = _unique_texts(self.old_classes, "old classes")
        new = tuple(_require_text(value, "new classes") for value in self.new_classes)
        if len(set(new)) != len(new):
            raise D105Target25RunnerError("new classes must be unique")
        if set(support).intersection(query):
            raise D105Target25RunnerError("support/query physical IDs must be disjoint")
        if set(old).intersection(new) or registered != old + new:
            raise D105Target25RunnerError(
                "registered classes must be the ordered old/new lifecycle union"
            )
        if (self.stage == "S_B" and new) or (self.stage == "S_C" and not new):
            raise D105Target25RunnerError("state lifecycle does not match S_B/S_C")
        object.__setattr__(self, "support_physical_ids", support)
        object.__setattr__(self, "query_physical_ids", query)
        object.__setattr__(self, "registered_classes", registered)
        object.__setattr__(self, "old_classes", old)
        object.__setattr__(self, "new_classes", new)
        object.__setattr__(
            self,
            "prediction_context_sha256",
            _require_sha256(
                self.prediction_context_sha256, "prediction_context_sha256"
            ),
        )

    @property
    def support_physical_root_sha256(self) -> str:
        return _physical_root(self.support_physical_ids)

    @property
    def registration_state(self) -> str:
        return (
            "BEFORE_REGISTRATION" if self.stage == "S_B" else "AFTER_REGISTRATION"
        )

    @property
    def query_physical_root_sha256(self) -> str:
        return _physical_root(self.query_physical_ids)

    def receipt_payload(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "registration_state": self.registration_state,
            "capsule_id": self.capsule_id,
            "split_id": self.split_id,
            "authority_receipt_sha256": self.authority_receipt_sha256,
            "authority_envelope_sha256": self.authority_envelope_sha256,
            "data_feature_runtime_sha256": self.data_feature_runtime_sha256,
            "data_materialization_lock_sha256": self.data_materialization_lock_sha256,
            "d105_candidate_runtime_manifest_sha256": self.d105_candidate_runtime_manifest_sha256,
            "d105_candidate_method_lock_sha256": self.d105_candidate_method_lock_sha256,
            "protocol_schema": PROTOCOL_SCHEMA,
            "phase2_data_status": "VALIDATED_ONCE",
            "single_leo_observation": True,
            "clean_source_runtime_access": False,
            "query_fit_access": False,
            "query_decision_policy": "per_sample_all_registered_classes",
            "support_physical_ids": list(self.support_physical_ids),
            "support_physical_root_sha256": self.support_physical_root_sha256,
            "query_physical_ids": list(self.query_physical_ids),
            "query_physical_root_sha256": self.query_physical_root_sha256,
            "registered_classes": list(self.registered_classes),
            "old_classes": list(self.old_classes),
            "new_classes": list(self.new_classes),
            "prediction_context_sha256": self.prediction_context_sha256,
        }


@dataclass(frozen=True, slots=True)
class D105Target25ScenarioPlan:
    """One LEO scenario with separately sealed before/after state inputs."""

    scenario: str
    before: D105Target25StatePlan
    after: D105Target25StatePlan

    def __post_init__(self) -> None:
        if self.scenario not in LEO_SCENARIOS:
            raise D105Target25RunnerError("scenario must be one frozen leo_*_weak value")
        if (
            type(self.before) is not D105Target25StatePlan
            or type(self.after) is not D105Target25StatePlan
            or self.before.stage != "S_B"
            or self.after.stage != "S_C"
            or self.before.old_classes != self.after.old_classes
            or self.before.registered_classes != self.before.old_classes
            or not set(self.before.query_physical_ids) < set(
                self.after.query_physical_ids
            )
        ):
            raise D105Target25RunnerError(
                "scenario before/after state binding or old-query nesting drift"
            )

    def receipt_payload(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "before": self.before.receipt_payload(),
            "after": self.after.receipt_payload(),
        }


@dataclass(frozen=True, slots=True)
class D105Target25OuterRow:
    row_id: str
    receiver: str
    k_shot: int
    new_count: int
    scenarios: tuple[D105Target25ScenarioPlan, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_id", _require_text(self.row_id, "row_id"))
        object.__setattr__(self, "receiver", _require_text(self.receiver, "receiver"))
        if (self.k_shot, self.new_count) not in TARGET25_SLICES:
            raise D105Target25RunnerError("row uses a non-frozen Target25 slice")
        scenarios = tuple(self.scenarios)
        if tuple(item.scenario for item in scenarios) != LEO_SCENARIOS:
            raise D105Target25RunnerError("row scenarios must use frozen LEO order")
        object.__setattr__(self, "scenarios", scenarios)

    def receipt_payload(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "receiver": self.receiver,
            "k_shot": self.k_shot,
            "new_count": self.new_count,
            "scenarios": [item.receipt_payload() for item in self.scenarios],
        }

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self.receipt_payload())


@dataclass(frozen=True, slots=True)
class D105Target25Plan:
    seed: int
    claim_scope: str
    formal_launch_authority: bool
    authority_envelope_root_sha256: str
    data_feature_runtime_sha256: str
    data_materialization_lock_sha256: str
    d105_candidate_runtime_manifest_sha256: str
    d105_candidate_method_lock_sha256: str
    candidate_runtime_manifest_path: Path
    candidate_method_lock_path: Path
    rows: tuple[D105Target25OuterRow, ...]
    plan_receipt_sha256: str


def _plan_identity_payload(plan: D105Target25Plan) -> dict[str, Any]:
    return {
        "claim_scope": plan.claim_scope,
        "formal_launch_authority": plan.formal_launch_authority,
        "authority_envelope_root_sha256": plan.authority_envelope_root_sha256,
        "data_feature_runtime_sha256": plan.data_feature_runtime_sha256,
        "data_materialization_lock_sha256": plan.data_materialization_lock_sha256,
        "d105_candidate_runtime_manifest_sha256": plan.d105_candidate_runtime_manifest_sha256,
        "d105_candidate_method_lock_sha256": plan.d105_candidate_method_lock_sha256,
    }


def _plan_payload(
    *,
    seed: int,
    claim_scope: str,
    formal_launch_authority: bool,
    authority_envelope_root_sha256: str,
    data_feature_runtime_sha256: str,
    data_materialization_lock_sha256: str,
    d105_candidate_runtime_manifest_sha256: str,
    d105_candidate_method_lock_sha256: str,
    rows: Sequence[D105Target25OuterRow],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA + ".plan",
        "seed": seed,
        "claim_scope": claim_scope,
        "formal_launch_authority": formal_launch_authority,
        "authority_envelope_root_sha256": authority_envelope_root_sha256,
        "data_feature_runtime_sha256": data_feature_runtime_sha256,
        "data_materialization_lock_sha256": data_materialization_lock_sha256,
        "d105_candidate_runtime_manifest_sha256": d105_candidate_runtime_manifest_sha256,
        "d105_candidate_method_lock_sha256": d105_candidate_method_lock_sha256,
        "arms": list(ARMS),
        "leo_scenarios": list(LEO_SCENARIOS),
        "target25_slices": [list(item) for item in TARGET25_SLICES],
        "rows": [row.receipt_payload() for row in rows],
    }


def _validate_outer_row(row: D105Target25OuterRow) -> None:
    if len(row.scenarios) != 3:
        raise D105Target25RunnerError("each outer row must contain three scenarios")
    all_physical_ids: set[str] = set()
    first = row.scenarios[0]
    for scenario in row.scenarios:
        before = scenario.before
        after = scenario.after
        if len(after.new_classes) != row.new_count:
            raise D105Target25RunnerError("row new-count/lifecycle binding drift")
        for state in (before, after):
            if len(state.support_physical_ids) != row.k_shot * len(
                state.registered_classes
            ):
                raise D105Target25RunnerError(
                    "row support count is not exact K-shot coverage"
                )
        if (
            before.registered_classes != first.before.registered_classes
            or before.old_classes != first.before.old_classes
            or after.registered_classes != first.after.registered_classes
            or after.old_classes != first.after.old_classes
            or after.new_classes != first.after.new_classes
        ):
            raise D105Target25RunnerError("row registry/lifecycle must match across scenes")
        physical_ids = set(before.support_physical_ids).union(
            before.query_physical_ids,
            after.support_physical_ids,
            after.query_physical_ids,
        )
        if all_physical_ids.intersection(physical_ids):
            raise D105Target25RunnerError(
                "three scenario physical-ID sets must be pairwise disjoint"
            )
        all_physical_ids.update(physical_ids)


def _validate_plan_semantics(
    *,
    seed: int,
    claim_scope: str,
    formal_launch_authority: bool,
    authority_envelope_root_sha256: str,
    data_feature_runtime_sha256: str,
    data_materialization_lock_sha256: str,
    d105_candidate_runtime_manifest_sha256: str,
    d105_candidate_method_lock_sha256: str,
    rows: Sequence[D105Target25OuterRow],
) -> None:
    if type(seed) is not int or seed != TARGET25_SEED:
        raise D105Target25RunnerError(f"Target25 seed is frozen to {TARGET25_SEED}")
    if claim_scope not in CLAIM_SCOPES or type(formal_launch_authority) is not bool:
        raise D105Target25RunnerError("Target25 claim scope/authority mode drift")
    if (claim_scope == FORMAL_CLAIM_SCOPE) is not formal_launch_authority:
        raise D105Target25RunnerError(
            "formal claim scope requires formal launch authority and development scope forbids it"
        )
    authority_envelope_root_sha256 = _require_sha256(
        authority_envelope_root_sha256, "authority_envelope_root_sha256"
    )
    identity = {
        "data_feature_runtime_sha256": _require_sha256(
            data_feature_runtime_sha256, "data_feature_runtime_sha256"
        ),
        "data_materialization_lock_sha256": _require_sha256(
            data_materialization_lock_sha256, "data_materialization_lock_sha256"
        ),
        "d105_candidate_runtime_manifest_sha256": _require_sha256(
            d105_candidate_runtime_manifest_sha256,
            "d105_candidate_runtime_manifest_sha256",
        ),
        "d105_candidate_method_lock_sha256": _require_sha256(
            d105_candidate_method_lock_sha256,
            "d105_candidate_method_lock_sha256",
        ),
    }
    if len(rows) != OUTER_ROW_COUNT or len({row.row_id for row in rows}) != OUTER_ROW_COUNT:
        raise D105Target25RunnerError("Target25 must contain exactly 25 unique outer rows")
    receivers = tuple(row.receiver for row in rows[:: len(TARGET25_SLICES)])
    if len(receivers) != 5 or len(set(receivers)) != 5:
        raise D105Target25RunnerError("Target25 must contain exactly five receivers")
    expected = [
        (receiver, k_shot, new_count)
        for receiver in receivers
        for k_shot, new_count in TARGET25_SLICES
    ]
    actual = [(row.receiver, row.k_shot, row.new_count) for row in rows]
    if actual != expected:
        raise D105Target25RunnerError("Target25 row order/coverage drift")
    for row in rows:
        _validate_outer_row(row)
        for scenario in row.scenarios:
            for state in (scenario.before, scenario.after):
                if any(getattr(state, name) != value for name, value in identity.items()):
                    raise D105Target25RunnerError(
                        "Target25 state identity-plane binding drift"
                    )
    observed_envelopes = sorted(
        {
            state.authority_envelope_sha256
            for row in rows
            for scenario in row.scenarios
            for state in (scenario.before, scenario.after)
        }
    )
    if canonical_sha256(observed_envelopes) != authority_envelope_root_sha256:
        raise D105Target25RunnerError("Target25 authority envelope root drift")
    by_key = {(row.receiver, row.k_shot, row.new_count): row for row in rows}
    for receiver in receivers:
        k5 = by_key[(receiver, 5, 20)]
        k10 = by_key[(receiver, 10, 20)]
        k10_by_scene = {item.scenario: item for item in k10.scenarios}
        for k5_scene in k5.scenarios:
            k10_scene = k10_by_scene[k5_scene.scenario]
            for short, long in (
                (k5_scene.before, k10_scene.before),
                (k5_scene.after, k10_scene.after),
            ):
                if (
                    short.capsule_id != long.capsule_id
                    or short.registered_classes != long.registered_classes
                    or short.old_classes != long.old_classes
                    or short.new_classes != long.new_classes
                    or not set(short.support_physical_ids).issubset(
                        long.support_physical_ids
                    )
                    or short.query_physical_root_sha256
                    != long.query_physical_root_sha256
                    or set(short.query_physical_ids)
                    != set(long.query_physical_ids)
                ):
                    raise D105Target25RunnerError(
                        "K5/new20 requires same-capsule K10 subset support and query root"
                    )


def _load_candidate_identity(
    candidate_runtime_manifest_path: Path,
    candidate_method_lock_path: Path,
) -> tuple[Path, Path, str, str, dict[str, Any]]:
    from .stage2_d105_phase1_bundle import (
        load_d105_candidate_method_lock,
        load_d105_candidate_runtime_manifest,
    )

    runtime_source = Path(candidate_runtime_manifest_path)
    lock_source = Path(candidate_method_lock_path)
    for source, name in (
        (runtime_source, "candidate runtime manifest"),
        (lock_source, "candidate method lock"),
    ):
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_mode
            & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise D105Target25RunnerError(
                f"{name} must be a read-only regular non-symlink file"
            )
    runtime_path = runtime_source.resolve(strict=True)
    lock_path = lock_source.resolve(strict=True)
    try:
        runtime = load_d105_candidate_runtime_manifest(runtime_path)
        lock = load_d105_candidate_method_lock(
            lock_path,
            expected_checkpoint_sha256=runtime["checkpoint_sha256"],
            expected_runtime_sha256=runtime[
                "d105_candidate_runtime_manifest_sha256"
            ],
        )
    except ValueError as error:
        raise D105Target25RunnerError(
            "canonical D105 candidate identity preflight failed"
        ) from error
    return (
        runtime_path,
        lock_path,
        runtime["d105_candidate_runtime_manifest_sha256"],
        lock["d105_candidate_method_lock_sha256"],
        dict(lock["lock"]),
    )


def _validate_candidate_target25_claim(
    candidate_lock: Mapping[str, Any],
    *,
    claim_scope: str,
    formal_launch_authority: bool,
) -> None:
    target25 = candidate_lock.get("target25")
    if (
        not isinstance(target25, Mapping)
        or target25.get("claim_scope") != claim_scope
        or target25.get("formal_launch_authority") is not formal_launch_authority
    ):
        raise D105Target25RunnerError(
            "Target25 claim scope/formal authority differs from candidate method lock"
        )


def _verify_plan(plan: D105Target25Plan) -> None:
    if type(plan) is not D105Target25Plan:
        raise D105Target25RunnerError("exact D105Target25Plan required")
    runtime_path, lock_path, runtime_sha, lock_sha, candidate_lock = _load_candidate_identity(
        plan.candidate_runtime_manifest_path, plan.candidate_method_lock_path
    )
    if (
        runtime_path != plan.candidate_runtime_manifest_path
        or lock_path != plan.candidate_method_lock_path
        or runtime_sha != plan.d105_candidate_runtime_manifest_sha256
        or lock_sha != plan.d105_candidate_method_lock_sha256
    ):
        raise D105Target25RunnerError("D105 candidate identity source drift")
    _validate_candidate_target25_claim(
        candidate_lock,
        claim_scope=plan.claim_scope,
        formal_launch_authority=plan.formal_launch_authority,
    )
    _validate_plan_semantics(
        seed=plan.seed,
        claim_scope=plan.claim_scope,
        formal_launch_authority=plan.formal_launch_authority,
        authority_envelope_root_sha256=plan.authority_envelope_root_sha256,
        data_feature_runtime_sha256=plan.data_feature_runtime_sha256,
        data_materialization_lock_sha256=plan.data_materialization_lock_sha256,
        d105_candidate_runtime_manifest_sha256=runtime_sha,
        d105_candidate_method_lock_sha256=lock_sha,
        rows=plan.rows,
    )
    expected = canonical_sha256(
        _plan_payload(
            seed=plan.seed,
            claim_scope=plan.claim_scope,
            formal_launch_authority=plan.formal_launch_authority,
            authority_envelope_root_sha256=plan.authority_envelope_root_sha256,
            data_feature_runtime_sha256=plan.data_feature_runtime_sha256,
            data_materialization_lock_sha256=plan.data_materialization_lock_sha256,
            d105_candidate_runtime_manifest_sha256=runtime_sha,
            d105_candidate_method_lock_sha256=lock_sha,
            rows=plan.rows,
        )
    )
    if expected != plan.plan_receipt_sha256:
        raise D105Target25RunnerError("Target25 plan receipt drift")


def freeze_d105_target25_plan(
    *,
    candidate_runtime_manifest_path: Path,
    candidate_method_lock_path: Path,
    receivers: Sequence[str],
    scenario_plans: Mapping[
        tuple[str, int, int], Sequence[D105Target25ScenarioPlan]
    ],
    seed: int = TARGET25_SEED,
    claim_scope: str = DEVELOPMENT_CLAIM_SCOPE,
    formal_launch_authority: bool = False,
) -> D105Target25Plan:
    """Freeze the exact 5 receiver × 5 slice Target25 matrix.

    ``scenario_plans`` contains truth-free validated D92-style authority
    identities keyed by ``(receiver, K, new_count)``.  No prediction or metric
    is read while the matrix is frozen.
    """

    receiver_ids = _unique_texts(receivers, "receiver")
    if len(receiver_ids) != 5:
        raise D105Target25RunnerError("Target25 requires exactly five receivers")
    expected_keys = {
        (receiver, k_shot, new_count)
        for receiver in receiver_ids
        for k_shot, new_count in TARGET25_SLICES
    }
    if set(scenario_plans) != expected_keys:
        raise D105Target25RunnerError("Target25 scenario-plan keys do not close at 25")
    rows: list[D105Target25OuterRow] = []
    for receiver_index, receiver in enumerate(receiver_ids):
        for k_shot, new_count in TARGET25_SLICES:
            key = (receiver, k_shot, new_count)
            rows.append(
                D105Target25OuterRow(
                    row_id=f"row-{receiver_index:02d}-k{k_shot}-n{new_count}",
                    receiver=receiver,
                    k_shot=k_shot,
                    new_count=new_count,
                    scenarios=tuple(scenario_plans[key]),
                )
            )
    runtime_path, lock_path, candidate_runtime_sha, candidate_lock_sha, candidate_lock = (
        _load_candidate_identity(
            candidate_runtime_manifest_path, candidate_method_lock_path
        )
    )
    _validate_candidate_target25_claim(
        candidate_lock,
        claim_scope=claim_scope,
        formal_launch_authority=formal_launch_authority,
    )
    first_state = rows[0].scenarios[0].before
    data_runtime_sha = first_state.data_feature_runtime_sha256
    data_lock_sha = first_state.data_materialization_lock_sha256
    authority_envelope_root_sha = canonical_sha256(
        sorted(
            {
                state.authority_envelope_sha256
                for row in rows
                for scenario in row.scenarios
                for state in (scenario.before, scenario.after)
            }
        )
    )
    _validate_plan_semantics(
        seed=seed,
        claim_scope=claim_scope,
        formal_launch_authority=formal_launch_authority,
        authority_envelope_root_sha256=authority_envelope_root_sha,
        data_feature_runtime_sha256=data_runtime_sha,
        data_materialization_lock_sha256=data_lock_sha,
        d105_candidate_runtime_manifest_sha256=candidate_runtime_sha,
        d105_candidate_method_lock_sha256=candidate_lock_sha,
        rows=rows,
    )
    plan = D105Target25Plan(
        seed=seed,
        claim_scope=claim_scope,
        formal_launch_authority=formal_launch_authority,
        authority_envelope_root_sha256=authority_envelope_root_sha,
        data_feature_runtime_sha256=data_runtime_sha,
        data_materialization_lock_sha256=data_lock_sha,
        d105_candidate_runtime_manifest_sha256=candidate_runtime_sha,
        d105_candidate_method_lock_sha256=candidate_lock_sha,
        candidate_runtime_manifest_path=runtime_path,
        candidate_method_lock_path=lock_path,
        rows=tuple(rows),
        plan_receipt_sha256=canonical_sha256(
            _plan_payload(
                seed=seed,
                claim_scope=claim_scope,
                formal_launch_authority=formal_launch_authority,
                authority_envelope_root_sha256=authority_envelope_root_sha,
                data_feature_runtime_sha256=data_runtime_sha,
                data_materialization_lock_sha256=data_lock_sha,
                d105_candidate_runtime_manifest_sha256=candidate_runtime_sha,
                d105_candidate_method_lock_sha256=candidate_lock_sha,
                rows=rows,
            )
        ),
    )
    _verify_plan(plan)
    return plan


def _state_plan_from_receipt_payload(value: Any) -> D105Target25StatePlan:
    """Rebuild one state only from a sealed plan-manifest receipt payload."""

    expected_keys = {
        "stage",
        "registration_state",
        "capsule_id",
        "split_id",
        "authority_receipt_sha256",
        "authority_envelope_sha256",
        "data_feature_runtime_sha256",
        "data_materialization_lock_sha256",
        "d105_candidate_runtime_manifest_sha256",
        "d105_candidate_method_lock_sha256",
        "protocol_schema",
        "phase2_data_status",
        "single_leo_observation",
        "clean_source_runtime_access",
        "query_fit_access",
        "query_decision_policy",
        "support_physical_ids",
        "support_physical_root_sha256",
        "query_physical_ids",
        "query_physical_root_sha256",
        "registered_classes",
        "old_classes",
        "new_classes",
        "prediction_context_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise D105Target25RunnerError("sealed state-plan manifest field closure drift")
    state = D105Target25StatePlan(
        stage=str(value["stage"]),
        capsule_id=value["capsule_id"],
        split_id=value["split_id"],
        authority_receipt_sha256=value["authority_receipt_sha256"],
        authority_envelope_sha256=value["authority_envelope_sha256"],
        data_feature_runtime_sha256=value["data_feature_runtime_sha256"],
        data_materialization_lock_sha256=value[
            "data_materialization_lock_sha256"
        ],
        d105_candidate_runtime_manifest_sha256=value[
            "d105_candidate_runtime_manifest_sha256"
        ],
        d105_candidate_method_lock_sha256=value[
            "d105_candidate_method_lock_sha256"
        ],
        support_physical_ids=tuple(value["support_physical_ids"]),
        query_physical_ids=tuple(value["query_physical_ids"]),
        registered_classes=tuple(value["registered_classes"]),
        old_classes=tuple(value["old_classes"]),
        new_classes=tuple(value["new_classes"]),
        prediction_context_sha256=value["prediction_context_sha256"],
    )
    if state.receipt_payload() != dict(value):
        raise D105Target25RunnerError("sealed state-plan manifest receipt drift")
    return state


def _scenario_plan_from_receipt_payload(value: Any) -> D105Target25ScenarioPlan:
    if not isinstance(value, Mapping) or set(value) != {"scenario", "before", "after"}:
        raise D105Target25RunnerError("sealed scenario-plan manifest field closure drift")
    scenario = D105Target25ScenarioPlan(
        scenario=str(value["scenario"]),
        before=_state_plan_from_receipt_payload(value["before"]),
        after=_state_plan_from_receipt_payload(value["after"]),
    )
    if scenario.receipt_payload() != dict(value):
        raise D105Target25RunnerError("sealed scenario-plan manifest receipt drift")
    return scenario


def write_d105_target25_plan_manifest(path: Path, plan: D105Target25Plan) -> str:
    """Publish a read-only, reconstructable Target25 plan before execution."""

    _verify_plan(plan)
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable output already exists: {destination}")
    identity_dir = destination.parent / f"{destination.name}.inputs"
    if identity_dir.exists() or identity_dir.is_symlink():
        raise FileExistsError(
            f"immutable plan identity input already exists: {identity_dir}"
        )
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise D105Target25RunnerError("unsafe plan-manifest parent")
    identity_dir.mkdir()
    runtime_name = "d105_candidate_runtime_manifest.json"
    lock_name = "d105_candidate_method_lock.json"
    _write_bytes_new_readonly(
        identity_dir / runtime_name,
        plan.candidate_runtime_manifest_path.read_bytes(),
    )
    _write_bytes_new_readonly(
        identity_dir / lock_name,
        plan.candidate_method_lock_path.read_bytes(),
    )
    document: dict[str, Any] = {
        "schema": PLAN_MANIFEST_SCHEMA,
        "plan_payload": _plan_payload(
            seed=plan.seed,
            claim_scope=plan.claim_scope,
            formal_launch_authority=plan.formal_launch_authority,
            authority_envelope_root_sha256=plan.authority_envelope_root_sha256,
            data_feature_runtime_sha256=plan.data_feature_runtime_sha256,
            data_materialization_lock_sha256=plan.data_materialization_lock_sha256,
            d105_candidate_runtime_manifest_sha256=plan.d105_candidate_runtime_manifest_sha256,
            d105_candidate_method_lock_sha256=plan.d105_candidate_method_lock_sha256,
            rows=plan.rows,
        ),
        "candidate_identity_sources": {
            "candidate_runtime_manifest_path": (
                f"{identity_dir.name}/{runtime_name}"
            ),
            "candidate_method_lock_path": f"{identity_dir.name}/{lock_name}",
        },
        "plan_receipt_sha256": plan.plan_receipt_sha256,
    }
    document["plan_manifest_receipt_sha256"] = canonical_sha256(document)
    return _write_json_new(destination, document)


def load_d105_target25_plan_manifest(path: Path) -> D105Target25Plan:
    """Fail closed while reconstructing the exact frozen Target25 plan."""

    document = _read_json_regular(Path(path))
    expected_document_keys = {
        "schema",
        "plan_payload",
        "candidate_identity_sources",
        "plan_receipt_sha256",
        "plan_manifest_receipt_sha256",
    }
    if set(document) != expected_document_keys or document.get("schema") != PLAN_MANIFEST_SCHEMA:
        raise D105Target25RunnerError("Target25 plan manifest field closure drift")
    if document.get("plan_manifest_receipt_sha256") != canonical_sha256(
        {
            key: value
            for key, value in document.items()
            if key != "plan_manifest_receipt_sha256"
        }
    ):
        raise D105Target25RunnerError("Target25 plan manifest receipt drift")
    payload = document.get("plan_payload")
    expected_payload_keys = {
        "schema",
        "seed",
        "claim_scope",
        "formal_launch_authority",
        "authority_envelope_root_sha256",
        "data_feature_runtime_sha256",
        "data_materialization_lock_sha256",
        "d105_candidate_runtime_manifest_sha256",
        "d105_candidate_method_lock_sha256",
        "arms",
        "leo_scenarios",
        "target25_slices",
        "rows",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_payload_keys or (
        payload.get("schema") != SCHEMA + ".plan"
        or payload.get("arms") != list(ARMS)
        or payload.get("leo_scenarios") != list(LEO_SCENARIOS)
        or payload.get("target25_slices") != [list(item) for item in TARGET25_SLICES]
        or not isinstance(payload.get("rows"), list)
    ):
        raise D105Target25RunnerError("Target25 plan payload closure drift")
    identity_sources = document.get("candidate_identity_sources")
    if not isinstance(identity_sources, Mapping) or set(identity_sources) != {
        "candidate_runtime_manifest_path",
        "candidate_method_lock_path",
    }:
        raise D105Target25RunnerError("candidate identity source closure drift")
    expected_runtime_relative = (
        f"{Path(path).name}.inputs/d105_candidate_runtime_manifest.json"
    )
    expected_lock_relative = (
        f"{Path(path).name}.inputs/d105_candidate_method_lock.json"
    )
    if (
        identity_sources["candidate_runtime_manifest_path"]
        != expected_runtime_relative
        or identity_sources["candidate_method_lock_path"] != expected_lock_relative
    ):
        raise D105Target25RunnerError(
            "candidate identity source portable-relative path drift"
        )
    runtime_source = _relative_file(
        Path(path).parent,
        expected_runtime_relative,
        "candidate_runtime_manifest_path",
    )
    lock_source = _relative_file(
        Path(path).parent,
        expected_lock_relative,
        "candidate_method_lock_path",
    )
    runtime_path, lock_path, runtime_sha, lock_sha, candidate_lock = _load_candidate_identity(
        runtime_source, lock_source
    )
    if (
        payload.get("d105_candidate_runtime_manifest_sha256") != runtime_sha
        or payload.get("d105_candidate_method_lock_sha256") != lock_sha
    ):
        raise D105Target25RunnerError("plan/candidate identity source drift")
    _validate_candidate_target25_claim(
        candidate_lock,
        claim_scope=str(payload.get("claim_scope")),
        formal_launch_authority=payload.get("formal_launch_authority"),
    )
    rows: list[D105Target25OuterRow] = []
    for value in payload["rows"]:
        expected_row_keys = {"row_id", "receiver", "k_shot", "new_count", "scenarios"}
        if (
            not isinstance(value, Mapping)
            or set(value) != expected_row_keys
            or not isinstance(value.get("scenarios"), list)
        ):
            raise D105Target25RunnerError("sealed outer-row manifest field closure drift")
        row = D105Target25OuterRow(
            row_id=str(value["row_id"]),
            receiver=str(value["receiver"]),
            k_shot=value["k_shot"],
            new_count=value["new_count"],
            scenarios=tuple(
                _scenario_plan_from_receipt_payload(item)
                for item in value["scenarios"]
            ),
        )
        if row.receipt_payload() != dict(value):
            raise D105Target25RunnerError("sealed outer-row manifest receipt drift")
        rows.append(row)
    plan = D105Target25Plan(
        seed=payload["seed"],
        claim_scope=payload["claim_scope"],
        formal_launch_authority=payload["formal_launch_authority"],
        authority_envelope_root_sha256=payload["authority_envelope_root_sha256"],
        data_feature_runtime_sha256=payload["data_feature_runtime_sha256"],
        data_materialization_lock_sha256=payload[
            "data_materialization_lock_sha256"
        ],
        d105_candidate_runtime_manifest_sha256=runtime_sha,
        d105_candidate_method_lock_sha256=lock_sha,
        candidate_runtime_manifest_path=runtime_path,
        candidate_method_lock_path=lock_path,
        rows=tuple(rows),
        plan_receipt_sha256=_require_sha256(
            document.get("plan_receipt_sha256"), "plan_receipt_sha256"
        ),
    )
    _verify_plan(plan)
    if canonical_sha256(dict(payload)) != plan.plan_receipt_sha256:
        raise D105Target25RunnerError("Target25 plan payload/receipt binding drift")
    return plan


@dataclass(frozen=True, slots=True)
class D105Target25GPUSchedule:
    """Deterministic scheduler metadata; local tests never require a GPU."""

    gpu_ids: tuple[int, ...]
    workers_per_gpu: int = 1

    def __post_init__(self) -> None:
        gpu_ids = tuple(self.gpu_ids)
        if (
            not gpu_ids
            or any(type(item) is not int or item < 0 for item in gpu_ids)
            or len(set(gpu_ids)) != len(gpu_ids)
            or type(self.workers_per_gpu) is not int
            or self.workers_per_gpu < 1
        ):
            raise D105Target25RunnerError("invalid deterministic GPU schedule metadata")
        object.__setattr__(self, "gpu_ids", gpu_ids)


@dataclass(frozen=True, slots=True)
class D105Target25Assignment:
    row_id: str
    gpu_id: int
    worker_slot: int
    dispatch_index: int


def _assignments(
    plan: D105Target25Plan, schedule: D105Target25GPUSchedule
) -> tuple[D105Target25Assignment, ...]:
    capacity = len(schedule.gpu_ids) * schedule.workers_per_gpu
    result: list[D105Target25Assignment] = []
    for index, row in enumerate(plan.rows):
        slot = index % capacity
        result.append(
            D105Target25Assignment(
                row_id=row.row_id,
                gpu_id=schedule.gpu_ids[slot % len(schedule.gpu_ids)],
                worker_slot=slot // len(schedule.gpu_ids),
                dispatch_index=index,
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class D105Target25Run:
    plan: D105Target25Plan
    run_id: str
    run_root: Path
    schedule: D105Target25GPUSchedule
    assignments: tuple[D105Target25Assignment, ...]
    run_manifest_sha256: str


def _require_run_id(value: str) -> str:
    text = _require_text(value, "run_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,127}", text):
        raise D105Target25RunnerError("run_id has unsafe characters")
    return text


def prepare_d105_target25_run(
    plan: D105Target25Plan,
    *,
    output_root: Path,
    run_id: str,
    schedule: D105Target25GPUSchedule,
) -> D105Target25Run:
    """Create a new, non-overwritable local prediction root for one Target25."""

    _verify_plan(plan)
    if type(schedule) is not D105Target25GPUSchedule:
        raise D105Target25RunnerError("exact GPU schedule metadata required")
    base = Path(output_root).resolve(strict=True)
    if base.is_symlink() or not base.is_dir():
        raise D105Target25RunnerError("output_root must be an existing regular directory")
    safe_run_id = _require_run_id(run_id)
    run_root = base / safe_run_id
    if run_root.exists() or run_root.is_symlink():
        raise FileExistsError(f"immutable run ID already exists: {safe_run_id}")
    run_root.mkdir()
    (run_root / "predictions").mkdir()
    (run_root / "row_logs").mkdir()
    assignments = _assignments(plan, schedule)
    manifest = {
        "schema": SCHEMA + ".run_manifest",
        "status": "PREPARED_LOCAL_ONLY",
        "run_id": safe_run_id,
        "plan_receipt_sha256": plan.plan_receipt_sha256,
        **_plan_identity_payload(plan),
        "seed": plan.seed,
        "outer_row_count": OUTER_ROW_COUNT,
        "scenario_row_count": SCENARIO_ROW_COUNT,
                "scenario_arm_pair_count": SCENARIO_ARM_PAIR_COUNT,
                "state_prediction_surface_count": STATE_PREDICTION_SURFACE_COUNT,
        "arms": list(ARMS),
        "gpu_schedule": {
            "gpu_ids": list(schedule.gpu_ids),
            "workers_per_gpu": schedule.workers_per_gpu,
        },
        "assignments": [
            {
                "row_id": item.row_id,
                "gpu_id": item.gpu_id,
                "worker_slot": item.worker_slot,
                "dispatch_index": item.dispatch_index,
            }
            for item in assignments
        ],
        "n607_access": False,
        "query_truth_access": False,
        "performance_selection_used": False,
    }
    manifest_sha = _write_json_new(run_root / "run_manifest.json", manifest)
    return D105Target25Run(
        plan=plan,
        run_id=safe_run_id,
        run_root=run_root,
        schedule=schedule,
        assignments=assignments,
        run_manifest_sha256=manifest_sha,
    )


def _verify_run(run: D105Target25Run) -> None:
    if type(run) is not D105Target25Run:
        raise D105Target25RunnerError("exact D105Target25Run required")
    _verify_plan(run.plan)
    if len(run.assignments) != OUTER_ROW_COUNT:
        raise D105Target25RunnerError("run assignment coverage drift")
    manifest_path = run.run_root / "run_manifest.json"
    if _sha256_file(manifest_path) != run.run_manifest_sha256:
        raise D105Target25RunnerError("immutable run manifest SHA drift")
    manifest = _read_json_regular(manifest_path)
    if (
        manifest.get("schema") != SCHEMA + ".run_manifest"
        or manifest.get("run_id") != run.run_id
        or manifest.get("plan_receipt_sha256") != run.plan.plan_receipt_sha256
        or any(
            manifest.get(name) != value
            for name, value in _plan_identity_payload(run.plan).items()
        )
        or manifest.get("status") != "PREPARED_LOCAL_ONLY"
        or manifest.get("scenario_arm_pair_count") != SCENARIO_ARM_PAIR_COUNT
        or manifest.get("state_prediction_surface_count")
        != STATE_PREDICTION_SURFACE_COUNT
    ):
        raise D105Target25RunnerError("run manifest binding drift")


def load_d105_target25_run(
    plan: D105Target25Plan, run_root: Path
) -> D105Target25Run:
    """Reconstruct one sealed run for an independent formal scoring process."""

    _verify_plan(plan)
    source_root = Path(run_root)
    if source_root.is_symlink():
        raise D105Target25RunnerError("run_root must be an existing regular directory")
    root = source_root.resolve(strict=True)
    if not root.is_dir():
        raise D105Target25RunnerError("run_root must be an existing regular directory")
    manifest_path = root / "run_manifest.json"
    manifest = _read_json_regular(manifest_path)
    expected_top_keys = {
        "schema",
        "status",
        "run_id",
        "plan_receipt_sha256",
        "claim_scope",
        "formal_launch_authority",
        "authority_envelope_root_sha256",
        "data_feature_runtime_sha256",
        "data_materialization_lock_sha256",
        "d105_candidate_runtime_manifest_sha256",
        "d105_candidate_method_lock_sha256",
        "seed",
        "outer_row_count",
        "scenario_row_count",
        "scenario_arm_pair_count",
        "state_prediction_surface_count",
        "arms",
        "gpu_schedule",
        "assignments",
        "n607_access",
        "query_truth_access",
        "performance_selection_used",
    }
    if set(manifest) != expected_top_keys:
        raise D105Target25RunnerError("run manifest top-level schema closure drift")
    schedule_payload = manifest.get("gpu_schedule")
    if not isinstance(schedule_payload, dict) or set(schedule_payload) != {
        "gpu_ids",
        "workers_per_gpu",
    }:
        raise D105Target25RunnerError("run manifest GPU schedule schema closure drift")
    gpu_ids = schedule_payload.get("gpu_ids")
    if not isinstance(gpu_ids, list):
        raise D105Target25RunnerError("run manifest GPU IDs must be a list")
    schedule = D105Target25GPUSchedule(
        gpu_ids=tuple(gpu_ids),
        workers_per_gpu=schedule_payload.get("workers_per_gpu"),
    )
    run_id = _require_run_id(manifest.get("run_id"))
    if root.name != run_id:
        raise D105Target25RunnerError("run root/run ID binding drift")
    assignments = _assignments(plan, schedule)
    expected_assignments = [
        {
            "row_id": item.row_id,
            "gpu_id": item.gpu_id,
            "worker_slot": item.worker_slot,
            "dispatch_index": item.dispatch_index,
        }
        for item in assignments
    ]
    if (
        manifest.get("schema") != SCHEMA + ".run_manifest"
        or manifest.get("status") != "PREPARED_LOCAL_ONLY"
        or manifest.get("plan_receipt_sha256") != plan.plan_receipt_sha256
        or any(
            manifest.get(name) != value
            for name, value in _plan_identity_payload(plan).items()
        )
        or manifest.get("seed") != plan.seed
        or manifest.get("outer_row_count") != OUTER_ROW_COUNT
        or manifest.get("scenario_row_count") != SCENARIO_ROW_COUNT
        or manifest.get("scenario_arm_pair_count") != SCENARIO_ARM_PAIR_COUNT
        or manifest.get("state_prediction_surface_count")
        != STATE_PREDICTION_SURFACE_COUNT
        or manifest.get("arms") != list(ARMS)
        or manifest.get("assignments") != expected_assignments
        or manifest.get("n607_access") is not False
        or manifest.get("query_truth_access") is not False
        or manifest.get("performance_selection_used") is not False
    ):
        raise D105Target25RunnerError("run manifest binding drift")
    run = D105Target25Run(
        plan=plan,
        run_id=run_id,
        run_root=root,
        schedule=schedule,
        assignments=assignments,
        run_manifest_sha256=_sha256_file(manifest_path),
    )
    _verify_run(run)
    return run


@dataclass(frozen=True, slots=True)
class D105Target25PredictionRequest:
    """The sole predictor input surface; it intentionally contains no truth.

    ``prediction_context_sha256`` is the explicit immutable hook for a caller-owned
    ``stage2_d105_feature_tap`` output.  This runner never opens IQ or creates
    a second backbone/tap path.
    """

    run_id: str
    row_id: str
    receiver: str
    seed: int
    k_shot: int
    new_count: int
    scenario: str
    stage: str
    registration_state: str
    capsule_id: str
    split_id: str
    authority_receipt_sha256: str
    authority_envelope_sha256: str
    data_feature_runtime_sha256: str
    data_materialization_lock_sha256: str
    d105_candidate_runtime_manifest_sha256: str
    d105_candidate_method_lock_sha256: str
    support_physical_ids: tuple[str, ...]
    query_physical_ids: tuple[str, ...]
    registered_classes: tuple[str, ...]
    old_classes: tuple[str, ...]
    new_classes: tuple[str, ...]
    prediction_context_sha256: str
    gpu_id: int
    worker_slot: int


@dataclass(frozen=True, slots=True)
class D105Target25PredictionOutput:
    """One truth-free D105StatePrediction returned by the evaluator hook."""

    stage: str
    registration_state: str
    arm_predictions: Mapping[str, Sequence[str]]
    state_receipt_sha256: str
    predictor_receipt_sha256: str
    feature_receipt_sha256: str
    resource_receipt_sha256: str
    logit_sha256_by_arm: Mapping[str, str]
    arm_prediction_sha256_by_arm: Mapping[str, str] | None = None
    query_rows_used_for_fit: int = 0
    query_state_updates: int = 0
    all_registered_classes_compete: bool = True
    per_query_independent: bool = True

    def __post_init__(self) -> None:
        if self.stage not in ("S_B", "S_C") or self.registration_state != (
            "BEFORE_REGISTRATION"
            if self.stage == "S_B"
            else "AFTER_REGISTRATION"
        ):
            raise D105Target25RunnerError("evaluator state/registration binding drift")
        if set(self.arm_predictions) != set(ARMS):
            raise D105Target25RunnerError("prediction output must contain exactly four arms")
        if set(self.logit_sha256_by_arm) != set(ARMS):
            raise D105Target25RunnerError("evaluator logit receipt arms drift")
        if self.arm_prediction_sha256_by_arm is not None and (
            not isinstance(self.arm_prediction_sha256_by_arm, Mapping)
            or set(self.arm_prediction_sha256_by_arm) != set(ARMS)
        ):
            raise D105Target25RunnerError(
                "evaluator arm-prediction receipt arms drift"
            )
        frozen = {arm: tuple(str(item) for item in self.arm_predictions[arm]) for arm in ARMS}
        object.__setattr__(self, "arm_predictions", MappingProxyType(frozen))
        prediction_hashes = (
            {
                arm: canonical_sha256(list(frozen[arm]))
                for arm in ARMS
            }
            if self.arm_prediction_sha256_by_arm is None
            else {
                arm: _require_sha256(
                    self.arm_prediction_sha256_by_arm[arm],
                    f"{arm} arm_prediction_sha256",
                )
                for arm in ARMS
            }
        )
        if set(prediction_hashes) != set(ARMS):
            raise D105Target25RunnerError("evaluator arm-prediction receipt arms drift")
        object.__setattr__(
            self,
            "arm_prediction_sha256_by_arm",
            MappingProxyType(prediction_hashes),
        )
        object.__setattr__(
            self,
            "state_receipt_sha256",
            _require_sha256(self.state_receipt_sha256, "state_receipt_sha256"),
        )
        object.__setattr__(
            self,
            "predictor_receipt_sha256",
            _require_sha256(self.predictor_receipt_sha256, "predictor_receipt_sha256"),
        )
        object.__setattr__(
            self,
            "feature_receipt_sha256",
            _require_sha256(self.feature_receipt_sha256, "feature_receipt_sha256"),
        )
        object.__setattr__(
            self,
            "resource_receipt_sha256",
            _require_sha256(self.resource_receipt_sha256, "resource_receipt_sha256"),
        )
        object.__setattr__(
            self,
            "logit_sha256_by_arm",
            MappingProxyType(
                {
                    arm: _require_sha256(
                        self.logit_sha256_by_arm[arm], f"{arm} logit_sha256"
                    )
                    for arm in ARMS
                }
            ),
        )


def _prediction_request(
    run: D105Target25Run,
    row: D105Target25OuterRow,
    scenario: D105Target25ScenarioPlan,
    state: D105Target25StatePlan,
    assignment: D105Target25Assignment,
) -> D105Target25PredictionRequest:
    return D105Target25PredictionRequest(
        run_id=run.run_id,
        row_id=row.row_id,
        receiver=row.receiver,
        seed=run.plan.seed,
        k_shot=row.k_shot,
        new_count=row.new_count,
        scenario=scenario.scenario,
        stage=state.stage,
        registration_state=state.registration_state,
        capsule_id=state.capsule_id,
        split_id=state.split_id,
        authority_receipt_sha256=state.authority_receipt_sha256,
        authority_envelope_sha256=state.authority_envelope_sha256,
        data_feature_runtime_sha256=state.data_feature_runtime_sha256,
        data_materialization_lock_sha256=state.data_materialization_lock_sha256,
        d105_candidate_runtime_manifest_sha256=state.d105_candidate_runtime_manifest_sha256,
        d105_candidate_method_lock_sha256=state.d105_candidate_method_lock_sha256,
        support_physical_ids=state.support_physical_ids,
        query_physical_ids=state.query_physical_ids,
        registered_classes=state.registered_classes,
        old_classes=state.old_classes,
        new_classes=state.new_classes,
        prediction_context_sha256=state.prediction_context_sha256,
        gpu_id=assignment.gpu_id,
        worker_slot=assignment.worker_slot,
    )


def _state_scorer_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": value["stage"],
        "registration_state": value["registration_state"],
        "capsule_id": value["capsule_id"],
        "split_id": value["split_id"],
        "authority_receipt_sha256": value["authority_receipt_sha256"],
        "authority_envelope_sha256": value["authority_envelope_sha256"],
        "data_feature_runtime_sha256": value["data_feature_runtime_sha256"],
        "data_materialization_lock_sha256": value[
            "data_materialization_lock_sha256"
        ],
        "d105_candidate_runtime_manifest_sha256": value[
            "d105_candidate_runtime_manifest_sha256"
        ],
        "d105_candidate_method_lock_sha256": value[
            "d105_candidate_method_lock_sha256"
        ],
        "protocol_schema": value["protocol_schema"],
        "phase2_data_status": value["phase2_data_status"],
        "support_physical_root_sha256": value["support_physical_root_sha256"],
        "query_physical_ids": value["query_physical_ids"],
        "query_physical_root_sha256": value["query_physical_root_sha256"],
        "registered_classes": value["registered_classes"],
        "old_classes": value["old_classes"],
        "new_classes": value["new_classes"],
        "prediction_context_sha256": value["prediction_context_sha256"],
        "arm_predictions": value["arm_predictions"],
        "arm_prediction_sha256_by_arm": value["arm_prediction_sha256_by_arm"],
        "arm_prediction_receipts": value["arm_prediction_receipts"],
        "state_receipt_sha256": value["state_receipt_sha256"],
        "predictor_receipt_sha256": value["predictor_receipt_sha256"],
        "feature_receipt_sha256": value["feature_receipt_sha256"],
        "resource_receipt_sha256": value["resource_receipt_sha256"],
        "logit_sha256_by_arm": value["logit_sha256_by_arm"],
        "query_rows_used_for_fit": value["query_rows_used_for_fit"],
        "query_state_updates": value["query_state_updates"],
        "all_registered_classes_compete": value["all_registered_classes_compete"],
        "per_query_independent": value["per_query_independent"],
        "query_truth_access": value["query_truth_access"],
        "query_role_access": value["query_role_access"],
        "query_quota_access": value["query_quota_access"],
        "global_assignment": value["global_assignment"],
    }


def _make_state_prediction(
    *,
    run: D105Target25Run,
    row: D105Target25OuterRow,
    scenario: D105Target25ScenarioPlan,
    state: D105Target25StatePlan,
    output: D105Target25PredictionOutput,
) -> dict[str, Any]:
    if type(output) is not D105Target25PredictionOutput:
        raise D105Target25RunnerError("predictor must return D105Target25PredictionOutput")
    if (
        output.stage != state.stage
        or output.registration_state != state.registration_state
        or output.query_rows_used_for_fit != 0
        or output.query_state_updates != 0
        or output.all_registered_classes_compete is not True
        or output.per_query_independent is not True
    ):
        raise D105Target25RunnerError("predictor query-isolation receipt drift")
    predictions = {arm: list(output.arm_predictions[arm]) for arm in ARMS}
    if any(
        len(predictions[arm]) != len(state.query_physical_ids)
        or any(item not in state.registered_classes for item in predictions[arm])
        for arm in ARMS
    ):
        raise D105Target25RunnerError("prediction labels do not close on registered classes")
    if row.k_shot == 1 and (
        predictions["M_HEAD"] != predictions["M0"]
        or predictions["M_JOINT"] != predictions["M_DA"]
    ):
        raise D105Target25RunnerError("K1 requires exact HEAD/base and JOINT/DA identity")
    result: dict[str, Any] = {
        "stage": state.stage,
        "registration_state": state.registration_state,
        "capsule_id": state.capsule_id,
        "split_id": state.split_id,
        "authority_receipt_sha256": state.authority_receipt_sha256,
        "authority_envelope_sha256": state.authority_envelope_sha256,
        "data_feature_runtime_sha256": state.data_feature_runtime_sha256,
        "data_materialization_lock_sha256": state.data_materialization_lock_sha256,
        "d105_candidate_runtime_manifest_sha256": state.d105_candidate_runtime_manifest_sha256,
        "d105_candidate_method_lock_sha256": state.d105_candidate_method_lock_sha256,
        "protocol_schema": PROTOCOL_SCHEMA,
        "phase2_data_status": "VALIDATED_ONCE",
        "single_leo_observation": True,
        "clean_source_runtime_access": False,
        "support_physical_root_sha256": state.support_physical_root_sha256,
        "query_physical_ids": list(state.query_physical_ids),
        "query_physical_root_sha256": state.query_physical_root_sha256,
        "registered_classes": list(state.registered_classes),
        "old_classes": list(state.old_classes),
        "new_classes": list(state.new_classes),
        "prediction_context_sha256": state.prediction_context_sha256,
        "arm_predictions": predictions,
        "arm_prediction_sha256_by_arm": dict(
            output.arm_prediction_sha256_by_arm
        ),
        "arm_prediction_receipts": {
            arm: canonical_sha256(
                {
                    "run_id": run.run_id,
                    "row_id": row.row_id,
                    "scenario": scenario.scenario,
                    "stage": state.stage,
                    "registration_state": state.registration_state,
                    "arm": arm,
                    "query_physical_ids": list(state.query_physical_ids),
                    "predictions": predictions[arm],
                    "arm_prediction_sha256": output.arm_prediction_sha256_by_arm[
                        arm
                    ],
                    "prediction_context_sha256": state.prediction_context_sha256,
                    "data_feature_runtime_sha256": state.data_feature_runtime_sha256,
                    "data_materialization_lock_sha256": state.data_materialization_lock_sha256,
                    "d105_candidate_runtime_manifest_sha256": state.d105_candidate_runtime_manifest_sha256,
                    "d105_candidate_method_lock_sha256": state.d105_candidate_method_lock_sha256,
                    "state_receipt_sha256": output.state_receipt_sha256,
                    "predictor_receipt_sha256": output.predictor_receipt_sha256,
                    "feature_receipt_sha256": output.feature_receipt_sha256,
                    "resource_receipt_sha256": output.resource_receipt_sha256,
                    "logit_sha256": output.logit_sha256_by_arm[arm],
                }
            )
            for arm in ARMS
        },
        "state_receipt_sha256": output.state_receipt_sha256,
        "predictor_receipt_sha256": output.predictor_receipt_sha256,
        "feature_receipt_sha256": output.feature_receipt_sha256,
        "resource_receipt_sha256": output.resource_receipt_sha256,
        "logit_sha256_by_arm": dict(output.logit_sha256_by_arm),
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "all_registered_classes_compete": True,
        "per_query_independent": True,
        "query_truth_access": False,
        "query_role_access": False,
        "query_quota_access": False,
        "global_assignment": False,
    }
    result["scorer_input_seal_sha256"] = canonical_sha256(_state_scorer_payload(result))
    return result


def _make_scenario_prediction_pair(
    *,
    run: D105Target25Run,
    row: D105Target25OuterRow,
    scenario: D105Target25ScenarioPlan,
    before_output: D105Target25PredictionOutput,
    after_output: D105Target25PredictionOutput,
) -> dict[str, Any]:
    before = _make_state_prediction(
        run=run,
        row=row,
        scenario=scenario,
        state=scenario.before,
        output=before_output,
    )
    after = _make_state_prediction(
        run=run,
        row=row,
        scenario=scenario,
        state=scenario.after,
        output=after_output,
    )
    arm_pair_receipts = {
        arm: canonical_sha256(
            {
                "run_id": run.run_id,
                "row_id": row.row_id,
                "scenario": scenario.scenario,
                "arm": arm,
                "before_arm_prediction_receipt_sha256": before[
                    "arm_prediction_receipts"
                ][arm],
                "after_arm_prediction_receipt_sha256": after[
                    "arm_prediction_receipts"
                ][arm],
                "before_state_receipt_sha256": before["state_receipt_sha256"],
                "after_state_receipt_sha256": after["state_receipt_sha256"],
                "before_prediction_context_sha256": before[
                    "prediction_context_sha256"
                ],
                "after_prediction_context_sha256": after[
                    "prediction_context_sha256"
                ],
                "before_top1_prediction_sha256": canonical_sha256(
                    before["arm_predictions"][arm]
                ),
                "after_top1_prediction_sha256": canonical_sha256(
                    after["arm_predictions"][arm]
                ),
                "before_arm_prediction_sha256": before[
                    "arm_prediction_sha256_by_arm"
                ][arm],
                "after_arm_prediction_sha256": after[
                    "arm_prediction_sha256_by_arm"
                ][arm],
                "before_logit_sha256": before["logit_sha256_by_arm"][arm],
                "after_logit_sha256": after["logit_sha256_by_arm"][arm],
                "before_query_physical_root_sha256": before[
                    "query_physical_root_sha256"
                ],
                "after_query_physical_root_sha256": after[
                    "query_physical_root_sha256"
                ],
            }
        )
        for arm in ARMS
    }
    pair = {
        "scenario": scenario.scenario,
        "state_predictions": [before, after],
        "arm_pair_receipts": arm_pair_receipts,
    }
    pair["scenario_pair_receipt_sha256"] = canonical_sha256(pair)
    return pair


def _make_prediction_artifact(
    *,
    run: D105Target25Run,
    row: D105Target25OuterRow,
    scenario_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "schema": PREDICTION_SCHEMA,
        "run_id": run.run_id,
        "row_id": row.row_id,
        "outer_row_receipt_sha256": row.receipt_sha256,
        "plan_receipt_sha256": run.plan.plan_receipt_sha256,
        **_plan_identity_payload(run.plan),
        "scenario_predictions": [dict(item) for item in scenario_predictions],
        "query_truth_access": False,
        "query_rows_used_for_fit": 0,
        "query_state_updates": 0,
        "all_registered_classes_compete": True,
        "per_query_independent": True,
        "performance_selection_used": False,
    }
    artifact["prediction_receipt_sha256"] = canonical_sha256(artifact)
    return artifact


def validate_d105_target25_prediction_artifact_without_truth(
    artifact: Mapping[str, Any],
    *,
    run: D105Target25Run,
    row: D105Target25OuterRow,
) -> dict[str, Any]:
    """Fail closed on one row artifact without reading a single truth label."""

    if not isinstance(artifact, Mapping):
        raise D105Target25RunnerError("prediction artifact must be an object")
    expected_keys = {
        "schema",
        "run_id",
        "row_id",
        "outer_row_receipt_sha256",
        "plan_receipt_sha256",
        "claim_scope",
        "formal_launch_authority",
        "authority_envelope_root_sha256",
        "data_feature_runtime_sha256",
        "data_materialization_lock_sha256",
        "d105_candidate_runtime_manifest_sha256",
        "d105_candidate_method_lock_sha256",
        "scenario_predictions",
        "query_truth_access",
        "query_rows_used_for_fit",
        "query_state_updates",
        "all_registered_classes_compete",
        "per_query_independent",
        "performance_selection_used",
        "prediction_receipt_sha256",
    }
    if set(artifact) != expected_keys:
        raise D105Target25RunnerError("prediction artifact field closure drift")
    if (
        artifact.get("schema") != PREDICTION_SCHEMA
        or artifact.get("run_id") != run.run_id
        or artifact.get("row_id") != row.row_id
        or artifact.get("outer_row_receipt_sha256") != row.receipt_sha256
        or artifact.get("plan_receipt_sha256") != run.plan.plan_receipt_sha256
        or any(
            artifact.get(name) != value
            for name, value in _plan_identity_payload(run.plan).items()
        )
        or artifact.get("query_truth_access") is not False
        or artifact.get("query_rows_used_for_fit") != 0
        or artifact.get("query_state_updates") != 0
        or artifact.get("all_registered_classes_compete") is not True
        or artifact.get("per_query_independent") is not True
        or artifact.get("performance_selection_used") is not False
    ):
        raise D105Target25RunnerError("prediction artifact truth-free binding drift")
    receipt = artifact.get("prediction_receipt_sha256")
    if receipt != canonical_sha256(
        {key: value for key, value in artifact.items() if key != "prediction_receipt_sha256"}
    ):
        raise D105Target25RunnerError("prediction artifact receipt drift")
    scenarios = artifact.get("scenario_predictions")
    if not isinstance(scenarios, list) or len(scenarios) != 3:
        raise D105Target25RunnerError("prediction artifact must close at three scenarios")
    expected_by_scene = {item.scenario: item for item in row.scenarios}
    if tuple(item.get("scenario") for item in scenarios if isinstance(item, Mapping)) != LEO_SCENARIOS:
        raise D105Target25RunnerError("prediction scenario order drift")
    state_arm_receipts: list[str] = []
    arm_pair_receipts: list[str] = []
    scorer_seals: list[str] = []
    scenario_keys = {
        "scenario",
        "state_predictions",
        "arm_pair_receipts",
        "scenario_pair_receipt_sha256",
    }
    state_keys = {
        "stage",
        "registration_state",
        "capsule_id",
        "split_id",
        "authority_receipt_sha256",
        "authority_envelope_sha256",
        "data_feature_runtime_sha256",
        "data_materialization_lock_sha256",
        "d105_candidate_runtime_manifest_sha256",
        "d105_candidate_method_lock_sha256",
        "protocol_schema",
        "phase2_data_status",
        "single_leo_observation",
        "clean_source_runtime_access",
        "support_physical_root_sha256",
        "query_physical_ids",
        "query_physical_root_sha256",
        "registered_classes",
        "old_classes",
        "new_classes",
        "prediction_context_sha256",
        "arm_predictions",
        "arm_prediction_sha256_by_arm",
        "arm_prediction_receipts",
        "state_receipt_sha256",
        "predictor_receipt_sha256",
        "feature_receipt_sha256",
        "resource_receipt_sha256",
        "logit_sha256_by_arm",
        "query_rows_used_for_fit",
        "query_state_updates",
        "all_registered_classes_compete",
        "per_query_independent",
        "query_truth_access",
        "query_role_access",
        "query_quota_access",
        "global_assignment",
        "scorer_input_seal_sha256",
    }
    for value in scenarios:
        if not isinstance(value, Mapping) or set(value) != scenario_keys:
            raise D105Target25RunnerError("scenario prediction field closure drift")
        expected = expected_by_scene.get(str(value.get("scenario")))
        if expected is None:
            raise D105Target25RunnerError("unexpected scenario prediction")
        states = value.get("state_predictions")
        if not isinstance(states, list) or [
            item.get("stage") for item in states if isinstance(item, Mapping)
        ] != ["S_B", "S_C"]:
            raise D105Target25RunnerError("scenario state surface order drift")
        expected_states = {"S_B": expected.before, "S_C": expected.after}
        for state_value in states:
            if not isinstance(state_value, Mapping) or set(state_value) != state_keys:
                raise D105Target25RunnerError("state prediction field closure drift")
            state = expected_states.get(str(state_value.get("stage")))
            if state is None or (
                state_value.get("registration_state") != state.registration_state
                or state_value.get("capsule_id") != state.capsule_id
                or state_value.get("split_id") != state.split_id
                or state_value.get("authority_receipt_sha256")
                != state.authority_receipt_sha256
                or state_value.get("authority_envelope_sha256")
                != state.authority_envelope_sha256
                or state_value.get("data_feature_runtime_sha256")
                != state.data_feature_runtime_sha256
                or state_value.get("data_materialization_lock_sha256")
                != state.data_materialization_lock_sha256
                or state_value.get("d105_candidate_runtime_manifest_sha256")
                != state.d105_candidate_runtime_manifest_sha256
                or state_value.get("d105_candidate_method_lock_sha256")
                != state.d105_candidate_method_lock_sha256
                or state_value.get("protocol_schema") != PROTOCOL_SCHEMA
                or state_value.get("phase2_data_status") != "VALIDATED_ONCE"
                or state_value.get("single_leo_observation") is not True
                or state_value.get("clean_source_runtime_access") is not False
                or state_value.get("support_physical_root_sha256")
                != state.support_physical_root_sha256
                or tuple(state_value.get("query_physical_ids", ()))
                != state.query_physical_ids
                or state_value.get("query_physical_root_sha256")
                != state.query_physical_root_sha256
                or tuple(state_value.get("registered_classes", ()))
                != state.registered_classes
                or tuple(state_value.get("old_classes", ())) != state.old_classes
                or tuple(state_value.get("new_classes", ())) != state.new_classes
                or state_value.get("prediction_context_sha256")
                != state.prediction_context_sha256
                or state_value.get("query_rows_used_for_fit") != 0
                or state_value.get("query_state_updates") != 0
                or state_value.get("all_registered_classes_compete") is not True
                or state_value.get("per_query_independent") is not True
                or state_value.get("query_truth_access") is not False
                or state_value.get("query_role_access") is not False
                or state_value.get("query_quota_access") is not False
                or state_value.get("global_assignment") is not False
            ):
                raise D105Target25RunnerError("state prediction protocol binding drift")
            arms = state_value.get("arm_predictions")
            arm_receipt_map = state_value.get("arm_prediction_receipts")
            arm_prediction_hashes = state_value.get("arm_prediction_sha256_by_arm")
            logits = state_value.get("logit_sha256_by_arm")
            if (
                not isinstance(arms, Mapping)
                or not isinstance(arm_receipt_map, Mapping)
                or not isinstance(arm_prediction_hashes, Mapping)
                or not isinstance(logits, Mapping)
                or tuple(arms) != ARMS
                or tuple(arm_receipt_map) != ARMS
                or tuple(arm_prediction_hashes) != ARMS
                or tuple(logits) != ARMS
                or any(
                    not isinstance(arms[arm], list)
                    or len(arms[arm]) != len(state.query_physical_ids)
                    or any(item not in state.registered_classes for item in arms[arm])
                    for arm in ARMS
                )
            ):
                raise D105Target25RunnerError("state four-arm prediction closure drift")
            for name in (
                "state_receipt_sha256",
                "predictor_receipt_sha256",
                "feature_receipt_sha256",
                "resource_receipt_sha256",
            ):
                _require_sha256(state_value.get(name), name)
            if any(_require_sha256(logits[arm], f"{arm} logit SHA") != logits[arm] for arm in ARMS):
                raise D105Target25RunnerError("state logit SHA drift")
            if any(
                _require_sha256(
                    arm_prediction_hashes[arm], f"{arm} arm prediction SHA"
                )
                != arm_prediction_hashes[arm]
                for arm in ARMS
            ):
                raise D105Target25RunnerError("state arm-prediction SHA drift")
            if row.k_shot == 1 and (
                arms["M_HEAD"] != arms["M0"]
                or arms["M_JOINT"] != arms["M_DA"]
            ):
                raise D105Target25RunnerError("K1 identity in immutable state artifact drift")
            for arm in ARMS:
                expected_arm_receipt = canonical_sha256(
                    {
                        "run_id": run.run_id,
                        "row_id": row.row_id,
                        "scenario": expected.scenario,
                        "stage": state.stage,
                        "registration_state": state.registration_state,
                        "arm": arm,
                        "query_physical_ids": list(state.query_physical_ids),
                        "predictions": arms[arm],
                        "arm_prediction_sha256": arm_prediction_hashes[arm],
                        "prediction_context_sha256": state.prediction_context_sha256,
                        "data_feature_runtime_sha256": state.data_feature_runtime_sha256,
                        "data_materialization_lock_sha256": state.data_materialization_lock_sha256,
                        "d105_candidate_runtime_manifest_sha256": state.d105_candidate_runtime_manifest_sha256,
                        "d105_candidate_method_lock_sha256": state.d105_candidate_method_lock_sha256,
                        "state_receipt_sha256": state_value.get("state_receipt_sha256"),
                        "predictor_receipt_sha256": state_value.get(
                            "predictor_receipt_sha256"
                        ),
                        "feature_receipt_sha256": state_value.get(
                            "feature_receipt_sha256"
                        ),
                        "resource_receipt_sha256": state_value.get(
                            "resource_receipt_sha256"
                        ),
                        "logit_sha256": logits[arm],
                    }
                )
                if arm_receipt_map.get(arm) != expected_arm_receipt:
                    raise D105Target25RunnerError("state arm receipt drift")
                state_arm_receipts.append(expected_arm_receipt)
            if state_value.get("scorer_input_seal_sha256") != canonical_sha256(
                _state_scorer_payload(state_value)
            ):
                raise D105Target25RunnerError("state scorer input seal drift")
            scorer_seals.append(str(state_value["scorer_input_seal_sha256"]))
        pair_receipts = value.get("arm_pair_receipts")
        if not isinstance(pair_receipts, Mapping) or tuple(pair_receipts) != ARMS:
            raise D105Target25RunnerError("scenario arm-pair receipt closure drift")
        before, after = states
        for arm in ARMS:
            expected_pair = canonical_sha256(
                {
                    "run_id": run.run_id,
                    "row_id": row.row_id,
                    "scenario": expected.scenario,
                    "arm": arm,
                    "before_arm_prediction_receipt_sha256": before[
                        "arm_prediction_receipts"
                    ][arm],
                    "after_arm_prediction_receipt_sha256": after[
                        "arm_prediction_receipts"
                    ][arm],
                    "before_state_receipt_sha256": before["state_receipt_sha256"],
                    "after_state_receipt_sha256": after["state_receipt_sha256"],
                    "before_prediction_context_sha256": before[
                        "prediction_context_sha256"
                    ],
                    "after_prediction_context_sha256": after[
                        "prediction_context_sha256"
                    ],
                    "before_top1_prediction_sha256": canonical_sha256(
                        before["arm_predictions"][arm]
                    ),
                    "after_top1_prediction_sha256": canonical_sha256(
                        after["arm_predictions"][arm]
                    ),
                    "before_arm_prediction_sha256": before[
                        "arm_prediction_sha256_by_arm"
                    ][arm],
                    "after_arm_prediction_sha256": after[
                        "arm_prediction_sha256_by_arm"
                    ][arm],
                    "before_logit_sha256": before["logit_sha256_by_arm"][arm],
                    "after_logit_sha256": after["logit_sha256_by_arm"][arm],
                    "before_query_physical_root_sha256": before[
                        "query_physical_root_sha256"
                    ],
                    "after_query_physical_root_sha256": after[
                        "query_physical_root_sha256"
                    ],
                }
            )
            if pair_receipts.get(arm) != expected_pair:
                raise D105Target25RunnerError("scenario arm-pair receipt drift")
            arm_pair_receipts.append(expected_pair)
        if value.get("scenario_pair_receipt_sha256") != canonical_sha256(
            {
                "scenario": value["scenario"],
                "state_predictions": value["state_predictions"],
                "arm_pair_receipts": value["arm_pair_receipts"],
            }
        ):
            raise D105Target25RunnerError("scenario pair receipt drift")
    if (
        len(set(state_arm_receipts)) != 24
        or len(set(arm_pair_receipts)) != 12
        or len(set(scorer_seals)) != 6
    ):
        raise D105Target25RunnerError("state/pair receipt uniqueness drift")
    return {
        "prediction_receipt_sha256": str(receipt),
        "state_arm_prediction_receipts": state_arm_receipts,
        "arm_pair_receipts": arm_pair_receipts,
        "scorer_input_seals": scorer_seals,
    }


def normalise_exception_fingerprint(error: Exception) -> dict[str, str]:
    """Normalize row-specific numbers, paths, and hex tokens before grouping."""

    message = " ".join(str(error).strip().lower().split())
    message = re.sub(r"0x[0-9a-f]+", "<hex>", message)
    message = re.sub(r"[0-9a-f]{16,}", "<hash>", message)
    message = re.sub(r"\d+", "<n>", message)
    message = re.sub(r"[a-z]:\\[^ ]+", "<path>", message)
    kind = f"{type(error).__module__}.{type(error).__qualname__}"
    return {
        "exception_type": kind,
        "normalized_message": message or "<empty>",
        "fingerprint_sha256": canonical_sha256(
            {"exception_type": kind, "normalized_message": message or "<empty>"}
        ),
    }


def _write_row_event(
    *,
    run: D105Target25Run,
    row: D105Target25OuterRow,
    assignment: D105Target25Assignment,
    status: str,
    started_ns: int,
    completed_ns: int,
    artifact_relative_path: str | None,
    fingerprint: Mapping[str, str] | None,
) -> tuple[str, str]:
    row_dir = run.run_root / "row_logs" / row.row_id
    if row_dir.exists() or row_dir.is_symlink():
        raise FileExistsError(f"immutable row log root already exists: {row_dir}")
    row_dir.mkdir()
    event = {
        "schema": SCHEMA + ".row_log",
        "run_id": run.run_id,
        "row_id": row.row_id,
        "status": status,
        "dispatch_index": assignment.dispatch_index,
        "gpu_id": assignment.gpu_id,
        "worker_slot": assignment.worker_slot,
        "started_unix_ns": started_ns,
        "completed_unix_ns": completed_ns,
        "scenario_row_count": 3,
        "scenario_arm_pair_count": 12 if status == "SUCCEEDED" else 0,
        "state_prediction_surface_count": 24 if status == "SUCCEEDED" else 0,
        "artifact_relative_path": artifact_relative_path,
        "normalized_exception_fingerprint": fingerprint,
        "query_truth_access": False,
        "performance_selection_used": False,
    }
    exit_event = {
        "schema": SCHEMA + ".row_exit",
        "run_id": run.run_id,
        "row_id": row.row_id,
        "status": status,
        "exit_code": 0 if status == "SUCCEEDED" else 1,
        "prediction_published": status == "SUCCEEDED",
        "normalized_exception_fingerprint": fingerprint,
    }
    return (
        _write_json_new(row_dir / "row_log.json", event),
        _write_json_new(row_dir / "exit.json", exit_event),
    )


@dataclass(frozen=True, slots=True)
class D105Target25ExecutionSummary:
    status: str
    launched_outer_rows: int
    completed_outer_rows: int
    succeeded_outer_rows: int
    failed_outer_rows: int
    scenario_arm_pair_count: int
    state_prediction_surface_count: int
    manifest_path: Path
    stop_dispatch: bool
    stop_fingerprint_sha256: str | None


def _records_payload(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in records]


def execute_d105_target25_predictions(
    run: D105Target25Run,
    predictor: Callable[[D105Target25PredictionRequest], D105Target25PredictionOutput],
) -> D105Target25ExecutionSummary:
    """Seal all predictions, or preserve an immutable technical-failure partial run.

    The callback receives only ``D105Target25PredictionRequest``.  It is never
    passed true labels, query role, query quota, score, or any performance gate.
    """

    _verify_run(run)
    if not callable(predictor):
        raise D105Target25RunnerError("predictor callback is required")
    if any(
        (run.run_root / name).exists()
        for name in ("prediction_manifest.json", "partial_prediction_manifest.json")
    ) or any((run.run_root / "predictions").iterdir()) or any(
        (run.run_root / "row_logs").iterdir()
    ):
        raise FileExistsError("run root already contains immutable execution outputs")
    assignment_by_row = {item.row_id: item for item in run.assignments}
    records: list[dict[str, Any]] = []
    fingerprints: dict[str, set[str]] = {}
    stop_fingerprint: str | None = None
    for row in run.plan.rows:
        assignment = assignment_by_row[row.row_id]
        started_ns = time.time_ns()
        try:
            scenario_predictions = []
            for scenario in row.scenarios:
                before_request = _prediction_request(
                    run, row, scenario, scenario.before, assignment
                )
                after_request = _prediction_request(
                    run, row, scenario, scenario.after, assignment
                )
                scenario_predictions.append(
                    _make_scenario_prediction_pair(
                        run=run,
                        row=row,
                        scenario=scenario,
                        before_output=predictor(before_request),
                        after_output=predictor(after_request),
                    )
                )
            artifact = _make_prediction_artifact(
                run=run, row=row, scenario_predictions=scenario_predictions
            )
            validated = validate_d105_target25_prediction_artifact_without_truth(
                artifact, run=run, row=row
            )
            relative_path = f"predictions/{row.row_id}.json"
            artifact_path = run.run_root / relative_path
            artifact_sha = _write_json_new(artifact_path, artifact)
            completed_ns = time.time_ns()
            log_sha, exit_sha = _write_row_event(
                run=run,
                row=row,
                assignment=assignment,
                status="SUCCEEDED",
                started_ns=started_ns,
                completed_ns=completed_ns,
                artifact_relative_path=relative_path,
                fingerprint=None,
            )
            records.append(
                {
                    "row_id": row.row_id,
                    "status": "SUCCEEDED",
                    "artifact_relative_path": relative_path,
                    "artifact_sha256": artifact_sha,
                    "prediction_receipt_sha256": validated[
                        "prediction_receipt_sha256"
                    ],
                    "state_arm_prediction_receipts": validated[
                        "state_arm_prediction_receipts"
                    ],
                    "arm_pair_receipts": validated["arm_pair_receipts"],
                    "scorer_input_seals": validated["scorer_input_seals"],
                    "row_log_sha256": log_sha,
                    "exit_sha256": exit_sha,
                    "normalized_exception_fingerprint": None,
                }
            )
        except Exception as error:
            completed_ns = time.time_ns()
            fingerprint = normalise_exception_fingerprint(error)
            log_sha, exit_sha = _write_row_event(
                run=run,
                row=row,
                assignment=assignment,
                status="FAILED_BEFORE_PREDICTION",
                started_ns=started_ns,
                completed_ns=completed_ns,
                artifact_relative_path=None,
                fingerprint=fingerprint,
            )
            records.append(
                {
                    "row_id": row.row_id,
                    "status": "FAILED_BEFORE_PREDICTION",
                    "artifact_relative_path": None,
                    "artifact_sha256": None,
                    "prediction_receipt_sha256": None,
                    "state_arm_prediction_receipts": [],
                    "arm_pair_receipts": [],
                    "scorer_input_seals": [],
                    "row_log_sha256": log_sha,
                    "exit_sha256": exit_sha,
                    "normalized_exception_fingerprint": fingerprint,
                }
            )
            key = fingerprint["fingerprint_sha256"]
            fingerprints.setdefault(key, set()).add(row.row_id)
            if len(fingerprints[key]) >= 2:
                stop_fingerprint = key
                break
    succeeded = [item for item in records if item["status"] == "SUCCEEDED"]
    failed = [item for item in records if item["status"] != "SUCCEEDED"]
    if len(succeeded) == OUTER_ROW_COUNT:
        state_arm_receipts = [
            receipt
            for item in succeeded
            for receipt in item["state_arm_prediction_receipts"]
        ]
        arm_pair_receipts = [
            receipt for item in succeeded for receipt in item["arm_pair_receipts"]
        ]
        if (
            len(state_arm_receipts) != STATE_PREDICTION_SURFACE_COUNT
            or len(set(state_arm_receipts)) != STATE_PREDICTION_SURFACE_COUNT
            or len(arm_pair_receipts) != SCENARIO_ARM_PAIR_COUNT
            or len(set(arm_pair_receipts)) != SCENARIO_ARM_PAIR_COUNT
        ):
            raise D105Target25RunnerError("complete prediction pair/surface coverage drift")
        manifest: dict[str, Any] = {
            "schema": PREDICTION_MANIFEST_SCHEMA,
            "status": "PREDICTIONS_COMPLETE",
            "run_id": run.run_id,
            "plan_receipt_sha256": run.plan.plan_receipt_sha256,
            **_plan_identity_payload(run.plan),
            "seed": run.plan.seed,
            "outer_row_count": OUTER_ROW_COUNT,
            "scenario_row_count": SCENARIO_ROW_COUNT,
            "scenario_arm_pair_count": SCENARIO_ARM_PAIR_COUNT,
            "state_prediction_surface_count": STATE_PREDICTION_SURFACE_COUNT,
            "rows": _records_payload(records),
            "all_state_arm_prediction_receipts_unique": True,
            "all_arm_pair_receipts_unique": True,
            "query_truth_access": False,
            "performance_selection_used": False,
        }
        manifest["prediction_manifest_receipt_sha256"] = canonical_sha256(manifest)
        manifest_path = run.run_root / "prediction_manifest.json"
        _write_json_new(manifest_path, manifest)
        verify_d105_target25_prediction_manifest(run)
        return D105Target25ExecutionSummary(
            status="PREDICTIONS_COMPLETE",
            launched_outer_rows=len(records),
            completed_outer_rows=len(records),
            succeeded_outer_rows=len(succeeded),
            failed_outer_rows=0,
            scenario_arm_pair_count=SCENARIO_ARM_PAIR_COUNT,
            state_prediction_surface_count=STATE_PREDICTION_SURFACE_COUNT,
            manifest_path=manifest_path,
            stop_dispatch=False,
            stop_fingerprint_sha256=None,
        )
    status = (
        "STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT"
        if stop_fingerprint is not None
        else "TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT"
    )
    partial: dict[str, Any] = {
        "schema": PARTIAL_MANIFEST_SCHEMA,
        "status": status,
        "run_id": run.run_id,
        "plan_receipt_sha256": run.plan.plan_receipt_sha256,
        **_plan_identity_payload(run.plan),
        "launched_outer_rows": len(records),
        "completed_outer_rows": len(records),
        "succeeded_outer_rows": len(succeeded),
        "failed_outer_rows": len(failed),
        "scenario_arm_pair_count": len(succeeded) * 12,
        "state_prediction_surface_count": len(succeeded) * 24,
        "stop_dispatch": stop_fingerprint is not None,
        "stop_fingerprint_sha256": stop_fingerprint,
        "rows": _records_payload(records),
        "query_truth_access": False,
        "performance_selection_used": False,
    }
    partial["partial_manifest_receipt_sha256"] = canonical_sha256(partial)
    manifest_path = run.run_root / "partial_prediction_manifest.json"
    _write_json_new(manifest_path, partial)
    return D105Target25ExecutionSummary(
        status=status,
        launched_outer_rows=len(records),
        completed_outer_rows=len(records),
        succeeded_outer_rows=len(succeeded),
        failed_outer_rows=len(failed),
        scenario_arm_pair_count=len(succeeded) * 12,
        state_prediction_surface_count=len(succeeded) * 24,
        manifest_path=manifest_path,
        stop_dispatch=stop_fingerprint is not None,
        stop_fingerprint_sha256=stop_fingerprint,
    )


def verify_d105_target25_prediction_manifest(run: D105Target25Run) -> dict[str, Any]:
    """Validate all 25 sealed artifacts before a truth-side caller can open truth."""

    _verify_run(run)
    path = run.run_root / "prediction_manifest.json"
    manifest = _read_json_regular(path)
    expected_keys = {
        "schema",
        "status",
        "run_id",
        "plan_receipt_sha256",
        "claim_scope",
        "formal_launch_authority",
        "authority_envelope_root_sha256",
        "data_feature_runtime_sha256",
        "data_materialization_lock_sha256",
        "d105_candidate_runtime_manifest_sha256",
        "d105_candidate_method_lock_sha256",
        "seed",
        "outer_row_count",
        "scenario_row_count",
        "scenario_arm_pair_count",
        "state_prediction_surface_count",
        "rows",
        "all_state_arm_prediction_receipts_unique",
        "all_arm_pair_receipts_unique",
        "query_truth_access",
        "performance_selection_used",
        "prediction_manifest_receipt_sha256",
    }
    if set(manifest) != expected_keys or (
        manifest.get("schema") != PREDICTION_MANIFEST_SCHEMA
        or manifest.get("status") != "PREDICTIONS_COMPLETE"
        or manifest.get("run_id") != run.run_id
        or manifest.get("plan_receipt_sha256") != run.plan.plan_receipt_sha256
        or any(
            manifest.get(name) != value
            for name, value in _plan_identity_payload(run.plan).items()
        )
        or manifest.get("seed") != TARGET25_SEED
        or manifest.get("outer_row_count") != OUTER_ROW_COUNT
        or manifest.get("scenario_row_count") != SCENARIO_ROW_COUNT
        or manifest.get("scenario_arm_pair_count") != SCENARIO_ARM_PAIR_COUNT
        or manifest.get("state_prediction_surface_count")
        != STATE_PREDICTION_SURFACE_COUNT
        or manifest.get("all_state_arm_prediction_receipts_unique") is not True
        or manifest.get("all_arm_pair_receipts_unique") is not True
        or manifest.get("query_truth_access") is not False
        or manifest.get("performance_selection_used") is not False
    ):
        raise D105Target25RunnerError("prediction manifest closure drift")
    if manifest.get("prediction_manifest_receipt_sha256") != canonical_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "prediction_manifest_receipt_sha256"
        }
    ):
        raise D105Target25RunnerError("prediction manifest receipt drift")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != OUTER_ROW_COUNT:
        raise D105Target25RunnerError("prediction manifest row count drift")
    row_by_id = {row.row_id: row for row in run.plan.rows}
    if [item.get("row_id") for item in rows if isinstance(item, Mapping)] != [
        row.row_id for row in run.plan.rows
    ]:
        raise D105Target25RunnerError("prediction manifest frozen row order drift")
    manifest_mtime = path.stat().st_mtime_ns
    all_state_arm_receipts: list[str] = []
    all_arm_pair_receipts: list[str] = []
    artifact_paths: list[str] = []
    for entry in rows:
        if not isinstance(entry, Mapping) or set(entry) != {
            "row_id",
            "status",
            "artifact_relative_path",
            "artifact_sha256",
            "prediction_receipt_sha256",
            "state_arm_prediction_receipts",
            "arm_pair_receipts",
            "scorer_input_seals",
            "row_log_sha256",
            "exit_sha256",
            "normalized_exception_fingerprint",
        }:
            raise D105Target25RunnerError("prediction manifest row field closure drift")
        row_id = str(entry["row_id"])
        if entry.get("status") != "SUCCEEDED" or row_id not in row_by_id:
            raise D105Target25RunnerError("prediction manifest includes non-success row")
        raw_path = entry.get("artifact_relative_path")
        if not isinstance(raw_path, str) or raw_path in artifact_paths:
            raise D105Target25RunnerError("prediction artifact path uniqueness drift")
        artifact_paths.append(raw_path)
        artifact_path = _relative_file(run.run_root, raw_path, "artifact_relative_path")
        if _sha256_file(artifact_path) != entry.get("artifact_sha256"):
            raise D105Target25RunnerError("prediction artifact SHA drift")
        if artifact_path.stat().st_mtime_ns > manifest_mtime:
            raise D105Target25RunnerError("prediction artifact postdates final manifest")
        row_log_path = run.run_root / "row_logs" / row_id / "row_log.json"
        exit_path = run.run_root / "row_logs" / row_id / "exit.json"
        if (
            _sha256_file(row_log_path) != entry.get("row_log_sha256")
            or _sha256_file(exit_path) != entry.get("exit_sha256")
        ):
            raise D105Target25RunnerError("per-row log/exit SHA drift")
        row_log = _read_json_regular(row_log_path)
        exit_record = _read_json_regular(exit_path)
        if (
            row_log.get("status") != "SUCCEEDED"
            or row_log.get("row_id") != row_id
            or row_log.get("artifact_relative_path") != raw_path
            or row_log.get("scenario_arm_pair_count") != 12
            or row_log.get("state_prediction_surface_count") != 24
            or row_log.get("query_truth_access") is not False
            or row_log.get("performance_selection_used") is not False
            or exit_record.get("status") != "SUCCEEDED"
            or exit_record.get("row_id") != row_id
            or exit_record.get("exit_code") != 0
            or exit_record.get("prediction_published") is not True
        ):
            raise D105Target25RunnerError("per-row log/exit binding drift")
        validated = validate_d105_target25_prediction_artifact_without_truth(
            _read_json_regular(artifact_path), run=run, row=row_by_id[row_id]
        )
        if (
            entry.get("prediction_receipt_sha256")
            != validated["prediction_receipt_sha256"]
            or entry.get("state_arm_prediction_receipts")
            != validated["state_arm_prediction_receipts"]
            or entry.get("arm_pair_receipts") != validated["arm_pair_receipts"]
            or entry.get("scorer_input_seals") != validated["scorer_input_seals"]
            or entry.get("normalized_exception_fingerprint") is not None
        ):
            raise D105Target25RunnerError("prediction manifest row receipt binding drift")
        all_state_arm_receipts.extend(validated["state_arm_prediction_receipts"])
        all_arm_pair_receipts.extend(validated["arm_pair_receipts"])
    if (
        len(all_state_arm_receipts) != STATE_PREDICTION_SURFACE_COUNT
        or len(set(all_state_arm_receipts)) != STATE_PREDICTION_SURFACE_COUNT
        or len(all_arm_pair_receipts) != SCENARIO_ARM_PAIR_COUNT
        or len(set(all_arm_pair_receipts)) != SCENARIO_ARM_PAIR_COUNT
    ):
        raise D105Target25RunnerError("prediction manifest pair/surface receipt coverage drift")
    return manifest


@dataclass(frozen=True, slots=True)
class D105Target25TruthSideManifest:
    run_id: str
    plan_receipt_sha256: str
    truth_catalog_sha256: str
    scenario_references: tuple[Mapping[str, Any], ...]
    receipt_sha256: str


def _truth_manifest_payload(
    *,
    run_id: str,
    plan_receipt_sha256: str,
    truth_catalog_sha256: str,
    scenario_references: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": TRUTH_MANIFEST_SCHEMA,
        "run_id": run_id,
        "plan_receipt_sha256": plan_receipt_sha256,
        "truth_catalog_sha256": truth_catalog_sha256,
        "predictor_truth_access": False,
        "scenario_references": [dict(item) for item in scenario_references],
    }


def build_d105_target25_truth_side_manifest(
    run: D105Target25Run, *, truth_catalog_sha256: str
) -> D105Target25TruthSideManifest:
    """Create a label-free reference manifest for an independently owned scorer."""

    _verify_run(run)
    references = []
    for row in run.plan.rows:
        for scenario in row.scenarios:
            references.append(
                {
                    "row_id": row.row_id,
                    "scenario": scenario.scenario,
                    "before": {
                        "stage": scenario.before.stage,
                        "registration_state": scenario.before.registration_state,
                        "query_physical_ids": tuple(
                            scenario.before.query_physical_ids
                        ),
                        "query_physical_root_sha256": scenario.before.query_physical_root_sha256,
                        "registered_classes": tuple(
                            scenario.before.registered_classes
                        ),
                        "old_classes": tuple(scenario.before.old_classes),
                        "new_classes": tuple(scenario.before.new_classes),
                    },
                    "after": {
                        "stage": scenario.after.stage,
                        "registration_state": scenario.after.registration_state,
                        "query_physical_ids": tuple(scenario.after.query_physical_ids),
                        "query_physical_root_sha256": scenario.after.query_physical_root_sha256,
                        "registered_classes": tuple(scenario.after.registered_classes),
                        "old_classes": tuple(scenario.after.old_classes),
                        "new_classes": tuple(scenario.after.new_classes),
                    },
                }
            )
    truth_sha = _require_sha256(truth_catalog_sha256, "truth_catalog_sha256")
    payload = _truth_manifest_payload(
        run_id=run.run_id,
        plan_receipt_sha256=run.plan.plan_receipt_sha256,
        truth_catalog_sha256=truth_sha,
        scenario_references=references,
    )
    return D105Target25TruthSideManifest(
        run_id=run.run_id,
        plan_receipt_sha256=run.plan.plan_receipt_sha256,
        truth_catalog_sha256=truth_sha,
        scenario_references=tuple(references),
        receipt_sha256=canonical_sha256(payload),
    )


def write_d105_target25_truth_side_manifest(
    path: Path, manifest: D105Target25TruthSideManifest
) -> str:
    if type(manifest) is not D105Target25TruthSideManifest:
        raise D105Target25RunnerError("exact truth-side manifest required")
    payload = _truth_manifest_payload(
        run_id=manifest.run_id,
        plan_receipt_sha256=manifest.plan_receipt_sha256,
        truth_catalog_sha256=manifest.truth_catalog_sha256,
        scenario_references=manifest.scenario_references,
    )
    if canonical_sha256(payload) != manifest.receipt_sha256:
        raise D105Target25RunnerError("truth-side manifest receipt drift")
    document = dict(payload)
    document["truth_side_manifest_receipt_sha256"] = manifest.receipt_sha256
    return _write_json_new(Path(path), document)


@dataclass(frozen=True, slots=True)
class D105Target25TruthReadRequest:
    run_id: str
    row_id: str
    scenario: str
    stage: str
    registration_state: str
    query_physical_ids: tuple[str, ...]
    query_physical_root_sha256: str
    registered_classes: tuple[str, ...]
    old_classes: tuple[str, ...]
    new_classes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class D105Target25TruthLabels:
    query_physical_ids: tuple[str, ...]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "query_physical_ids", _unique_texts(self.query_physical_ids, "truth query IDs")
        )
        object.__setattr__(self, "labels", tuple(str(item) for item in self.labels))


@dataclass(frozen=True, slots=True)
class D105Target25LoadedTruthCatalog:
    """A schema-closed, externally SHA-bound truth catalog opened after prediction."""

    run_id: str
    plan_receipt_sha256: str
    file_sha256: str
    catalog_receipt_sha256: str
    truth_side_manifest: D105Target25TruthSideManifest
    state_plans: Mapping[tuple[str, str, str], D105Target25StatePlan]
    labels_by_state: Mapping[tuple[str, str, str], D105Target25TruthLabels]

    def read_labels(
        self, request: D105Target25TruthReadRequest
    ) -> D105Target25TruthLabels:
        if type(request) is not D105Target25TruthReadRequest:
            raise D105Target25RunnerError("exact truth read request required")
        if request.run_id != self.run_id:
            raise D105Target25RunnerError("truth catalog run binding drift")
        key = (request.row_id, request.scenario, request.stage)
        state = self.state_plans.get(key)
        truth = self.labels_by_state.get(key)
        if state is None or truth is None:
            raise D105Target25RunnerError("truth catalog state key drift")
        if (
            request.registration_state != state.registration_state
            or request.query_physical_ids != state.query_physical_ids
            or request.query_physical_root_sha256
            != state.query_physical_root_sha256
            or request.registered_classes != state.registered_classes
            or request.old_classes != state.old_classes
            or request.new_classes != state.new_classes
        ):
            raise D105Target25RunnerError("truth catalog read request binding drift")
        return truth


def seal_d105_target25_truth_catalog_manifest(
    path: Path, document: Mapping[str, Any]
) -> str:
    """Seal a new catalog and add its canonical content receipt."""

    if not isinstance(document, Mapping):
        raise D105Target25RunnerError("truth catalog document must be a mapping")
    payload = dict(document)
    if "truth_catalog_receipt_sha256" in payload:
        raise D105Target25RunnerError("caller must not supply truth catalog receipt")
    if set(payload) != {"schema", "plan_receipt_sha256", "states"}:
        raise D105Target25RunnerError("truth catalog top-level schema closure drift")
    if payload.get("schema") != TRUTH_CATALOG_SCHEMA:
        raise D105Target25RunnerError("truth catalog schema drift")
    payload["truth_catalog_receipt_sha256"] = canonical_sha256(payload)
    return _write_json_new(Path(path), payload)


def load_d105_target25_truth_catalog_manifest(
    run: D105Target25Run,
    path: Path,
    *,
    expected_file_sha256: str,
) -> D105Target25LoadedTruthCatalog:
    """Open and validate an externally SHA-bound, schema-closed truth catalog."""

    _verify_run(run)
    expected_sha = _require_sha256(
        expected_file_sha256, "expected truth catalog file SHA256"
    )
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise D105Target25RunnerError(
            f"expected regular immutable JSON file: {source}"
        )
    mode = source.stat().st_mode
    if not stat.S_ISREG(mode):
        raise D105Target25RunnerError(f"non-regular immutable JSON file: {source}")
    if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise D105Target25RunnerError(f"immutable JSON file is writable: {source}")
    raw = source.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise D105Target25RunnerError("truth catalog external file SHA drift")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D105Target25RunnerError("truth catalog is not canonical UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise D105Target25RunnerError("truth catalog must be a JSON object")
    if raw != _canonical_bytes(document) + b"\n":
        raise D105Target25RunnerError("truth catalog canonical encoding drift")
    if set(document) != {
        "schema",
        "plan_receipt_sha256",
        "states",
        "truth_catalog_receipt_sha256",
    }:
        raise D105Target25RunnerError("truth catalog top-level schema closure drift")
    if (
        document.get("schema") != TRUTH_CATALOG_SCHEMA
        or document.get("plan_receipt_sha256") != run.plan.plan_receipt_sha256
    ):
        raise D105Target25RunnerError("truth catalog plan/schema binding drift")
    receipt = _require_sha256(
        document.get("truth_catalog_receipt_sha256"),
        "truth_catalog_receipt_sha256",
    )
    receipt_payload = dict(document)
    del receipt_payload["truth_catalog_receipt_sha256"]
    if canonical_sha256(receipt_payload) != receipt:
        raise D105Target25RunnerError("truth catalog receipt drift")
    states_value = document.get("states")
    if not isinstance(states_value, list):
        raise D105Target25RunnerError("truth catalog states must be a list")
    expected_states = [
        (row.row_id, scenario.scenario, state)
        for row in run.plan.rows
        for scenario in row.scenarios
        for state in (scenario.before, scenario.after)
    ]
    if len(states_value) != SCENARIO_ROW_COUNT * 2:
        raise D105Target25RunnerError("truth catalog state coverage drift")
    state_plans: dict[tuple[str, str, str], D105Target25StatePlan] = {}
    labels_by_state: dict[tuple[str, str, str], D105Target25TruthLabels] = {}
    expected_state_keys = {
        "row_id",
        "scenario",
        "stage",
        "registration_state",
        "query_physical_ids",
        "query_physical_root_sha256",
        "registered_classes",
        "old_classes",
        "new_classes",
        "labels",
    }
    for value, (row_id, scenario_name, state) in zip(
        states_value, expected_states, strict=True
    ):
        if not isinstance(value, dict) or set(value) != expected_state_keys:
            raise D105Target25RunnerError("truth catalog state schema closure drift")
        query_ids = value.get("query_physical_ids")
        registered = value.get("registered_classes")
        old = value.get("old_classes")
        new = value.get("new_classes")
        labels = value.get("labels")
        if (
            value.get("row_id") != row_id
            or value.get("scenario") != scenario_name
            or value.get("stage") != state.stage
            or value.get("registration_state") != state.registration_state
            or query_ids != list(state.query_physical_ids)
            or value.get("query_physical_root_sha256")
            != state.query_physical_root_sha256
            or registered != list(state.registered_classes)
            or old != list(state.old_classes)
            or new != list(state.new_classes)
        ):
            raise D105Target25RunnerError("truth catalog state/plan binding drift")
        if (
            not isinstance(labels, list)
            or len(labels) != len(state.query_physical_ids)
            or any(
                not isinstance(label, str) or label not in state.registered_classes
                for label in labels
            )
        ):
            raise D105Target25RunnerError(
                "truth catalog labels are outside the frozen state registry"
            )
        key = (row_id, scenario_name, state.stage)
        state_plans[key] = state
        labels_by_state[key] = D105Target25TruthLabels(
            query_physical_ids=state.query_physical_ids,
            labels=tuple(labels),
        )
    for row in run.plan.rows:
        for scenario in row.scenarios:
            before = labels_by_state[(row.row_id, scenario.scenario, "S_B")]
            after = labels_by_state[(row.row_id, scenario.scenario, "S_C")]
            before_by_id = dict(
                zip(before.query_physical_ids, before.labels, strict=True)
            )
            after_by_id = dict(zip(after.query_physical_ids, after.labels, strict=True))
            if any(
                before_by_id[query_id] != after_by_id.get(query_id)
                or before_by_id[query_id] not in scenario.before.old_classes
                for query_id in scenario.before.query_physical_ids
            ):
                raise D105Target25RunnerError(
                    "truth catalog old-query label lifecycle drift"
                )
            if any(
                after_by_id[query_id] not in scenario.after.new_classes
                for query_id in (
                    set(scenario.after.query_physical_ids)
                    - set(scenario.before.query_physical_ids)
                )
            ):
                raise D105Target25RunnerError(
                    "truth catalog new-query label lifecycle drift"
                )
    truth_manifest = build_d105_target25_truth_side_manifest(
        run, truth_catalog_sha256=actual_sha
    )
    return D105Target25LoadedTruthCatalog(
        run_id=run.run_id,
        plan_receipt_sha256=run.plan.plan_receipt_sha256,
        file_sha256=actual_sha,
        catalog_receipt_sha256=receipt,
        truth_side_manifest=truth_manifest,
        state_plans=MappingProxyType(state_plans),
        labels_by_state=MappingProxyType(labels_by_state),
    )


def _verify_truth_manifest(
    manifest: D105Target25TruthSideManifest, run: D105Target25Run
) -> None:
    if type(manifest) is not D105Target25TruthSideManifest:
        raise D105Target25RunnerError("exact truth-side manifest required")
    expected = _truth_manifest_payload(
        run_id=manifest.run_id,
        plan_receipt_sha256=manifest.plan_receipt_sha256,
        truth_catalog_sha256=manifest.truth_catalog_sha256,
        scenario_references=manifest.scenario_references,
    )
    if (
        manifest.run_id != run.run_id
        or manifest.plan_receipt_sha256 != run.plan.plan_receipt_sha256
        or canonical_sha256(expected) != manifest.receipt_sha256
        or len(manifest.scenario_references) != SCENARIO_ROW_COUNT
    ):
        raise D105Target25RunnerError("truth-side manifest binding drift")
    expected_refs = []
    for row in run.plan.rows:
        for scenario in row.scenarios:
            expected_refs.append(
                {
                    "row_id": row.row_id,
                    "scenario": scenario.scenario,
                    "before": {
                        "stage": scenario.before.stage,
                        "registration_state": scenario.before.registration_state,
                        "query_physical_ids": tuple(
                            scenario.before.query_physical_ids
                        ),
                        "query_physical_root_sha256": scenario.before.query_physical_root_sha256,
                        "registered_classes": tuple(
                            scenario.before.registered_classes
                        ),
                        "old_classes": tuple(scenario.before.old_classes),
                        "new_classes": tuple(scenario.before.new_classes),
                    },
                    "after": {
                        "stage": scenario.after.stage,
                        "registration_state": scenario.after.registration_state,
                        "query_physical_ids": tuple(scenario.after.query_physical_ids),
                        "query_physical_root_sha256": scenario.after.query_physical_root_sha256,
                        "registered_classes": tuple(scenario.after.registered_classes),
                        "old_classes": tuple(scenario.after.old_classes),
                        "new_classes": tuple(scenario.after.new_classes),
                    },
                }
            )
    actual_refs = [dict(item) for item in manifest.scenario_references]
    if actual_refs != expected_refs:
        raise D105Target25RunnerError("truth-side scenario reference closure drift")


def _score_arm(
    *, predictions: Sequence[str], labels: Sequence[str], classes: Sequence[str]
) -> dict[str, Any]:
    if len(predictions) != len(labels):
        raise D105Target25RunnerError("truth/prediction row length drift")
    if any(label not in classes for label in labels):
        raise D105Target25RunnerError("truth contains a class outside the frozen registry")
    per_class = {}
    for class_id in classes:
        count = sum(label == class_id for label in labels)
        correct = sum(
            label == class_id and predicted == class_id
            for label, predicted in zip(labels, predictions, strict=True)
        )
        per_class[class_id] = {"count": count, "correct": correct}
    correct_count = sum(
        label == predicted for label, predicted in zip(labels, predictions, strict=True)
    )
    return {
        "correct_count": correct_count,
        "query_count": len(labels),
        "accuracy": correct_count / len(labels),
        "per_class": per_class,
    }


def score_d105_target25_truth_side(
    run: D105Target25Run,
    truth_manifest: D105Target25TruthSideManifest,
    truth_provider: Callable[[D105Target25TruthReadRequest], D105Target25TruthLabels],
    *,
    score_root: Path,
) -> Path:
    """Read truth only after full immutable prediction-manifest validation."""

    prediction_manifest = verify_d105_target25_prediction_manifest(run)
    _verify_truth_manifest(truth_manifest, run)
    if not callable(truth_provider):
        raise D105Target25RunnerError("truth-side provider callback is required")
    destination = Path(score_root)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable score root already exists: {destination}")
    destination.mkdir()
    (destination / "rows").mkdir()
    manifest_path = run.run_root / "prediction_manifest.json"
    opened_ns = time.time_ns()
    if manifest_path.stat().st_mtime_ns >= opened_ns:
        raise D105Target25RunnerError("prediction manifest did not precede truth open")
    event = {
        "schema": TRUTH_OPEN_SCHEMA,
        "run_id": run.run_id,
        "prediction_manifest_sha256": _sha256_file(manifest_path),
        "prediction_manifest_receipt_sha256": prediction_manifest[
            "prediction_manifest_receipt_sha256"
        ],
        "truth_side_manifest_receipt_sha256": truth_manifest.receipt_sha256,
        "opened_unix_ns": opened_ns,
        "all_predictions_committed_before_truth_open": True,
        "sealed_outer_row_count": OUTER_ROW_COUNT,
        "sealed_scenario_arm_pair_count": SCENARIO_ARM_PAIR_COUNT,
        "sealed_state_prediction_surface_count": STATE_PREDICTION_SURFACE_COUNT,
        "predictor_truth_access": False,
    }
    event_sha = _write_json_new(destination / "truth_open_event.json", event)
    row_by_id = {row.row_id: row for row in run.plan.rows}
    artifacts = {}
    for entry in prediction_manifest["rows"]:
        artifacts[str(entry["row_id"])] = _read_json_regular(
            _relative_file(run.run_root, str(entry["artifact_relative_path"]), "artifact_relative_path")
        )
    score_rows = []
    for row in run.plan.rows:
        artifact = artifacts[row.row_id]
        scored_scenarios = []
        for scenario_prediction in artifact["scenario_predictions"]:
            scenario_name = str(scenario_prediction["scenario"])
            plan_scenario = next(
                item for item in row.scenarios if item.scenario == scenario_name
            )
            states = {
                str(state_prediction["stage"]): state_prediction
                for state_prediction in scenario_prediction["state_predictions"]
            }
            before_prediction = states["S_B"]
            after_prediction = states["S_C"]

            def read_truth(state: D105Target25StatePlan) -> D105Target25TruthLabels:
                request = D105Target25TruthReadRequest(
                    run_id=run.run_id,
                    row_id=row.row_id,
                    scenario=scenario_name,
                    stage=state.stage,
                    registration_state=state.registration_state,
                    query_physical_ids=state.query_physical_ids,
                    query_physical_root_sha256=state.query_physical_root_sha256,
                    registered_classes=state.registered_classes,
                    old_classes=state.old_classes,
                    new_classes=state.new_classes,
                )
                truth = truth_provider(request)
                if type(truth) is not D105Target25TruthLabels or (
                    truth.query_physical_ids != state.query_physical_ids
                    or len(truth.labels) != len(state.query_physical_ids)
                ):
                    raise D105Target25RunnerError(
                        "truth-side provider identity/length drift"
                    )
                return truth

            before_truth = read_truth(plan_scenario.before)
            after_truth = read_truth(plan_scenario.after)
            before_labels_by_id = dict(
                zip(
                    before_truth.query_physical_ids,
                    before_truth.labels,
                    strict=True,
                )
            )
            after_labels_by_id = dict(
                zip(
                    after_truth.query_physical_ids,
                    after_truth.labels,
                    strict=True,
                )
            )
            before_ids = set(plan_scenario.before.query_physical_ids)
            after_old_indices = [
                index
                for index, physical_id in enumerate(plan_scenario.after.query_physical_ids)
                if physical_id in before_ids
            ]
            after_new_indices = [
                index
                for index, physical_id in enumerate(plan_scenario.after.query_physical_ids)
                if physical_id not in before_ids
            ]
            if (
                any(label not in plan_scenario.before.old_classes for label in before_truth.labels)
                or any(
                    after_labels_by_id[physical_id]
                    != before_labels_by_id[physical_id]
                    for physical_id in plan_scenario.before.query_physical_ids
                )
                or any(
                    after_truth.labels[index] not in plan_scenario.after.old_classes
                    for index in after_old_indices
                )
                or not after_new_indices
                or any(
                    after_truth.labels[index] not in plan_scenario.after.new_classes
                    for index in after_new_indices
                )
            ):
                raise D105Target25RunnerError(
                    "before/after truth lifecycle or physical-ID alignment drift"
                )
            after_old_labels = tuple(after_truth.labels[index] for index in after_old_indices)
            after_new_labels = tuple(after_truth.labels[index] for index in after_new_indices)
            arm_pair_scores = {}
            for arm in ARMS:
                before_old = _score_arm(
                    predictions=before_prediction["arm_predictions"][arm],
                    labels=before_truth.labels,
                    classes=plan_scenario.before.old_classes,
                )
                after_old = _score_arm(
                    predictions=tuple(
                        after_prediction["arm_predictions"][arm][index]
                        for index in after_old_indices
                    ),
                    labels=after_old_labels,
                    classes=plan_scenario.after.old_classes,
                )
                after_new = _score_arm(
                    predictions=tuple(
                        after_prediction["arm_predictions"][arm][index]
                        for index in after_new_indices
                    ),
                    labels=after_new_labels,
                    classes=plan_scenario.after.new_classes,
                )
                after_all = _score_arm(
                    predictions=after_prediction["arm_predictions"][arm],
                    labels=after_truth.labels,
                    classes=plan_scenario.after.registered_classes,
                )
                b_old = before_old["accuracy"]
                a_old = after_old["accuracy"]
                new_score = after_new["accuracy"]
                arm_pair_scores[arm] = {
                    "before_arm_prediction_receipt_sha256": before_prediction[
                        "arm_prediction_receipts"
                    ][arm],
                    "after_arm_prediction_receipt_sha256": after_prediction[
                        "arm_prediction_receipts"
                    ][arm],
                    "arm_pair_receipt_sha256": scenario_prediction[
                        "arm_pair_receipts"
                    ][arm],
                    "before_old": before_old,
                    "after_old": after_old,
                    "after_new": after_new,
                    "after_all": after_all,
                    "B_old": b_old,
                    "A_old": a_old,
                    "N": new_score,
                    "H_old_new": (
                        0.0
                        if a_old + new_score == 0
                        else 2 * a_old * new_score / (a_old + new_score)
                    ),
                    "forgetting": b_old - a_old,
                }
            scored_scenarios.append(
                {
                    "scenario": scenario_name,
                    "before_query_physical_root_sha256": plan_scenario.before.query_physical_root_sha256,
                    "after_query_physical_root_sha256": plan_scenario.after.query_physical_root_sha256,
                    "before_scorer_input_seal_sha256": before_prediction[
                        "scorer_input_seal_sha256"
                    ],
                    "after_scorer_input_seal_sha256": after_prediction[
                        "scorer_input_seal_sha256"
                    ],
                    "arm_pair_scores": arm_pair_scores,
                }
            )
        row_score: dict[str, Any] = {
            "schema": SCORE_SCHEMA,
            "run_id": run.run_id,
            "row_id": row.row_id,
            **_plan_identity_payload(run.plan),
            "prediction_receipt_sha256": artifact["prediction_receipt_sha256"],
            "truth_open_event_sha256": event_sha,
            "scenario_pairs": scored_scenarios,
        }
        row_score["score_receipt_sha256"] = canonical_sha256(row_score)
        relative_path = f"rows/{row.row_id}.json"
        score_sha = _write_json_new(destination / relative_path, row_score)
        score_rows.append(
            {
                "row_id": row.row_id,
                "score_relative_path": relative_path,
                "score_sha256": score_sha,
                "score_receipt_sha256": row_score["score_receipt_sha256"],
            }
        )
    score_manifest: dict[str, Any] = {
        "schema": SCORE_MANIFEST_SCHEMA,
        "status": "SCORES_COMPLETE",
        "run_id": run.run_id,
        "plan_receipt_sha256": run.plan.plan_receipt_sha256,
        **_plan_identity_payload(run.plan),
        "prediction_manifest_sha256": _sha256_file(manifest_path),
        "truth_open_event_sha256": event_sha,
        "outer_row_count": OUTER_ROW_COUNT,
        "scenario_row_count": SCENARIO_ROW_COUNT,
        "scenario_arm_pair_count": SCENARIO_ARM_PAIR_COUNT,
        "state_prediction_surface_count": STATE_PREDICTION_SURFACE_COUNT,
        "state_score_surface_count": STATE_PREDICTION_SURFACE_COUNT,
        "rows": score_rows,
        "performance_selection_used": False,
    }
    score_manifest["score_manifest_receipt_sha256"] = canonical_sha256(score_manifest)
    output = destination / "score_manifest.json"
    _write_json_new(output, score_manifest)
    verify_d105_target25_score_manifest(run, destination)
    return output


def score_d105_target25_from_catalog_file(
    run: D105Target25Run,
    *,
    truth_catalog_path: Path,
    expected_truth_catalog_sha256: str,
    score_root: Path,
) -> Path:
    """Formal score path: close predictions before the first truth-file open."""

    # This ordering is a security boundary.  The catalog loader is the first
    # operation allowed to open truth and is unreachable until all 25 immutable
    # prediction artifacts and the complete prediction manifest validate.
    verify_d105_target25_prediction_manifest(run)
    catalog = load_d105_target25_truth_catalog_manifest(
        run,
        truth_catalog_path,
        expected_file_sha256=expected_truth_catalog_sha256,
    )
    return score_d105_target25_truth_side(
        run,
        catalog.truth_side_manifest,
        catalog.read_labels,
        score_root=score_root,
    )


def verify_d105_target25_score_manifest(
    run: D105Target25Run, score_root: Path
) -> dict[str, Any]:
    """Fail closed on all 25 score artifacts without ranking any metric."""

    _verify_run(run)
    prediction_manifest = verify_d105_target25_prediction_manifest(run)
    root = Path(score_root).resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise D105Target25RunnerError("score root is not a regular directory")
    truth_event_path = root / "truth_open_event.json"
    truth_event = _read_json_regular(truth_event_path)
    truth_event_keys = {
        "schema",
        "run_id",
        "prediction_manifest_sha256",
        "prediction_manifest_receipt_sha256",
        "truth_side_manifest_receipt_sha256",
        "opened_unix_ns",
        "all_predictions_committed_before_truth_open",
        "sealed_outer_row_count",
        "sealed_scenario_arm_pair_count",
        "sealed_state_prediction_surface_count",
        "predictor_truth_access",
    }
    if set(truth_event) != truth_event_keys or (
        truth_event.get("schema") != TRUTH_OPEN_SCHEMA
        or truth_event.get("run_id") != run.run_id
        or truth_event.get("prediction_manifest_sha256")
        != _sha256_file(run.run_root / "prediction_manifest.json")
        or truth_event.get("prediction_manifest_receipt_sha256")
        != prediction_manifest["prediction_manifest_receipt_sha256"]
        or type(truth_event.get("opened_unix_ns")) is not int
        or truth_event["opened_unix_ns"] <= 0
        or truth_event.get("all_predictions_committed_before_truth_open") is not True
        or truth_event.get("sealed_outer_row_count") != OUTER_ROW_COUNT
        or truth_event.get("sealed_scenario_arm_pair_count")
        != SCENARIO_ARM_PAIR_COUNT
        or truth_event.get("sealed_state_prediction_surface_count")
        != STATE_PREDICTION_SURFACE_COUNT
        or truth_event.get("predictor_truth_access") is not False
    ):
        raise D105Target25RunnerError("truth-open event closure drift")
    _require_sha256(
        truth_event.get("truth_side_manifest_receipt_sha256"),
        "truth_side_manifest_receipt_sha256",
    )
    manifest_path = root / "score_manifest.json"
    manifest = _read_json_regular(manifest_path)
    expected_keys = {
        "schema",
        "status",
        "run_id",
        "plan_receipt_sha256",
        "claim_scope",
        "formal_launch_authority",
        "authority_envelope_root_sha256",
        "data_feature_runtime_sha256",
        "data_materialization_lock_sha256",
        "d105_candidate_runtime_manifest_sha256",
        "d105_candidate_method_lock_sha256",
        "prediction_manifest_sha256",
        "truth_open_event_sha256",
        "outer_row_count",
        "scenario_row_count",
        "scenario_arm_pair_count",
        "state_prediction_surface_count",
        "state_score_surface_count",
        "rows",
        "performance_selection_used",
        "score_manifest_receipt_sha256",
    }
    if set(manifest) != expected_keys or (
        manifest.get("schema") != SCORE_MANIFEST_SCHEMA
        or manifest.get("status") != "SCORES_COMPLETE"
        or manifest.get("run_id") != run.run_id
        or manifest.get("plan_receipt_sha256") != run.plan.plan_receipt_sha256
        or any(
            manifest.get(name) != value
            for name, value in _plan_identity_payload(run.plan).items()
        )
        or manifest.get("prediction_manifest_sha256")
        != _sha256_file(run.run_root / "prediction_manifest.json")
        or manifest.get("truth_open_event_sha256")
        != _sha256_file(truth_event_path)
        or manifest.get("outer_row_count") != OUTER_ROW_COUNT
        or manifest.get("scenario_row_count") != SCENARIO_ROW_COUNT
        or manifest.get("scenario_arm_pair_count") != SCENARIO_ARM_PAIR_COUNT
        or manifest.get("state_prediction_surface_count")
        != STATE_PREDICTION_SURFACE_COUNT
        or manifest.get("state_score_surface_count")
        != STATE_PREDICTION_SURFACE_COUNT
        or manifest.get("performance_selection_used") is not False
    ):
        raise D105Target25RunnerError("score manifest closure drift")
    if manifest.get("score_manifest_receipt_sha256") != canonical_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "score_manifest_receipt_sha256"
        }
    ):
        raise D105Target25RunnerError("score manifest receipt drift")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != OUTER_ROW_COUNT:
        raise D105Target25RunnerError("score manifest row count drift")
    expected_row_ids = [row.row_id for row in run.plan.rows]
    if [item.get("row_id") for item in rows if isinstance(item, Mapping)] != expected_row_ids:
        raise D105Target25RunnerError("score manifest row order drift")
    score_units = 0
    prediction_receipt_by_row = {
        str(item["row_id"]): str(item["prediction_receipt_sha256"])
        for item in prediction_manifest["rows"]
    }
    artifact_by_row = {
        row_id: _read_json_regular(
            _relative_file(
                run.run_root,
                str(entry["artifact_relative_path"]),
                "artifact_relative_path",
            )
        )
        for row_id, entry in (
            (str(item["row_id"]), item) for item in prediction_manifest["rows"]
        )
    }

    def verify_metric(
        metrics: Any, *, classes: tuple[str, ...], query_count: int
    ) -> None:
        if (
            not isinstance(metrics, Mapping)
            or set(metrics) != {"correct_count", "query_count", "accuracy", "per_class"}
            or metrics.get("query_count") != query_count
            or type(metrics.get("correct_count")) is not int
            or metrics["correct_count"] < 0
            or metrics["correct_count"] > query_count
            or type(metrics.get("accuracy")) not in (int, float)
            or isinstance(metrics.get("accuracy"), bool)
            or metrics["accuracy"] != metrics["correct_count"] / query_count
            or not isinstance(metrics.get("per_class"), Mapping)
            or set(metrics["per_class"]) != set(classes)
        ):
            raise D105Target25RunnerError("row score metric closure drift")
        per_class_count = 0
        per_class_correct = 0
        for class_id in classes:
            class_metrics = metrics["per_class"][class_id]
            if (
                not isinstance(class_metrics, Mapping)
                or set(class_metrics) != {"count", "correct"}
                or type(class_metrics.get("count")) is not int
                or type(class_metrics.get("correct")) is not int
                or class_metrics["count"] < 0
                or class_metrics["correct"] < 0
                or class_metrics["correct"] > class_metrics["count"]
            ):
                raise D105Target25RunnerError("row score per-class closure drift")
            per_class_count += class_metrics["count"]
            per_class_correct += class_metrics["correct"]
        if (
            per_class_count != query_count
            or per_class_correct != metrics["correct_count"]
        ):
            raise D105Target25RunnerError("row score aggregate/per-class closure drift")

    for entry, row in zip(rows, run.plan.rows, strict=True):
        if not isinstance(entry, Mapping) or set(entry) != {
            "row_id",
            "score_relative_path",
            "score_sha256",
            "score_receipt_sha256",
        }:
            raise D105Target25RunnerError("score manifest row field closure drift")
        score_path = _relative_file(
            root, str(entry["score_relative_path"]), "score_relative_path"
        )
        if _sha256_file(score_path) != entry.get("score_sha256"):
            raise D105Target25RunnerError("row score SHA drift")
        score = _read_json_regular(score_path)
        if (
            set(score)
            != {
                "schema",
                "run_id",
                "row_id",
                "claim_scope",
                "formal_launch_authority",
                "authority_envelope_root_sha256",
                "data_feature_runtime_sha256",
                "data_materialization_lock_sha256",
                "d105_candidate_runtime_manifest_sha256",
                "d105_candidate_method_lock_sha256",
                "prediction_receipt_sha256",
                "truth_open_event_sha256",
                "scenario_pairs",
                "score_receipt_sha256",
            }
            or score.get("schema") != SCORE_SCHEMA
            or score.get("run_id") != run.run_id
            or score.get("row_id") != row.row_id
            or any(
                score.get(name) != value
                for name, value in _plan_identity_payload(run.plan).items()
            )
            or score.get("prediction_receipt_sha256")
            != prediction_receipt_by_row[row.row_id]
            or score.get("truth_open_event_sha256")
            != manifest["truth_open_event_sha256"]
            or score.get("score_receipt_sha256")
            != canonical_sha256(
                {
                    key: value
                    for key, value in score.items()
                    if key != "score_receipt_sha256"
                }
            )
            or entry.get("score_receipt_sha256")
            != score.get("score_receipt_sha256")
        ):
            raise D105Target25RunnerError("row score receipt/binding drift")
        scenarios = score.get("scenario_pairs")
        if not isinstance(scenarios, list) or [
            item.get("scenario") for item in scenarios if isinstance(item, Mapping)
        ] != list(LEO_SCENARIOS):
            raise D105Target25RunnerError("row score scenario closure drift")
        artifact_scenarios = {
            str(value["scenario"]): value
            for value in artifact_by_row[row.row_id]["scenario_predictions"]
        }
        for scenario_score, planned in zip(scenarios, row.scenarios, strict=True):
            if not isinstance(scenario_score, Mapping) or set(scenario_score) != {
                "scenario",
                "before_query_physical_root_sha256",
                "after_query_physical_root_sha256",
                "before_scorer_input_seal_sha256",
                "after_scorer_input_seal_sha256",
                "arm_pair_scores",
            }:
                raise D105Target25RunnerError("row score scenario-pair field closure drift")
            prediction_pair = artifact_scenarios.get(planned.scenario)
            if prediction_pair is None:
                raise D105Target25RunnerError("row score lacks sealed prediction pair")
            states = {
                str(value["stage"]): value
                for value in prediction_pair["state_predictions"]
            }
            before_prediction = states["S_B"]
            after_prediction = states["S_C"]
            arm_scores = scenario_score.get("arm_pair_scores")
            if (
                scenario_score.get("scenario") != planned.scenario
                or scenario_score.get("before_query_physical_root_sha256")
                != planned.before.query_physical_root_sha256
                or scenario_score.get("after_query_physical_root_sha256")
                != planned.after.query_physical_root_sha256
                or scenario_score.get("before_scorer_input_seal_sha256")
                != before_prediction["scorer_input_seal_sha256"]
                or scenario_score.get("after_scorer_input_seal_sha256")
                != after_prediction["scorer_input_seal_sha256"]
                or not isinstance(arm_scores, Mapping)
                or tuple(arm_scores) != ARMS
            ):
                raise D105Target25RunnerError("row score arm closure drift")
            before_count = len(planned.before.query_physical_ids)
            after_new_count = len(planned.after.query_physical_ids) - before_count
            for arm in ARMS:
                metrics = arm_scores[arm]
                if (
                    not isinstance(metrics, Mapping)
                    or set(metrics)
                    != {
                        "before_arm_prediction_receipt_sha256",
                        "after_arm_prediction_receipt_sha256",
                        "arm_pair_receipt_sha256",
                        "before_old",
                        "after_old",
                        "after_new",
                        "after_all",
                        "B_old",
                        "A_old",
                        "N",
                        "H_old_new",
                        "forgetting",
                    }
                    or metrics.get("before_arm_prediction_receipt_sha256")
                    != before_prediction["arm_prediction_receipts"][arm]
                    or metrics.get("after_arm_prediction_receipt_sha256")
                    != after_prediction["arm_prediction_receipts"][arm]
                    or metrics.get("arm_pair_receipt_sha256")
                    != prediction_pair["arm_pair_receipts"][arm]
                ):
                    raise D105Target25RunnerError("row score pair-receipt closure drift")
                verify_metric(
                    metrics["before_old"],
                    classes=planned.before.old_classes,
                    query_count=before_count,
                )
                verify_metric(
                    metrics["after_old"],
                    classes=planned.after.old_classes,
                    query_count=before_count,
                )
                verify_metric(
                    metrics["after_new"],
                    classes=planned.after.new_classes,
                    query_count=after_new_count,
                )
                verify_metric(
                    metrics["after_all"],
                    classes=planned.after.registered_classes,
                    query_count=len(planned.after.query_physical_ids),
                )
                if (
                    any(
                        type(metrics.get(name)) not in (int, float)
                        or isinstance(metrics.get(name), bool)
                        for name in ("B_old", "A_old", "N", "H_old_new", "forgetting")
                    )
                    or metrics["B_old"] != metrics["before_old"]["accuracy"]
                    or metrics["A_old"] != metrics["after_old"]["accuracy"]
                    or metrics["N"] != metrics["after_new"]["accuracy"]
                    or metrics["H_old_new"]
                    != (
                        0.0
                        if metrics["A_old"] + metrics["N"] == 0
                        else 2
                        * metrics["A_old"]
                        * metrics["N"]
                        / (metrics["A_old"] + metrics["N"])
                    )
                    or metrics["forgetting"] != metrics["B_old"] - metrics["A_old"]
                ):
                    raise D105Target25RunnerError("row score paired-metric closure drift")
                score_units += 1
    if score_units != SCENARIO_ARM_PAIR_COUNT:
        raise D105Target25RunnerError("score manifest arm-pair coverage drift")
    return manifest


def summarize_d105_target25_outputs(run: D105Target25Run) -> dict[str, Any]:
    """Return immutable coverage/status only; it never ranks performance."""

    _verify_run(run)
    complete = run.run_root / "prediction_manifest.json"
    partial = run.run_root / "partial_prediction_manifest.json"
    if complete.exists():
        manifest = verify_d105_target25_prediction_manifest(run)
        return {
            "status": manifest["status"],
            "claim_scope": manifest["claim_scope"],
            "formal_launch_authority": manifest["formal_launch_authority"],
            "authority_envelope_root_sha256": manifest[
                "authority_envelope_root_sha256"
            ],
            "outer_row_count": manifest["outer_row_count"],
            "scenario_row_count": manifest["scenario_row_count"],
            "scenario_arm_pair_count": manifest["scenario_arm_pair_count"],
            "state_prediction_surface_count": manifest[
                "state_prediction_surface_count"
            ],
            "performance_selection_used": False,
        }
    if partial.exists():
        document = _read_json_regular(partial)
        if any(
            document.get(name) != value
            for name, value in _plan_identity_payload(run.plan).items()
        ):
            raise D105Target25RunnerError(
                "partial summary claim/identity binding drift"
            )
        return {
            "status": document.get("status"),
            "claim_scope": document["claim_scope"],
            "formal_launch_authority": document["formal_launch_authority"],
            "authority_envelope_root_sha256": document[
                "authority_envelope_root_sha256"
            ],
            "outer_row_count": document.get("completed_outer_rows"),
            "scenario_row_count": int(document.get("succeeded_outer_rows", 0)) * 3,
            "scenario_arm_pair_count": document.get("scenario_arm_pair_count"),
            "state_prediction_surface_count": document.get(
                "state_prediction_surface_count"
            ),
            "performance_selection_used": False,
        }
    raise D105Target25RunnerError("run has no immutable execution manifest")


__all__ = [
    "ARMS",
    "CLAIM_SCOPES",
    "DEVELOPMENT_CLAIM_SCOPE",
    "D105Target25Assignment",
    "D105Target25ExecutionSummary",
    "D105Target25GPUSchedule",
    "D105Target25OuterRow",
    "D105Target25Plan",
    "D105Target25PredictionOutput",
    "D105Target25PredictionRequest",
    "D105Target25Run",
    "D105Target25RunnerError",
    "D105Target25ScenarioPlan",
    "D105Target25StatePlan",
    "D105Target25LoadedTruthCatalog",
    "D105Target25TruthLabels",
    "D105Target25TruthReadRequest",
    "D105Target25TruthSideManifest",
    "FORMAL_CLAIM_SCOPE",
    "LEO_SCENARIOS",
    "OUTER_ROW_COUNT",
    "SCENARIO_ARM_COUNT",
    "SCENARIO_ARM_PAIR_COUNT",
    "SCENARIO_ROW_COUNT",
    "STATE_PREDICTION_SURFACE_COUNT",
    "TARGET25_SEED",
    "TARGET25_SLICES",
    "build_d105_target25_truth_side_manifest",
    "canonical_sha256",
    "execute_d105_target25_predictions",
    "freeze_d105_target25_plan",
    "load_d105_target25_plan_manifest",
    "load_d105_target25_run",
    "load_d105_target25_truth_catalog_manifest",
    "normalise_exception_fingerprint",
    "prepare_d105_target25_run",
    "score_d105_target25_truth_side",
    "score_d105_target25_from_catalog_file",
    "seal_d105_target25_truth_catalog_manifest",
    "summarize_d105_target25_outputs",
    "validate_d105_target25_prediction_artifact_without_truth",
    "verify_d105_target25_score_manifest",
    "verify_d105_target25_prediction_manifest",
    "write_d105_target25_truth_side_manifest",
    "write_d105_target25_plan_manifest",
]
