"""Same-row source scoring, non-compensatory MIRAGE Gates, and arm selection.

The module consumes only frozen decision tables and explicit receipts.  Gate 4
is intentionally a pure summary-to-receipt helper and cannot feed target
results back into source candidate selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Sequence

from sklearn.metrics import roc_auc_score

from .calibration import CalibrationProtocolError, FrozenDecisionTable, _validate_frozen_decision_table
from .protocol import ProxyRole, SourcePartition


class ScoringProtocolError(ValueError):
    """Raised when a metric, receipt, gate input, or source-selection boundary is invalid."""


class NoPromotedArmError(RuntimeError):
    """Raised when source evidence contains no arm passing all Gates 1--3."""


def _require_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScoringProtocolError(f"{field_name} must be a non-empty string")
    return value


def _require_fold(value: object, field_name: str = "fold") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScoringProtocolError(f"{field_name} must be a non-negative integer")
    return value


def _require_unit_interval(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ScoringProtocolError(f"{field_name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ScoringProtocolError(f"{field_name} must be finite")
    if not 0.0 <= numeric <= 1.0:
        raise ScoringProtocolError(f"{field_name} must lie in [0, 1]")
    return numeric


def _require_nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScoringProtocolError(f"{field_name} must be a non-negative integer")
    return value


def _freeze_metric_mapping(value: object, field_name: str) -> Mapping[object, float]:
    if not isinstance(value, Mapping) or not value:
        raise ScoringProtocolError(f"{field_name} must be a non-empty mapping")
    frozen: dict[object, float] = {}
    for key, metric in value.items():
        try:
            hash(key)
        except TypeError as error:
            raise ScoringProtocolError(f"{field_name} keys must be hashable") from error
        frozen[key] = _require_unit_interval(metric, f"{field_name}[{key!r}]")
    return MappingProxyType(frozen)


def _at_least(value: float, limit: float) -> bool:
    return value > limit or math.isclose(value, limit, rel_tol=0.0, abs_tol=1e-12)


@dataclass(frozen=True)
class SameRowMetrics:
    """All source metrics derived from one frozen decision table and one fold."""

    candidate_id: str
    fold: int
    macro_accuracy: float
    per_class_accuracy: Mapping[object, float]
    min_class_accuracy: float
    receiver_accuracy: Mapping[object, float]
    day_accuracy: Mapping[object, float]
    scene_accuracy: Mapping[object, float]
    worst_scene_accuracy: float
    known_frr: float
    proxy_explicit_rejection: float
    proxy_false_accept: float
    proxy_defer: float
    proxy_coverage: float
    proxy_auroc: float
    proxy_update_count: int
    decision_table_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _require_identifier(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "fold", _require_fold(self.fold))
        for field_name in (
            "macro_accuracy",
            "min_class_accuracy",
            "worst_scene_accuracy",
            "known_frr",
            "proxy_explicit_rejection",
            "proxy_false_accept",
            "proxy_defer",
            "proxy_coverage",
            "proxy_auroc",
        ):
            object.__setattr__(self, field_name, _require_unit_interval(getattr(self, field_name), field_name))
        object.__setattr__(
            self,
            "per_class_accuracy",
            _freeze_metric_mapping(self.per_class_accuracy, "per_class_accuracy"),
        )
        object.__setattr__(
            self,
            "receiver_accuracy",
            _freeze_metric_mapping(self.receiver_accuracy, "receiver_accuracy"),
        )
        object.__setattr__(self, "day_accuracy", _freeze_metric_mapping(self.day_accuracy, "day_accuracy"))
        object.__setattr__(
            self,
            "scene_accuracy",
            _freeze_metric_mapping(self.scene_accuracy, "scene_accuracy"),
        )
        if not math.isclose(
            self.min_class_accuracy,
            min(self.per_class_accuracy.values()),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ScoringProtocolError("min_class_accuracy must match per_class_accuracy")
        if not math.isclose(
            self.worst_scene_accuracy,
            min(self.scene_accuracy.values()),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ScoringProtocolError("worst_scene_accuracy must match scene_accuracy")
        if not math.isclose(
            self.proxy_coverage,
            self.proxy_explicit_rejection + self.proxy_false_accept,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ScoringProtocolError("proxy_coverage must equal explicit rejection plus false accept")
        if not math.isclose(
            self.proxy_defer + self.proxy_coverage,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ScoringProtocolError("proxy coverage and defer must sum to one")
        object.__setattr__(
            self,
            "proxy_update_count",
            _require_nonnegative_int(self.proxy_update_count, "proxy_update_count"),
        )
        object.__setattr__(self, "decision_table_id", _require_identifier(self.decision_table_id, "decision_table_id"))


def _correct_known(row: object) -> bool:
    return bool(row.registered and row.row.predicted_class == row.row.true_class)


def _macro_accuracy(rows: Sequence[object]) -> float:
    classes: dict[object, list[bool]] = {}
    for row in rows:
        classes.setdefault(row.row.true_class, []).append(_correct_known(row))
    if not classes:
        raise ScoringProtocolError("known score rows must not be empty")
    return sum(sum(values) / len(values) for values in classes.values()) / len(classes)


def _per_class_accuracy(rows: Sequence[object]) -> Mapping[object, float]:
    by_class_scene: dict[object, dict[object, list[bool]]] = {}
    for row in rows:
        by_class_scene.setdefault(row.row.true_class, {}).setdefault(row.row.scene, []).append(_correct_known(row))
    return MappingProxyType(
        {
            class_id: sum(sum(values) / len(values) for values in scene_values.values()) / len(scene_values)
            for class_id, scene_values in by_class_scene.items()
        }
    )


def _group_macro_accuracy(rows: Sequence[object], attribute: str) -> Mapping[object, float]:
    groups: dict[object, list[object]] = {}
    for row in rows:
        groups.setdefault(getattr(row.row, attribute), []).append(row)
    return MappingProxyType({key: _macro_accuracy(values) for key, values in groups.items()})


def score_same_row(decision_table: FrozenDecisionTable) -> SameRowMetrics:
    """Score one frozen V_select/P_select fold without mixing decision rows."""

    if not isinstance(decision_table, FrozenDecisionTable):
        raise ScoringProtocolError("same-row scoring requires a FrozenDecisionTable")
    try:
        decision_table = _validate_frozen_decision_table(decision_table)
    except CalibrationProtocolError as error:
        raise ScoringProtocolError("same-row scoring rejected an unsealed or mismatched source receipt") from error
    if (
        decision_table.known_role is not SourcePartition.V_SELECT
        or decision_table.proxy_role is not ProxyRole.P_SELECT
    ):
        raise ScoringProtocolError("same-row scoring accepts only frozen V_select/P_select decisions")
    if decision_table.proxy_update_count != 0:
        raise ScoringProtocolError("P_select update_count must be zero for source scoring")
    known_rows = decision_table.known_rows
    proxy_rows = decision_table.proxy_rows
    per_class = _per_class_accuracy(known_rows)
    receiver = _group_macro_accuracy(known_rows, "receiver")
    day = _group_macro_accuracy(known_rows, "day")
    scene = _group_macro_accuracy(known_rows, "scene")
    known_correct = [_correct_known(row) for row in known_rows]
    labels = [0] * len(known_rows) + [1] * len(proxy_rows)
    risks = [row.row.unknown_risk for row in known_rows] + [row.row.unknown_risk for row in proxy_rows]
    try:
        proxy_auroc = float(roc_auc_score(labels, risks))
    except ValueError as error:
        raise ScoringProtocolError("proxy AUROC requires both known and proxy rows") from error
    explicit_rejection = sum(row.explicit_unknown for row in proxy_rows) / len(proxy_rows)
    false_accept = sum(row.registered for row in proxy_rows) / len(proxy_rows)
    defer = sum(row.deferred for row in proxy_rows) / len(proxy_rows)
    return SameRowMetrics(
        candidate_id=decision_table.candidate_id,
        fold=decision_table.fold,
        macro_accuracy=sum(scene.values()) / len(scene),
        per_class_accuracy=per_class,
        min_class_accuracy=min(per_class.values()),
        receiver_accuracy=receiver,
        day_accuracy=day,
        scene_accuracy=scene,
        worst_scene_accuracy=min(scene.values()),
        known_frr=sum(not row.registered for row in known_rows) / len(known_rows),
        proxy_explicit_rejection=explicit_rejection,
        proxy_false_accept=false_accept,
        proxy_defer=defer,
        proxy_coverage=explicit_rejection + false_accept,
        proxy_auroc=proxy_auroc,
        proxy_update_count=decision_table.proxy_update_count,
        decision_table_id=decision_table.table_id,
    )


@dataclass(frozen=True)
class SourceArmSummary:
    """Exactly six same-row fold metrics for one arm, never marginal maxima."""

    arm_id: str
    fold_metrics: tuple[SameRowMetrics, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_id", _require_identifier(self.arm_id, "arm_id"))
        if not isinstance(self.fold_metrics, tuple) or len(self.fold_metrics) != 6:
            raise ScoringProtocolError("source arm summary requires exactly six fold metrics")
        if any(not isinstance(metric, SameRowMetrics) for metric in self.fold_metrics):
            raise ScoringProtocolError("source arm summary contains an invalid fold metric")
        folds = [metric.fold for metric in self.fold_metrics]
        if len(set(folds)) != 6:
            raise ScoringProtocolError("source arm summary requires six unique folds")
        if any(metric.candidate_id != self.arm_id for metric in self.fold_metrics):
            raise ScoringProtocolError("all fold metrics must belong to the same arm")
        table_ids = [metric.decision_table_id for metric in self.fold_metrics]
        if len(set(table_ids)) != 6:
            raise ScoringProtocolError("each source fold must use one unique frozen decision table")

    @property
    def by_fold(self) -> Mapping[int, SameRowMetrics]:
        return MappingProxyType({metric.fold: metric for metric in self.fold_metrics})

    @property
    def macro_accuracy(self) -> float:
        return sum(metric.macro_accuracy for metric in self.fold_metrics) / 6

    @property
    def min_class_accuracy(self) -> float:
        return sum(metric.min_class_accuracy for metric in self.fold_metrics) / 6

    @property
    def worst_scene_accuracy(self) -> float:
        return sum(metric.worst_scene_accuracy for metric in self.fold_metrics) / 6

    @property
    def proxy_auroc(self) -> float:
        return sum(metric.proxy_auroc for metric in self.fold_metrics) / 6

    @property
    def known_frr(self) -> float:
        return sum(metric.known_frr for metric in self.fold_metrics) / 6

    @property
    def proxy_update_count(self) -> int:
        return sum(metric.proxy_update_count for metric in self.fold_metrics)


def aggregate_sixfold(arm_id: str, metrics: Sequence[SameRowMetrics]) -> SourceArmSummary:
    """Create an equal-fold source summary; source selection never chooses one fold."""

    return SourceArmSummary(arm_id=arm_id, fold_metrics=tuple(metrics))


@dataclass(frozen=True)
class TrainerFoldReceipt:
    """Read-only trainer completion receipt paired with its pre-registered fold."""

    fold: int
    receipt: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fold", _require_fold(self.fold))
        if not isinstance(self.receipt, Mapping):
            raise ScoringProtocolError("trainer receipt must be a mapping")
        object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))

    @property
    def valid(self) -> bool:
        receipt = self.receipt
        if receipt.get("schema") != "phase1_mirage_completion_receipt_v1":
            return False
        if receipt.get("status") != "COMPLETED" or receipt.get("selection_source") != "V_select":
            return False
        digest = receipt.get("checkpoint_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            return False
        if isinstance(receipt.get("epochs_completed"), bool) or not isinstance(receipt.get("epochs_completed"), int):
            return False
        if isinstance(receipt.get("selected_epoch"), bool) or not isinstance(receipt.get("selected_epoch"), int):
            return False
        try:
            _require_unit_interval(receipt["v_select_known_macro"], "v_select_known_macro")
            _require_unit_interval(receipt["v_select_worst_scene"], "v_select_worst_scene")
        except (KeyError, ScoringProtocolError):
            return False
        return True


@dataclass(frozen=True)
class Gate1Evidence:
    """Independent Boolean receipts required for protocol/training closure."""

    split_receipt_valid: bool
    receiver_tx_disjoint: bool
    proxy_origin_valid: bool
    target_training_access_count: int
    target_calibration_access_count: int
    target_selection_access_count: int
    checkpoint_forward_complete: bool
    trainer_receipts: tuple[TrainerFoldReceipt, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "split_receipt_valid",
            "receiver_tx_disjoint",
            "proxy_origin_valid",
            "checkpoint_forward_complete",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ScoringProtocolError(f"{field_name} must be bool")
        for field_name in (
            "target_training_access_count",
            "target_calibration_access_count",
            "target_selection_access_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_nonnegative_int(getattr(self, field_name), field_name),
            )
        if not isinstance(self.trainer_receipts, tuple) or any(
            not isinstance(receipt, TrainerFoldReceipt) for receipt in self.trainer_receipts
        ):
            raise ScoringProtocolError("trainer_receipts must be an immutable tuple of TrainerFoldReceipt")


@dataclass(frozen=True)
class Gate1Receipt:
    """Gate 1 pass/fail with immutable explicit checks."""

    passed: bool
    checks: Mapping[str, bool]

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise ScoringProtocolError("Gate 1 passed must be bool")
        if not isinstance(self.checks, Mapping):
            raise ScoringProtocolError("Gate 1 checks must be a mapping")
        frozen = {str(name): bool(value) for name, value in self.checks.items()}
        object.__setattr__(self, "checks", MappingProxyType(frozen))


def evaluate_gate1(evidence: Gate1Evidence, *, expected_folds: frozenset[int]) -> Gate1Receipt:
    """Evaluate protocol/training closure from explicit receipts, without target reads."""

    if not isinstance(evidence, Gate1Evidence):
        raise ScoringProtocolError("Gate 1 requires Gate1Evidence")
    if len(expected_folds) != 6:
        raise ScoringProtocolError("Gate 1 requires exactly six expected folds")
    receipt_folds = [receipt.fold for receipt in evidence.trainer_receipts]
    receipt_fold_set = set(receipt_folds)
    checks = {
        "split_receipt": evidence.split_receipt_valid,
        "receiver_tx_disjoint": evidence.receiver_tx_disjoint,
        "proxy_origin": evidence.proxy_origin_valid,
        "target_access_zero": (
            evidence.target_training_access_count == 0
            and evidence.target_calibration_access_count == 0
            and evidence.target_selection_access_count == 0
        ),
        "checkpoint_forward_complete": evidence.checkpoint_forward_complete,
        "trainer_receipts_complete": (
            len(receipt_folds) == 6
            and receipt_fold_set == set(expected_folds)
            and len(receipt_fold_set) == len(receipt_folds)
            and all(receipt.valid for receipt in evidence.trainer_receipts)
        ),
    }
    return Gate1Receipt(passed=all(checks.values()), checks=checks)


@dataclass(frozen=True)
class SourceGateReceipt:
    """All source Gate 1--3 decisions for a single arm relative to B0."""

    candidate: SourceArmSummary
    baseline: SourceArmSummary
    gate1_pass: bool
    gate2_pass: bool
    gate3_pass: bool
    promoted: bool
    gate1_checks: Mapping[str, bool]
    gate2_checks: Mapping[str, bool]
    gate3_checks: Mapping[str, bool]
    gate_margins: Mapping[str, float]
    normalized_gate_slacks: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, SourceArmSummary) or not isinstance(self.baseline, SourceArmSummary):
            raise ScoringProtocolError("source Gate receipt requires source arm summaries")
        for field_name in ("gate1_pass", "gate2_pass", "gate3_pass", "promoted"):
            if not isinstance(getattr(self, field_name), bool):
                raise ScoringProtocolError(f"{field_name} must be bool")
        if self.promoted != (self.gate1_pass and self.gate2_pass and self.gate3_pass):
            raise ScoringProtocolError("promoted must equal Gate1 and Gate2 and Gate3")
        for field_name in ("gate1_checks", "gate2_checks", "gate3_checks"):
            value = getattr(self, field_name)
            if not isinstance(value, Mapping):
                raise ScoringProtocolError(f"{field_name} must be a mapping")
            object.__setattr__(self, field_name, MappingProxyType({str(key): bool(item) for key, item in value.items()}))
        if not isinstance(self.gate_margins, Mapping) or not self.gate_margins:
            raise ScoringProtocolError("gate_margins must be a non-empty mapping")
        object.__setattr__(
            self,
            "gate_margins",
            MappingProxyType(
                {str(key): _require_unit_or_signed(value, f"gate_margins[{key!r}]") for key, value in self.gate_margins.items()}
            ),
        )
        if not isinstance(self.normalized_gate_slacks, Mapping) or not self.normalized_gate_slacks:
            raise ScoringProtocolError("normalized_gate_slacks must be a non-empty mapping")
        expected_slacks = {
            "gate2_macro_delta",
            "gate2_minimum_delta",
            "gate2_worst_scene_delta",
            "gate2_fold_nondegrade",
            "gate3_proxy_auroc",
            "gate3_proxy_auroc_delta",
            "gate3_known_frr",
        }
        if set(self.normalized_gate_slacks) != expected_slacks:
            raise ScoringProtocolError("normalized_gate_slacks must contain the pre-registered Gate 2/3 constraints")
        object.__setattr__(
            self,
            "normalized_gate_slacks",
            MappingProxyType(
                {
                    str(key): _require_unit_or_signed(value, f"normalized_gate_slacks[{key!r}]")
                    for key, value in self.normalized_gate_slacks.items()
                }
            ),
        )

    @property
    def weakest_gate_margin(self) -> float:
        """Return the minimum pre-registered dimensionless Gate 2/3 slack.

        Gate 1 and P_select update-count are Boolean closure checks and are
        deliberately excluded.  Selection already filters to promoted arms;
        the remaining continuous constraints use their approved normalizers.
        """

        return min(self.normalized_gate_slacks.values())


def _require_unit_or_signed(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ScoringProtocolError(f"{field_name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ScoringProtocolError(f"{field_name} must be finite")
    return numeric


def _evaluate_gate2(
    candidate: SourceArmSummary,
    baseline: SourceArmSummary,
) -> tuple[Mapping[str, bool], Mapping[str, float], Mapping[str, float]]:
    candidate_folds = candidate.by_fold
    baseline_folds = baseline.by_fold
    if set(candidate_folds) != set(baseline_folds):
        raise ScoringProtocolError("candidate and B0 must use the same six folds")
    nondegraded = []
    for fold in sorted(candidate_folds):
        candidate_metric = candidate_folds[fold]
        baseline_metric = baseline_folds[fold]
        nondegraded.append(
            _at_least(candidate_metric.macro_accuracy, baseline_metric.macro_accuracy - 0.005)
            and _at_least(candidate_metric.min_class_accuracy, baseline_metric.min_class_accuracy - 0.01)
            and _at_least(candidate_metric.worst_scene_accuracy, baseline_metric.worst_scene_accuracy - 0.005)
        )
    count = sum(nondegraded)
    macro_delta = candidate.macro_accuracy - baseline.macro_accuracy
    minimum_delta = candidate.min_class_accuracy - baseline.min_class_accuracy
    worst_delta = candidate.worst_scene_accuracy - baseline.worst_scene_accuracy
    checks = {
        "macro_delta_at_least_0_02": _at_least(macro_delta, 0.02),
        "minimum_delta_at_least_0_01": _at_least(minimum_delta, 0.01),
        "worst_scene_nonlower": _at_least(worst_delta, 0.0),
        "fold_nondegrade_5_of_6": count >= 5,
    }
    margins = {
        "gate2_macro_delta": macro_delta - 0.02,
        "gate2_minimum_delta": minimum_delta - 0.01,
        "gate2_worst_scene_delta": worst_delta,
        "gate2_fold_nondegrade": (count - 5) / 6,
    }
    normalized_slacks = {
        "gate2_macro_delta": (macro_delta - 0.02) / 0.02,
        "gate2_minimum_delta": (minimum_delta - 0.01) / 0.01,
        "gate2_worst_scene_delta": worst_delta / 0.005,
        "gate2_fold_nondegrade": (count - 5) / 1,
    }
    return MappingProxyType(checks), MappingProxyType(margins), MappingProxyType(normalized_slacks)


def _evaluate_gate3(
    candidate: SourceArmSummary,
    baseline: SourceArmSummary,
) -> tuple[Mapping[str, bool], Mapping[str, float], Mapping[str, float]]:
    auroc_delta = candidate.proxy_auroc - baseline.proxy_auroc
    checks = {
        "proxy_auroc_at_least_0_85": _at_least(candidate.proxy_auroc, 0.85),
        "proxy_auroc_delta_at_least_0_05": _at_least(auroc_delta, 0.05),
        "v_select_known_frr_at_most_0_10": _at_least(0.10 - candidate.known_frr, 0.0),
        "proxy_update_count_zero": candidate.proxy_update_count == 0,
    }
    margins = {
        "gate3_proxy_auroc": candidate.proxy_auroc - 0.85,
        "gate3_proxy_auroc_delta": auroc_delta - 0.05,
        "gate3_known_frr": 0.10 - candidate.known_frr,
    }
    normalized_slacks = {
        "gate3_proxy_auroc": (candidate.proxy_auroc - 0.85) / 0.15,
        "gate3_proxy_auroc_delta": (auroc_delta - 0.05) / 0.05,
        "gate3_known_frr": (0.10 - candidate.known_frr) / 0.10,
    }
    return MappingProxyType(checks), MappingProxyType(margins), MappingProxyType(normalized_slacks)


def evaluate_source_gates(
    candidate: SourceArmSummary,
    baseline: SourceArmSummary,
    gate1: Gate1Evidence,
) -> SourceGateReceipt:
    """Apply the non-compensatory source Gates 1--3 to one complete sixfold arm."""

    if not isinstance(candidate, SourceArmSummary) or not isinstance(baseline, SourceArmSummary):
        raise ScoringProtocolError("source Gates require SourceArmSummary inputs")
    if candidate.arm_id == baseline.arm_id:
        raise ScoringProtocolError("candidate arm and B0 arm must differ")
    gate1_receipt = evaluate_gate1(gate1, expected_folds=frozenset(candidate.by_fold))
    gate2_checks, gate2_margins, gate2_slacks = _evaluate_gate2(candidate, baseline)
    gate3_checks, gate3_margins, gate3_slacks = _evaluate_gate3(candidate, baseline)
    gate2_pass = all(gate2_checks.values())
    gate3_pass = all(gate3_checks.values())
    margins = dict(gate2_margins)
    margins.update(gate3_margins)
    normalized_slacks = dict(gate2_slacks)
    normalized_slacks.update(gate3_slacks)
    return SourceGateReceipt(
        candidate=candidate,
        baseline=baseline,
        gate1_pass=gate1_receipt.passed,
        gate2_pass=gate2_pass,
        gate3_pass=gate3_pass,
        promoted=gate1_receipt.passed and gate2_pass and gate3_pass,
        gate1_checks=gate1_receipt.checks,
        gate2_checks=gate2_checks,
        gate3_checks=gate3_checks,
        gate_margins=margins,
        normalized_gate_slacks=normalized_slacks,
    )


_TARGET_UNKNOWN_SCENES = frozenset({"global", "clear", "low_elev", "rain"})


@dataclass(frozen=True)
class SealedTargetSummary:
    """A file-free immutable input to the one-shot Gate 4 scorer."""

    arm_id: str
    seal_id: str
    known_macro_accuracy: float
    min_class_accuracy: float
    worst_scene_accuracy: float
    explicit_unknown_rejection: Mapping[str, float]
    known_frr: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_id", _require_identifier(self.arm_id, "arm_id"))
        object.__setattr__(self, "seal_id", _require_identifier(self.seal_id, "seal_id"))
        for field_name in (
            "known_macro_accuracy",
            "min_class_accuracy",
            "worst_scene_accuracy",
            "known_frr",
        ):
            object.__setattr__(self, field_name, _require_unit_interval(getattr(self, field_name), field_name))
        if not isinstance(self.explicit_unknown_rejection, Mapping):
            raise ScoringProtocolError("explicit_unknown_rejection must be a mapping")
        values = dict(self.explicit_unknown_rejection)
        if set(values) != _TARGET_UNKNOWN_SCENES:
            raise ScoringProtocolError("explicit_unknown_rejection must contain exactly global, clear, low_elev, rain")
        object.__setattr__(
            self,
            "explicit_unknown_rejection",
            MappingProxyType(
                {
                    scene: _require_unit_interval(values[scene], f"explicit_unknown_rejection[{scene}]")
                    for scene in sorted(_TARGET_UNKNOWN_SCENES)
                }
            ),
        )


@dataclass(frozen=True)
class Gate4Receipt:
    """One-shot target confirmation receipt; never an input to source selection."""

    candidate_arm_id: str
    baseline_arm_id: str
    passed: bool
    checks: Mapping[str, bool]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_arm_id", _require_identifier(self.candidate_arm_id, "candidate_arm_id"))
        object.__setattr__(self, "baseline_arm_id", _require_identifier(self.baseline_arm_id, "baseline_arm_id"))
        if not isinstance(self.passed, bool) or not isinstance(self.checks, Mapping):
            raise ScoringProtocolError("Gate 4 receipt is malformed")
        object.__setattr__(self, "checks", MappingProxyType({str(key): bool(value) for key, value in self.checks.items()}))


def evaluate_gate4(candidate: SealedTargetSummary, baseline: SealedTargetSummary) -> Gate4Receipt:
    """Score only two sealed target summaries under the exact non-feedback Gate 4 rules."""

    if not isinstance(candidate, SealedTargetSummary) or not isinstance(baseline, SealedTargetSummary):
        raise ScoringProtocolError("Gate 4 accepts only sealed target summaries")
    if candidate.arm_id == baseline.arm_id or candidate.seal_id == baseline.seal_id:
        raise ScoringProtocolError("Gate 4 candidate and B0 summaries must be distinct sealed artifacts")
    unknown_scene_checks = {
        scene: _at_least(candidate.explicit_unknown_rejection[scene], 0.70)
        for scene in _TARGET_UNKNOWN_SCENES
    }
    checks = {
        "known_macro_delta_at_least_0_02": _at_least(
            candidate.known_macro_accuracy - baseline.known_macro_accuracy,
            0.02,
        ),
        "minimum_class_nonlower": _at_least(candidate.min_class_accuracy - baseline.min_class_accuracy, 0.0),
        "worst_scene_nonlower": _at_least(candidate.worst_scene_accuracy - baseline.worst_scene_accuracy, 0.0),
        "all_unknown_scenes": all(unknown_scene_checks.values()),
        "known_frr_at_most_0_10": _at_least(0.10 - candidate.known_frr, 0.0),
    }
    return Gate4Receipt(
        candidate_arm_id=candidate.arm_id,
        baseline_arm_id=baseline.arm_id,
        passed=all(checks.values()),
        checks=checks,
    )


@dataclass(frozen=True)
class ArmSelectionCandidate:
    """A promoted source arm and its source-only deployment-size receipt."""

    arm_id: str
    gate_receipt: SourceGateReceipt
    bundle_bytes: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_id", _require_identifier(self.arm_id, "arm_id"))
        if not isinstance(self.gate_receipt, SourceGateReceipt):
            raise ScoringProtocolError("arm selection requires a SourceGateReceipt")
        if self.gate_receipt.candidate.arm_id != self.arm_id:
            raise ScoringProtocolError("arm selection candidate ID must match its source Gate receipt")
        if self.bundle_bytes is not None:
            object.__setattr__(self, "bundle_bytes", _require_nonnegative_int(self.bundle_bytes, "bundle_bytes"))


@dataclass(frozen=True)
class ArmSelectionReceipt:
    """Unique source-only arm decision and the final tie-break actually used."""

    selected_arm_id: str
    tie_break: str
    ranked_arm_ids: tuple[str, ...]
    used_target_data: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_arm_id", _require_identifier(self.selected_arm_id, "selected_arm_id"))
        object.__setattr__(self, "tie_break", _require_identifier(self.tie_break, "tie_break"))
        if not isinstance(self.ranked_arm_ids, tuple) or not self.ranked_arm_ids:
            raise ScoringProtocolError("ranked_arm_ids must be a non-empty tuple")
        if self.selected_arm_id not in self.ranked_arm_ids:
            raise ScoringProtocolError("selected arm must appear in ranked arm IDs")
        if self.used_target_data is not False:
            raise ScoringProtocolError("source arm selection must not use target data")


def _max_filter(
    candidates: tuple[ArmSelectionCandidate, ...],
    value,
) -> tuple[ArmSelectionCandidate, ...]:
    best_value = max(value(candidate) for candidate in candidates)
    return tuple(candidate for candidate in candidates if math.isclose(value(candidate), best_value, rel_tol=0.0, abs_tol=1e-12))


def select_unique_arm(candidates: Sequence[ArmSelectionCandidate]) -> ArmSelectionReceipt:
    """Choose one promoted arm by normalized Gate 2/3 slack, macro, AUROC, bytes, then stable ID.

    The first comparison is the pre-registered dimensionless minimum slack
    described by :attr:`SourceGateReceipt.weakest_gate_margin`; Boolean Gate 1
    and proxy-update closure do not collapse otherwise distinct promoted arms.
    Missing bundle-size receipts are deterministically ranked as infinity. This
    avoids target fallback while still yielding a stable result whenever at
    least one source-promoted arm exists.
    """

    if isinstance(candidates, (str, bytes)):
        raise ScoringProtocolError("arm selection candidates must be a sequence")
    all_candidates = tuple(candidates)
    if not all_candidates or any(not isinstance(candidate, ArmSelectionCandidate) for candidate in all_candidates):
        raise ScoringProtocolError("arm selection requires ArmSelectionCandidate inputs")
    arm_ids = [candidate.arm_id for candidate in all_candidates]
    if len(set(arm_ids)) != len(arm_ids):
        raise ScoringProtocolError("arm selection IDs must be unique")
    promoted = tuple(candidate for candidate in all_candidates if candidate.gate_receipt.promoted)
    if not promoted:
        raise NoPromotedArmError("NO_PROMOTED_SOURCE_ARM")
    remaining = _max_filter(promoted, lambda candidate: candidate.gate_receipt.weakest_gate_margin)
    if len(remaining) == 1:
        tie_break = "weakest_gate_margin"
    else:
        remaining = _max_filter(remaining, lambda candidate: candidate.gate_receipt.candidate.macro_accuracy)
        if len(remaining) == 1:
            tie_break = "source_macro_accuracy"
        else:
            remaining = _max_filter(remaining, lambda candidate: candidate.gate_receipt.candidate.proxy_auroc)
            if len(remaining) == 1:
                tie_break = "proxy_auroc"
            else:
                smallest_bytes = min(
                    math.inf if candidate.bundle_bytes is None else candidate.bundle_bytes for candidate in remaining
                )
                remaining = tuple(
                    candidate
                    for candidate in remaining
                    if (math.inf if candidate.bundle_bytes is None else candidate.bundle_bytes) == smallest_bytes
                )
                if len(remaining) == 1:
                    tie_break = "bundle_bytes"
                else:
                    tie_break = "stable_arm_id"
    ranked = tuple(
        candidate.arm_id
        for candidate in sorted(
            promoted,
            key=lambda candidate: (
                -candidate.gate_receipt.weakest_gate_margin,
                -candidate.gate_receipt.candidate.macro_accuracy,
                -candidate.gate_receipt.candidate.proxy_auroc,
                math.inf if candidate.bundle_bytes is None else candidate.bundle_bytes,
                candidate.arm_id,
            ),
        )
    )
    selected = min(remaining, key=lambda candidate: candidate.arm_id)
    return ArmSelectionReceipt(
        selected_arm_id=selected.arm_id,
        tie_break=tie_break,
        ranked_arm_ids=ranked,
        used_target_data=False,
    )


__all__ = [
    "ArmSelectionCandidate",
    "ArmSelectionReceipt",
    "Gate1Evidence",
    "Gate1Receipt",
    "Gate4Receipt",
    "NoPromotedArmError",
    "SameRowMetrics",
    "ScoringProtocolError",
    "SealedTargetSummary",
    "SourceArmSummary",
    "SourceGateReceipt",
    "TrainerFoldReceipt",
    "aggregate_sixfold",
    "evaluate_gate1",
    "evaluate_gate4",
    "evaluate_source_gates",
    "score_same_row",
    "select_unique_arm",
]
