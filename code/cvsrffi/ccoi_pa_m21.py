"""Pure contracts and diagnostics for the PA-M2.1 theta-transfer audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
import hashlib
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .ccoi_pa import (
    CCOIPASidecar,
    PAChallengeEncoder,
    nonoverlap_anchor_indices,
    raw_support_holdout_masks,
)
from .ccoi_causal_audit import group_paired_bootstrap


SIDECAR_V2_SCHEMA = "cvs.phase1.ccoi_pa_sidecar.v2"
SIDECAR_V3_SCHEMA = "cvs.phase1.ccoi_pa_sidecar.v3"
GATE_FEATURE_ALLOWLIST = frozenset(
    {
        "base_margin",
        "base_entropy",
        "operator_margin",
        "operator_entropy",
        "js_divergence",
        "top1_disagreement",
        "rms",
        "papr",
        "pa_condition_number",
        "spectral_null_ratio",
        "clipping_ratio",
        "snr_proxy",
        "residual_cfo",
        "phase_instability",
        "challenge_coverage",
    }
)


@dataclass(frozen=True)
class RetroSplit:
    block_size: int
    fit_ratio: float
    fit_indices: tuple[int, ...]
    audit_indices: tuple[int, ...]
    guard_indices: tuple[int, ...]
    role_by_group: Mapping[tuple[int, int, int, int, int], str]
    base_index_overlap_count: int
    cell_count: int
    min_blocks_per_cell: int


@dataclass(frozen=True)
class FoldRecords:
    base_index: Tensor
    fold_id: Tensor
    theta: Tensor
    q_holdout: Tensor
    target: Tensor
    support_raw_mask: Tensor
    holdout_raw_mask: Tensor
    fold_count: int


@dataclass(frozen=True)
class RelationMapping:
    relation: str
    index: tuple[int, ...]
    valid: tuple[bool, ...]
    candidate_seed: int
    selection_policy: str
    selection_uses_learned_q: bool
    fallback_count: int


@dataclass(frozen=True)
class FactorRow:
    inputs: Tensor
    target: Tensor
    valid: Tensor
    common_anchor: Tensor
    base_index: Tensor
    fold_id: Tensor


@dataclass(frozen=True)
class StageAVerdict:
    status: str
    next_route: str
    reasons: tuple[str, ...]
    stage_b_allowed: bool


@dataclass(frozen=True)
class StageBVerdict:
    status: str
    next_route: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TruthBlindGateFit:
    feature_names: tuple[str, ...]
    eta: float
    clip_norm: float
    tau: float
    lambda_h: float
    rescue_mean: Tensor
    rescue_scale: Tensor
    rescue_weight: Tensor
    harm_mean: Tensor
    harm_scale: Tensor
    harm_weight: Tensor
    oof_sample_count: int
    oof_coverage: float
    oof_weighted_utility: float
    group_overlap_count: int
    positive_receiver_cv_count: int
    receiver_cv_count: int
    audit_labels_consumed: bool


@dataclass(frozen=True)
class FactorMatrixResult:
    payload: Mapping[str, Any]
    squared_errors: Mapping[int, Mapping[str, Tensor]]
    valid_masks: Mapping[str, Tensor]


class _FactorHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, target_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(target_dim)),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


def build_fold_records(
    sidecar: CCOIPASidecar,
    x: Tensor,
    pa_map: Tensor,
    *,
    conditioned: bool,
    base_index: Tensor,
) -> FoldRecords:
    """Evaluate every non-overlapping holdout fold for each packet."""

    if x.ndim != 3 or x.size(1) != 2:
        raise ValueError("x must have shape [B,2,L]")
    if pa_map.ndim != 3 or pa_map.size(0) != x.size(0):
        raise ValueError("pa_map must have shape [B,C,L] aligned with x")
    base_index = torch.as_tensor(base_index).detach().view(-1).long().cpu()
    if base_index.numel() != x.size(0):
        raise ValueError("base_index must align with x")
    encoder = sidecar.challenge_encoder
    token_count = 1 + (int(x.size(-1)) - int(encoder.token_length)) // int(encoder.stride)
    anchors = nonoverlap_anchor_indices(token_count, encoder.token_length, encoder.stride)
    fold_count = int(anchors.numel())
    if fold_count != 4:
        raise ValueError(f"PA-M2.1 requires four non-overlap folds, got {fold_count}")

    base_rows: list[Tensor] = []
    fold_rows: list[Tensor] = []
    theta_rows: list[Tensor] = []
    q_rows: list[Tensor] = []
    target_rows: list[Tensor] = []
    support_raw_rows: list[Tensor] = []
    holdout_raw_rows: list[Tensor] = []
    with torch.no_grad():
        for fold in range(fold_count):
            output = sidecar(x, pa_map, conditioned=bool(conditioned), holdout_fold=fold)
            support_raw, holdout_raw = raw_support_holdout_masks(
                int(x.size(-1)),
                token_count,
                token_length=encoder.token_length,
                stride=encoder.stride,
                fold=fold,
            )
            base_rows.append(base_index)
            fold_rows.append(torch.full_like(base_index, fold))
            theta_rows.append(output["support_theta"].detach().cpu())
            q_rows.append(output["q_holdout"].detach().cpu())
            target_rows.append(output["heldout_target"].detach().cpu())
            support_raw_rows.append(support_raw.unsqueeze(0).expand(x.size(0), -1))
            holdout_raw_rows.append(holdout_raw.unsqueeze(0).expand(x.size(0), -1))
    return FoldRecords(
        base_index=torch.cat(base_rows, dim=0),
        fold_id=torch.cat(fold_rows, dim=0),
        theta=torch.cat(theta_rows, dim=0),
        q_holdout=torch.cat(q_rows, dim=0),
        target=torch.cat(target_rows, dim=0),
        support_raw_mask=torch.cat(support_raw_rows, dim=0),
        holdout_raw_mask=torch.cat(holdout_raw_rows, dim=0),
        fold_count=fold_count,
    )


def _stable_candidate_position(base_index: int, relation: str, seed: int, count: int) -> int:
    payload = f"{int(base_index)}|{relation}|{int(seed)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % int(count)


def build_relation_indices(
    audit_metadata: Mapping[str, Any],
    bank_metadata: Mapping[str, Any],
    relation: str,
    *,
    seed: int,
    audit_physical_features: Tensor | None = None,
    bank_physical_features: Tensor | None = None,
) -> RelationMapping:
    """Map audit anchors to a separate support bank using strict metadata relations."""

    audit = _metadata_columns(audit_metadata)
    bank = _metadata_columns(bank_metadata)
    relation = str(relation).upper()
    if relation not in {"F2", "F3", "F4", "F5", "F6", "F7"}:
        raise ValueError(f"unsupported relation {relation}")
    if relation == "F6":
        if audit_physical_features is None or bank_physical_features is None:
            raise ValueError("F6 requires fixed physical features")
        audit_features = torch.as_tensor(audit_physical_features).detach().float().cpu()
        bank_features = torch.as_tensor(bank_physical_features).detach().float().cpu()
        if audit_features.size(0) != len(audit["tx"]) or bank_features.size(0) != len(bank["tx"]):
            raise ValueError("F6 physical features must align with metadata")
    else:
        audit_features = bank_features = None

    selected: list[int] = []
    valid: list[bool] = []
    for audit_index in range(len(audit["tx"])):
        atx = audit["tx"][audit_index]
        arx = audit["receiver"][audit_index]
        aday = audit["day"][audit_index]
        abase = audit["base_index"][audit_index]
        candidates: list[int] = []
        if relation != "F7":
            for bank_index in range(len(bank["tx"])):
                btx = bank["tx"][bank_index]
                brx = bank["receiver"][bank_index]
                bday = bank["day"][bank_index]
                bbase = bank["base_index"][bank_index]
                matches = {
                    "F2": btx == atx and brx == arx and bday == aday and bbase != abase,
                    "F3": btx == atx and brx != arx and bday == aday,
                    "F4": btx == atx and brx == arx and bday != aday,
                    "F5": btx != atx and brx == arx and bday == aday,
                    "F6": btx == atx and brx != arx and bday == aday,
                }[relation]
                if matches:
                    candidates.append(bank_index)
        candidates.sort(key=lambda index: bank["base_index"][index])
        if not candidates:
            selected.append(-1)
            valid.append(False)
            continue
        if relation == "F6":
            assert audit_features is not None and bank_features is not None
            distances = (bank_features[candidates] - audit_features[audit_index]).square().sum(dim=1)
            minimum = float(distances.min().item())
            tied = [
                candidate
                for candidate, distance in zip(candidates, distances.tolist())
                if abs(float(distance) - minimum) <= 1e-12
            ]
            position = _stable_candidate_position(abase, relation, seed, len(tied))
            choice = tied[position]
        else:
            position = _stable_candidate_position(abase, relation, seed, len(candidates))
            choice = candidates[position]
        selected.append(int(choice))
        valid.append(True)
    policy = (
        "strict_metadata_then_fixed_pa_distance"
        if relation == "F6"
        else ("unavailable_no_synchronized_cross_receiver_event_id" if relation == "F7" else "strict_metadata_then_stable_seed")
    )
    return RelationMapping(
        relation=relation,
        index=tuple(selected),
        valid=tuple(valid),
        candidate_seed=int(seed),
        selection_policy=policy,
        selection_uses_learned_q=False,
        fallback_count=0,
    )


def common_anchor_mask(
    mappings: Mapping[str, RelationMapping],
    *,
    required: Sequence[str] = ("F2", "F3", "F5"),
) -> Tensor:
    missing = [name for name in required if name not in mappings]
    if missing:
        raise ValueError(f"missing relation mappings: {missing}")
    counts = {len(mappings[name].valid) for name in required}
    if len(counts) != 1:
        raise ValueError("relation mappings must align")
    count = counts.pop()
    mask = torch.ones(count, dtype=torch.bool)
    for name in required:
        mask &= torch.tensor(mappings[name].valid, dtype=torch.bool)
    return mask


def compose_factor_rows(
    audit_records: FoldRecords,
    bank_records: FoldRecords,
    mappings: Mapping[str, RelationMapping],
) -> dict[str, FactorRow]:
    """Compose F0--F9 from audit targets and a disjoint support bank."""

    required = ("F2", "F3", "F4", "F5", "F6", "F7")
    missing = [name for name in required if name not in mappings]
    if missing:
        raise ValueError(f"missing factor mappings: {missing}")
    audit_sample_count = len(mappings["F2"].index)
    if audit_records.base_index.numel() != audit_sample_count * audit_records.fold_count:
        raise ValueError("audit fold records do not align with relation mappings")
    if bank_records.base_index.numel() % bank_records.fold_count != 0:
        raise ValueError("bank fold records are incomplete")
    bank_sample_count = bank_records.base_index.numel() // bank_records.fold_count
    for name in required:
        if len(mappings[name].index) != audit_sample_count:
            raise ValueError("factor mappings must align")

    operator_dim = int(audit_records.theta.size(1))
    audit_q = audit_records.q_holdout.reshape(audit_records.q_holdout.size(0), -1)
    bank_q = bank_records.q_holdout.reshape(bank_records.q_holdout.size(0), -1)
    common_sample = common_anchor_mask(mappings)
    common_expanded = common_sample.repeat(audit_records.fold_count)
    all_valid = torch.ones(audit_records.base_index.numel(), dtype=torch.bool)
    zero_theta = torch.zeros_like(audit_records.theta)

    def row(inputs: Tensor, valid: Tensor, common: Tensor | None = None) -> FactorRow:
        return FactorRow(
            inputs=inputs,
            target=audit_records.target.reshape(audit_records.target.size(0), -1),
            valid=valid,
            common_anchor=common_expanded if common is None else common,
            base_index=audit_records.base_index,
            fold_id=audit_records.fold_id,
        )

    rows: dict[str, FactorRow] = {
        "F0": row(torch.cat((zero_theta, audit_q), dim=1), all_valid),
        "F1": row(torch.cat((audit_records.theta, audit_q), dim=1), all_valid),
    }
    mapped_theta: dict[str, Tensor] = {}
    mapped_q: dict[str, Tensor] = {}
    mapped_valid: dict[str, Tensor] = {}
    for name in required:
        theta = torch.zeros_like(audit_records.theta)
        q = torch.zeros_like(audit_q)
        valid = torch.tensor(mappings[name].valid, dtype=torch.bool).repeat(audit_records.fold_count)
        for fold in range(audit_records.fold_count):
            for audit_index, bank_index in enumerate(mappings[name].index):
                if bank_index < 0:
                    continue
                audit_row = fold * audit_sample_count + audit_index
                bank_row = fold * bank_sample_count + int(bank_index)
                theta[audit_row] = bank_records.theta[bank_row]
                q[audit_row] = bank_q[bank_row]
        mapped_theta[name] = theta
        mapped_q[name] = q
        mapped_valid[name] = valid
    for name in ("F2", "F3", "F4", "F5", "F6"):
        rows[name] = row(
            torch.cat((mapped_theta[name], audit_q), dim=1),
            mapped_valid[name],
        )
    rows["F7"] = row(
        torch.cat((mapped_theta["F7"], audit_q), dim=1),
        mapped_valid["F7"],
        torch.zeros_like(common_expanded),
    )
    rows["F8"] = row(
        torch.cat((audit_records.theta, mapped_q["F5"]), dim=1),
        mapped_valid["F5"],
    )
    rows["F9"] = row(torch.cat((zero_theta, torch.zeros_like(audit_q)), dim=1), all_valid)
    if any(factor.inputs.size(1) != operator_dim + audit_q.size(1) for factor in rows.values()):
        raise RuntimeError("factor rows do not have equal capacity")
    return rows


def fold_macro_nmse(
    squared_error: Tensor,
    target_energy: Tensor,
    fold_id: Tensor,
    valid: Tensor,
) -> dict[str, Any]:
    """Compute an equal-weight mean over the four fold-specific NMSE values."""

    squared_error = torch.as_tensor(squared_error).detach().view(-1).double().cpu()
    target_energy = torch.as_tensor(target_energy).detach().view(-1).double().cpu()
    fold_id = torch.as_tensor(fold_id).detach().view(-1).long().cpu()
    valid = torch.as_tensor(valid).detach().view(-1).bool().cpu()
    if not (
        squared_error.numel()
        == target_energy.numel()
        == fold_id.numel()
        == valid.numel()
    ):
        raise ValueError("fold metric inputs must align")
    per_fold: dict[str, float] = {}
    for fold in range(4):
        mask = valid & fold_id.eq(fold)
        if not bool(mask.any()):
            raise ValueError(f"fold {fold} contains no valid rows")
        per_fold[str(fold)] = float(
            squared_error[mask].sum().item() / max(1e-12, target_energy[mask].sum().item())
        )
    return {
        "macro_nmse": float(sum(per_fold.values()) / 4.0),
        "per_fold_nmse": per_fold,
        "valid_count": int(valid.sum().item()),
    }


def evaluate_stage_a(
    c1_metrics: Mapping[str, Any],
    c4_metrics: Mapping[str, Any],
    conditioning_comparison: Mapping[str, Any],
    coverage: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
) -> StageAVerdict:
    """Apply the preregistered transfer, coverage, conditioning and sensitivity gates."""

    del c1_metrics  # C1 values enter through the paired conditioning comparison.
    reasons: list[str] = []
    if float(c4_metrics["f3_vs_f0_relative_gain"]) < 0.05 or float(
        c4_metrics["f3_vs_f0_ci_low"]
    ) <= 0.0:
        reasons.append("F3_NOT_BETTER_THAN_F0")
    if float(c4_metrics["f3_vs_f5_relative_gain"]) < 0.05 or float(
        c4_metrics["f3_vs_f5_ci_low"]
    ) <= 0.0:
        reasons.append("F3_NOT_BETTER_THAN_F5")
    if float(c4_metrics["f3_vs_f2_relative_degradation"]) > 0.10:
        reasons.append("F3_CROSS_RX_DEGRADATION_GT_10PCT")
    if float(coverage["f3"]) < 0.80:
        reasons.append("F3_COVERAGE_LT_80PCT")
    if not bool(coverage["each_tx_two_cross_receiver_relations"]):
        reasons.append("F3_TX_CROSS_RX_RELATION_COVERAGE_FAILED")
    if not bool(coverage["major_cell_minimum_pass"]):
        reasons.append("F3_MAJOR_CELL_MINIMUM_FAILED")
    if int(sensitivity["head_seed_direction_count"]) < 2:
        reasons.append("HEAD_SEED_DIRECTION_UNSTABLE")
    if int(sensitivity["candidate_seed_direction_count"]) < 2:
        reasons.append("CANDIDATE_SEED_DIRECTION_UNSTABLE")
    if bool(sensitivity["satellite_seed_conclusion_reversal"]):
        reasons.append("SATELLITE_SEED_CONCLUSION_REVERSED")
    if reasons:
        return StageAVerdict(
            status="A_FAIL",
            next_route="STOP_CURRENT_PA_THETA_TRANSFER",
            reasons=tuple(reasons),
            stage_b_allowed=False,
        )
    conditioning_pass = (
        float(conditioning_comparison["c4_vs_c1_f3_relative_gain"]) >= 0.03
        and float(conditioning_comparison["c4_vs_c1_f3_ci_low"]) > 0.0
    )
    if not conditioning_pass:
        return StageAVerdict(
            status="A_PARTIAL",
            next_route="KEEP_PA_OPERATOR_STOP_CURRENT_CHALLENGE_CONDITIONING",
            reasons=("C4_CONDITIONING_GAIN_LT_3PCT_OR_CI_CROSSES_ZERO",),
            stage_b_allowed=False,
        )
    return StageAVerdict(
        status="A_PASS",
        next_route="RUN_M21B_TRUTH_BLIND_EXPERT_GATE",
        reasons=(),
        stage_b_allowed=True,
    )


def run_factor_matrix(
    train_rows: Mapping[str, FactorRow],
    eval_rows: Mapping[str, FactorRow],
    *,
    eval_groups: Tensor,
    head_seeds: Sequence[int] = (20260824, 20260825, 20260826),
    steps: int = 800,
    batch_size: int = 128,
    hidden_dim: int = 64,
    bootstrap_resamples: int = 1000,
    device: torch.device,
) -> FactorMatrixResult:
    """Fit equal-capacity heads for F0--F8 and an unconditional F9 mean baseline."""

    expected_rows = tuple(f"F{index}" for index in range(10))
    if set(train_rows) != set(expected_rows) or set(eval_rows) != set(expected_rows):
        raise ValueError("factor matrix requires F0--F9 for train and evaluation")
    seeds = tuple(int(seed) for seed in head_seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("factor matrix requires exactly three distinct head seeds")
    eval_count = eval_rows["F0"].target.size(0)
    groups = torch.as_tensor(eval_groups).detach().cpu()
    if groups.ndim == 1:
        groups = groups[:, None]
    if groups.size(0) != eval_count:
        raise ValueError("eval_groups must align with factor rows")
    target_reference = train_rows["F0"].target.detach().float().cpu()
    target_mean = target_reference.mean(dim=0, keepdim=True)
    target_scale = target_reference.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    squared_errors: dict[int, dict[str, Tensor]] = {}
    valid_masks = {name: eval_rows[name].valid.detach().bool().cpu() for name in expected_rows}
    per_seed_payload: dict[str, Any] = {}
    comparison_rows: dict[int, dict[str, Any]] = {}
    row_order = {name: index for index, name in enumerate(expected_rows)}

    for seed in seeds:
        seed_errors: dict[str, Tensor] = {}
        seed_rows: dict[str, Any] = {}
        for name in expected_rows:
            train_row = train_rows[name]
            eval_row = eval_rows[name]
            train_valid = train_row.valid.detach().bool().cpu()
            eval_valid = eval_row.valid.detach().bool().cpu()
            if name == "F7" or not bool(eval_valid.any()):
                seed_rows[name] = {"status": "UNAVAILABLE", "valid_count": int(eval_valid.sum().item())}
                continue
            if name == "F9":
                prediction = target_mean.expand(eval_row.target.size(0), -1).clone()
            else:
                valid_train_indices = torch.nonzero(train_valid, as_tuple=False).flatten()
                if valid_train_indices.numel() < 2:
                    raise ValueError(f"factor row {name} has fewer than two training rows")
                torch.manual_seed(seed + 1009 * row_order[name])
                head = _FactorHead(
                    train_row.inputs.size(1),
                    int(hidden_dim),
                    train_row.target.size(1),
                ).to(device)
                optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
                generator = torch.Generator().manual_seed(seed + 7001 * (row_order[name] + 1))
                train_inputs = train_row.inputs.detach().float().cpu()
                normalized_target = (train_row.target.detach().float().cpu() - target_mean) / target_scale
                head.train()
                for _ in range(max(1, int(steps))):
                    positions = torch.randint(
                        valid_train_indices.numel(),
                        (min(int(batch_size), int(valid_train_indices.numel())),),
                        generator=generator,
                    )
                    selected = valid_train_indices[positions]
                    output = head(train_inputs[selected].to(device))
                    loss = F.mse_loss(output, normalized_target[selected].to(device))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                head.eval()
                with torch.no_grad():
                    normalized_prediction = head(eval_row.inputs.detach().float().to(device)).cpu()
                prediction = normalized_prediction * target_scale + target_mean
            target = eval_row.target.detach().float().cpu()
            error = (prediction - target).square().sum(dim=1)
            energy = target.square().sum(dim=1)
            seed_errors[name] = error
            all_metrics = fold_macro_nmse(error, energy, eval_row.fold_id, eval_valid)
            common_valid = eval_valid & eval_row.common_anchor.detach().bool().cpu()
            common_metrics = (
                fold_macro_nmse(error, energy, eval_row.fold_id, common_valid)
                if bool(common_valid.any())
                else {"status": "UNAVAILABLE_EMPTY", "valid_count": 0}
            )
            seed_rows[name] = {
                "status": "COMPLETE",
                "all_valid": all_metrics,
                "common_anchor": common_metrics,
            }
        comparisons: dict[str, Any] = {}
        for label, reference, candidate in (
            ("f3_vs_f0", "F0", "F3"),
            ("f3_vs_f5", "F5", "F3"),
            ("f3_vs_f2", "F2", "F3"),
        ):
            valid = (
                eval_rows[reference].valid.detach().bool().cpu()
                & eval_rows[candidate].valid.detach().bool().cpu()
                & eval_rows["F3"].common_anchor.detach().bool().cpu()
            )
            if bool(valid.any()):
                comparison = group_paired_bootstrap(
                    seed_errors[reference][valid],
                    seed_errors[candidate][valid],
                    groups[valid],
                    resamples=int(bootstrap_resamples),
                    seed=seed + row_order[candidate] * 31,
                )
                comparison["status"] = "COMPLETE"
            else:
                comparison = {
                    "status": "UNAVAILABLE_EMPTY_COMMON_ANCHOR",
                    "sample_count": 0,
                    "relative_gain": -1.0,
                    "ci95_low": -1.0,
                    "ci95_high": -1.0,
                }
            comparisons[label] = comparison
        squared_errors[seed] = seed_errors
        comparison_rows[seed] = comparisons
        per_seed_payload[str(seed)] = {"rows": seed_rows, "comparisons": comparisons}

    row_summary: dict[str, Any] = {}
    for name in expected_rows:
        complete = [
            per_seed_payload[str(seed)]["rows"][name]
            for seed in seeds
            if per_seed_payload[str(seed)]["rows"][name]["status"] == "COMPLETE"
        ]
        if not complete:
            row_summary[name] = {"status": "UNAVAILABLE"}
            continue
        common_complete = [
            item for item in complete if item["common_anchor"].get("status") != "UNAVAILABLE_EMPTY"
        ]
        row_summary[name] = {
            "status": "COMPLETE" if common_complete else "UNAVAILABLE_EMPTY_COMMON_ANCHOR",
            "macro_nmse_mean": (
                float(sum(item["common_anchor"]["macro_nmse"] for item in common_complete) / len(common_complete))
                if common_complete
                else None
            ),
            "head_seed_count": len(complete),
        }
    f30 = [comparison_rows[seed]["f3_vs_f0"] for seed in seeds]
    f35 = [comparison_rows[seed]["f3_vs_f5"] for seed in seeds]
    f32 = [comparison_rows[seed]["f3_vs_f2"] for seed in seeds]
    summary = {
        "f3_vs_f0_relative_gain": float(sum(row["relative_gain"] for row in f30) / len(f30)),
        "f3_vs_f0_ci_low": float(min(row["ci95_low"] for row in f30)),
        "f3_vs_f5_relative_gain": float(sum(row["relative_gain"] for row in f35) / len(f35)),
        "f3_vs_f5_ci_low": float(min(row["ci95_low"] for row in f35)),
        "f3_vs_f2_relative_degradation": float(
            -sum(row["relative_gain"] for row in f32) / len(f32)
        ),
        "f3_common_nmse": row_summary["F3"]["macro_nmse_mean"],
        "common_anchor_status": (
            "COMPLETE" if row_summary["F3"]["macro_nmse_mean"] is not None else "UNAVAILABLE_EMPTY"
        ),
        "head_seed_direction_count": int(
            sum(
                row30["relative_gain"] > 0.0 and row35["relative_gain"] > 0.0
                for row30, row35 in zip(f30, f35)
            )
        ),
    }
    payload = {
        "status": "COMPLETE",
        "head_seeds": list(seeds),
        "same_capacity_heads": True,
        "fold_macro_policy": "equal_weight_mean_of_four_fold_nmse",
        "common_anchor_policy": "F2_intersection_F3_intersection_F5",
        "rows": row_summary,
        "per_seed": per_seed_payload,
        "summary": summary,
        "sample_level_state_persisted": False,
    }
    return FactorMatrixResult(payload=payload, squared_errors=squared_errors, valid_masks=valid_masks)


def m0_exact_pair_retrieval(
    *,
    clean_q: Tensor,
    satellite_q: Tensor,
    clean_theta: Tensor,
    satellite_theta: Tensor,
    base_index: Tensor,
    fold_id: Tensor,
    sample_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate exact clean/satellite retrieval inside nuisance-matched pools."""

    clean = F.normalize(torch.as_tensor(clean_q).detach().float().cpu().reshape(len(base_index), -1), dim=1)
    satellite = F.normalize(
        torch.as_tensor(satellite_q).detach().float().cpu().reshape(len(base_index), -1), dim=1
    )
    clean_theta = torch.as_tensor(clean_theta).detach().float().cpu()
    satellite_theta = torch.as_tensor(satellite_theta).detach().float().cpu()
    base_index = torch.as_tensor(base_index).detach().view(-1).long().cpu()
    fold_id = torch.as_tensor(fold_id).detach().view(-1).long().cpu()
    if not (
        clean.size(0)
        == satellite.size(0)
        == clean_theta.size(0)
        == satellite_theta.size(0)
        == base_index.numel()
        == fold_id.numel()
    ):
        raise ValueError("M0 inputs must align")
    metadata = _metadata_columns(sample_metadata)
    sample_by_base = {value: index for index, value in enumerate(metadata["base_index"])}
    ranks: list[int] = []
    reciprocal: list[float] = []
    pair_wins: list[float] = []
    theta_distances: list[float] = []
    margins: list[float] = []
    for row in range(base_index.numel()):
        sample = sample_by_base.get(int(base_index[row]))
        if sample is None:
            raise ValueError("M0 base_index is absent from sample metadata")
        candidates = []
        for candidate in range(base_index.numel()):
            candidate_sample = sample_by_base.get(int(base_index[candidate]))
            if candidate_sample is None or int(fold_id[candidate]) != int(fold_id[row]):
                continue
            if all(
                metadata[name][candidate_sample] == metadata[name][sample]
                for name in ("tx", "receiver", "day")
            ):
                candidates.append(candidate)
        exact = [candidate for candidate in candidates if int(base_index[candidate]) == int(base_index[row])]
        if len(exact) != 1:
            raise ValueError("M0 candidate pool must contain exactly one exact physical pair")
        similarities = satellite[candidates] @ clean[row]
        ordered = sorted(
            range(len(candidates)),
            key=lambda position: (-float(similarities[position]), int(base_index[candidates[position]])),
        )
        exact_position = candidates.index(exact[0])
        rank = ordered.index(exact_position) + 1
        ranks.append(rank)
        reciprocal.append(1.0 / rank)
        exact_similarity = float(similarities[exact_position].item())
        other = [float(similarities[position].item()) for position in range(len(candidates)) if position != exact_position]
        pair_wins.extend(1.0 if exact_similarity > value else (0.5 if exact_similarity == value else 0.0) for value in other)
        if other:
            margins.append(exact_similarity - max(other))
        theta_distances.append(
            float((clean_theta[row] - satellite_theta[exact[0]]).square().sum().sqrt().item())
        )
    rank_tensor = torch.tensor(ranks, dtype=torch.float32)
    return {
        "status": "COMPLETE",
        "candidate_pool_policy": "same_tx_same_rx_same_day_same_fold",
        "sample_count": len(ranks),
        "recall_at_1": float((rank_tensor <= 1).float().mean().item()),
        "recall_at_5": float((rank_tensor <= 5).float().mean().item()),
        "median_rank": float(rank_tensor.median().item()),
        "mrr": float(sum(reciprocal) / len(reciprocal)),
        "exact_pair_distance_auc": float(sum(pair_wins) / len(pair_wins)) if pair_wins else 1.0,
        "clean_satellite_theta_distance_mean": float(sum(theta_distances) / len(theta_distances)),
        "exact_pair_margin_mean": float(sum(margins) / len(margins)) if margins else 0.0,
        "sample_level_state_persisted": False,
    }


def _fit_regression_head(
    inputs: Tensor,
    target: Tensor,
    train_mask: Tensor,
    predict_mask: Tensor,
    *,
    seed: int,
    steps: int,
    hidden_dim: int,
    device: torch.device,
) -> Tensor:
    indices = torch.nonzero(train_mask, as_tuple=False).flatten()
    if indices.numel() < 2:
        raise ValueError("regression fold requires at least two training rows")
    torch.manual_seed(int(seed))
    head = _FactorHead(inputs.size(1), int(hidden_dim), target.size(1)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(int(seed) + 7001)
    for _ in range(max(1, int(steps))):
        selected = indices[
            torch.randint(indices.numel(), (min(64, int(indices.numel())),), generator=generator)
        ]
        output = head(inputs[selected].to(device))
        loss = F.mse_loss(output, target[selected].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    head.eval()
    with torch.no_grad():
        return head(inputs[predict_mask].to(device)).cpu()


def _centroid_probe_accuracy(features: Tensor, labels: Tensor) -> float:
    labels = labels.view(-1).long().cpu()
    features = F.normalize(features.float().cpu(), dim=1)
    unique = torch.unique(labels)
    centroids = torch.stack([features[labels.eq(label)].mean(dim=0) for label in unique])
    prediction = unique[(features @ F.normalize(centroids, dim=1).T).argmax(dim=1)]
    return float((prediction == labels).float().mean().item())


def _residual_centroid_distances(features: Tensor, tx: Tensor, receiver: Tensor) -> dict[str, Any]:
    features = torch.as_tensor(features).detach().float().cpu()
    tx = torch.as_tensor(tx).detach().view(-1).long().cpu()
    receiver = torch.as_tensor(receiver).detach().view(-1).long().cpu()
    tx_centroids = {
        int(label): features[tx.eq(label)].mean(dim=0)
        for label in torch.unique(tx)
    }
    between = [
        float((tx_centroids[left] - tx_centroids[right]).norm().item())
        for position, left in enumerate(sorted(tx_centroids))
        for right in sorted(tx_centroids)[position + 1 :]
    ]
    cell_centroids = {
        (int(tx_value), int(rx_value)): features[tx.eq(tx_value) & receiver.eq(rx_value)].mean(dim=0)
        for tx_value in torch.unique(tx)
        for rx_value in torch.unique(receiver[tx.eq(tx_value)])
    }
    same_cross_rx = []
    for tx_value in sorted(set(key[0] for key in cell_centroids)):
        receivers = sorted(key[1] for key in cell_centroids if key[0] == tx_value)
        same_cross_rx.extend(
            float((cell_centroids[(tx_value, left)] - cell_centroids[(tx_value, right)]).norm().item())
            for position, left in enumerate(receivers)
            for right in receivers[position + 1 :]
        )
    return {
        "between_tx_mean": float(sum(between) / len(between)) if between else None,
        "same_tx_cross_receiver_mean": float(sum(same_cross_rx) / len(same_cross_rx)) if same_cross_rx else None,
        "between_tx_pair_count": len(between),
        "same_tx_cross_receiver_pair_count": len(same_cross_rx),
    }


def run_loto_residual(
    *,
    common_inputs: Tensor,
    operator_inputs: Tensor,
    target: Tensor,
    tx: Tensor,
    receiver: Tensor,
    day: Tensor,
    fold_id: Tensor,
    seed: int,
    steps: int,
    hidden_dim: int,
    device: torch.device,
) -> dict[str, Any]:
    """Cross-fit common and residual predictors with whole transmitters held out."""

    common_inputs = torch.as_tensor(common_inputs).detach().float().cpu()
    operator_inputs = torch.as_tensor(operator_inputs).detach().float().cpu()
    target = torch.as_tensor(target).detach().float().cpu()
    tx = torch.as_tensor(tx).detach().view(-1).long().cpu()
    receiver = torch.as_tensor(receiver).detach().view(-1).long().cpu()
    day = torch.as_tensor(day).detach().view(-1).long().cpu()
    fold_id = torch.as_tensor(fold_id).detach().view(-1).long().cpu()
    count = target.size(0)
    if any(value.size(0) != count for value in (common_inputs, operator_inputs, tx, receiver, day, fold_id)):
        raise ValueError("LOTO inputs must align")
    unique_tx = torch.unique(tx).tolist()
    if len(unique_tx) < 2:
        raise ValueError("LOTO requires at least two transmitters")
    common_prediction = torch.empty_like(target)
    folds: list[dict[str, Any]] = []
    for position, held_out in enumerate(unique_tx):
        train_mask = tx.ne(int(held_out))
        predict_mask = tx.eq(int(held_out))
        common_prediction[predict_mask] = _fit_regression_head(
            common_inputs,
            target,
            train_mask,
            predict_mask,
            seed=int(seed) + 100 * position,
            steps=steps,
            hidden_dim=hidden_dim,
            device=device,
        )
        train_txs = sorted(int(value) for value in torch.unique(tx[train_mask]).tolist())
        folds.append(
            {
                "held_out_tx": int(held_out),
                "common_train_txs": train_txs,
                "residual_train_txs": train_txs,
                "eval_count": int(predict_mask.sum().item()),
            }
        )
    residual_target = target - common_prediction
    residual_prediction = torch.empty_like(target)
    for position, held_out in enumerate(unique_tx):
        train_mask = tx.ne(int(held_out))
        predict_mask = tx.eq(int(held_out))
        residual_prediction[predict_mask] = _fit_regression_head(
            operator_inputs,
            residual_target,
            train_mask,
            predict_mask,
            seed=int(seed) + 1000 + 100 * position,
            steps=steps,
            hidden_dim=hidden_dim,
            device=device,
        )
    final_prediction = common_prediction + residual_prediction
    squared_error = (final_prediction - target).square().sum(dim=1)
    energy = target.square().sum(dim=1)
    metrics = fold_macro_nmse(squared_error, energy, fold_id, torch.ones(count, dtype=torch.bool))
    return {
        "status": "COMPLETE",
        "cross_fit_policy": "leave_one_tx_out_for_common_and_residual_heads",
        "folds": folds,
        "metrics": metrics,
        "residual_probe_scope": {
            "tx": _centroid_probe_accuracy(residual_target, tx),
            "receiver": _centroid_probe_accuracy(residual_target, receiver),
            "day": _centroid_probe_accuracy(residual_target, day),
        },
        "residual_distance_scope": _residual_centroid_distances(
            residual_target, tx, receiver
        ),
        "non_finite_count": int((~torch.isfinite(final_prediction)).sum().item()),
        "sample_level_state_persisted": False,
    }


def _balanced_accuracy(prediction: Tensor, truth: Tensor) -> float:
    truth = truth.view(-1).long().cpu()
    prediction = prediction.view(-1).long().cpu()
    recalls = [
        float((prediction[truth.eq(label)] == label).float().mean().item())
        for label in torch.unique(truth)
    ]
    return float(sum(recalls) / len(recalls))


def _classification_probe(
    train_x: Tensor,
    train_y: Tensor,
    eval_x: Tensor,
    eval_y: Tensor,
    *,
    seed: int,
    steps: int,
    hidden_dim: int,
    device: torch.device,
) -> float | None:
    train_y = train_y.view(-1).long().cpu()
    eval_y = eval_y.view(-1).long().cpu()
    classes = torch.unique(train_y)
    if classes.numel() < 2 or not set(torch.unique(eval_y).tolist()).issubset(set(classes.tolist())):
        return None
    class_to_local = {int(value): index for index, value in enumerate(classes.tolist())}
    local_train = torch.tensor([class_to_local[int(value)] for value in train_y.tolist()])
    torch.manual_seed(int(seed))
    head = nn.Sequential(
        nn.Linear(train_x.size(1), int(hidden_dim)),
        nn.GELU(),
        nn.Linear(int(hidden_dim), len(classes)),
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(int(seed) + 1701)
    for _ in range(max(1, int(steps))):
        selected = torch.randint(train_x.size(0), (min(64, train_x.size(0)),), generator=generator)
        logits = head(train_x[selected].to(device))
        loss = F.cross_entropy(logits, local_train[selected].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    head.eval()
    with torch.no_grad():
        local_prediction = head(eval_x.to(device)).argmax(dim=1).cpu()
    prediction = classes[local_prediction]
    return _balanced_accuracy(prediction, eval_y)


def conditional_q_probe(
    *,
    train_q: Tensor,
    eval_q: Tensor,
    train_labels: Mapping[str, Tensor],
    eval_labels: Mapping[str, Tensor],
    seed: int,
    steps: int,
    hidden_dim: int,
    device: torch.device,
) -> dict[str, Any]:
    """Probe ordered, token-shuffled and permutation-invariant q leakage."""

    train_q = torch.as_tensor(train_q).detach().float().cpu()
    eval_q = torch.as_tensor(eval_q).detach().float().cpu()
    if train_q.ndim != 3 or eval_q.ndim != 3 or train_q.shape[1:] != eval_q.shape[1:]:
        raise ValueError("q probes require aligned [N,T,Q] tensors")
    labels = ("tx", "receiver", "day")
    for name in labels:
        if name not in train_labels or name not in eval_labels:
            raise ValueError(f"q probe labels are missing {name}")
    ordered_train = train_q.flatten(1)
    ordered_eval = eval_q.flatten(1)
    generator = torch.Generator().manual_seed(int(seed) + 41)

    def shuffled(value: Tensor) -> Tensor:
        rows = []
        for row in value:
            rows.append(row[torch.randperm(row.size(0), generator=generator)])
        return torch.stack(rows).flatten(1)

    shuffled_train = shuffled(train_q)
    shuffled_eval = shuffled(eval_q)

    def invariant(value: Tensor) -> Tensor:
        return torch.cat(
            (value.mean(dim=1), value.std(dim=1, unbiased=False), value.amax(dim=1), value.amin(dim=1)),
            dim=1,
        )

    representations = {
        "ordered_sequence": (ordered_train, ordered_eval),
        "token_shuffled_sequence": (shuffled_train, shuffled_eval),
        "permutation_invariant": (invariant(train_q), invariant(eval_q)),
    }
    result: dict[str, Any] = {}
    for representation, (train_x, eval_x) in representations.items():
        metrics: dict[str, Any] = {}
        for offset, name in enumerate(labels):
            accuracy = _classification_probe(
                train_x,
                torch.as_tensor(train_labels[name]),
                eval_x,
                torch.as_tensor(eval_labels[name]),
                seed=int(seed) + offset * 101,
                steps=steps,
                hidden_dim=hidden_dim,
                device=device,
            )
            metrics[f"{name}_balanced_accuracy"] = accuracy
        result[representation] = metrics
    ordered_tx = result["ordered_sequence"]["tx_balanced_accuracy"]
    shuffled_tx = result["token_shuffled_sequence"]["tx_balanced_accuracy"]
    result["ordered_minus_shuffled_tx_accuracy"] = (
        float(ordered_tx - shuffled_tx) if ordered_tx is not None and shuffled_tx is not None else None
    )
    conditional_specs = {
        "tx_within_fixed_receiver_day": ("tx", ("receiver", "day")),
        "receiver_within_fixed_tx_day": ("receiver", ("tx", "day")),
        "day_within_fixed_tx_receiver": ("day", ("tx", "receiver")),
    }
    conditional: dict[str, Any] = {}
    for output_name, (target_name, fixed_names) in conditional_specs.items():
        train_fixed = torch.stack([torch.as_tensor(train_labels[name]).view(-1) for name in fixed_names], dim=1)
        eval_fixed = torch.stack([torch.as_tensor(eval_labels[name]).view(-1) for name in fixed_names], dim=1)
        scores = []
        for group in torch.unique(eval_fixed, dim=0):
            train_mask = (train_fixed == group).all(dim=1)
            eval_mask = (eval_fixed == group).all(dim=1)
            if int(train_mask.sum()) < 2 or int(eval_mask.sum()) < 1:
                continue
            score = _classification_probe(
                ordered_train[train_mask],
                torch.as_tensor(train_labels[target_name])[train_mask],
                ordered_eval[eval_mask],
                torch.as_tensor(eval_labels[target_name])[eval_mask],
                seed=int(seed) + len(scores) * 211,
                steps=steps,
                hidden_dim=hidden_dim,
                device=device,
            )
            if score is not None:
                scores.append(score)
        conditional[output_name] = {
            "balanced_accuracy_macro": float(sum(scores) / len(scores)) if scores else None,
            "valid_group_count": len(scores),
        }
    result["conditional"] = conditional
    result["sample_level_state_persisted"] = False
    return result


def build_gate_feature_matrix(features: Mapping[str, Tensor]) -> tuple[Tensor, tuple[str, ...]]:
    """Create a gate matrix from the explicit deployment-time feature allowlist."""

    observed = set(features)
    forbidden = sorted(observed.difference(GATE_FEATURE_ALLOWLIST))
    if forbidden:
        raise ValueError(f"forbidden gate features: {forbidden}")
    missing = sorted(GATE_FEATURE_ALLOWLIST.difference(observed))
    if missing:
        raise ValueError(f"missing gate features: {missing}")
    names = tuple(sorted(GATE_FEATURE_ALLOWLIST))
    columns = [torch.as_tensor(features[name]).detach().float().cpu().view(-1) for name in names]
    counts = {column.numel() for column in columns}
    if len(counts) != 1 or not counts or next(iter(counts)) == 0:
        raise ValueError("gate feature columns must have equal non-zero length")
    matrix = torch.stack(columns, dim=1)
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("gate features contain non-finite values")
    return matrix, names


def _gate_polynomial_features(matrix: Tensor) -> Tensor:
    value = torch.as_tensor(matrix).detach().float().cpu()
    if value.ndim != 2:
        raise ValueError("gate matrix must be two-dimensional")
    return torch.cat((value, value.square()), dim=1)


def _fit_binary_logistic(
    matrix: Tensor,
    labels: Tensor,
    *,
    steps: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    x = _gate_polynomial_features(matrix)
    y = torch.as_tensor(labels).detach().float().cpu().view(-1)
    if x.size(0) != y.numel() or x.size(0) == 0:
        raise ValueError("gate labels must match a non-empty feature matrix")
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (x - mean) / scale
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(int(seed))
    try:
        model = nn.Linear(normalized.size(1), 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-3)
        positive = float(y.sum().item())
        negative = float(y.numel() - positive)
        positive_weight = max(negative / max(positive, 1.0), 1e-3)
        for _ in range(max(1, int(steps))):
            logits = model(normalized).view(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                y,
                pos_weight=logits.new_tensor(positive_weight),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        weight = torch.cat((model.weight.detach().view(-1), model.bias.detach().view(1)))
    finally:
        torch.random.set_rng_state(generator_state)
    return mean, scale, weight


def _predict_binary_logistic(matrix: Tensor, mean: Tensor, scale: Tensor, weight: Tensor) -> Tensor:
    x = _gate_polynomial_features(matrix)
    normalized = (x - mean) / scale
    return torch.sigmoid(normalized @ weight[:-1] + weight[-1])


def _gate_group_folds(groups: Sequence[Any], folds: int) -> tuple[Tensor, int]:
    if int(folds) < 2:
        raise ValueError("gate group CV requires at least two folds")
    stable_groups = [repr(group) for group in groups]
    if not stable_groups:
        raise ValueError("gate groups cannot be empty")
    unique = sorted(
        set(stable_groups),
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    if len(unique) < int(folds):
        raise ValueError("gate group count is smaller than fold count")
    fold_by_group = {group: index % int(folds) for index, group in enumerate(unique)}
    assignments = torch.tensor([fold_by_group[group] for group in stable_groups], dtype=torch.long)
    overlap = 0
    for fold in range(int(folds)):
        train_groups = {stable_groups[index] for index in torch.nonzero(assignments != fold).view(-1).tolist()}
        eval_groups = {stable_groups[index] for index in torch.nonzero(assignments == fold).view(-1).tolist()}
        overlap += len(train_groups.intersection(eval_groups))
    return assignments, overlap


def _cross_fitted_gate_probabilities(
    matrix: Tensor,
    rescue: Tensor,
    harm: Tensor,
    assignments: Tensor,
    *,
    steps: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    rescue_probability = torch.zeros(matrix.size(0), dtype=torch.float32)
    harm_probability = torch.zeros_like(rescue_probability)
    for fold in sorted(set(assignments.tolist())):
        train = assignments != int(fold)
        evaluate = assignments == int(fold)
        r_mean, r_scale, r_weight = _fit_binary_logistic(
            matrix[train], rescue[train], steps=steps, seed=int(seed) + 37 * int(fold)
        )
        h_mean, h_scale, h_weight = _fit_binary_logistic(
            matrix[train], harm[train], steps=steps, seed=int(seed) + 37 * int(fold) + 17
        )
        rescue_probability[evaluate] = _predict_binary_logistic(
            matrix[evaluate], r_mean, r_scale, r_weight
        )
        harm_probability[evaluate] = _predict_binary_logistic(
            matrix[evaluate], h_mean, h_scale, h_weight
        )
    return rescue_probability, harm_probability


def fit_truth_blind_gate(
    features: Mapping[str, Tensor],
    *,
    outcomes: Mapping[tuple[float, float], Mapping[str, Tensor]],
    groups: Sequence[Any],
    receivers: Tensor,
    tau_candidates: Sequence[float] = (-0.10, 0.0, 0.05, 0.10),
    lambda_h_candidates: Sequence[float] = (1.5, 2.0, 3.0),
    folds: int = 5,
    steps: int = 200,
    seed: int = 0,
) -> TruthBlindGateFit:
    """Fit and freeze a low-capacity rescue/harm gate using V_cal labels only."""

    matrix, names = build_gate_feature_matrix(features)
    count = matrix.size(0)
    if len(groups) != count:
        raise ValueError("gate group count must match feature rows")
    receiver = torch.as_tensor(receivers).detach().long().cpu().view(-1)
    if receiver.numel() != count:
        raise ValueError("receiver count must match feature rows")
    assignments, overlap = _gate_group_folds(groups, int(folds))
    if not outcomes:
        raise ValueError("gate calibration requires at least one frozen fusion candidate")

    candidate_cache: dict[tuple[float, float], tuple[Tensor, Tensor, Tensor, Tensor]] = {}
    best: tuple[tuple[float, float, float, float], tuple[float, ...]] | None = None
    for candidate, labels in sorted(outcomes.items()):
        eta, clip_norm = (float(candidate[0]), float(candidate[1]))
        if eta not in {0.05, 0.10, 0.20} or clip_norm <= 0.0:
            raise ValueError("invalid frozen fusion candidate")
        if set(labels) != {"rescue", "harm"}:
            raise ValueError("each fusion candidate requires rescue and harm labels")
        rescue = torch.as_tensor(labels["rescue"]).detach().bool().cpu().view(-1)
        harm = torch.as_tensor(labels["harm"]).detach().bool().cpu().view(-1)
        if rescue.numel() != count or harm.numel() != count or bool((rescue & harm).any()):
            raise ValueError("rescue/harm labels must be disjoint and match V_cal rows")
        p_rescue, p_harm = _cross_fitted_gate_probabilities(
            matrix,
            rescue,
            harm,
            assignments,
            steps=steps,
            seed=int(seed) + int(round(eta * 1000.0)) + int(round(clip_norm * 100.0)),
        )
        candidate_cache[(eta, clip_norm)] = (rescue, harm, p_rescue, p_harm)
        for lambda_h in lambda_h_candidates:
            if float(lambda_h) <= 1.0:
                raise ValueError("lambda_h candidates must be greater than one")
            for tau in tau_candidates:
                selected = (p_rescue - float(lambda_h) * p_harm) > float(tau)
                weighted = rescue.float() - float(lambda_h) * harm.float()
                utility = float((selected.float() * weighted).sum().item())
                coverage = float(selected.float().mean().item())
                ranking = (utility, coverage, eta, -float(lambda_h), -abs(float(tau)))
                if best is None or ranking > best[1]:
                    best = ((eta, clip_norm, float(tau), float(lambda_h)), ranking)
    assert best is not None
    eta, clip_norm, tau, lambda_h = best[0]
    rescue, harm, p_rescue, p_harm = candidate_cache[(eta, clip_norm)]
    selected = (p_rescue - lambda_h * p_harm) > tau
    weighted = rescue.float() - lambda_h * harm.float()

    receiver_utilities: list[float] = []
    for rx in sorted(set(receiver.tolist())):
        train = receiver != int(rx)
        evaluate = receiver == int(rx)
        if not bool(train.any()) or not bool(evaluate.any()):
            continue
        r_mean, r_scale, r_weight = _fit_binary_logistic(
            matrix[train], rescue[train], steps=steps, seed=int(seed) + 1009 + int(rx)
        )
        h_mean, h_scale, h_weight = _fit_binary_logistic(
            matrix[train], harm[train], steps=steps, seed=int(seed) + 2003 + int(rx)
        )
        rx_selected = (
            _predict_binary_logistic(matrix[evaluate], r_mean, r_scale, r_weight)
            - lambda_h * _predict_binary_logistic(matrix[evaluate], h_mean, h_scale, h_weight)
        ) > tau
        receiver_utilities.append(float((rx_selected.float() * weighted[evaluate]).sum().item()))

    r_mean, r_scale, r_weight = _fit_binary_logistic(
        matrix, rescue, steps=steps, seed=int(seed) + 3001
    )
    h_mean, h_scale, h_weight = _fit_binary_logistic(
        matrix, harm, steps=steps, seed=int(seed) + 4001
    )
    return TruthBlindGateFit(
        feature_names=names,
        eta=eta,
        clip_norm=clip_norm,
        tau=tau,
        lambda_h=lambda_h,
        rescue_mean=r_mean,
        rescue_scale=r_scale,
        rescue_weight=r_weight,
        harm_mean=h_mean,
        harm_scale=h_scale,
        harm_weight=h_weight,
        oof_sample_count=count,
        oof_coverage=float(selected.float().mean().item()),
        oof_weighted_utility=float((selected.float() * weighted).sum().item()),
        group_overlap_count=int(overlap),
        positive_receiver_cv_count=sum(value > 0.0 for value in receiver_utilities),
        receiver_cv_count=len(receiver_utilities),
        audit_labels_consumed=False,
    )


def predict_truth_blind_gate(fitted: TruthBlindGateFit, features: Mapping[str, Tensor]) -> Tensor:
    """Apply the frozen V_cal gate without labels or role metadata."""

    matrix, names = build_gate_feature_matrix(features)
    if names != fitted.feature_names:
        raise ValueError("gate feature order does not match the frozen model")
    p_rescue = _predict_binary_logistic(
        matrix, fitted.rescue_mean, fitted.rescue_scale, fitted.rescue_weight
    )
    p_harm = _predict_binary_logistic(
        matrix, fitted.harm_mean, fitted.harm_scale, fitted.harm_weight
    )
    return ((p_rescue - fitted.lambda_h * p_harm) > fitted.tau).float()


def bounded_residual_fusion(
    base_logits: Tensor,
    operator_logits: Tensor,
    *,
    gate: Tensor,
    eta: float,
    scale: float,
    clip_norm: float,
) -> Tensor:
    """Apply a gate-weighted residual correction with a per-sample L2 cap."""

    base = torch.as_tensor(base_logits)
    operator = torch.as_tensor(operator_logits).to(device=base.device, dtype=base.dtype)
    if base.ndim != 2 or operator.shape != base.shape:
        raise ValueError("base and operator logits must share [N,C] geometry")
    if float(eta) not in {0.05, 0.10, 0.20}:
        raise ValueError("eta must be one of 0.05, 0.10, 0.20")
    if float(clip_norm) <= 0.0 or float(scale) <= 0.0:
        raise ValueError("scale and clip_norm must be positive")
    gate = torch.as_tensor(gate, device=base.device, dtype=base.dtype).view(-1)
    if gate.numel() != base.size(0) or bool(((gate < 0.0) | (gate > 1.0)).any()):
        raise ValueError("gate must contain one value in [0,1] per sample")
    raw = float(scale) * operator - base
    norm = raw.norm(dim=1, keepdim=True).clamp_min(1e-12)
    clipped = raw * torch.clamp(raw.new_tensor(float(clip_norm)) / norm, max=1.0)
    return base + gate.unsqueeze(1) * float(eta) * clipped


def evaluate_stage_b(metrics: Mapping[str, Any], *, stage_a_status: str) -> StageBVerdict:
    """Apply the preregistered safety criteria or emit a clean A-gated skip."""

    if str(stage_a_status) != "A_PASS":
        return StageBVerdict(
            status="NOT_RUN_A_GATE",
            next_route="STAGE_B_NOT_AUTHORIZED_BY_STAGE_A",
            reasons=(f"STAGE_A_STATUS_{stage_a_status}",),
        )
    required = {
        "leo_mean_gain_pp",
        "leo_gain_ci_low_pp",
        "clean_gain_pp",
        "worst_receiver_gain_pp",
        "selected_weighted_utility",
        "gate_coverage",
        "gate_coverage_min",
        "positive_receiver_cv_count",
        "receiver_cv_count",
    }
    missing = sorted(required.difference(metrics))
    if missing:
        raise ValueError(f"stage B metrics missing: {missing}")
    reasons: list[str] = []
    if float(metrics["leo_mean_gain_pp"]) < 0.20:
        reasons.append("LEO_MEAN_GAIN_LT_0_20PP")
    if float(metrics["leo_gain_ci_low_pp"]) <= 0.0:
        reasons.append("LEO_GAIN_CI_CROSSES_ZERO")
    if float(metrics["clean_gain_pp"]) < -0.10:
        reasons.append("CLEAN_DROP_GT_0_10PP")
    if float(metrics["worst_receiver_gain_pp"]) < -0.05:
        reasons.append("WORST_RECEIVER_DROP_GT_0_05PP")
    if float(metrics["selected_weighted_utility"]) <= 0.0:
        reasons.append("SELECTED_WEIGHTED_UTILITY_NOT_POSITIVE")
    if float(metrics["gate_coverage"]) < float(metrics["gate_coverage_min"]):
        reasons.append("GATE_COVERAGE_BELOW_PREREGISTERED_MINIMUM")
    receiver_count = int(metrics["receiver_cv_count"])
    if receiver_count <= 0 or int(metrics["positive_receiver_cv_count"]) <= receiver_count // 2:
        reasons.append("LEAVE_ONE_RECEIVER_CV_MAJORITY_NOT_POSITIVE")
    if reasons:
        return StageBVerdict(
            status="B_FAIL",
            next_route="KEEP_PA_MECHANISM_OUT_OF_PHASE1_CLASSIFICATION_LOGITS",
            reasons=tuple(reasons),
        )
    return StageBVerdict(
        status="B_PASS",
        next_route="DESIGN_CONTINUOUS_CHALLENGE_V3",
        reasons=(),
    )


def _metadata_columns(metadata: Mapping[str, Any]) -> dict[str, list[int]]:
    required = ("tx", "receiver", "day", "eq", "sig_i", "base_index")
    result: dict[str, list[int]] = {}
    expected_count: int | None = None
    for name in required:
        if name not in metadata:
            raise ValueError(f"metadata is missing {name}")
        value = metadata[name]
        if isinstance(value, Tensor):
            rows = value.detach().view(-1).cpu().tolist()
        elif isinstance(value, Sequence):
            rows = list(value)
        else:
            raise ValueError(f"metadata {name} must be a tensor or sequence")
        result[name] = [int(item) for item in rows]
        if expected_count is None:
            expected_count = len(rows)
        elif len(rows) != expected_count:
            raise ValueError("metadata columns must have equal length")
    if not expected_count:
        raise ValueError("metadata must contain at least one sample")
    if len(set(result["base_index"])) != expected_count:
        raise ValueError("metadata base_index must be unique")
    return result


def _stable_bit(seed: int, cell: tuple[int, int, int, int]) -> int:
    payload = f"{int(seed)}|{'|'.join(str(value) for value in cell)}".encode("ascii")
    return hashlib.sha256(payload).digest()[0] & 1


def split_v_select_retro(
    metadata: Mapping[str, Any],
    *,
    seed: int,
    block_candidates: Sequence[int] = (10, 20, 25),
    fit_ratio: float = 0.65,
    minimum_blocks_per_cell: int = 4,
) -> RetroSplit:
    """Split V_select by contiguous capture blocks with one guard block per cell."""

    columns = _metadata_columns(metadata)
    if not 0.5 <= float(fit_ratio) <= 0.8:
        raise ValueError("fit_ratio must be between 0.5 and 0.8")
    candidates = tuple(sorted({int(value) for value in block_candidates if int(value) > 0}))
    if not candidates:
        raise ValueError("block_candidates must contain a positive size")

    selected_size: int | None = None
    selected_groups: dict[tuple[int, int, int, int], set[int]] | None = None
    for block_size in candidates:
        groups: dict[tuple[int, int, int, int], set[int]] = defaultdict(set)
        for index in range(len(columns["tx"])):
            cell = (
                columns["tx"][index],
                columns["receiver"][index],
                columns["day"][index],
                columns["eq"][index],
            )
            groups[cell].add(columns["sig_i"][index] // block_size)
        if groups and min(len(blocks) for blocks in groups.values()) >= int(minimum_blocks_per_cell):
            selected_size = block_size
            selected_groups = groups
            break
    if selected_size is None or selected_groups is None:
        raise ValueError("no capture block size preserves the minimum blocks per cell")

    role_by_group: dict[tuple[int, int, int, int, int], str] = {}
    for cell, block_set in sorted(selected_groups.items()):
        blocks = sorted(block_set)
        usable_count = len(blocks) - 1
        fit_count = max(1, min(usable_count - 1, int(round(usable_count * float(fit_ratio)))))
        audit_count = usable_count - fit_count
        if _stable_bit(seed, cell) == 0:
            fit_blocks = blocks[:fit_count]
            guard_block = blocks[fit_count]
            audit_blocks = blocks[fit_count + 1 :]
        else:
            audit_blocks = blocks[:audit_count]
            guard_block = blocks[audit_count]
            fit_blocks = blocks[audit_count + 1 :]
        for block in fit_blocks:
            role_by_group[(*cell, block)] = "fit"
        role_by_group[(*cell, guard_block)] = "guard"
        for block in audit_blocks:
            role_by_group[(*cell, block)] = "audit"

    role_indices: dict[str, list[int]] = {"fit": [], "audit": [], "guard": []}
    for index in range(len(columns["tx"])):
        group = (
            columns["tx"][index],
            columns["receiver"][index],
            columns["day"][index],
            columns["eq"][index],
            columns["sig_i"][index] // selected_size,
        )
        role_indices[role_by_group[group]].append(index)
    if not role_indices["fit"] or not role_indices["audit"]:
        raise ValueError("retro split produced an empty fit or audit role")
    fit_base = {columns["base_index"][index] for index in role_indices["fit"]}
    audit_base = {columns["base_index"][index] for index in role_indices["audit"]}
    return RetroSplit(
        block_size=selected_size,
        fit_ratio=float(fit_ratio),
        fit_indices=tuple(role_indices["fit"]),
        audit_indices=tuple(role_indices["audit"]),
        guard_indices=tuple(role_indices["guard"]),
        role_by_group=role_by_group,
        base_index_overlap_count=len(fit_base.intersection(audit_base)),
        cell_count=len(selected_groups),
        min_blocks_per_cell=min(len(blocks) for blocks in selected_groups.values()),
    )


def _quantiles(values: Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {name: float("nan") for name in ("q50", "q90", "q95", "q99", "max")}
    values = values.float().view(-1)
    return {
        "q50": float(torch.quantile(values, 0.50).item()),
        "q90": float(torch.quantile(values, 0.90).item()),
        "q95": float(torch.quantile(values, 0.95).item()),
        "q99": float(torch.quantile(values, 0.99).item()),
        "max": float(values.max().item()),
    }


def duplicate_audit(
    iq: Tensor,
    metadata: Mapping[str, Any],
    split: RetroSplit,
    *,
    projection_dim: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """Return aggregate exact and near-duplicate evidence without sample-level state."""

    columns = _metadata_columns(metadata)
    packets = torch.as_tensor(iq).detach().float().cpu()
    if packets.size(0) != len(columns["tx"]):
        raise ValueError("iq and metadata must have the same sample count")
    flattened = packets.reshape(packets.size(0), -1)
    centered = flattened - flattened.mean(dim=1, keepdim=True)
    normalized = centered / centered.norm(dim=1, keepdim=True).clamp_min(1e-8)
    if int(projection_dim) <= 0:
        raise ValueError("projection_dim must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    projection_matrix = torch.randn(
        normalized.size(1),
        int(projection_dim),
        generator=generator,
    )
    projected = normalized @ projection_matrix
    projected = projected / projected.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def cell(index: int) -> tuple[int, int, int, int]:
        return (
            columns["tx"][index],
            columns["receiver"][index],
            columns["day"][index],
            columns["eq"][index],
        )

    fit_by_cell: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    audit_by_cell: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for index in split.fit_indices:
        fit_by_cell[cell(index)].append(index)
    for index in split.audit_indices:
        audit_by_cell[cell(index)].append(index)

    exact_pairs = 0
    nearest_similarities: list[Tensor] = []
    nearest_sig_gaps: list[int] = []
    for key in sorted(set(fit_by_cell).intersection(audit_by_cell)):
        fit_indices = fit_by_cell[key]
        audit_indices = audit_by_cell[key]
        fit_hashes = Counter(
            hashlib.sha256(normalized[index].numpy().tobytes()).hexdigest()
            for index in fit_indices
        )
        audit_hashes = Counter(
            hashlib.sha256(normalized[index].numpy().tobytes()).hexdigest()
            for index in audit_indices
        )
        exact_pairs += sum(
            fit_hashes[value] * audit_hashes[value]
            for value in set(fit_hashes).intersection(audit_hashes)
        )
        fit_tensor = torch.tensor(fit_indices, dtype=torch.long)
        audit_tensor = torch.tensor(audit_indices, dtype=torch.long)
        similarity = projected[fit_tensor] @ projected[audit_tensor].T
        maximum, positions = similarity.max(dim=1)
        nearest_similarities.extend(maximum.unbind())
        for row, position in enumerate(positions.tolist()):
            fit_index = fit_indices[row]
            audit_index = audit_indices[int(position)]
            nearest_sig_gaps.append(abs(columns["sig_i"][fit_index] - columns["sig_i"][audit_index]))

    similarity_values = (
        torch.stack(nearest_similarities) if nearest_similarities else torch.empty(0)
    )
    sig_gap_values = torch.tensor(nearest_sig_gaps, dtype=torch.float32)
    fit_base = {columns["base_index"][index] for index in split.fit_indices}
    audit_base = {columns["base_index"][index] for index in split.audit_indices}
    return {
        "status": "COMPLETE",
        "sample_level_state_persisted": False,
        "sig_i_semantics": "within_tx_rx_day_eq_temporal_index_only_not_cross_receiver_event_id",
        "block_size": int(split.block_size),
        "role_counts": {
            "V_select_fit": len(split.fit_indices),
            "V_audit_retro": len(split.audit_indices),
            "guard": len(split.guard_indices),
        },
        "base_index_overlap_count": len(fit_base.intersection(audit_base)),
        "exact_duplicate_pair_count": int(exact_pairs),
        "near_similarity_pair_count": int(similarity_values.numel()),
        "near_similarity_gt_0_999_rate": (
            float((similarity_values > 0.999).float().mean().item())
            if similarity_values.numel()
            else 0.0
        ),
        "near_similarity_gt_0_995_rate": (
            float((similarity_values > 0.995).float().mean().item())
            if similarity_values.numel()
            else 0.0
        ),
        "nearest_similarity_quantiles": _quantiles(similarity_values),
        "nearest_sig_gap_quantiles": _quantiles(sig_gap_values),
        "guard_block_effective": True,
    }


@dataclass(frozen=True)
class SidecarArchitectureConfig:
    input_length: int
    token_length: int
    stride: int
    q_dim: int
    challenge_hidden_dim: int
    codebook_size: int
    response_dim: int
    operator_dim: int
    pa_channels: int
    num_classes: int
    num_domains: int
    holdout_anchor_policy: str
    conditioned: bool
    pa_map_contract: str

    def __post_init__(self) -> None:
        positive = (
            "input_length",
            "token_length",
            "stride",
            "q_dim",
            "challenge_hidden_dim",
            "codebook_size",
            "response_dim",
            "operator_dim",
            "pa_channels",
            "num_classes",
            "num_domains",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"architecture_config {name} must be positive")
        if self.token_length > self.input_length:
            raise ValueError("architecture_config token_length exceeds input_length")
        if self.holdout_anchor_policy != "all_nonoverlap_folds":
            raise ValueError("architecture_config holdout_anchor_policy must be all_nonoverlap_folds")
        if self.pa_map_contract != "core90_pa_token_map_v1":
            raise ValueError("architecture_config pa_map_contract must be core90_pa_token_map_v1")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SidecarArchitectureConfig":
        required = {field.name for field in fields(cls)}
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError(f"architecture_config missing fields: {missing}")
        return cls(**{name: value[name] for name in required})


def _build_sidecar(config: SidecarArchitectureConfig, device: torch.device) -> CCOIPASidecar:
    encoder = PAChallengeEncoder(
        token_length=config.token_length,
        stride=config.stride,
        q_dim=config.q_dim,
        codebook_size=config.codebook_size,
        hidden_dim=config.challenge_hidden_dim,
        num_tx=config.num_classes,
        num_rx=config.num_domains,
    )
    return CCOIPASidecar(
        pa_channels=config.pa_channels,
        num_classes=config.num_classes,
        challenge_encoder=encoder,
        q_dim=config.q_dim,
        response_dim=config.response_dim,
        operator_dim=config.operator_dim,
    ).to(device)


def _validate_sidecar_against_config(
    sidecar: CCOIPASidecar,
    config: SidecarArchitectureConfig,
) -> None:
    observed = {
        "token_length": int(sidecar.challenge_encoder.token_length),
        "stride": int(sidecar.challenge_encoder.stride),
        "q_dim": int(sidecar.challenge_encoder.q_dim),
        "codebook_size": int(sidecar.challenge_encoder.codebook_size),
        "response_dim": int(sidecar.response_head.pa_proj.out_features),
        "operator_dim": int(sidecar.operator_pool.value.out_features),
        "pa_channels": int(sidecar.response_head.pa_proj.in_features),
        "num_classes": int(sidecar.classifier.out_features),
        "num_domains": (
            int(sidecar.challenge_encoder.rx_probe.out_features)
            if sidecar.challenge_encoder.rx_probe is not None
            else 0
        ),
    }
    for name, actual in observed.items():
        expected = int(getattr(config, name))
        if actual != expected:
            raise ValueError(f"architecture_config {name}={expected} does not match sidecar={actual}")


def build_sidecar_v3_payload(
    sidecar: CCOIPASidecar,
    *,
    row: str,
    base_checkpoint: str,
    architecture_config: SidecarArchitectureConfig,
    fusion_alpha: float,
    fusion_scale: float,
) -> dict[str, Any]:
    """Serialize parameter and non-parameter sidecar semantics together."""

    _validate_sidecar_against_config(sidecar, architecture_config)
    return {
        "schema": SIDECAR_V3_SCHEMA,
        "row": str(row),
        "base_checkpoint": str(base_checkpoint),
        "fusion_alpha": float(fusion_alpha),
        "fusion_scale": float(fusion_scale),
        "architecture_config": asdict(architecture_config),
        "state_dict": sidecar.state_dict(),
        "sample_level_source_state_included": False,
    }


def load_sidecar_v3(
    payload: Mapping[str, Any],
    *,
    expected_config: SidecarArchitectureConfig,
    device: torch.device,
) -> CCOIPASidecar:
    """Strictly restore a V3 sidecar only when all semantics match."""

    if str(payload.get("schema", "")) != SIDECAR_V3_SCHEMA:
        raise ValueError(f"sidecar schema must be {SIDECAR_V3_SCHEMA}")
    raw_config = payload.get("architecture_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("sidecar V3 requires architecture_config")
    observed_config = SidecarArchitectureConfig.from_mapping(raw_config)
    for field in fields(SidecarArchitectureConfig):
        name = field.name
        observed = getattr(observed_config, name)
        expected = getattr(expected_config, name)
        if observed != expected:
            raise ValueError(f"architecture_config {name} mismatch: saved={observed!r} expected={expected!r}")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("sidecar V3 requires state_dict")
    sidecar = _build_sidecar(observed_config, device)
    try:
        sidecar.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"sidecar V3 strict state load failed: {exc}") from exc
    _validate_sidecar_against_config(sidecar, observed_config)
    return sidecar


def migrate_v2_challenge_encoder(
    payload: Mapping[str, Any],
    *,
    architecture_config: SidecarArchitectureConfig,
    device: torch.device,
    legacy_migration_mode: bool,
) -> PAChallengeEncoder:
    """Load only the frozen challenge encoder from an explicitly accepted V2 payload."""

    if not bool(legacy_migration_mode):
        raise ValueError("legacy_migration_mode=true is required for a V2 sidecar")
    if str(payload.get("schema", "")) != SIDECAR_V2_SCHEMA:
        raise ValueError(f"legacy sidecar schema must be {SIDECAR_V2_SCHEMA}")
    if bool(payload.get("sample_level_source_state_included", True)):
        raise ValueError("legacy migration rejects sample-level source state")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("legacy sidecar requires state_dict")
    prefix = "challenge_encoder."
    encoder_state: dict[str, Tensor] = {
        str(key)[len(prefix) :]: value
        for key, value in state_dict.items()
        if str(key).startswith(prefix) and isinstance(value, Tensor)
    }
    if not encoder_state:
        raise ValueError("legacy sidecar contains no challenge encoder state")
    encoder = PAChallengeEncoder(
        token_length=architecture_config.token_length,
        stride=architecture_config.stride,
        q_dim=architecture_config.q_dim,
        codebook_size=architecture_config.codebook_size,
        hidden_dim=architecture_config.challenge_hidden_dim,
        num_tx=architecture_config.num_classes,
        num_rx=architecture_config.num_domains,
    ).to(device)
    try:
        encoder.load_state_dict(encoder_state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"legacy challenge encoder strict state load failed: {exc}") from exc
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder


__all__ = [
    "FactorRow",
    "FactorMatrixResult",
    "FoldRecords",
    "GATE_FEATURE_ALLOWLIST",
    "RelationMapping",
    "RetroSplit",
    "SIDECAR_V2_SCHEMA",
    "SIDECAR_V3_SCHEMA",
    "SidecarArchitectureConfig",
    "StageAVerdict",
    "StageBVerdict",
    "TruthBlindGateFit",
    "bounded_residual_fusion",
    "build_fold_records",
    "build_gate_feature_matrix",
    "build_relation_indices",
    "build_sidecar_v3_payload",
    "common_anchor_mask",
    "compose_factor_rows",
    "conditional_q_probe",
    "duplicate_audit",
    "evaluate_stage_a",
    "evaluate_stage_b",
    "fit_truth_blind_gate",
    "fold_macro_nmse",
    "load_sidecar_v3",
    "m0_exact_pair_retrieval",
    "migrate_v2_challenge_encoder",
    "predict_truth_blind_gate",
    "run_factor_matrix",
    "run_loto_residual",
    "split_v_select_retro",
]
