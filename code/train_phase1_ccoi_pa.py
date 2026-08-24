"""Source-only Phase1 C0--C4 runner for CCOI-PA-V1.

The Core90 checkpoint is reconstructed exactly and remains frozen. The runner
writes prediction and truth streams separately; scoring is performed by the
independent ``score_phase1_ccoi_pa.py`` process after prediction closure.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from cvsrffi.ccoi_losses import (
    ccoi_did_loss,
    ccoi_supcon_loss,
    challenge_pair_masks,
    conditional_distance_diagnostics,
    ordinary_pair_masks,
)
from cvsrffi.ccoi_pa import (
    CCOIPASidecar,
    PAChallengeEncoder,
    challenge_pretrain_losses,
    raw_support_holdout_masks,
)
from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint


SCENARIOS = ("clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
SOURCE_ROLE_RATIOS = (0.07, 0.63, 0.15, 0.15)


@dataclass(frozen=True)
class MatrixSpec:
    row: str
    train_sidecar: bool
    conditioned: bool
    challenge_pairs: bool
    use_did: bool
    use_holdout: bool
    parameter_profile: str


def build_matrix_specs() -> Dict[str, MatrixSpec]:
    profile = "ccoi_pa_v1_capacity_q32_code48_r64_theta64"
    return {
        "C0": MatrixSpec("C0", False, False, False, False, False, "frozen_core90"),
        "C1": MatrixSpec("C1", True, False, False, False, False, profile),
        "C2": MatrixSpec("C2", True, True, True, False, False, profile),
        "C3": MatrixSpec("C3", True, True, True, True, False, profile),
        "C4": MatrixSpec("C4", True, True, True, True, True, profile),
    }


def freeze_base_model(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def validate_output_root(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable output root: {path}")


def validate_source_roles(args: argparse.Namespace, split_info: Mapping[str, Any]) -> None:
    actual = (
        float(args.labeled_ratio),
        float(args.unlabeled_ratio),
        float(args.source_cal_ratio),
        float(args.source_select_ratio),
    )
    if any(abs(a - b) > 1e-9 for a, b in zip(actual, SOURCE_ROLE_RATIOS)):
        raise ValueError("Phase1 source roles must be exactly 0.07/0.63/0.15/0.15")
    if str(args.phase1_source_role_protocol) != "l_s_u_s_v_cal_v_select":
        raise ValueError("Phase1 requires source role protocol l_s_u_s_v_cal_v_select")
    if str(args.split_mode) != "tx_rx_day_1_7_2":
        raise ValueError("Phase1 source roles require split_mode=tx_rx_day_1_7_2")
    realized = float(split_info.get("rho_label", float("nan")))
    if not math.isfinite(realized) or realized > 0.10 + 1e-9:
        raise ValueError(f"Phase1 realized rho_label must be <=0.10, got {realized}")
    recorded = split_info.get("source_role_ratios", {}) or {}
    for key, expected in zip(("L_s", "U_s", "V_cal", "V_select"), SOURCE_ROLE_RATIOS):
        if abs(float(recorded.get(key, float("nan"))) - expected) > 1e-9:
            raise ValueError(f"split_info source role mismatch for {key}")


def paired_challenge_batch(
    clean: Tensor,
    satellite: Tensor,
    labels: Tensor,
    domains: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if clean.shape != satellite.shape or clean.size(0) != labels.numel() or labels.shape != domains.shape:
        raise ValueError("clean/satellite/label/domain batch geometry mismatch")
    return (
        torch.cat((clean, satellite), dim=0),
        torch.cat((labels, labels), dim=0),
        torch.cat((domains, domains), dim=0),
    )


class FrozenCore90CCOI(nn.Module):
    """Evaluation-compatible wrapper around frozen Core90 plus one sidecar row."""

    def __init__(
        self,
        base: nn.Module,
        sidecar: Optional[CCOIPASidecar],
        *,
        row: str,
        fusion_alpha: float,
    ) -> None:
        super().__init__()
        self.base = freeze_base_model(base)
        self.sidecar = sidecar
        self.row = str(row)
        self.fusion_alpha = float(fusion_alpha)
        self.spec = build_matrix_specs()[self.row]

    def forward(
        self,
        x: Tensor,
        y_tx: Optional[Tensor] = None,
        grl_lambda: float = 1.0,
        return_aux: bool = False,
        domain_labels: Optional[Tensor] = None,
        **_: Any,
    ):
        with torch.no_grad():
            base_out = self.base(
                x,
                y_tx=y_tx,
                grl_lambda=float(grl_lambda),
                return_aux=True,
                domain_labels=domain_labels,
            )
        if not isinstance(base_out, Mapping) or "tx_logits" not in base_out:
            raise TypeError("Core90 must return the SSDG auxiliary-output mapping")
        final_logits = base_out["tx_logits"]
        side_out: Dict[str, Any] = {}
        if self.sidecar is not None:
            pa_map = (base_out.get("aux_id", {}) or {}).get("pa_token_map")
            if not torch.is_tensor(pa_map):
                raise KeyError("Core90 auxiliary output is missing pa_token_map")
            support_pa_map = None
            target_pa_map = None
            if self.spec.use_holdout:
                token_count = 1 + (int(x.size(-1)) - self.sidecar.challenge_encoder.token_length) // self.sidecar.challenge_encoder.stride
                support_raw, holdout_raw = raw_support_holdout_masks(
                    signal_length=int(x.size(-1)),
                    token_count=token_count,
                    token_length=self.sidecar.challenge_encoder.token_length,
                    stride=self.sidecar.challenge_encoder.stride,
                    fold=0,
                )
                support_view = x * support_raw.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
                holdout_view = x * holdout_raw.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
                with torch.no_grad():
                    support_out = self.base(
                        support_view,
                        y_tx=None,
                        grl_lambda=float(grl_lambda),
                        return_aux=True,
                        domain_labels=domain_labels,
                    )
                    target_out = self.base(
                        holdout_view,
                        y_tx=None,
                        grl_lambda=float(grl_lambda),
                        return_aux=True,
                        domain_labels=domain_labels,
                    )
                support_pa_map = (support_out.get("aux_id", {}) or {}).get("pa_token_map")
                target_pa_map = (target_out.get("aux_id", {}) or {}).get("pa_token_map")
                if not torch.is_tensor(support_pa_map) or not torch.is_tensor(target_pa_map):
                    raise KeyError("Core90 holdout isolation forward is missing pa_token_map")
            side_out = self.sidecar(
                x,
                pa_map.detach(),
                conditioned=self.spec.conditioned,
                holdout_support_pa_map=None if support_pa_map is None else support_pa_map.detach(),
                holdout_target_pa_map=None if target_pa_map is None else target_pa_map.detach(),
            )
            final_logits = final_logits + self.fusion_alpha * side_out["logit_correction"]
        if not return_aux:
            return final_logits
        merged = dict(base_out)
        merged["base_tx_logits"] = base_out["tx_logits"]
        merged["tx_logits"] = final_logits
        merged["ccoi"] = side_out
        merged["ccoi_row"] = self.row
        merged["fusion_alpha"] = self.fusion_alpha
        return merged


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _torch_load(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _prepare_ssdg_args(args: argparse.Namespace, checkpoint: Mapping[str, Any]):
    from SSDG import train_ssdg as ssdg

    data_args = ssdg.build_arg_parser().parse_args(["--output_dir", str(args.output_dir)])
    for key, value in dict(checkpoint.get("args") or {}).items():
        setattr(data_args, key, value)
    data_args.output_dir = str(args.output_dir)
    data_args.wisig_pkl = str(args.wisig_pkl)
    data_args.seed = int(args.seed)
    data_args.batch_size = int(args.batch_size)
    data_args.eval_batch_size = int(args.eval_batch_size)
    data_args.num_workers = int(args.num_workers)
    data_args.prefetch_factor = int(args.prefetch_factor)
    data_args.phase1_source_role_protocol = "l_s_u_s_v_cal_v_select"
    data_args.split_mode = "tx_rx_day_1_7_2"
    data_args.labeled_ratio = SOURCE_ROLE_RATIOS[0]
    data_args.unlabeled_ratio = SOURCE_ROLE_RATIOS[1]
    data_args.source_cal_ratio = SOURCE_ROLE_RATIOS[2]
    data_args.source_select_ratio = SOURCE_ROLE_RATIOS[3]
    data_args.source_val_ratio = SOURCE_ROLE_RATIOS[2] + SOURCE_ROLE_RATIOS[3]
    data_args.eval_sat_on = str(args.eval_sat_on)
    data_args.sat_seed = int(args.sat_seed)
    return ssdg, data_args


def _move_batch(ssdg, batch, device: torch.device, domain_label_map):
    x, y, extra = ssdg.move_batch(batch, device)
    domain = ssdg.domain_from_extra(extra, domain_label_map, device)
    if domain is None:
        domain = torch.full_like(y, -1)
    return x.float(), y.long(), domain.long(), extra


def _satellite_view(ssdg, x: Tensor, scenario: str, data_args, generator) -> Tensor:
    sat, _ = ssdg.apply_sat_channel_for_scenario(
        x,
        str(scenario),
        data_args,
        gen=generator,
        return_meta=False,
    )
    return torch.nan_to_num(sat.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _limited(loader: Iterable, max_batches: int):
    for batch_index, batch in enumerate(loader, start=1):
        if int(max_batches) > 0 and batch_index > int(max_batches):
            break
        yield batch_index, batch


def _pretrain_challenge(
    encoder: PAChallengeEncoder,
    data_ctx,
    ssdg,
    data_args,
    args,
    device: torch.device,
) -> list[dict[str, float]]:
    encoder.train()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=float(args.q_lr), weight_decay=1e-4)
    generator = ssdg.make_torch_generator(device, int(args.seed) + 1771)
    history = []
    loaders = (("L_s", data_ctx["train_loader"]), ("U_s", data_ctx["unlabeled_loader"]))
    scenarios = SCENARIOS[1:]
    for epoch in range(1, int(args.q_epochs) + 1):
        sums: Dict[str, float] = {}
        steps = 0
        for role, loader in loaders:
            for batch_index, batch in _limited(loader, int(args.max_train_batches)):
                x, y, domain, _ = _move_batch(ssdg, batch, device, data_ctx["domain_label_map"])
                scenario = scenarios[(epoch + batch_index - 2) % len(scenarios)]
                with torch.no_grad():
                    satellite = _satellite_view(ssdg, x, scenario, data_args, generator)
                losses = challenge_pretrain_losses(
                    encoder,
                    x,
                    satellite,
                    tx_labels=y if role == "L_s" else None,
                    rx_labels=domain,
                )
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError(f"non-finite q pretraining loss at {role}/{epoch}/{batch_index}")
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()
                for key, value in losses.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach().item())
                steps += 1
        if steps == 0:
            raise RuntimeError("challenge pretraining produced zero batches")
        history.append({"epoch": epoch, "steps": steps, **{k: v / steps for k, v in sums.items()}})
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return history


def _source_accuracy(model, loader, ssdg, data_ctx, device, max_batches: int) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _, batch in _limited(loader, max_batches):
            x, y, domain, _ = _move_batch(ssdg, batch, device, data_ctx["domain_label_map"])
            logits = model(x, domain_labels=domain)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
    return 100.0 * correct / max(1, total)


def _train_sidecar(
    model: FrozenCore90CCOI,
    data_ctx,
    ssdg,
    data_args,
    args,
    device: torch.device,
) -> list[dict[str, float]]:
    if model.sidecar is None:
        return []
    parameters = [parameter for parameter in model.sidecar.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(args.head_lr), weight_decay=1e-4)
    history = []
    best_accuracy = -float("inf")
    best_state = None
    generator = ssdg.make_torch_generator(device, int(args.seed) + 2881)
    scenarios = SCENARIOS[1:]
    for epoch in range(1, int(args.head_epochs) + 1):
        model.sidecar.train()
        model.sidecar.challenge_encoder.eval()
        sums: Dict[str, float] = {}
        steps = 0
        for batch_index, batch in _limited(data_ctx["train_loader"], int(args.max_train_batches)):
            x, y, domain, _ = _move_batch(ssdg, batch, device, data_ctx["domain_label_map"])
            scenario = scenarios[(epoch + batch_index - 2) % len(scenarios)]
            with torch.no_grad():
                satellite = _satellite_view(ssdg, x, scenario, data_args, generator)
            paired_x, paired_y, paired_domain = paired_challenge_batch(x, satellite, y, domain)
            out = model(paired_x, y_tx=None, return_aux=True, domain_labels=paired_domain)
            ccoi = out["ccoi"]
            classification = F.cross_entropy(out["tx_logits"], paired_y)
            q_summary = ccoi["q"].mean(dim=1)
            masks = (
                challenge_pair_masks(
                    q_summary,
                    paired_y,
                    paired_domain,
                    min_cosine=float(args.min_match_cosine),
                )
                if model.spec.challenge_pairs
                else ordinary_pair_masks(paired_y, paired_domain)
            )
            pair = ccoi_supcon_loss(ccoi["theta"], masks, temperature=float(args.temperature))
            did, rectangle_count = ccoi_did_loss(ccoi["theta"], paired_y, paired_domain)
            holdout = F.mse_loss(ccoi["heldout_prediction"], ccoi["heldout_target"])
            total = classification + 0.15 * pair.loss
            if model.spec.use_did:
                total = total + 0.10 * did
            if model.spec.use_holdout:
                total = total + 0.20 * holdout
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite sidecar loss row={model.row} epoch={epoch}")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            values = {
                "total": total,
                "classification": classification,
                "pair": pair.loss,
                "did": did,
                "holdout": holdout,
            }
            for key, value in values.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().item())
            sums["positive_pairs"] = sums.get("positive_pairs", 0.0) + pair.positive_count
            sums["rectangles"] = sums.get("rectangles", 0.0) + rectangle_count
            steps += 1
        if steps == 0:
            raise RuntimeError(f"sidecar row {model.row} produced zero batches")
        select_accuracy = _source_accuracy(
            model,
            data_ctx["val_loader"],
            ssdg,
            data_ctx,
            device,
            int(args.max_eval_batches),
        )
        history.append(
            {
                "epoch": epoch,
                "steps": steps,
                "v_select_accuracy": select_accuracy,
                **{key: value / steps for key, value in sums.items()},
            }
        )
        if select_accuracy > best_accuracy:
            best_accuracy = select_accuracy
            best_state = deepcopy(model.sidecar.state_dict())
    if best_state is None:
        raise RuntimeError(f"row {model.row} has no selected sidecar state")
    model.sidecar.load_state_dict(best_state, strict=True)
    return history


def _calibrate_alpha(model, data_ctx, ssdg, args, device: torch.device) -> dict[str, Any]:
    if model.sidecar is None:
        return {"selected_alpha": 0.0, "grid": {"0.0": _source_accuracy(model, data_ctx["source_calibration_loader"], ssdg, data_ctx, device, int(args.max_eval_batches))}}
    original = model.fusion_alpha
    results = {}
    for alpha in (0.0, 0.05, 0.10, 0.15, 0.20):
        model.fusion_alpha = alpha
        results[f"{alpha:.2f}"] = _source_accuracy(
            model,
            data_ctx["source_calibration_loader"],
            ssdg,
            data_ctx,
            device,
            int(args.max_eval_batches),
        )
    best = max(results, key=lambda key: (results[key], -abs(float(key) - original)))
    model.fusion_alpha = float(best)
    return {"selected_alpha": model.fusion_alpha, "grid": results}


def _meta_value(extra: Any, key: str, index: int, default: Any = None) -> Any:
    if isinstance(extra, Mapping) and key in extra:
        value = extra[key]
        if torch.is_tensor(value):
            return value[index].detach().cpu().item()
        if isinstance(value, (list, tuple)):
            return value[index]
        return value
    return default


def _source_challenge_audit(model, data_ctx, ssdg, args, device) -> dict[str, Any]:
    if model.sidecar is None:
        return {"status": "NOT_APPLICABLE_C0", "truth_scope": "source_V_select_only"}
    model.eval()
    relation_rows = []
    code_counts = torch.zeros(48, dtype=torch.long)
    control_error = {"real": [0.0, 0.0], "shuffle": [0.0, 0.0], "random": [0.0, 0.0], "constant": [0.0, 0.0]}
    coverage_correct = {1.0: 0, 0.75: 0, 0.50: 0, 0.25: 0}
    coverage_total = 0
    with torch.no_grad():
        for _, batch in _limited(data_ctx["val_loader"], int(args.max_eval_batches)):
            x, y, domain, _ = _move_batch(ssdg, batch, device, data_ctx["domain_label_map"])
            out = model(x, return_aux=True, domain_labels=domain)
            ccoi = out["ccoi"]
            relation_rows.append(
                conditional_distance_diagnostics(
                    ccoi["response"], ccoi["q"], y, domain, min_cosine=float(args.min_match_cosine)
                )
            )
            bins = ccoi["code_prob"].mean(dim=1).argmax(dim=1).detach().cpu()
            code_counts += torch.bincount(bins, minlength=48)
            target = ccoi["heldout_target"]
            q_holdout = ccoi["q_holdout"]
            controls = {
                "real": ccoi["heldout_prediction"],
                "shuffle": model.sidecar.heldout_predictor(
                    ccoi["support_theta"], q_holdout[torch.arange(q_holdout.size(0) - 1, -1, -1, device=device)]
                ),
                "random": model.sidecar.heldout_predictor(
                    ccoi["support_theta"], F.normalize(torch.randn_like(q_holdout), dim=-1)
                ),
                "constant": model.sidecar.heldout_predictor(
                    ccoi["support_theta"],
                    model.sidecar.response_head.constant_condition.view(1, 1, -1).expand_as(q_holdout),
                ),
            }
            target_energy = float(target.square().sum().item())
            for name, prediction in controls.items():
                control_error[name][0] += float((prediction - target).square().sum().item())
                control_error[name][1] += target_energy

            attention = ccoi["attention"]
            token_count = int(attention.size(1))
            for fraction in coverage_correct:
                keep = max(1, int(math.ceil(token_count * fraction)))
                indices = attention.topk(keep, dim=1).indices
                mask = torch.zeros_like(attention, dtype=torch.bool).scatter_(1, indices, True)
                pooled = model.sidecar.operator_pool(ccoi["response"], ccoi["condition_q"], mask)
                logits = out["base_tx_logits"] + model.fusion_alpha * model.sidecar.classifier(pooled.theta)
                coverage_correct[fraction] += int((logits.argmax(dim=1) == y).sum().item())
            coverage_total += int(y.numel())
    relation = {}
    for name in ("d1", "d2", "d3"):
        count_key = f"{name}_count"
        total_count = sum(int(row[count_key]) for row in relation_rows)
        weighted = sum(float(row[name]) * int(row[count_key]) for row in relation_rows if int(row[count_key]) > 0)
        relation[name] = weighted / total_count if total_count else float("nan")
        relation[count_key] = total_count
    return {
        "status": "COMPLETE" if coverage_total else "EMPTY",
        "truth_scope": "source_V_select_only",
        "target_or_query_truth_used": False,
        "conditional_distance": relation,
        "holdout_nmse_controls": {
            name: error / max(1e-12, energy) for name, (error, energy) in control_error.items()
        },
        "coverage_accuracy": {
            f"{int(fraction * 100)}%": 100.0 * correct / max(1, coverage_total)
            for fraction, correct in coverage_correct.items()
        },
        "challenge_code_histogram": code_counts.tolist(),
        "challenge_code_observed": int((code_counts > 0).sum().item()),
        "challenge_code_unobserved": int((code_counts == 0).sum().item()),
        "challenge_ood_interpretation": "read_only_sparse_or_unobserved_code_diagnostic_not_semantic_content_OOD",
    }


def _write_predictions(
    model,
    row_dir: Path,
    data_ctx,
    ssdg,
    data_args,
    args,
    device,
) -> dict[str, Any]:
    prediction_path = row_dir / "prediction.jsonl"
    truth_path = row_dir / "truth.jsonl"
    if prediction_path.exists() or truth_path.exists():
        raise FileExistsError(f"refusing to overwrite prediction stream in {row_dir}")
    model.eval()
    prediction_count = 0
    generator = ssdg.make_torch_generator(device, int(args.sat_seed) + 9917)
    with prediction_path.open("w", encoding="utf-8", newline="\n") as pred_file, torch.no_grad():
        for scenario in SCENARIOS:
            for loader_name, loader in data_ctx["named_test_loaders"].items():
                for batch_index, batch in _limited(loader, int(args.max_eval_batches)):
                    x, _truth_ignored, domain, extra = _move_batch(ssdg, batch, device, data_ctx["domain_label_map"])
                    view = x if scenario == "clean" else _satellite_view(ssdg, x, scenario, data_args, generator)
                    out = model(view, return_aux=True, domain_labels=domain)
                    logits = out["tx_logits"]
                    predicted = logits.argmax(dim=1)
                    for sample_index in range(int(x.size(0))):
                        sample_id = f"{scenario}:{loader_name}:{batch_index}:{sample_index}"
                        receiver = _meta_value(extra, "rx_i", sample_index, int(domain[sample_index].item()))
                        pred_record = {
                            "sample_id": sample_id,
                            "row": model.row,
                            "scenario": scenario,
                            "loader": loader_name,
                            "predicted_class": int(predicted[sample_index].item()),
                            "receiver": receiver,
                            "logits": logits[sample_index].detach().float().cpu().tolist(),
                        }
                        pred_file.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
                        prediction_count += 1
    if prediction_count == 0:
        raise RuntimeError(f"row {model.row} produced no predictions")

    truth_count = 0
    with truth_path.open("w", encoding="utf-8", newline="\n") as truth_file:
        for scenario in SCENARIOS:
            for loader_name, loader in data_ctx["named_test_loaders"].items():
                for batch_index, batch in _limited(loader, int(args.max_eval_batches)):
                    _x, y, _extra = ssdg.move_batch(batch, device)
                    for sample_index in range(int(y.numel())):
                        sample_id = f"{scenario}:{loader_name}:{batch_index}:{sample_index}"
                        truth_file.write(
                            json.dumps(
                                {"sample_id": sample_id, "true_class": int(y[sample_index].item())},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        truth_count += 1
    if truth_count != prediction_count:
        raise RuntimeError(
            f"prediction/truth closure mismatch before scoring: {prediction_count} != {truth_count}"
        )
    return {
        "prediction_path": str(prediction_path),
        "truth_path": str(truth_path),
        "prediction_count": prediction_count,
        "truth_written_after_prediction_close": True,
    }


def _infer_base_dimensions(base, data_ctx, ssdg, device):
    _, batch = next(iter(_limited(data_ctx["train_loader"], 1)))
    x, _, domain, _ = _move_batch(ssdg, batch, device, data_ctx["domain_label_map"])
    with torch.no_grad():
        out = base(x, y_tx=None, return_aux=True, domain_labels=domain)
    pa_map = (out.get("aux_id", {}) or {}).get("pa_token_map")
    if not torch.is_tensor(pa_map) or pa_map.ndim != 3 or pa_map.size(1) == 0:
        raise RuntimeError("real checkpoint smoke did not expose a valid PA map")
    return int(pa_map.size(1)), int(out["tx_logits"].size(1)), {
        "batch_size": int(x.size(0)),
        "pa_map_shape": list(pa_map.shape),
        "logit_shape": list(out["tx_logits"].shape),
        "finite_logits": bool(torch.isfinite(out["tx_logits"]).all().item()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase1 CCOI-PA-V1 frozen-Core90 matrix")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wisig_pkl", required=True)
    parser.add_argument("--rows", default="C0,C1,C2,C3,C4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--sat_seed", type=int, default=20260824)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--q_epochs", type=int, default=10)
    parser.add_argument("--head_epochs", type=int, default=20)
    parser.add_argument("--q_lr", type=float, default=3e-4)
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--fusion_alpha", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=0.12)
    parser.add_argument("--min_match_cosine", type=float, default=0.70)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--eval_sat_on", default="main")
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    validate_output_root(output_dir)
    rows = [item.strip().upper() for item in str(args.rows).split(",") if item.strip()]
    specs = build_matrix_specs()
    unknown = [row for row in rows if row not in specs]
    if unknown:
        raise ValueError(f"unknown matrix rows: {unknown}")
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "rows": rows, "scenarios": SCENARIOS, "ratios": SOURCE_ROLE_RATIOS}))
        return 0

    checkpoint_path = Path(args.checkpoint).resolve()
    wisig_path = Path(args.wisig_pkl).resolve()
    if not checkpoint_path.is_file() or not wisig_path.is_file():
        raise FileNotFoundError("checkpoint and wisig_pkl must exist")
    output_dir.mkdir(parents=True, exist_ok=False)
    _seed_all(args.seed)
    device = torch.device(args.device)
    checkpoint = _torch_load(checkpoint_path, device)
    ssdg, data_args = _prepare_ssdg_args(args, checkpoint)
    data_ctx = ssdg._build_ssdg_wisig_data(data_args, device)
    validate_source_roles(data_args, data_ctx["split_info"])
    base, checkpoint_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint,
        input_len=int(data_ctx["input_len"]),
        device=device,
        ssdg_module=ssdg,
    )
    base = freeze_base_model(base)
    pa_channels, num_classes, smoke = _infer_base_dimensions(base, data_ctx, ssdg, device)
    if not smoke["finite_logits"]:
        raise FloatingPointError("real checkpoint smoke produced non-finite logits")
    _json_write(
        output_dir / "protocol_and_smoke.json",
        {
            "protocol": "Phase1_source_only",
            "source_roles": data_ctx["split_info"],
            "checkpoint_audit": checkpoint_audit,
            "real_checkpoint_no_query_smoke": smoke,
            "target_or_query_access": False,
        },
    )
    if args.smoke_only:
        print(f"[CCOI-SMOKE] PASS output={output_dir}", flush=True)
        return 0

    challenge_encoder = PAChallengeEncoder(
        q_dim=32,
        codebook_size=48,
        num_tx=num_classes,
        num_rx=int(data_ctx["num_domains"]),
    ).to(device)
    q_history = _pretrain_challenge(challenge_encoder, data_ctx, ssdg, data_args, args, device)
    _json_write(output_dir / "challenge_pretrain_history.json", q_history)

    _seed_all(args.seed + 404)
    template = CCOIPASidecar(
        pa_channels=pa_channels,
        num_classes=num_classes,
        challenge_encoder=deepcopy(challenge_encoder),
    ).to(device)
    template_state = deepcopy(template.state_dict())
    manifest = {
        "status": "PREDICTIONS_PENDING",
        "rows": {},
        "scenarios": list(SCENARIOS),
        "seed": int(args.seed),
        "checkpoint": str(checkpoint_path),
        "checkpoint_audit": checkpoint_audit,
        "split_info": data_ctx["split_info"],
    }
    _json_write(output_dir / "matrix_manifest.json", manifest)
    for row in rows:
        row_dir = output_dir / row
        row_dir.mkdir(parents=False, exist_ok=False)
        spec = specs[row]
        sidecar = None
        if spec.train_sidecar:
            sidecar = CCOIPASidecar(
                pa_channels=pa_channels,
                num_classes=num_classes,
                challenge_encoder=deepcopy(challenge_encoder),
            ).to(device)
            sidecar.load_state_dict(template_state, strict=True)
            sidecar.freeze_challenge_encoder()
        model = FrozenCore90CCOI(base, sidecar, row=row, fusion_alpha=float(args.fusion_alpha)).to(device)
        train_history = _train_sidecar(model, data_ctx, ssdg, data_args, args, device)
        calibration = _calibrate_alpha(model, data_ctx, ssdg, args, device)
        if sidecar is not None:
            torch.save(
                {
                    "schema": "cvs.phase1.ccoi_pa_sidecar.v1",
                    "row": row,
                    "base_checkpoint": str(checkpoint_path),
                    "fusion_alpha": model.fusion_alpha,
                    "state_dict": sidecar.state_dict(),
                    "sample_level_source_state_included": False,
                },
                row_dir / "sidecar.pth",
            )
        _json_write(row_dir / "train_history.json", train_history)
        _json_write(row_dir / "calibration.json", calibration)
        challenge_audit = _source_challenge_audit(model, data_ctx, ssdg, args, device)
        _json_write(row_dir / "challenge_audit.json", challenge_audit)
        prediction_info = _write_predictions(model, row_dir, data_ctx, ssdg, data_args, args, device)
        manifest["rows"][row] = {
            "spec": asdict(spec),
            "fusion_alpha": model.fusion_alpha,
            "prediction_path": prediction_info["prediction_path"],
            "truth_path": prediction_info["truth_path"],
            "prediction_count": prediction_info["prediction_count"],
            "prediction_complete": True,
            "truth_written_after_prediction_close": prediction_info["truth_written_after_prediction_close"],
        }
        _json_write(output_dir / "matrix_manifest.json", manifest)
    manifest["status"] = "PREDICTIONS_COMPLETE_TRUTH_NOT_SCORED"
    _json_write(output_dir / "matrix_manifest.json", manifest)
    print(f"[CCOI-PREDICTIONS] COMPLETE output={output_dir}", flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
