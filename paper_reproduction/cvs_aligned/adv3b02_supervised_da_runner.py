"""CVS Stage2-B DA methods sharing the exact ADV3B02 identity backbone."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
for value in (str(CODE_ROOT), str(PROJECT_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from SSDG import train_ssdg as ssdg_mod  # noqa: E402
from model_dual_cvsincnet import backbone_forward_compat  # noqa: E402
from eval_feature_diagnosis import infer_num_domains, strip_module_prefix  # noqa: E402
from paper_reproduction.common.config import load_json_config  # noqa: E402
from paper_reproduction.common.wisig_runtime import load_wisig_compact_pkl, set_seed, write_json  # noqa: E402
from paper_reproduction.cvs_aligned.class_incremental import (  # noqa: E402
    _compact_source_labels,
    _cycle_batches,
    _detailed_breakdown,
    _source_loader,
    _trace_loss,
)
from paper_reproduction.cvs_aligned.evaluate import (  # noqa: E402
    _apply_scenario,
    _build_stage2_tensors,
    _select_target_sets,
)
from paper_reproduction.cvs_aligned.supervised_da import (  # noqa: E402
    dadda_sda_objective,
    mrior_sda_batch_step,
    validate_supervised_da_manifest,
)


METHODS = {"protonet_cda", "mrior_sda", "dadda_sda"}
SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_adv3b02(checkpoint_path: Path, *, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = strip_module_prefix(checkpoint["model"])
    checkpoint_args = dict(checkpoint.get("args") or {})
    num_domains = infer_num_domains(
        None, state=state, split_info={}, ckpt_args=checkpoint_args, cli_num_domains=None
    )
    parser = ssdg_mod.build_arg_parser()
    model_args = parser.parse_args(["--output_dir", str(PROJECT_ROOT / ".tmp_adv3b02_da")])
    for key, value in checkpoint_args.items():
        setattr(model_args, key, value)
    model_args.device = str(device)
    merged = ssdg_mod.merge_checkpoint_args(
        checkpoint, model_args, input_len=256, num_domains=int(num_domains)
    )
    merged = ssdg_mod._apply_model_cli_args(merged, model_args)
    model = ssdg_mod.build_baseline_model(merged, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"strict ADV3B02 reconstruction failed: missing={list(missing)} unexpected={list(unexpected)}"
        )
    if not hasattr(model, "id_backbone") or not callable(getattr(model, "_pick_z_id", None)):
        raise ValueError("ADV3B02 checkpoint does not expose the identity backbone/z_id interface")
    return model, {
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_load_strict": True,
        "checkpoint_load_audit": {"missing_keys": 0, "unexpected_keys": 0, "shape_mismatch": 0},
        "num_domains": int(num_domains),
        "checkpoint_args": checkpoint_args,
    }


class ADV3B02MethodModel(nn.Module):
    def __init__(self, exact_model: nn.Module, *, method: str, feature_dim: int) -> None:
        super().__init__()
        self.method = str(method)
        self.id_backbone = copy.deepcopy(exact_model.id_backbone)
        self.feature_key = str(exact_model.id_feature_key)
        self.estimate_network = (
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim), nn.ELU(),
                nn.Linear(feature_dim, feature_dim), nn.ELU(), nn.Linear(feature_dim, 1),
            )
            if self.method == "mrior_sda" else None
        )

    def _identity(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        aux = backbone_forward_compat(
            self.id_backbone, x, y=None, return_aux=True, domain_labels=None
        )
        keys = [self.feature_key, "feat_joint", "feat_cls", "feat_con", "base"]
        features = next((aux[key] for key in keys if torch.is_tensor(aux.get(key))), None)
        if not torch.is_tensor(features) or not torch.is_tensor(aux.get("logits")):
            raise KeyError(f"ADV3B02 identity output misses z_id/logits; keys={sorted(aux)}")
        return features, aux["logits"], aux

    def forward(self, x: torch.Tensor) -> Any:
        features, logits, aux = self._identity(x)
        if self.method == "protonet_cda":
            return features
        if self.method == "mrior_sda":
            assert self.estimate_network is not None
            return {
                "features": features, "tx_logits": logits,
                "estimate_logits": self.estimate_network(features),
            }
        local_parts = [
            aux[key] for key in ("feat_cls", "feat_dac", "feat_pa", "feat_imp")
            if torch.is_tensor(aux.get(key))
        ]
        if not local_parts:
            local_parts = [features]
        return {
            "global_features": features,
            "local_features": torch.cat(local_parts, dim=1),
            "logits": logits,
        }


def _validate_config(config: dict[str, Any]) -> None:
    method = str(config.get("method_id", "")).lower()
    if method not in METHODS:
        raise ValueError(f"method_id must be one of {sorted(METHODS)}")
    if str(config.get("stage")) != "Stage2-B":
        raise ValueError("ADV3B02 supervised DA requires Stage2-B")
    if config.get("target_new_tx_labels") or config.get("target_unknown_tx_labels"):
        raise ValueError("Stage2-B permits target-old classes only")
    if len(config.get("target_receiver_labels", [])) != 1:
        raise ValueError("each run must adapt exactly one target receiver")
    if tuple(config.get("target_channel_scenarios", [])) != SCENARIOS:
        raise ValueError(f"formal scenarios must be exactly {SCENARIOS}")
    if int(config.get("k_shot", 0)) <= 0:
        raise ValueError("k_shot must be positive")
    if method != "protonet_cda" and int(config.get("adapt_steps", 0)) <= 0:
        raise ValueError("parametric DA methods require positive adapt_steps")


def _nearest_prototype(support: torch.Tensor, labels: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    class_ids = torch.unique(labels, sorted=True)
    prototypes = torch.stack([support[labels == class_id].mean(dim=0) for class_id in class_ids])
    return class_ids[torch.cdist(query.float(), prototypes.float()).argmin(dim=1)]


def _predict_logits(model: ADV3B02MethodModel, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        _, logits, _ = model._identity(x.to(device))
        return logits.argmax(dim=1).cpu()


def _adapt(
    config: dict[str, Any], model: ADV3B02MethodModel, source_loader: DataLoader,
    source_ids: list[int], support_x: torch.Tensor, support_y: torch.Tensor,
    *, scenario: str, device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    method = str(config["method_id"])
    steps = int(config["adapt_steps"])
    target_loader = DataLoader(
        TensorDataset(support_x, support_y),
        batch_size=min(int(config.get("target_batch_size", 64)), int(support_y.numel())),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    source_batches = _cycle_batches(source_loader, steps)
    target_batches = _cycle_batches(target_loader, steps)
    trace: list[dict[str, Any]] = []
    if method == "mrior_sda":
        learning_rate = float(config.get("mrior_adapt_learning_rate", 6.0e-4))
        optimizer_ec = torch.optim.Adam(model.id_backbone.parameters(), lr=learning_rate)
        assert model.estimate_network is not None
        optimizer_t = torch.optim.Adam(model.estimate_network.parameters(), lr=learning_rate)
        optimizer = None
    else:
        learning_rate = float(config.get("dadda_adapt_learning_rate", 1.0e-4))
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate,
            momentum=float(config.get("dadda_momentum", 0.9)),
            weight_decay=float(config.get("dadda_weight_decay", 5.0e-4)),
        )
    last: dict[str, float] = {}
    for step, (source_batch, (target_x, target_y)) in enumerate(
        zip(source_batches, target_batches), start=1
    ):
        source_x = source_batch["iq"].to(device)
        source_y = _compact_source_labels(source_batch["label"], source_ids, device)
        target_x, target_y = target_x.to(device), target_y.to(device)
        if method == "mrior_sda":
            losses = mrior_sda_batch_step(
                model, source_x, source_y, target_x, target_y,
                optimizer_t=optimizer_t, optimizer_ec=optimizer_ec,
                estimate_steps=int(config.get("mrior_estimate_steps", 7)),
                target_ce_weight=float(config.get("target_ce_weight", 1.0)),
                dvkl_weight=float(config.get("dvkl_weight", 0.005)),
                mu=float(config.get("mrior_mu", 0.5)),
                class_balance_smoothing=float(config.get("class_balance_smoothing", 0.0)),
            )
        else:
            progress = float(step - 1) / float(max(1, steps - 1))
            current_lr = learning_rate / ((1.0 + 10.0 * progress) ** 0.75)
            assert optimizer is not None
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            losses = dadda_sda_objective(
                model(source_x), model(target_x), source_labels=source_y,
                target_support_labels=target_y,
                target_ce_weight=float(config.get("target_ce_weight", 1.0)),
                alignment_weight=float(config.get("alignment_weight", 1.0)),
                bandwidth=config.get("bandwidth"), detach_dynamic_alpha=True,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
        last = {
            key: float(value.detach().cpu()) for key, value in losses.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        }
        _trace_loss(
            trace, {**config, "method": method, "_active_scenario": scenario},
            phase="target_support_adaptation", step=step, total_steps=steps,
            losses={key: value for key, value in losses.items() if value.numel() == 1},
        )
    return trace, {
        "adapt_steps": steps, "final_adaptation_losses": last,
        "optimizer": "Adam_minimax" if method == "mrior_sda" else "SGD_inverse",
        "learning_rate": learning_rate,
        "adv3b02_gradient_updates": steps,
    }


def _accuracy(predicted: torch.Tensor, truth: torch.Tensor) -> float:
    return float((predicted.cpu() == truth.cpu()).float().mean())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(config: dict[str, Any], *, run_dir: Path, device: torch.device) -> dict[str, Any]:
    _validate_config(config)
    seed = int(config["seed"])
    set_seed(seed)
    method = str(config["method_id"])
    checkpoint_path = Path(config["adv3b02_checkpoint"])
    exact_model, checkpoint_info = _exact_adv3b02(checkpoint_path, device=device)
    feature_dim = int(config.get("adv3b02_feature_dim", getattr(exact_model, "emb_dim", 160)))
    template = ADV3B02MethodModel(exact_model, method=method, feature_dim=feature_dim).cpu()
    del exact_model
    manysig = load_wisig_compact_pkl(str(config["manysig_pkl"]))
    manytx = load_wisig_compact_pkl(str(config["manytx_pkl"]))
    target_info = _select_target_sets(config, manysig, manytx)
    tensors = _build_stage2_tensors(config, manysig, manytx)
    if tensors["new_class_ids"] or tensors["support_query_overlap"]:
        raise ValueError("Stage2-B requires old-only disjoint support/query")
    source_loader, source_ids = _source_loader(config, manysig)
    scenarios: dict[str, dict[str, Any]] = {}
    predictions: dict[str, torch.Tensor] = {}
    score_rows: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    updates = 0
    for scenario_index, scenario in enumerate(SCENARIOS):
        set_seed(seed)
        support_x = _apply_scenario(
            tensors["support_x"].to(device), scenario, seed=seed + 1000 + scenario_index
        ).cpu()
        query_x = _apply_scenario(
            tensors["query_x"].to(device), scenario, seed=seed + 2000 + scenario_index
        ).cpu()
        model = copy.deepcopy(template).to(device)
        before = _predict_logits(model, query_x, device)
        started = time.perf_counter()
        if method == "protonet_cda":
            model.eval()
            with torch.no_grad():
                support_z = model(support_x.to(device)).cpu()
                query_z = model(query_x.to(device)).cpu()
            predicted = _nearest_prototype(support_z, tensors["support_y"], query_z)
            method_info = {
                "adapt_steps": 0, "adv3b02_gradient_updates": 0,
                "adaptation_objective": "labeled_target_support_prototype_registration",
            }
            trace.append({
                "method": method, "scenario": scenario, "phase": "support_prototype_registration",
                "step": 1, "total_steps": 1, "loss": 0.0, "gradient_updates": 0,
            })
        else:
            scenario_trace, method_info = _adapt(
                config, model, source_loader, source_ids, support_x, tensors["support_y"],
                scenario=scenario, device=device,
            )
            trace.extend(scenario_trace)
            updates += int(method_info["adv3b02_gradient_updates"])
            predicted = _predict_logits(model, query_x, device)
        elapsed = time.perf_counter() - started
        after_acc, before_acc = _accuracy(predicted, tensors["query_y"]), _accuracy(before, tensors["query_y"])
        scenarios[scenario] = {
            "target_old_accuracy": after_acc,
            "target_old_accuracy_before_adaptation": before_acc,
            "target_old_accuracy_delta": after_acc - before_acc,
            "adaptation_latency_sec": elapsed,
            "latency_per_query_ms": elapsed * 1000.0 / int(tensors["query_y"].numel()),
            **method_info,
        }
        predictions[scenario] = predicted
        detailed.extend(_detailed_breakdown(
            predicted, tensors["query_y"], tensors["query_meta"], scenario=scenario
        ))
        for meta, truth, prediction in zip(
            tensors["query_meta"], tensors["query_y"].tolist(), predicted.tolist()
        ):
            score_rows.append({
                "sample_id": meta["sample_id"], "receiver_label": meta.get("rx_label", ""),
                "transmitter_label": meta.get("tx_label", ""), "day_i": meta.get("day_i", ""),
                "sig_i": meta.get("sig_i", ""), "role": meta["role"],
                "true_label": truth, "predicted_label": prediction,
                "correct": int(truth == prediction), "scenario": scenario,
            })
    support_ids = [row["sample_id"] for row in tensors["support_meta"]]
    query_ids = [row["sample_id"] for row in tensors["query_meta"]]
    manifest = validate_supervised_da_manifest({
        **config, **target_info, "method_id": method, "stage": "Stage2-B", "cvs_extension": True,
        "target_old_support_sample_ids": support_ids,
        "target_old_query_sample_ids": query_ids,
        "target_labels_scope": "registered_support_only",
        "target_query_used_for_training": False,
        "target_query_used_for_model_selection": False,
    })
    manifest.update({
        **checkpoint_info, "feature_extractor": "ADV3B02 identity backbone",
        "adv3b02_feature_dim": feature_dim, "adv3b02_frozen": method == "protonet_cda",
        "adv3b02_gradient_updates": updates, "method_architecture_claim": "ADV3B02-backbone extension",
        "paper_faithful_architecture": False, "split_seed": int(config["split_seed"]),
        "support_query_overlap": bool(set(support_ids) & set(query_ids)),
        "all_tests_satellite_augmented": True,
        "claim_boundary": "Stage2-B target-old adaptation only",
    })
    aggregate = {
        key + "_mean": float(sum(float(row[key]) for row in scenarios.values()) / len(scenarios))
        for key in (
            "target_old_accuracy", "target_old_accuracy_before_adaptation",
            "target_old_accuracy_delta", "adaptation_latency_sec",
        )
    }
    result = {
        "schema": "adv3b02_stage2b_supervised_da_v1",
        "experiment_id": config["experiment_id"], "method_id": method,
        "seed": seed, "target_receiver_label": config["target_receiver_labels"][0],
        "k_shot": int(config["k_shot"]), "metrics": aggregate,
        "metrics_by_scenario": scenarios,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in (
        ("metrics.json", result), ("split_manifest.json", manifest),
        ("resolved_config.json", config), ("detailed_metrics.json", detailed),
        ("loss_trace.json", trace),
    ):
        write_json(run_dir / filename, payload)
    _write_csv(run_dir / "score_table.csv", score_rows)
    _write_csv(run_dir / "detailed_metrics.csv", detailed)
    _write_csv(run_dir / "loss_trace.csv", trace)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--method", choices=sorted(METHODS), default=None)
    parser.add_argument("--target-receiver", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--k-shot", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_json_config(args.config)
    for key, value in (
        ("experiment_id", args.experiment_id), ("method_id", args.method),
        ("seed", args.seed), ("split_seed", args.split_seed), ("k_shot", args.k_shot),
    ):
        if value is not None:
            config[key] = value
    if args.target_receiver is not None:
        config["target_receiver_labels"] = [args.target_receiver]
    _validate_config(config)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_pass", "config": config}, ensure_ascii=False, default=str))
        return 0
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result = run(config, run_dir=args.run_dir, device=device)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
