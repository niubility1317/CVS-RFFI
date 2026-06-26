from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from post_stage_cli import MAIN_SAT_EVAL_ON, add_common_data_args, add_sat_eval_args, str2bool
from post_stage_common import (
    build_standard_data,
    ensure_dir,
    load_baseline_from_checkpoint,
    mean_logs,
    move_batch,
    resolve_device,
    save_payload,
    set_seed,
)
from target_domain_adaptation import (
    TargetAdaptLossConfig,
    build_target_adapter,
    compute_target_adaptation_loss,
    configure_target_adaptation_parameters,
    select_fewshot_indices,
    select_target_indices_per_rx_tx,
    select_unlabeled_target_indices,
    select_unlabeled_target_indices_per_rx,
)
from cvsrffi.adaptation_safety import DEFAULT_TARGET_ROLLBACK_RULES, evaluate_rollback_gate, rules_from_policy
from cvsrffi.eval import (
    aggregate_named_stats,
    apply_sat_channel_for_scenario,
    evaluate_loader,
    evaluate_named_loaders,
    evaluate_sat_scenarios,
    format_named_test_lines,
    format_sat_test_lines,
    make_loader,
)
from cvsrffi.tensors import (
    make_torch_generator,
)
from training_controls import parse_sat_scenarios, sat_channel_config_for_scenario


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Few-shot target-domain adaptation on a trained strongest baseline checkpoint."
    )
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--target_loader", type=str, default="test_unseen_day_unseen_rx")
    parser.add_argument(
        "--target_channel_view",
        type=str,
        default="provided_satellite",
        choices=["provided_satellite", "leo_satellite", "satellite", "clean"],
        help="Domain of the few-shot target samples. provided_satellite means the loader already returns satellite-channel target samples.",
    )
    parser.add_argument(
        "--target_label_mode",
        type=str,
        default="unlabeled",
        choices=["unlabeled", "labeled"],
        help="Whether the selected target samples have transmitter labels available for adaptation.",
    )
    parser.add_argument(
        "--target_train_scenarios",
        type=str,
        default="clear_leo,low_elev_leo,rain_leo",
        help="Satellite scenarios used to synthesize target-domain few-shot views during adaptation.",
    )
    parser.add_argument(
        "--target_num_samples",
        type=int,
        default=64,
        help="Unlabeled target sample budget. Randomly select this many target samples without reading labels.",
    )
    parser.add_argument(
        "--target_samples_per_rx",
        type=int,
        default=0,
        help="Select this many target adaptation samples from each target receiver. Takes precedence over --target_num_samples.",
    )
    parser.add_argument(
        "--target_samples_per_rx_tx",
        type=int,
        default=0,
        help="Select this many target adaptation samples for each transmitter inside each target receiver. Takes precedence over --target_samples_per_rx.",
    )
    parser.add_argument(
        "--target_samples_per_class",
        type=int,
        default=0,
        help="Deprecated labeled-selection mode. Keep 0 for the unlabeled target-domain setting.",
    )
    parser.add_argument("--target_max_samples", type=int, default=0)
    parser.add_argument("--target_batch_size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--adapt_steps_per_epoch",
        type=int,
        default=0,
        help="Fine-tuning updates per epoch. 0 means one pass over the selected target subset.",
    )
    parser.add_argument("--lr_adapt", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--sat_train_scenario", type=str, default="mixed_orbit")
    parser.add_argument("--update_norm", type=str2bool, default=True)
    parser.add_argument("--update_classifier", type=str2bool, default=False)
    parser.add_argument(
        "--target_adapter_type",
        type=str,
        default="logit_calibration",
        choices=["logit_calibration", "feature_residual", "logit_lora"],
        help="On-orbit update backend. Non-logit backends keep the backbone frozen and train only lightweight delta modules.",
    )
    parser.add_argument("--freeze_base_stats", type=str2bool, default=False)
    parser.add_argument("--adapter_rank", type=int, default=4)
    parser.add_argument("--adapter_bottleneck", type=int, default=16)
    parser.add_argument("--adapter_alpha", type=float, default=1.0)
    parser.add_argument("--adapter_dropout", type=float, default=0.0)
    parser.add_argument("--rollback_enabled", type=str2bool, default=True)
    parser.add_argument("--rollback_policy_json", type=str, default="")
    parser.add_argument("--entropy_weight", type=float, default=1.0)
    parser.add_argument("--consistency_weight", type=float, default=0.5)
    parser.add_argument("--pseudo_weight", type=float, default=0.5)
    parser.add_argument("--anchor_weight", type=float, default=0.05)
    parser.add_argument("--conf_threshold", type=float, default=0.90)
    parser.add_argument("--margin_threshold", type=float, default=0.20)
    parser.add_argument("--anchor_temperature", type=float, default=2.0)
    parser.add_argument(
        "--eval_detail_every",
        type=int,
        default=0,
        help="Print detailed TEST-SPLIT lines every N epochs. 0 keeps per-epoch evaluation concise.",
    )
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--dry_run", action="store_true")
    add_common_data_args(parser)
    add_sat_eval_args(parser)
    parser.set_defaults(eval_sat_channel=False, eval_sat_on=MAIN_SAT_EVAL_ON)
    return parser


def _load_json_mapping(path: str | None) -> dict | None:
    if path is None or str(path).strip() == "":
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _param_numel_safe(param: torch.nn.Parameter) -> int:
    try:
        return int(param.numel())
    except ValueError:
        return 0


def _main_test_keys(args, data_ctx: Mapping[str, Any]):
    if str(args.dataset).lower() == "wisig":
        return ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"]
    return list(data_ctx["named_test_loaders"].keys())


def _sat_score(sat_stats: Mapping[str, Mapping[str, Any]]) -> float:
    values = []
    for item in sat_stats.values():
        aggregate = item.get("aggregate", {}) if isinstance(item, Mapping) else {}
        if "tx_acc" in aggregate:
            values.append(float(aggregate["tx_acc"]))
    return sum(values) / max(1, len(values)) if values else float("nan")


def _aggregate_sat_target_stats(sat_stats: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    correct = 0.0
    total = 0.0
    values = []
    for item in sat_stats.values():
        aggregate = item.get("aggregate", {}) if isinstance(item, Mapping) else {}
        if "tx_correct" in aggregate and "tx_total" in aggregate:
            correct += float(aggregate.get("tx_correct", 0.0))
            total += float(aggregate.get("tx_total", 0.0))
        if "tx_acc" in aggregate:
            values.append(float(aggregate["tx_acc"]))
    if total > 0:
        return {"tx_correct": correct, "tx_total": total, "tx_acc": 100.0 * correct / total}
    return {"tx_correct": 0.0, "tx_total": 0.0, "tx_acc": sum(values) / max(1, len(values))}


def _evaluate(
    model,
    args,
    data_ctx: Mapping[str, Any],
    device: torch.device,
    *,
    target_loader_name: str,
) -> Dict[str, Any]:
    target_loader = data_ctx["named_test_loaders"][target_loader_name]
    target_sat_stats: Dict[str, Dict[str, Any]] = {}
    if str(getattr(args, "target_channel_view", "provided_satellite")) in {"clean", "provided_satellite"}:
        target_stats = evaluate_loader(
            model,
            target_loader,
            device,
            domain_label_map=data_ctx["domain_label_map"],
            max_batches=int(args.eval_max_batches),
        )
    else:
        sat_eval_max_batches = int(args.sat_eval_max_batches)
        if sat_eval_max_batches < 0:
            sat_eval_max_batches = int(args.eval_max_batches)
        previous_eval_on = getattr(args, "eval_sat_on", target_loader_name)
        args.eval_sat_on = target_loader_name
        target_sat_stats = evaluate_sat_scenarios(
            model,
            data_ctx["named_test_loaders"],
            device,
            domain_label_map=data_ctx["domain_label_map"],
            scenario_names=getattr(args, "target_train_scenario_list", []),
            args=args,
            max_batches=sat_eval_max_batches,
        )
        args.eval_sat_on = previous_eval_on
        target_stats = _aggregate_sat_target_stats(target_sat_stats)
    named = evaluate_named_loaders(
        model,
        data_ctx["named_test_loaders"],
        device,
        domain_label_map=data_ctx["domain_label_map"],
        max_batches=int(args.eval_max_batches),
    )
    test_stats = aggregate_named_stats(named, [k for k in _main_test_keys(args, data_ctx) if k in named])
    sat_stats: Dict[str, Dict[str, Any]] = {}
    if bool(args.eval_sat_channel) and len(getattr(args, "eval_sat_scenario_list", [])) > 0:
        sat_eval_max_batches = int(args.sat_eval_max_batches)
        if sat_eval_max_batches < 0:
            sat_eval_max_batches = int(args.eval_max_batches)
        sat_stats = evaluate_sat_scenarios(
            model,
            data_ctx["named_test_loaders"],
            device,
            domain_label_map=data_ctx["domain_label_map"],
            scenario_names=args.eval_sat_scenario_list,
            args=args,
            max_batches=sat_eval_max_batches,
        )
    return {
        "target": target_stats,
        "target_sat": target_sat_stats,
        "test": test_stats,
        "named": named,
        "sat": sat_stats,
        "sat_score": _sat_score(sat_stats),
    }


def _format_eval_block(
    tag: str,
    args,
    data_ctx: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    include_details: bool = True,
) -> str:
    target = metrics.get("target", {})
    test = metrics.get("test", {})
    lines = [
        "-" * 132,
        f"[{tag}] target_loader={args.target_loader} target_tx={float(target.get('tx_acc', float('nan'))):.2f}% "
        f"target_view={getattr(args, 'target_channel_view', 'clean')} "
        f"overall_tx={float(test.get('tx_acc', float('nan'))):.2f}% sat_mean={float(metrics.get('sat_score', float('nan'))):.2f}%",
    ]
    if not include_details:
        return "\n".join(lines)
    lines.append("[TEST-SPLIT]")
    lines.extend(format_named_test_lines(metrics.get("named", {}), data_ctx["named_test_meta"]))
    if metrics.get("target_sat"):
        lines.append("[TARGET-SAT]")
        lines.extend(format_sat_test_lines(metrics["target_sat"]))
    if metrics.get("sat"):
        lines.extend(format_sat_test_lines(metrics["sat"]))
    return "\n".join(lines)


def build_eval_data_ctx_excluding_target_indices(
    data_ctx: Mapping[str, Any],
    *,
    target_loader_name: str,
    adaptation_indices,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    prefetch_factor: int,
) -> Dict[str, Any]:
    if target_loader_name not in data_ctx["named_test_loaders"]:
        return dict(data_ctx)
    original_loader = data_ctx["named_test_loaders"][target_loader_name]
    target_ds = original_loader.dataset
    excluded = {int(i) for i in adaptation_indices}
    kept = [idx for idx in range(len(target_ds)) if idx not in excluded]
    eval_ds = Subset(target_ds, kept)
    eval_loader = make_loader(
        eval_ds,
        int(batch_size),
        False,
        int(num_workers),
        device,
        False,
        int(prefetch_factor),
    )
    eval_ctx = dict(data_ctx)
    named_loaders = dict(data_ctx["named_test_loaders"])
    named_loaders[target_loader_name] = eval_loader
    eval_ctx["named_test_loaders"] = named_loaders
    named_meta = dict(data_ctx["named_test_meta"])
    meta = dict(named_meta.get(target_loader_name, {}))
    meta["size"] = len(eval_ds)
    meta["excluded_adaptation_samples"] = len(excluded)
    named_meta[target_loader_name] = meta
    eval_ctx["named_test_meta"] = named_meta
    return eval_ctx


def _validate_target_scenarios(args) -> None:
    args.target_train_scenario_list = parse_sat_scenarios(args.target_train_scenarios)
    if str(args.target_channel_view) == "provided_satellite":
        return
    if not args.target_train_scenario_list:
        raise ValueError("--target_train_scenarios produced an empty scenario list.")
    for scenario in args.target_train_scenario_list:
        cfg = sat_channel_config_for_scenario(scenario)
        if str(args.target_channel_view) == "leo_satellite":
            orbit_probs = cfg.get("orbit_probs", {})
            leo = float(orbit_probs.get("LEO", 0.0)) if isinstance(orbit_probs, Mapping) else 0.0
            non_leo = 0.0
            if isinstance(orbit_probs, Mapping):
                non_leo = sum(float(v) for k, v in orbit_probs.items() if str(k).upper() != "LEO")
            if leo < 0.999 or non_leo > 1e-6:
                raise ValueError(
                    f"--target_channel_view=leo_satellite requires pure LEO scenarios; {scenario!r} has orbit_probs={orbit_probs}"
                )


def _make_target_views(x_base, args, *, step: int, gen_a, gen_b):
    if str(args.target_channel_view) == "provided_satellite":
        return x_base, None, "provided_satellite", ""
    if str(args.target_channel_view) == "clean":
        x_a = x_base
        scenario_b = str(args.sat_train_scenario)
        x_b, _ = apply_sat_channel_for_scenario(x_base, scenario_b, args, gen=gen_b, return_meta=False)
        return x_a, x_b, "clean", scenario_b

    scenarios = getattr(args, "target_train_scenario_list", [str(args.sat_train_scenario)])
    scenario_a = scenarios[step % len(scenarios)]
    scenario_b = scenarios[(step + 1) % len(scenarios)]
    x_a, _ = apply_sat_channel_for_scenario(x_base, scenario_a, args, gen=gen_a, return_meta=False)
    x_b, _ = apply_sat_channel_for_scenario(x_base, scenario_b, args, gen=gen_b, return_meta=False)
    return x_a, x_b, scenario_a, scenario_b


def _compute_labeled_target_loss(model, x_target: torch.Tensor, y: torch.Tensor, cfg: TargetAdaptLossConfig):
    out = model(x_target)
    logits = out["tx_logits"]
    ce = F.cross_entropy(logits.float(), y.long())
    anchor = logits.sum() * 0.0
    if "base_tx_logits" in out:
        temp = max(1e-6, float(cfg.anchor_temperature))
        log_q = F.log_softmax(logits.float() / temp, dim=-1)
        p = F.softmax(out["base_tx_logits"].detach().float() / temp, dim=-1)
        anchor = F.kl_div(log_q, p, reduction="batchmean") * (temp * temp)
    loss = ce + float(cfg.anchor_weight) * anchor
    pred = logits.argmax(dim=-1)
    logs = {
        "target_adapt/loss_total": loss.detach(),
        "target_adapt/loss_supervised_ce": ce.detach(),
        "target_adapt/loss_anchor": anchor.detach(),
        "target_adapt/train_tx_acc": (pred == y.long()).float().mean().detach() * 100.0,
        "target_adapt/pseudo_coverage": logits.new_tensor(1.0),
        "target_adapt/loss_entropy": logits.new_tensor(0.0),
        "target_adapt/loss_consistency": logits.new_tensor(0.0),
    }
    return loss, logs


def build_target_finetune_loader(target_few_ds, args, *, device: torch.device):
    target_batch = int(args.target_batch_size) if int(args.target_batch_size) > 0 else int(args.batch_size)
    return make_loader(
        target_few_ds,
        target_batch,
        True,
        int(args.num_workers),
        device,
        False,
        int(args.prefetch_factor),
    )


def train(args) -> int:
    args.sat_train_scenario = str(args.sat_train_scenario or "mixed_orbit").strip().lower().replace("-", "_")
    args.eval_sat_scenario_list = parse_sat_scenarios(args.eval_sat_scenarios) if bool(args.eval_sat_channel) else []
    _validate_target_scenarios(args)
    sat_channel_config_for_scenario(args.sat_train_scenario)
    for scenario in args.eval_sat_scenario_list:
        sat_channel_config_for_scenario(scenario)

    set_seed(int(args.seed))
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.output_dir)
    data_ctx = build_standard_data(args, device)
    if args.target_loader not in data_ctx["named_test_loaders"]:
        available = ",".join(sorted(data_ctx["named_test_loaders"].keys()))
        raise KeyError(f"--target_loader {args.target_loader!r} not found. Available: {available}")

    base_model, ckpt, model_args = load_baseline_from_checkpoint(args.teacher_ckpt, args, data_ctx, device, freeze=False)
    adapter = build_target_adapter(
        base_model,
        num_classes=int(args.num_classes),
        adapter_type=str(args.target_adapter_type),
        adapter_rank=int(args.adapter_rank),
        adapter_bottleneck=int(args.adapter_bottleneck),
        adapter_alpha=float(args.adapter_alpha),
        adapter_dropout=float(args.adapter_dropout),
        freeze_base_stats=bool(args.freeze_base_stats),
    ).to(device)
    trainable = configure_target_adaptation_parameters(
        adapter,
        update_norm=bool(args.update_norm),
        update_classifier=bool(args.update_classifier),
    )
    if not trainable:
        raise RuntimeError("No trainable target-adaptation parameters were selected.")
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr_adapt), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    sat_gen_a = make_torch_generator(device, int(args.seed) + 771)
    sat_gen_b = make_torch_generator(device, int(args.seed) + 1771)
    loss_cfg = TargetAdaptLossConfig(
        entropy_weight=float(args.entropy_weight),
        consistency_weight=float(args.consistency_weight),
        pseudo_weight=float(args.pseudo_weight),
        anchor_weight=float(args.anchor_weight),
        conf_threshold=float(args.conf_threshold),
        margin_threshold=float(args.margin_threshold),
        anchor_temperature=float(args.anchor_temperature),
    )
    rollback_policy = _load_json_mapping(getattr(args, "rollback_policy_json", ""))
    rollback_rules = rules_from_policy(rollback_policy, default=DEFAULT_TARGET_ROLLBACK_RULES)
    trainable_names = [name for name, param in adapter.named_parameters() if param.requires_grad]

    target_full_ds = data_ctx["named_test_loaders"][args.target_loader].dataset
    if int(args.target_samples_per_class) > 0:
        print(
            "[WARN] --target_samples_per_class uses target labels for selection and is not source-free; "
            "prefer --target_num_samples for unlabeled target adaptation.",
            flush=True,
        )
        few_indices = select_fewshot_indices(
            target_full_ds,
            samples_per_class=int(args.target_samples_per_class),
            max_samples=int(args.target_max_samples),
            seed=int(args.seed),
        )
    else:
        if int(args.target_samples_per_rx_tx) > 0:
            print(
                "[WARN] --target_samples_per_rx_tx uses transmitter labels for stratified sample selection; "
                "unlabeled adaptation still does not use labels in the loss.",
                flush=True,
            )
            few_indices = select_target_indices_per_rx_tx(
                target_full_ds,
                samples_per_rx_tx=int(args.target_samples_per_rx_tx),
                seed=int(args.seed),
            )
            if int(args.target_max_samples) > 0:
                few_indices = few_indices[: int(args.target_max_samples)]
        elif int(args.target_samples_per_rx) > 0:
            few_indices = select_unlabeled_target_indices_per_rx(
                target_full_ds,
                samples_per_rx=int(args.target_samples_per_rx),
                seed=int(args.seed),
            )
            if int(args.target_max_samples) > 0:
                few_indices = few_indices[: int(args.target_max_samples)]
        else:
            budget = int(args.target_max_samples) if int(args.target_max_samples) > 0 else int(args.target_num_samples)
            few_indices = select_unlabeled_target_indices(
                target_full_ds,
                num_samples=budget,
                seed=int(args.seed),
            )
    if not few_indices:
        raise RuntimeError("Target few-shot selection produced no samples.")
    eval_data_ctx = build_eval_data_ctx_excluding_target_indices(
        data_ctx,
        target_loader_name=args.target_loader,
        adaptation_indices=few_indices,
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        device=device,
        prefetch_factor=int(args.prefetch_factor),
    )
    target_few_ds = Subset(target_full_ds, few_indices)
    target_loader = build_target_finetune_loader(target_few_ds, args, device=device)
    if len(target_loader) <= 0:
        raise RuntimeError(
            "Target fine-tuning loader is empty. Use a smaller --target_batch_size or ensure drop_last=False."
        )

    print("=" * 132, flush=True)
    print("[CONFIG-BEGIN] target_domain_adaptation", flush=True)
    print(
        f"[CONFIG-RUN] time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')} run_name={args.run_name} "
        f"seed={args.seed} device={device} epochs={args.epochs} amp={int(bool(args.amp))} label_mode={args.target_label_mode}",
        flush=True,
    )
    print(
        f"[CONFIG-BASE] teacher_ckpt={args.teacher_ckpt} num_classes={args.num_classes} "
        f"update_norm={int(bool(args.update_norm))} update_classifier={int(bool(args.update_classifier))} "
        f"target_adapter_type={args.target_adapter_type} freeze_base_stats={int(bool(args.freeze_base_stats))} "
        f"trainable_params={sum(_param_numel_safe(p) for p in trainable)}",
        flush=True,
    )
    print(
        f"[CONFIG-TARGET] loader={args.target_loader} view={args.target_channel_view} "
        f"train_scenarios={','.join(args.target_train_scenario_list)} full_size={len(target_full_ds)} few_size={len(target_few_ds)} "
        f"eval_size_after_excluding_adapt={len(eval_data_ctx['named_test_loaders'][args.target_loader].dataset)} "
        f"target_num_samples={args.target_num_samples} samples_per_rx={args.target_samples_per_rx} "
        f"samples_per_rx_tx={args.target_samples_per_rx_tx} samples_per_class={args.target_samples_per_class} max_samples={args.target_max_samples} "
        f"legacy_sat_train={args.sat_train_scenario}",
        flush=True,
    )
    print(
        f"[CONFIG-LOSS] entropy={args.entropy_weight} consistency={args.consistency_weight} "
        f"pseudo={args.pseudo_weight} anchor={args.anchor_weight} conf={args.conf_threshold} margin={args.margin_threshold}",
        flush=True,
    )
    print(
        f"[CONFIG-ADAPTER] type={getattr(adapter, 'adapter_type', args.target_adapter_type)} "
        f"rank={args.adapter_rank} bottleneck={args.adapter_bottleneck} alpha={args.adapter_alpha} "
        f"dropout={args.adapter_dropout} trainable_names={','.join(trainable_names)}",
        flush=True,
    )
    print(
        f"[CONFIG-ROLLBACK] enabled={int(bool(args.rollback_enabled))} rules={len(rollback_rules)} "
        f"policy_json={args.rollback_policy_json or '<default>'}",
        flush=True,
    )
    print(
        f"[CONFIG-CKPT] latest={out_dir / 'latest_target_adapt.pth'} best={out_dir / 'best_target_adapt.pth'}",
        flush=True,
    )
    print("[CONFIG-END]", flush=True)
    print("=" * 132, flush=True)

    if args.dry_run:
        print("[DRY-RUN] Built target adaptation model and selected target few-shot subset.", flush=True)
        return 0

    adapter.eval()
    before_metrics = _evaluate(adapter, args, eval_data_ctx, device, target_loader_name=args.target_loader)
    print(_format_eval_block("BEFORE-ADAPT", args, eval_data_ctx, before_metrics), flush=True)

    best_score = float("-inf")
    best_epoch = 0
    best_path = out_dir / "best_target_adapt.pth"
    latest_path = out_dir / "latest_target_adapt.pth"
    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        adapter.train()
        logs_epoch = []
        if int(args.adapt_steps_per_epoch) > 0:
            from itertools import cycle, islice

            epoch_iter = islice(cycle(target_loader), int(args.adapt_steps_per_epoch))
            epoch_steps = int(args.adapt_steps_per_epoch)
        else:
            epoch_iter = iter(target_loader)
            epoch_steps = len(target_loader)
        for batch_idx, batch in enumerate(epoch_iter):
            x_base, y_target, _ = move_batch(batch, device)
            with torch.no_grad():
                x_view_a, x_view_b, _, _ = _make_target_views(
                    x_base,
                    args,
                    step=(epoch - 1) * max(1, len(target_loader)) + batch_idx,
                    gen_a=sat_gen_a,
                    gen_b=sat_gen_b,
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                if str(args.target_label_mode) == "labeled":
                    loss, logs = _compute_labeled_target_loss(adapter, x_view_a, y_target, loss_cfg)
                else:
                    loss, logs = compute_target_adaptation_loss(adapter, x_view_a, x_view_b, loss_cfg)
            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            logs_epoch.append(logs)

        logs_mean = mean_logs(logs_epoch)
        adapter.eval()
        metrics = _evaluate(adapter, args, eval_data_ctx, device, target_loader_name=args.target_loader)
        rollback_decision = evaluate_rollback_gate(
            before_metrics=before_metrics,
            after_metrics=metrics,
            rules=rollback_rules,
        ) if bool(args.rollback_enabled) else None
        target_score = float(metrics["target"].get("tx_acc", float("nan")))
        sat_score = float(metrics.get("sat_score", float("nan")))
        score = target_score if sat_score != sat_score else 0.5 * target_score + 0.5 * sat_score
        adapter_state = adapter.adapter_state_dict() if hasattr(adapter, "adapter_state_dict") else adapter.state_dict()
        payload = {
            "target_adapter": adapter_state,
            "epoch": epoch,
            "args": vars(args),
            "teacher_args": vars(model_args),
            "teacher_stats": ckpt.get("stats", {}),
            "adaptation_indices": few_indices,
            "eval_excludes_adaptation_samples": True,
            "target_adapter_type": getattr(adapter, "adapter_type", str(args.target_adapter_type)),
            "trainable_names": trainable_names,
            "logs": logs_mean,
            "metrics": metrics,
            "before_metrics": before_metrics,
            "rollback_decision": rollback_decision.to_dict() if rollback_decision is not None else {"accepted": True, "rollback_triggered": False},
            "score": score,
        }
        save_payload(latest_path, payload)
        best_updated = False
        deployable = rollback_decision is None or rollback_decision.accepted
        if deployable and score > best_score:
            best_score = score
            best_epoch = epoch
            best_updated = True
            save_payload(best_path, payload)

        print(
            f"[EPOCH] E{epoch:03d}/{int(args.epochs):03d} time={time.time() - t0:.1f}s "
            f"loss={float(logs_mean.get('target_adapt/loss_total', 0.0)):.4f} "
            f"ce={float(logs_mean.get('target_adapt/loss_supervised_ce', 0.0)):.4f} "
            f"entropy={float(logs_mean.get('target_adapt/loss_entropy', 0.0)):.4f} "
            f"cons={float(logs_mean.get('target_adapt/loss_consistency', 0.0)):.4f} "
            f"pseudo_cov={float(logs_mean.get('target_adapt/pseudo_coverage', 0.0)):.3f} "
            f"train_tx={float(logs_mean.get('target_adapt/train_tx_acc', float('nan'))):.2f}% "
            f"steps={epoch_steps} "
            f"target_tx={target_score:.2f}% sat_mean={sat_score:.2f}% "
            f"rollback={int(not deployable)} "
            f"best={best_score:.2f}@E{best_epoch:03d} latest={latest_path} "
            f"best_path={best_path}{' (updated)' if best_updated else ''}",
            flush=True,
        )
        detail_every = int(getattr(args, "eval_detail_every", 0))
        include_details = epoch == int(args.epochs) or (detail_every > 0 and epoch % detail_every == 0)
        print(_format_eval_block("AFTER-ADAPT", args, eval_data_ctx, metrics, include_details=include_details), flush=True)

    if best_epoch <= 0:
        print(
            "Training finished. No adapter checkpoint passed rollback gates; deploy the teacher checkpoint and inspect latest_target_adapt.pth.",
            flush=True,
        )
    else:
        print(f"Training finished. best_target_adapt_score={best_score:.2f} at epoch {best_epoch} -> {best_path}")
    return 0


def main() -> int:
    return train(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
