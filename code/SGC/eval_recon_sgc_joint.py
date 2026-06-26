from __future__ import annotations

import argparse
from typing import Mapping

import torch

from post_stage_cli import add_common_data_args, add_sat_eval_args, str2bool
from post_stage_common import build_standard_data, ensure_dir, load_baseline_from_checkpoint, move_batch, resolve_device, set_seed
from cvsrffi.eval import evaluate_loader, evaluate_named_loaders, evaluate_sat_scenarios, format_sat_test_lines
from training_controls import parse_sat_scenarios, sat_channel_config_for_scenario
from training_test_eval import evaluate_training_tests

from SGC.recon.cx_consistency import CxConsistency
from SGC.recon.cx_unet_1d import count_parameters
from SGC.train_recon_sgc_joint import ReconSGCJointEvalAdapter
from SGC.v3.sgc_v3_model import SGCv3Config, SGCv3Model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a finished PhyCon recon + SGC joint checkpoint.")
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--joint_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eval_recon_steps", type=int, default=2)
    parser.add_argument("--eval_recon_rho", type=float, default=-1.0)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--dry_run", action="store_true")
    add_common_data_args(parser)
    add_sat_eval_args(parser)
    return parser


def _infer_feature_dim(teacher: torch.nn.Module, x: torch.Tensor) -> int:
    with torch.no_grad():
        if hasattr(teacher, "extract_feature"):
            return int(teacher.extract_feature(x[:1]).shape[-1])
        out = teacher(x[:1], return_aux=True)
        if isinstance(out, Mapping):
            for key in ("z_id_raw", "z_id", "feat_joint", "feat_cls", "base"):
                value = out.get(key)
                if torch.is_tensor(value):
                    return int(value.shape[-1])
    raise AttributeError("Cannot infer teacher feature dimension.")


def _load_joint_models(args, data_ctx, device):
    joint_ckpt = torch.load(args.joint_ckpt, map_location=device)
    if not isinstance(joint_ckpt, Mapping):
        raise TypeError(f"Unsupported joint checkpoint payload in {args.joint_ckpt}")
    if "recon" not in joint_ckpt or "sgc_v3" not in joint_ckpt:
        raise KeyError("Joint checkpoint must contain 'recon' and 'sgc_v3' state dicts.")

    teacher, _, _ = load_baseline_from_checkpoint(args.teacher_ckpt, args, data_ctx, device, freeze=True)
    recon = CxConsistency().to(device)
    recon.load_state_dict(joint_ckpt["recon"], strict=False)
    recon.eval()

    first_batch = next(iter(data_ctx["train_loader"]))
    x0, _, _ = move_batch(first_batch, device)
    feature_dim = _infer_feature_dim(teacher, x0)
    cfg = SGCv3Config(num_classes=int(args.num_classes), feature_dim=int(feature_dim))
    sgc = SGCv3Model(teacher, cfg).to(device)
    sgc.load_state_dict(joint_ckpt["sgc_v3"], strict=False)
    sgc.eval()
    return joint_ckpt, recon, sgc


@torch.no_grad()
def evaluate(args) -> int:
    args.eval_sat_scenario_list = parse_sat_scenarios(args.eval_sat_scenarios) if bool(args.eval_sat_channel) else []
    for scenario in args.eval_sat_scenario_list:
        sat_channel_config_for_scenario(scenario)
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.output_dir)
    print(
        f"[RECON-SGC-EVAL] joint_ckpt={args.joint_ckpt} teacher={args.teacher_ckpt} "
        f"device={device} dry_run={int(bool(args.dry_run))}",
        flush=True,
    )
    if args.dry_run:
        return 0

    data_ctx = build_standard_data(args, device)
    joint_ckpt, recon, sgc = _load_joint_models(args, data_ctx, device)
    ckpt_args = joint_ckpt.get("args") or {}
    train_rho = float(ckpt_args.get("eval_recon_rho", ckpt_args.get("rho", -1.0))) if isinstance(ckpt_args, Mapping) else -1.0
    eval_rho = float(args.eval_recon_rho) if float(args.eval_recon_rho) > 0.0 else (train_rho if train_rho > 0.0 else 0.10)
    eval_model = ReconSGCJointEvalAdapter(
        recon,
        sgc,
        steps=int(args.eval_recon_steps),
        rho=eval_rho,
    )
    print(
        f"[RECON-SGC-EVAL] recon_params={count_parameters(recon):,} "
        f"epoch={int(joint_ckpt.get('epoch', 0))} eval_steps={int(args.eval_recon_steps)} "
        f"eval_rho={eval_rho:.3f} output_dir={out_dir}",
        flush=True,
    )

    result = evaluate_training_tests(
        model=eval_model,
        val_loader=data_ctx["val_loader"],
        named_test_loaders=data_ctx["named_test_loaders"],
        device=device,
        domain_label_map=data_ctx["domain_label_map"],
        named_test_meta=data_ctx["named_test_meta"],
        dataset=args.dataset,
        max_batches=int(args.eval_max_batches),
        evaluate_loader_fn=evaluate_loader,
        evaluate_named_loaders_fn=evaluate_named_loaders,
    )
    print(f"[VAL]   tx={result.val_stats['tx_acc']:.2f}% dom={result.val_stats['dom_acc']:.2f}%", flush=True)
    for line in result.lines:
        print(line, flush=True)

    if bool(args.eval_sat_channel) and len(args.eval_sat_scenario_list) > 0:
        sat_eval_max_batches = int(args.sat_eval_max_batches)
        if sat_eval_max_batches < 0:
            sat_eval_max_batches = int(args.eval_max_batches)
        sat_stats = evaluate_sat_scenarios(
            eval_model,
            data_ctx["named_test_loaders"],
            device,
            domain_label_map=data_ctx["domain_label_map"],
            scenario_names=args.eval_sat_scenario_list,
            args=args,
            max_batches=sat_eval_max_batches,
        )
        for line in format_sat_test_lines(sat_stats):
            print(line, flush=True)
    return 0


def main() -> int:
    return evaluate(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
