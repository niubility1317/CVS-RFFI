from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


def _add_project_paths() -> None:
    here = Path(__file__).resolve()
    root = here.parents[1]
    for path in (root / "code", root):
        if path.exists():
            sys.path.insert(0, str(path))


_add_project_paths()

import torch  # noqa: E402

from cvsrffi.eval import (  # noqa: E402
    evaluate_loader,
    evaluate_named_loaders,
    evaluate_sat_scenarios,
    format_sat_test_lines,
    make_loader,
)
from evaluation.collaborative_inference_eval import (  # noqa: E402
    build_model_from_checkpoint_args,
    build_wisig_context_from_checkpoint,
    load_model_state,
    sha256_file,
    validate_checkpoint_identity,
)
from training_controls import parse_sat_scenarios  # noqa: E402
from training_test_eval import aggregate_named_stats, format_named_test_lines  # noqa: E402


def _load_checkpoint(path: str | Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint payload must be a mapping, got {type(payload)}")
    return payload


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _main_test_keys(named_stats: Mapping[str, Mapping[str, Any]]) -> list[str]:
    keys = ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"]
    return [key for key in keys if key in named_stats] or list(named_stats)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a central CVS checkpoint on clean and satellite-channel tests.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path saved by train.py.")
    parser.add_argument("--output_json", default="", help="Optional JSON result path.")
    parser.add_argument("--output_txt", default="", help="Optional text summary path.")
    parser.add_argument("--expect_run_name", default="", help="Fail unless this token appears in checkpoint identity.")
    parser.add_argument("--expect_sha256", default="", help="Fail unless checkpoint SHA256 matches this value.")
    parser.add_argument("--max_missing_keys", type=int, default=0)
    parser.add_argument("--max_unexpected_keys", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--eval_sat_channel", action="store_true")
    parser.add_argument("--eval_sat_scenarios", default="leo_clear_weak,leo_low_elev_weak,leo_rain_weak")
    parser.add_argument("--eval_sat_on", default="main")
    parser.add_argument("--sat_eval_max_batches", type=int, default=0)
    parser.add_argument("--sat_seed", type=int, default=2027)
    parser.add_argument("--sat_fs_hz", type=float, default=25e6)
    parser.add_argument("--sat_fc_hz", type=float, default=2.462e9)
    parser.add_argument("--wisig_pkl", default="")
    parser.add_argument("--wisig_out_len", type=int, default=0)
    parser.add_argument("--wisig_domain", default="")
    parser.add_argument("--max_samples_per_combo_test", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--model_size", default="")
    parser.add_argument("--model_variant", default="")
    parser.add_argument("--branch_ablation", default="")
    parser.add_argument("--domain_branch_ablation", default="")
    parser.add_argument("--sample_rate_hz", type=float, default=0.0)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = _load_checkpoint(args.ckpt)
    ckpt_sha = sha256_file(args.ckpt)
    identity = validate_checkpoint_identity(
        payload,
        checkpoint_path=args.ckpt,
        expected_run_name=args.expect_run_name,
        checkpoint_sha256=ckpt_sha,
        expected_sha256=args.expect_sha256,
    )
    context = build_wisig_context_from_checkpoint(payload, args)
    model = build_model_from_checkpoint_args(context, args, device)
    load_report = load_model_state(
        model,
        payload,
        max_missing=int(args.max_missing_keys),
        max_unexpected=int(args.max_unexpected_keys),
    )
    model.eval()

    val_loader = make_loader(
        context.val_ds,
        int(args.eval_batch_size),
        False,
        int(args.num_workers),
        device,
        False,
        int(args.prefetch_factor),
    )
    named_loaders = {
        name: make_loader(ds, int(args.eval_batch_size), False, int(args.num_workers), device, False, int(args.prefetch_factor))
        for name, ds in context.named_tests.items()
    }
    val_stats = evaluate_loader(model, val_loader, device, context.domain_label_map, max_batches=int(args.eval_max_batches))
    named_stats = evaluate_named_loaders(
        model,
        named_loaders,
        device,
        context.domain_label_map,
        max_batches=int(args.eval_max_batches),
    )
    test_stats = aggregate_named_stats(named_stats, _main_test_keys(named_stats))

    sat_stats = {}
    if bool(args.eval_sat_channel):
        scenarios = parse_sat_scenarios(args.eval_sat_scenarios)
        sat_stats = evaluate_sat_scenarios(
            model,
            named_loaders,
            device,
            context.domain_label_map,
            scenario_names=scenarios,
            args=args,
            max_batches=int(args.sat_eval_max_batches),
        )

    lines = [
        f"[CVS-EVAL] ckpt={args.ckpt}",
        f"[CVS-EVAL] run_name={identity.get('run_name')} sha256={identity.get('checkpoint_sha256')}",
        f"[CVS-EVAL] device={device} eval_max_batches={int(args.eval_max_batches)} sat_eval_max_batches={int(args.sat_eval_max_batches)}",
        f"[CVS-EVAL] missing_keys={load_report['missing_count']} unexpected_keys={load_report['unexpected_count']}",
        f"[CVS-EVAL] val tx_acc={float(val_stats.get('tx_acc', 0.0)):.6f} dom_acc={float(val_stats.get('dom_acc', 0.0)):.6f}",
        f"[CVS-EVAL] test_aggregate tx_acc={float(test_stats.get('tx_acc', 0.0)):.6f} dom_acc={float(test_stats.get('dom_acc', 0.0)):.6f}",
    ]
    lines.extend(format_named_test_lines(named_stats, context.named_test_meta))
    if sat_stats:
        lines.extend(format_sat_test_lines(sat_stats))
    text = "\n".join(lines) + "\n"
    print(text, end="", flush=True)

    result = {
        "checkpoint": str(args.ckpt),
        "checkpoint_identity": identity,
        "checkpoint_epoch": payload.get("epoch"),
        "load_report": load_report,
        "eval_args": vars(args),
        "split_info": context.split_info,
        "val": val_stats,
        "named_test": named_stats,
        "test": test_stats,
        "sat_test_named": sat_stats,
    }
    if args.output_txt:
        out_txt = Path(args.output_txt)
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text(text, encoding="utf-8")
    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
