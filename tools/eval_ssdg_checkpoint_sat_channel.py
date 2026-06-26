from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping


def _add_project_paths() -> None:
    here = Path(__file__).resolve()
    root = here.parents[1]
    candidates = [root / "code", root]
    for path in candidates:
        if path.exists():
            sys.path.insert(0, str(path))


_add_project_paths()

from SSDG.train_ssdg import (  # noqa: E402
    _aggregate_main_test,
    _apply_model_cli_args,
    _build_ssdg_wisig_data,
    _evaluate,
    _evaluate_sat_if_enabled,
    _format_named_test_lines,
    _format_sat_test_lines,
    build_arg_parser,
)
from post_stage_common import build_baseline_model, load_checkpoint, merge_checkpoint_args, resolve_device, set_seed  # noqa: E402
from training_controls import parse_sat_scenarios  # noqa: E402


def _jsonable(value: Any) -> Any:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser()
    parser.description = "Evaluate an SSDG checkpoint on clean WiSig tests and optional satellite-channel overlays."
    parser.add_argument("--ckpt", type=str, required=True, help="SSDG checkpoint path to evaluate.")
    parser.add_argument("--output_json", type=str, default="", help="Optional JSON result path.")
    parser.add_argument("--output_txt", type=str, default="", help="Optional text summary path.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    set_seed(int(args.seed))
    args.eval_sat_scenario_list = parse_sat_scenarios(args.eval_sat_scenarios) if bool(args.eval_sat_channel) else []

    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.ckpt, device)

    data_ctx = _build_ssdg_wisig_data(args, device)
    model_args = merge_checkpoint_args(
        ckpt,
        args,
        input_len=int(data_ctx["input_len"]),
        num_domains=int(data_ctx["num_domains"]),
    )
    model_args = _apply_model_cli_args(model_args, args)
    model = build_baseline_model(model_args, device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    val_stats, named_stats = _evaluate(model, data_ctx, device, int(args.eval_max_batches))
    test_stats = _aggregate_main_test(named_stats, str(args.dataset))
    sat_stats = _evaluate_sat_if_enabled(model, data_ctx, device, args)

    lines = [
        f"[SSDG-EVAL] ckpt={args.ckpt}",
        f"[SSDG-EVAL] device={device} eval_max_batches={int(args.eval_max_batches)} sat_eval_max_batches={int(args.sat_eval_max_batches)}",
        f"[SSDG-EVAL] missing_keys={len(missing)} unexpected_keys={len(unexpected)}",
        f"[SSDG-EVAL] val tx_acc={float(val_stats.get('tx_acc', 0.0)):.6f} dom_acc={float(val_stats.get('dom_acc', 0.0)):.6f}",
        f"[SSDG-EVAL] test_aggregate tx_acc={float(test_stats.get('tx_acc', 0.0)):.6f} dom_acc={float(test_stats.get('dom_acc', 0.0)):.6f}",
    ]
    lines.extend(_format_named_test_lines(named_stats, data_ctx.get("named_test_meta", {})))
    if sat_stats:
        lines.extend(_format_sat_test_lines(sat_stats))
    text = "\n".join(lines) + "\n"
    print(text, end="", flush=True)

    payload: Dict[str, Any] = {
        "checkpoint": str(args.ckpt),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_metric": ckpt.get("best_metric"),
        "checkpoint_best_score": ckpt.get("best_score"),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "eval_args": vars(args),
        "model_args": vars(model_args),
        "split_info": data_ctx.get("split_info"),
        "checkpoint_split_info": ckpt.get("split_info"),
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
        out_json.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
