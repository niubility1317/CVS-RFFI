from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_CODE = PROJECT_ROOT / "code"
if str(PROJECT_CODE) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE))

from cvsrffi.eval import MAIN_SAT_EVAL_ON_NAMES, apply_sat_channel_for_scenario  # noqa: E402
from cvsrffi.tensors import make_torch_generator, unpack_batch  # noqa: E402
from dataset_wisig import load_wisig_compact_pkl  # noqa: E402
from scripts.eval_ssdg_sat_per_rx import _build_exact_ssdg_context  # noqa: E402
from training_controls import parse_sat_scenarios  # noqa: E402

from paper_reproduction.scripts.evaluate_cvs_phase1_detailed import (  # noqa: E402
    _label,
    _write_csv,
    aggregate_score_rows,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def run(args_cli: argparse.Namespace) -> dict[str, Any]:
    scenarios = tuple(parse_sat_scenarios(args_cli.scenarios))
    expected_scenarios = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    if scenarios != expected_scenarios:
        raise ValueError(f"formal Phase1 comparison requires {expected_scenarios}, got {scenarios}")
    device = torch.device(
        args_cli.device
        if torch.cuda.is_available() or not str(args_cli.device).startswith("cuda")
        else "cpu"
    )
    checkpoint_path = Path(args_cli.ckpt)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, named_loaders, named_meta, _domain_map, split_info, model_args, missing, unexpected = (
        _build_exact_ssdg_context(checkpoint, args_cli, device)
    )
    if missing or unexpected:
        raise ValueError(
            f"strict checkpoint reconstruction failed: missing={len(missing)} unexpected={len(unexpected)}"
        )
    selected = [name for name in MAIN_SAT_EVAL_ON_NAMES if name in named_loaders]
    if selected != list(MAIN_SAT_EVAL_ON_NAMES):
        raise ValueError(f"formal main OOD loaders missing: {selected}")
    dataset = load_wisig_compact_pkl(str(model_args.wisig_pkl))
    tx_labels = [str(value) for value in dataset["tx_list"]]
    rx_labels = [str(value) for value in dataset["rx_list"]]
    day_labels = [str(value) for value in dataset["capture_date_list"]]
    model.eval()
    score_rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        for split_index, split in enumerate(selected):
            generator = make_torch_generator(
                device, int(args_cli.sat_seed) + scenario_index * 1009 + split_index * 97
            )
            for batch in named_loaders[split]:
                x, y, extra = unpack_batch(batch)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                x, _ = apply_sat_channel_for_scenario(
                    x, scenario, model_args, gen=generator, return_meta=False
                )
                output = model(x, y_tx=None, grl_lambda=1.0, return_aux=True)
                probabilities = torch.softmax(output["tx_logits"], dim=1)
                confidence, predicted = probabilities.max(dim=1)
                rx_values = torch.as_tensor(extra["rx_i"]).long()
                day_values = torch.as_tensor(extra["day_i"]).long()
                sig_values = torch.as_tensor(extra["sig_i"]).long()
                for index in range(int(predicted.numel())):
                    truth_i = int(y[index].cpu())
                    prediction_i = int(predicted[index].cpu())
                    receiver_i = int(rx_values[index])
                    day_i = int(day_values[index])
                    sig_i = int(sig_values[index])
                    score_rows.append(
                        {
                            "sample_id": f"rx{receiver_i}:day{day_i}:tx{truth_i}:sig{sig_i}",
                            "scenario": scenario,
                            "split": split,
                            "receiver_index": receiver_i,
                            "receiver_label": _label(rx_labels, receiver_i, "rx"),
                            "transmitter_index": truth_i,
                            "transmitter_label": _label(tx_labels, truth_i, "tx"),
                            "day_index": day_i,
                            "day_label": _label(day_labels, day_i, "day"),
                            "predicted_transmitter_index": prediction_i,
                            "predicted_transmitter_label": _label(tx_labels, prediction_i, "tx"),
                            "confidence": float(confidence[index].cpu()),
                            "correct": int(prediction_i == truth_i),
                        }
                    )
    detailed_rows = aggregate_score_rows(score_rows)
    if len(score_rows) != 3 * 204000:
        raise ValueError(f"expected 612000 detailed score rows, got {len(score_rows)}")
    if not all(math.isfinite(float(row["confidence"])) for row in score_rows):
        raise FloatingPointError("non-finite confidence in CVS detailed score rows")

    scenario_metrics = {}
    for scenario in scenarios:
        rows = [row for row in score_rows if row["scenario"] == scenario]
        scenario_metrics[scenario] = {
            "accuracy": sum(int(row["correct"]) for row in rows) / len(rows),
            "correct_count": sum(int(row["correct"]) for row in rows),
            "sample_count": len(rows),
        }
    terminal_status = {}
    if args_cli.terminal_status:
        terminal_status = json.loads(Path(args_cli.terminal_status).read_text(encoding="utf-8"))
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "score_table.csv", score_rows)
    _write_csv(output_dir / "detailed_metrics.csv", detailed_rows)
    (output_dir / "detailed_metrics.json").write_text(
        json.dumps(detailed_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "schema": "cvs_phase1_ssdg_detailed_satellite_evaluation_v1",
        "method": str(args_cli.method_label),
        "seed": int(getattr(model_args, "seed", -1)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "scenarios": scenario_metrics,
        "score_row_count": len(score_rows),
        "detailed_row_count": len(detailed_rows),
        "terminal_verdict": terminal_status.get("verdict", terminal_status.get("status", "")),
        "claim_boundary": str(args_cli.claim_boundary),
    }
    manifest = {
        "schema": "cvs_phase1_ssdg_detailed_split_manifest_v1",
        "method": str(args_cli.method_label),
        "formal_satellite_scenarios": list(scenarios),
        "selected_test_splits": selected,
        "receiver_labels": rx_labels,
        "transmitter_labels": tx_labels,
        "day_labels": day_labels,
        "split_info": split_info,
        "all_tests_satellite_augmented": True,
        "clean_control_in_formal_result": False,
        "checkpoint_load_strict": True,
    }
    resolved = {
        **vars(args_cli),
        "checkpoint_args": dict(checkpoint.get("args") or {}),
        "terminal_status": terminal_status,
    }
    for name, payload in (
        ("metrics.json", metrics),
        ("split_manifest.json", manifest),
        ("resolved_config.json", resolved),
    ):
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Detailed satellite evaluation for CVS SSDG checkpoints")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--terminal-status", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-label", default="cvs_jointp0_j5")
    parser.add_argument(
        "--claim-boundary",
        default="raw_phase1_comparison_result_non_promotable_under_internal_cvs_guard",
    )
    parser.add_argument("--scenarios", default="leo_clear_weak,leo_low_elev_weak,leo_rain_weak")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--sat-seed", type=int, default=2027)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
