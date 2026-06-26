#!/usr/bin/env python
"""Generate a WiSig-subset transfer validation matrix for K-segmented DM control.

The completed domain-metric controller run showed that useful control is
K-segmented, not one global knob. This launcher changes the WiSig day/RX subset
and keeps the comparison structure explicit:

1. replay the old per-K anchor,
2. replay the learned per-K segmented action,
3. test a neighboring per-K guard,
4. test a shared global-controller surrogate,
5. test a wrong/negative boundary,
6. add seed or ablation checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import cen51_domain_metric_controller_matrix as dmctrl  # noqa: E402
import cen51_safd_anchor_fit_matrix as anchorfit  # noqa: E402


RUN_PREFIX = "CEN51_WSUB_KSEG"
SUBSET_ID = "wisig_d12_t03_rx23458910_t016711"
SUBSET_ARGS = {
    "--wisig_protocol": "cvs_day_rx",
    "--wisig_domain": "rx_day",
    "--wisig_train_ratio": "0.1",
    "--wisig_train_days": "1,2",
    "--wisig_test_days": "0,3",
    "--wisig_train_rxs": "2,3,4,5,8,9,10",
    "--wisig_test_rxs": "0,1,6,7,11",
}
SHOT_ORDER = [5, 10, 20, 30, 50, 100]
GPU_K_ORDERS = {
    0: [5, 30, 100, 10, 50, 20],
    1: [10, 50, 20, 5, 30, 100],
    2: [20, 5, 50, 100, 10, 30],
    3: [30, 100, 10, 50, 20, 5],
    4: [50, 20, 5, 30, 100, 10],
    5: [100, 10, 30, 20, 5, 50],
    6: [5, 100, 50, 20, 30, 10],
    7: [10, 30, 20, 100, 50, 5],
}


BASE_FD_CIDS = {
    5: "FS005_BEST_HINGE6_SATMIN_2030",
    10: "FS010_BEST_IDFIRST_LATE_2030",
    20: "FS020_BEST_RIEIFD_LIGHT_2028",
    30: "FS030_BEST_RXFLOOR_CAP_2029",
    50: "FS050_BEST_CAP_RELAX_1337",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "automation_reports" / "CV-SincNet")
    parser.add_argument("--scripts-dir", type=Path, default=REPO_ROOT / "code" / "scripts")
    parser.add_argument("--max-active-per-gpu", type=int, default=2)
    parser.add_argument("--scheduler-hours", type=float, default=14.0)
    return parser.parse_args()


def set_subset_args(args: list[str]) -> list[str]:
    out = list(args)
    for flag, value in SUBSET_ARGS.items():
        out = anchorfit.set_arg(out, flag, value)
    return out


def normalize_exp(exp: anchorfit.Experiment, *, gpu: int) -> anchorfit.Experiment:
    return replace(
        exp,
        run_name=f"{RUN_PREFIX}_{exp.cid}",
        gpu=gpu,
        args=set_subset_args(exp.args),
    )


def controller_pool_lookup() -> dict[str, anchorfit.Experiment]:
    pools = dmctrl.make_pools()
    lookup: dict[str, anchorfit.Experiment] = {}
    for queue in pools.values():
        for exp in list(queue):
            lookup[exp.cid] = exp
    return lookup


def fd_global(k: int, *, comp: dict[str, object]) -> anchorfit.Experiment:
    return dmctrl.fd_scale(
        comp[BASE_FD_CIDS[k]],
        cid=f"K{k:03d}_U01_GLOBAL_P090_S120_R108",
        axis="global_unified",
        action="same_pressure_sat_rx",
        pressure=0.90,
        sat=1.20,
        rx=1.08,
        hypothesis=(
            f"K{k} uses the same global controller surrogate as other K values; "
            "it should not consistently beat the K-segmented action."
        ),
        success_gate="If this wins broadly, the K-segmented conclusion is rejected.",
    )


def k100_global() -> anchorfit.Experiment:
    return dmctrl.lac_exp(
        "K100_U01_GLOBAL_P090_S120_R108",
        "global_unified",
        "same_pressure_sat_rx",
        "K100 receives the same global-controller surrogate instead of the K100 guard.",
        "If this beats K100_B03 on strict/floors, segmentation needs revision.",
        updates={
            "lambda_adv": 0.34,
            "lambda_cons": 0.060,
            "lambda_group_ce": 0.070,
            "lambda_proto": 0.010,
            "lambda_supcon_id": 0.012,
            "lambda_fishr": 0.0010,
            "group_ce_top_frac": 0.28,
            "groupdro_cap": 0.58,
            "sat_view_prob": 0.60,
            "concat_sat_ce_weight": 0.72,
            "sat_view_schedule": f"1@0.45:{dmctrl.LIGHT_SAT};130@0.62:{dmctrl.ALL_SAT}",
        },
    )


def make_variants_by_k() -> dict[int, list[anchorfit.Experiment]]:
    pool = controller_pool_lookup()
    comp = dmctrl.comp_by_cid()

    variants = {
        5: [
            pool["K005_A00_ANCHOR2030"],
            pool["K005_S02_SAT_P093"],
            pool["K005_B01_RX_SAT_BAL"],
            fd_global(5, comp=comp),
            pool["K005_R02_RX_STRONG"],
            pool["K005_S04_SAT_P093_SEED2029"],
            pool["K005_N01_NOSAT_NEG"],
            pool["K005_C01_PRESS_CLAMP"],
        ],
        10: [
            pool["K010_A00_ANCHOR2030"],
            pool["K010_S02_SAT122"],
            pool["K010_B01_RXSAT"],
            fd_global(10, comp=comp),
            pool["K010_C02_CLAMP082"],
            pool["K010_B02_RXSAT_SEED1337"],
            pool["K010_N01_NOSAT_NEG"],
            pool["K010_S03_SAT135"],
        ],
        20: [
            pool["K020_A00_ANCHOR2028"],
            pool["K020_S01_SAT115"],
            pool["K020_B01_BAL"],
            fd_global(20, comp=comp),
            pool["K020_R02_RX112_NEG"],
            pool["K020_A01_SEED1337"],
            pool["K020_C01_CLAMP085"],
            pool["K020_S02_SAT135"],
        ],
        30: [
            pool["K030_A00_ANCHOR2029"],
            pool["K030_C03_CLAMP092"],
            pool["K030_B07_CLAMP_RXSAT"],
            fd_global(30, comp=comp),
            pool["K030_S02_SAT200_NEG"],
            pool["K030_B04_CLAMP_SEED2028"],
            pool["K030_R01_RX108"],
            pool["K030_C04_EARLY_SWAD"],
        ],
        50: [
            pool["K050_A00_ANCHOR1337"],
            pool["K050_R02_RX110"],
            pool["K050_B01_RXSAT"],
            fd_global(50, comp=comp),
            pool["K050_S05_SAT140_NEG"],
            pool["K050_B03_RX_SEED2028"],
            pool["K050_S02_SAT120"],
            pool["K050_C02_CLAMP_RX"],
        ],
        100: [
            pool["K100_A00_ANCHOR1337"],
            pool["K100_B03_SAT_FLOOR_GUARD"],
            pool["K100_B02_STRICT_FIRST"],
            k100_global(),
            pool["K100_S04_BOOST_NEG"],
            pool["K100_S05_CONSERVE050_SEED2028"],
            pool["K100_C04_GLOBAL_CLAMP_STRONG"],
            pool["K100_B05_CLAMP_RX_SEED2028"],
        ],
    }
    for shot, rows in variants.items():
        if len(rows) != 8:
            raise AssertionError(f"K{shot} expected 8 variants, got {len(rows)}")
        seen = [row.cid for row in rows]
        if len(seen) != len(set(seen)):
            raise AssertionError(f"duplicate K{shot} variants: {seen}")
    return variants


def assign_rows() -> list[anchorfit.Experiment]:
    variants = make_variants_by_k()
    buckets: dict[int, list[anchorfit.Experiment]] = defaultdict(list)
    for shot_index, shot in enumerate(SHOT_ORDER):
        for variant_index, exp in enumerate(variants[shot]):
            gpu = (variant_index + shot_index) % 8
            buckets[gpu].append(normalize_exp(exp, gpu=gpu))

    rows: list[anchorfit.Experiment] = []
    for gpu in range(8):
        by_shot = {row.shot: row for row in buckets[gpu]}
        if sorted(by_shot) != SHOT_ORDER:
            raise AssertionError(f"GPU{gpu} does not have one candidate per K: {sorted(by_shot)}")
        for shot in GPU_K_ORDERS[gpu]:
            rows.append(by_shot[shot])

    run_names = [row.run_name for row in rows]
    if len(rows) != 48:
        raise AssertionError(f"expected 48 candidates, got {len(rows)}")
    if len(run_names) != len(set(run_names)):
        duplicates = [name for name, count in Counter(run_names).items() if count > 1]
        raise AssertionError(f"duplicate run names: {duplicates}")
    return rows


def write_manifest(path: Path, rows: Sequence[anchorfit.Experiment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "cid",
            "run_name",
            "shot",
            "gpu",
            "seed",
            "axis",
            "action",
            "anchor_name",
            "target_strict",
            "target_overall",
            "hypothesis",
            "success_gate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            data.pop("args", None)
            writer.writerow(data)


def render_launcher(run_id: str, rows: Sequence[anchorfit.Experiment], max_active_per_gpu: int, scheduler_hours: float) -> str:
    return dmctrl.render_launcher(run_id, rows, max_active_per_gpu, scheduler_hours).replace(
        "[CEN51-DMCTRL]", "[CEN51-WSUB-KSEG]"
    )


def render_report(
    run_id: str,
    rows: Sequence[anchorfit.Experiment],
    script_path: Path,
    matrix_path: Path,
    manifest_path: Path,
    max_active_per_gpu: int,
    scheduler_hours: float,
) -> str:
    gpu_counts = Counter(row.gpu for row in rows)
    shot_counts = Counter(row.shot for row in rows)
    axis_counts = Counter(row.axis for row in rows)
    gpu_shots: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        gpu_shots[row.gpu].append(row.shot)

    table = [
        "| ID | K | GPU | axis/action | target strict/overall | success gate |",
        "|---|---:|---:|---|---:|---|",
    ]
    for row in rows:
        table.append(
            f"| `{row.cid}` | {row.shot} | {row.gpu} | {row.axis}/{row.action} | "
            f"{row.target_strict:.2f}/{row.target_overall:.2f} | {row.success_gate} |"
        )

    return "\n".join(
        [
            f"# {run_id}",
            "",
            "## 目标",
            "",
            "验证域指标控制器在新的 WiSig day/RX 子集上是否仍然有效，并检验上一批结论："
            "控制律必须按 K 分段，而不能使用全局统一控制器。",
            "",
            "## 新 WiSig 子集",
            "",
            f"- subset id: `{SUBSET_ID}`",
            "- train days: `1,2`; test days: `0,3`",
            "- train RXs: `2,3,4,5,8,9,10`; test RXs: `0,1,6,7,11`",
            "- hard constraint: `--wisig_train_ratio 0.1`",
            "- strict target remains `test_unseen_day_unseen_rx` under disjoint test days and test RXs.",
            "",
            "## 假设",
            "",
            "把每个 K 的训练看成受约束的多目标优化：先最大化 strict UDU 和 overall，"
            "再用 receiver floor、satellite floor、late rollback 约束决定调哪个损失族。"
            "如果域指标是真有用的，新子集上应出现同方向的分段收益：K5/K10 主要靠轻量 sat gate，"
            "K30 靠压力夹紧，K50 靠 RX/group-DRO floor，K100 靠身份保护和保守 sat guard。",
            "",
            "## 对照结构",
            "",
            "- `anchor_replay`: 原图/原批次强锚点，检验换子集后的基线。",
            "- `sat_gate`, `rx_floor`, `pressure_clamp`, `balanced_metric`, `identity_guard`: 上一批按 K 选择的分段动作。",
            "- `global_unified`: 所有 K 使用同一个压力/sat/RX 方向的统一控制器替身，用来反证全局控制。",
            "- `paired_control` 和强边界候选：no-sat、过强 RX、过强 sat 或过强 clamp，检验指标是否能阻止错误动作。",
            "",
            "## 调度",
            "",
            f"- candidates: {len(rows)}",
            f"- max active per GPU: {max_active_per_gpu}",
            f"- scheduler launch window: {scheduler_hours:.1f} hours",
            "- each GPU has one candidate for every K, with small and large shots interleaved.",
            f"- GPU candidate counts: `{json.dumps(dict(sorted(gpu_counts.items())), ensure_ascii=False)}`",
            f"- shot counts: `{json.dumps(dict(sorted(shot_counts.items())), ensure_ascii=False)}`",
            f"- axis counts: `{json.dumps(dict(sorted(axis_counts.items())), ensure_ascii=False)}`",
            f"- per-GPU K order: `{json.dumps({gpu: vals for gpu, vals in sorted(gpu_shots.items())}, ensure_ascii=False)}`",
            "",
            "## 成功判据",
            "",
            "- 每个 K 先比较 segmented action 与 anchor；若 strict/primary 不低于 anchor 且 floor 改善，判为有效迁移。",
            "- `global_unified` 若只在少数 K 有效而在低 K 或 K100 失效，则支持 K 分段结论。",
            "- 负控若击败 segmented action，需要回看日志中的 `[CONFIG-LOSS]`, `[CONFIG-SAT]`, `[LOSS-*]`, late-drop 和 floor 指标，不直接推广。",
            "- 训练完成后必须做 full-log parse；不能只按最后一轮或单一 strict 排名决策。",
            "",
            "## 候选矩阵",
            "",
            *table,
            "",
            "## 完成后解析",
            "",
            "```powershell",
            "conda activate ssr-gpu",
            f"python tools\\cen51_domain_metric_full_log_analysis.py --log-dir <local-log-dir> --matrix-json {matrix_path} --out-dir analysis_tmp\\{run_id}\\full_log_analysis",
            f"python tools\\cen51_fewshot_stability_validator.py --log-dir <local-log-dir> --matrix-json {matrix_path} --out-dir analysis_tmp\\{run_id}\\stability_validation --late-window 30 --no-fail",
            "```",
            "",
            "## 本地/远端路径",
            "",
            f"- launcher: `{script_path}`",
            f"- matrix: `{matrix_path}`",
            f"- manifest: `{manifest_path}`",
            f"- remote logs: `{anchorfit.REMOTE_ROOT}/logs/{run_id}`",
            f"- remote runs: `{anchorfit.REMOTE_ROOT}/runs/{run_id}`",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"cen51_wisig_subset_kseg_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rows = assign_rows()

    report_dir = args.output_root / run_id
    script_path = args.scripts_dir / f"launch_{run_id}.sh"
    matrix_path = report_dir / "matrix.json"
    manifest_path = report_dir / "manifest.tsv"
    report_path = report_dir / "report.md"

    report_dir.mkdir(parents=True, exist_ok=True)
    args.scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path.write_text(render_launcher(run_id, rows, args.max_active_per_gpu, args.scheduler_hours), encoding="utf-8", newline="\n")
    matrix_path.write_text(json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manifest(manifest_path, rows)
    report_path.write_text(
        render_report(run_id, rows, script_path, matrix_path, manifest_path, args.max_active_per_gpu, args.scheduler_hours),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "subset_id": SUBSET_ID,
                "candidates": len(rows),
                "launcher": str(script_path),
                "matrix": str(matrix_path),
                "manifest": str(manifest_path),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
