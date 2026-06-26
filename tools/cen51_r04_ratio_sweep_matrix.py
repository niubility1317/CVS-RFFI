#!/usr/bin/env python3
"""Generate a CEN51_R04 WiSig train-ratio sweep launcher.

This is a centralized ratio sweep, not a few-shot cap experiment. The only
training-control field changed across jobs is ``wisig_train_ratio``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from cen51_lowshot_config_search import arg_pairs
from cen51_r04_config import cen51_r04_ratio_params


REMOTE_ROOT = "/home/szu2070436088/2510044040/CV-SincNet"
REMOTE_PYTHON = "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
RATIOS: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


@dataclass(frozen=True)
class RatioJob:
    ratio: float
    ratio_tag: str
    run_name: str
    gpu: int
    seed: int


def q(value: object) -> str:
    return shlex.quote(str(value))


def ratio_tag(ratio: float) -> str:
    return f"r{int(round(float(ratio) * 100)):03d}"


def make_jobs(seed: int = 1337) -> List[RatioJob]:
    return [
        RatioJob(
            ratio=float(ratio),
            ratio_tag=ratio_tag(float(ratio)),
            run_name=f"CEN51_R04_RATIO_{ratio_tag(float(ratio)).upper()}_seed{seed}",
            gpu=index,
            seed=int(seed),
        )
        for index, ratio in enumerate(RATIOS)
    ]


def job_params(job: RatioJob, run_id: str) -> Dict[str, object]:
    run_dir = f"{REMOTE_ROOT}/runs/{run_id}/{job.run_name}"
    params = cen51_r04_ratio_params(seed=job.seed)
    params.update(
        {
            "wisig_pkl": f"{REMOTE_ROOT}/Dataset_WigSig/ManySig.pkl",
            "wisig_train_ratio": job.ratio,
            "output_dir": run_dir,
            "run_name": job.run_name,
            "latest_save_path": f"{run_dir}/latest_model.pth",
            "best_save_path": f"{run_dir}/best_val_model.pth",
            "best_primary_save_path": f"{run_dir}/best_primary_ood_model.pth",
            "best_unseen_day_unseen_rx_save_path": f"{run_dir}/best_strict_udu_model.pth",
            "best_worst_rx_save_path": f"{run_dir}/best_worst_rx_model.pth",
            "ema_save_path": f"{run_dir}/ema_model.pth",
            "swa_save_path": f"{run_dir}/swa_model.pth",
            "swad_save_path": f"{run_dir}/swad_model.pth",
        }
    )
    return params


def command_for(job: RatioJob, run_id: str) -> str:
    args = [
        "env",
        f"CUDA_VISIBLE_DEVICES={job.gpu}",
        f"PYTHONPATH={REMOTE_ROOT}/code:{REMOTE_ROOT}/tools:{REMOTE_ROOT}",
        REMOTE_PYTHON,
        "-u",
        f"{REMOTE_ROOT}/code/train.py",
        *arg_pairs(job_params(job, run_id)),
    ]
    return " ".join(q(item) for item in args)


def render_launcher(run_id: str, jobs: Sequence[RatioJob]) -> str:
    lines: List[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'ROOT="${{ROOT:-{REMOTE_ROOT}}}"',
        f'PYTHON="${{PYTHON:-{REMOTE_PYTHON}}}"',
        'RUN_ID="${RUN_ID:-' + run_id + '}"',
        'LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"',
        'RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"',
        'DRY_RUN="${DRY_RUN:-0}"',
        'SKIP_DONE="${SKIP_DONE:-1}"',
        "",
        'while [ "$#" -gt 0 ]; do',
        '  case "$1" in',
        '    --dry-run) DRY_RUN=1; shift ;;',
        '    --no-skip-done) SKIP_DONE=0; shift ;;',
        '    *) echo "[ERROR] unknown argument: $1" >&2; exit 2 ;;',
        "  esac",
        "done",
        "",
        'export PYTHONPATH="${ROOT}/code:${ROOT}/tools:${ROOT}:${PYTHONPATH:-}"',
        'mkdir -p "${LOG_ROOT}" "${RUNS_ROOT}"',
        'MANIFEST="${RUNS_ROOT}/manifest.tsv"',
        'printf "ratio\\trun_name\\tgpu\\tlog_file\\toutput_dir\\tcommand\\n" > "${MANIFEST}"',
        "",
        "run_job() {",
        "  local ratio=\"$1\"",
        "  local run_name=\"$2\"",
        "  local gpu=\"$3\"",
        "  local cmd=\"$4\"",
        '  local log_file="${LOG_ROOT}/${run_name}.log"',
        '  local out_dir="${RUNS_ROOT}/${run_name}"',
        '  mkdir -p "${out_dir}"',
        '  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" "${ratio}" "${run_name}" "${gpu}" "${log_file}" "${out_dir}" "${cmd}" >> "${MANIFEST}"',
        '  if [[ "${SKIP_DONE}" == "1" && -f "${out_dir}/latest_model.pth" ]]; then',
        '    echo "[CEN51-R04-RATIO] skip_done ratio=${ratio} run=${run_name}"',
        "    return 0",
        "  fi",
        '  echo "[CEN51-R04-RATIO] launch ratio=${ratio} gpu=${gpu} run=${run_name} dry_run=${DRY_RUN}"',
        '  echo "[CEN51-R04-RATIO-CMD] ${cmd}"',
        '  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi',
        '  bash -lc "${cmd}" > "${log_file}" 2>&1 &',
        '  echo "${ratio}\t${run_name}\t${gpu}\t$!\t${log_file}" >> "${LOG_ROOT}/launch_pids.tsv"',
        "}",
        "",
        'cd "${ROOT}"',
        ': > "${LOG_ROOT}/launch_pids.tsv"',
    ]
    for job in jobs:
        lines.append(
            "run_job "
            + " ".join(
                [
                    q(f"{job.ratio:.1f}"),
                    q(job.run_name),
                    q(job.gpu),
                    q(command_for(job, run_id)),
                ]
            )
        )
    lines.extend(
        [
            'echo "[CEN51-R04-RATIO] submitted ${RUN_ID}"',
            'if [[ "${DRY_RUN}" != "1" ]]; then',
            '  echo "[CEN51-R04-RATIO] launch_pids=${LOG_ROOT}/launch_pids.tsv"',
            "fi",
            "",
        ]
    )
    return "\n".join(lines)


def render_report(run_id: str, jobs: Sequence[RatioJob], script_path: Path, matrix_path: Path) -> str:
    rows = [
        "| ratio | run name | GPU | seed |",
        "|---:|---|---:|---:|",
    ]
    for job in jobs:
        rows.append(f"| {job.ratio:.1f} | `{job.run_name}` | {job.gpu} | {job.seed} |")
    return "\n".join(
        [
            f"# {run_id}",
            "",
            "## Objective",
            "",
            "Run the unchanged centralized CEN51_R04 ratio path at WiSig train ratios 0.1 through 0.8. This is a data-quantity scaling experiment, not a few-shot cap experiment.",
            "",
            "## Hypothesis",
            "",
            "Increasing the available train ratio should improve source validation and may improve strict unseen-day/unseen-receiver performance, but the curve can expose whether additional source data increases receiver/day/channel shortcut learning. The sweep keeps the CEN51_R04 architecture, satellite schedule, MixStyle, GroupCE, prototype, SupCon, Fishr, SWAD, and checkpoint policy fixed.",
            "",
            "## Candidate Matrix",
            "",
            *rows,
            "",
            "## Configuration Contract",
            "",
            "- centralized `train.py` only; no federated training.",
            "- no `--wisig_max_train_per_combo` and no `--wisig_train_shots_per_class`.",
            "- `wisig_protocol=cvs_day_rx`, train days `0,1`, test days `2,3`, train receivers `0..6`, test receivers `7..11`.",
            "- satellite evaluation uses `clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit` on `test_unseen_day_unseen_rx`.",
            "- each job uses one GPU; ratios map to GPUs 0..7.",
            "",
            "## Metrics To Watch",
            "",
            "- validation TX accuracy and best-vs-final rollback.",
            "- clean `test_unseen_day_unseen_rx` strict UDU.",
            "- worst receiver floor and best-primary OOD score.",
            "- five-scenario satellite strict UDU and satellite floor.",
            "- whether higher train ratios improve clean strict monotonically or only source validation.",
            "",
            "## Artifacts",
            "",
            f"- local launcher: `{script_path}`",
            f"- local matrix: `{matrix_path}`",
            f"- remote logs: `{REMOTE_ROOT}/logs/{run_id}`",
            f"- remote runs: `{REMOTE_ROOT}/runs/{run_id}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "automation_reports" / "CV-SincNet")
    parser.add_argument("--scripts-dir", type=Path, default=REPO_ROOT / "code" / "scripts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"cen51_r04_ratio_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    jobs = make_jobs(seed=args.seed)
    report_dir = args.output_root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    args.scripts_dir.mkdir(parents=True, exist_ok=True)

    script_path = args.scripts_dir / f"launch_{run_id}.sh"
    matrix_path = report_dir / "matrix.json"
    report_path = report_dir / "report.md"
    manifest_path = report_dir / "manifest.tsv"

    script_path.write_text(render_launcher(run_id, jobs), encoding="utf-8", newline="\n")
    matrix_path.write_text(json.dumps([asdict(job) for job in jobs], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(run_id, jobs, script_path, matrix_path), encoding="utf-8", newline="\n")
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ratio\tratio_tag\trun_name\tgpu\tseed\n")
        for job in jobs:
            handle.write(f"{job.ratio:.1f}\t{job.ratio_tag}\t{job.run_name}\t{job.gpu}\t{job.seed}\n")

    print(f"[CEN51-R04-RATIO] run_id={run_id}")
    print(f"[CEN51-R04-RATIO] launcher={script_path}")
    print(f"[CEN51-R04-RATIO] report={report_path}")
    print(f"[CEN51-R04-RATIO] jobs={len(jobs)} ratios={','.join(f'{job.ratio:.1f}' for job in jobs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
