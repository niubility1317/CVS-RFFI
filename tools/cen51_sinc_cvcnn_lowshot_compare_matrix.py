#!/usr/bin/env python
"""Generate Sinc-CVCNN low-shot comparison experiments.

The matrix compares a lighter CVCNN family against the current physics-aware
CVS backbone under the same WiSig CVS split:

* `sinc_cvcnn`: CVCNN with a SincConv first layer.
* `cvcnn`: matched plain CVCNN control.
* direct low-shot, staged K-shot anchor -> 0.1 expansion, and direct 0.1.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/home/szu2070436088/2510044040/CV-SincNet"
DEFAULT_RUN_ID = "cen51_sinc_cvcnn_lowshot_compare_20260609_204500"
ALL_SAT = "clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"
LIGHT_SAT = "clear_leo,mixed_orbit"


@dataclass(frozen=True)
class Candidate:
    cid: str
    run_name: str
    arch_family: str
    mode: str
    gpu: int
    seed: int
    shots: int
    stage2_kind: str
    cap_strategy: str
    hypothesis: str


def make_candidates() -> list[Candidate]:
    rows = [
        # Direct low-shot: isolate architecture at K-shot, no staged warm-start.
        ("SCV_K5_CLEAN_2028", "sinc_cvcnn", "direct_lowshot", 0, 2028, 5, "clean", "random", "Sinc-CVCNN 5-shot clean low-shot control."),
        ("BCV_K5_CLEAN_2028", "cvcnn", "direct_lowshot", 1, 2028, 5, "clean", "random", "Plain CVCNN 5-shot control; isolates SincConv stem effect."),
        ("SCV_K10_CLEAN_2028", "sinc_cvcnn", "direct_lowshot", 2, 2028, 10, "clean", "random", "Sinc-CVCNN 10-shot clean low-shot control."),
        ("BCV_K10_CLEAN_2028", "cvcnn", "direct_lowshot", 3, 2028, 10, "clean", "random", "Plain CVCNN 10-shot control; isolates SincConv stem effect."),
        ("SCV_K20_LATESAT_2028", "sinc_cvcnn", "direct_lowshot", 4, 2028, 20, "late_sat", "random", "Sinc-CVCNN 20-shot with delayed full-DG satellite."),
        ("BCV_K20_LATESAT_2028", "cvcnn", "direct_lowshot", 5, 2028, 20, "late_sat", "random", "Plain CVCNN 20-shot delayed satellite control."),

        # Staged Sinc-CVCNN: test whether low-capacity SincConv anchor scales to 0.1.
        ("SCV_S5_CLEAN_2028", "sinc_cvcnn", "staged", 6, 2028, 5, "clean", "random", "5-shot Sinc-CVCNN clean anchor then 0.1 clean expansion."),
        ("SCV_S5_LATESAT_2028", "sinc_cvcnn", "staged", 7, 2028, 5, "late_sat", "random", "5-shot Sinc-CVCNN anchor then delayed full-DG satellite expansion."),
        ("SCV_S10_CLEAN_2028", "sinc_cvcnn", "staged", 0, 2028, 10, "clean", "random", "10-shot Sinc-CVCNN clean anchor then 0.1 clean expansion."),
        ("SCV_S10_LATESAT_2028", "sinc_cvcnn", "staged", 1, 2028, 10, "late_sat", "random", "10-shot Sinc-CVCNN anchor then delayed satellite expansion."),
        ("SCV_S20_LATESAT_2028", "sinc_cvcnn", "staged", 2, 2028, 20, "late_sat", "random", "20-shot Sinc-CVCNN staged delayed satellite expansion."),
        ("SCV_S50_BAL_2028", "sinc_cvcnn", "staged", 3, 2028, 50, "balanced", "random", "50-shot Sinc-CVCNN staged balanced expansion; checks higher-shot scaling."),

        # Direct 0.1 controls.
        ("SCV_D0P1_CLEAN_2028", "sinc_cvcnn", "direct_0p1", 4, 2028, 100, "clean", "random", "Sinc-CVCNN direct 0.1 clean control."),
        ("SCV_D0P1_LATESAT_2028", "sinc_cvcnn", "direct_0p1", 5, 2028, 100, "late_sat", "random", "Sinc-CVCNN direct 0.1 delayed satellite control."),
        ("BCV_D0P1_CLEAN_2028", "cvcnn", "direct_0p1", 6, 2028, 100, "clean", "random", "Plain CVCNN direct 0.1 clean control."),
        ("BCV_D0P1_LATESAT_2028", "cvcnn", "direct_0p1", 7, 2028, 100, "late_sat", "random", "Plain CVCNN direct 0.1 delayed satellite control."),
    ]
    return [
        Candidate(
            cid=cid,
            run_name=f"CEN51_{cid}_r010",
            arch_family=arch,
            mode=mode,
            gpu=gpu,
            seed=seed,
            shots=shots,
            stage2_kind=kind,
            cap_strategy=cap,
            hypothesis=hyp,
        )
        for cid, arch, mode, gpu, seed, shots, kind, cap, hyp in rows
    ]


def q(value: object) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


def bash_items(items: Iterable[str], indent: str = "  ") -> str:
    return "".join(f"{indent}{q(item)}\n" for item in items)


def common_args() -> list[str]:
    return [
        "--train_mode", "centralized",
        "--dataset", "wisig",
        "--wisig_protocol", "cvs_day_rx",
        "--wisig_domain", "rx_day",
        "--wisig_equalized", "1",
        "--wisig_train_ratio", "0.1",
        "--wisig_val_ratio", "-1.0",
        "--wisig_split_strategy", "random",
        "--wisig_train_days", "0,1",
        "--wisig_test_days", "2,3",
        "--wisig_train_rxs", "0,1,2,3,4,5,6",
        "--wisig_test_rxs", "7,8,9,10,11",
        "--eval_batch_size", "192",
        "--batch_size", "96",
        "--num_workers", "1",
        "--cpu_threads", "2",
        "--cpu_interop_threads", "1",
        "--prefetch_factor", "2",
        "--test_eval_policy", "interval_final",
        "--test_eval_interval", "25",
        "--model_variant", "lite_d",
        "--branch_ablation", "no_dac",
        "--domain_branch_ablation", "no_stats",
        "--domain_enhancer", "rcn_stats",
        "--domain_enhancer_strength", "0.35",
        "--id_time_stability_mode", "off",
        "--id_freq_stability_mode", "off",
        "--domain_time_stability_mode", "off",
        "--domain_freq_stability_mode", "off",
        "--exp_group", "sinc_cvcnn_lowshot_compare",
        "--pa_orders", "1,3,5",
        "--collapse_guard",
        "--collapse_guard_min_epoch", "25",
        "--collapse_guard_best_margin", "10.0",
        "--collapse_guard_max_skipped_delta", "2",
        "--use_ema_ckpt",
        "--ema_decay", "0.999",
        "--use_swad_ckpt",
        "--swad_interval", "1",
        "--swad_tolerance", "0.9",
        "--primary_udu_weight", "0.80",
        "--label_smoothing", "0.0",
        "--no_enable_pa_aux",
        "--no_enable_dac_aux",
        "--no_aug_enable_pa_normal",
        "--aug_p_pa", "0.0",
        "--aug_p_dac", "0.0",
        "--lambda_cls_pa", "0.0",
        "--lambda_pa_joint_inv", "0.0",
        "--lambda_pa_kl", "0.0",
        "--lambda_pa_reg", "0.0",
        "--no_use_mixstyle",
    ]


def clean_args(shots: int, *, eval_sat: bool) -> list[str]:
    fn = 2.0e-4 if shots <= 5 else 1.4e-4 if shots <= 10 else 8.0e-5
    dom = 0.30 if shots <= 5 else 0.36 if shots <= 10 else 0.42
    adv = 0.10 if shots <= 5 else 0.14 if shots <= 10 else 0.20
    args = [
        "--epochs", "90" if shots <= 10 else "120",
        "--wisig_max_train_per_combo", str(shots),
        "--test_eval_start_epoch", "50" if shots <= 10 else "70",
        "--no_use_aug",
        "--no_use_concat_sat_channel_aug",
        "--no_use_sat_consistency",
        "--lambda_sat_cls", "0.0",
        "--lambda_sat_cons", "0.0",
        "--concat_sat_ce_weight", "0.0",
        "--sat_cons_start_epoch", "999",
        "--lambda_dom", f"{dom:.4g}",
        "--lambda_adv", f"{adv:.4g}",
        "--grl_lambda", "1.0",
        "--lambda_orth", "0.01",
        "--lambda_cons", "0.0",
        "--lambda_group_ce", "0.0" if shots <= 10 else "0.012",
        "--lambda_proto", "0.0" if shots <= 10 else "0.002",
        "--lambda_supcon_id", "0.0" if shots <= 10 else "0.002",
        "--lambda_fishr", "0.0",
        "--lambda_feature_norm_guard", f"{fn:.4g}",
        "--feature_norm_guard_mode", "l2",
        "--feature_norm_guard_target", "0.0",
        "--no_use_proto_memory" if shots <= 10 else "--use_proto_memory",
        "--swad_start_epoch", "35" if shots <= 10 else "45",
    ]
    if eval_sat:
        args.extend(["--eval_sat_channel", "--eval_sat_on", "test_unseen_day_unseen_rx", "--eval_sat_scenarios", ALL_SAT, "--sat_eval_max_batches", "-1"])
    else:
        args.append("--no_eval_sat_channel")
    return args


def stage2_args(kind: str, shots: int) -> list[str]:
    low = shots <= 10
    if kind == "clean":
        sat, sat_p, sat_start, scenarios = False, 0.0, 999, LIGHT_SAT
        dom, adv, gce, proto, sup, fishr, fn = (0.40, 0.18, 0.016, 0.003, 0.003, 0.0, 8e-5) if low else (0.50, 0.26, 0.034, 0.007, 0.007, 0.0005, 5e-5)
        aug_min, aug_max = (0.02, 0.14) if low else (0.04, 0.22)
    elif kind == "late_sat":
        sat, sat_p, sat_start, scenarios = True, (0.14 if low else 0.28), (45 if low else 35), ALL_SAT
        dom, adv, gce, proto, sup, fishr, fn = (0.44, 0.22, 0.022, 0.004, 0.004, 0.0, 7e-5) if low else (0.52, 0.30, 0.040, 0.008, 0.008, 0.0008, 4.5e-5)
        aug_min, aug_max = (0.02, 0.16) if low else (0.05, 0.24)
    elif kind == "balanced":
        sat, sat_p, sat_start, scenarios = True, 0.32, 30, ALL_SAT
        dom, adv, gce, proto, sup, fishr, fn = 0.56, 0.32, 0.048, 0.010, 0.010, 0.001, 5e-5
        aug_min, aug_max = 0.06, 0.26
    else:
        raise ValueError(f"unknown kind: {kind}")

    args = [
        "--epochs", "160" if low else "180",
        "--wisig_max_train_per_combo", "100",
        "--test_eval_start_epoch", "86" if low else "106",
        "--eval_sat_channel",
        "--eval_sat_on", "test_unseen_day_unseen_rx",
        "--eval_sat_scenarios", ALL_SAT,
        "--sat_eval_max_batches", "-1",
        "--use_aug",
        "--aug_scale_min", f"{aug_min:.3f}",
        "--aug_scale_max", f"{aug_max:.3f}",
        "--late_aug_min_scale", f"{aug_min:.3f}",
        "--lambda_dom", f"{dom:.4g}",
        "--lambda_adv", f"{adv:.4g}",
        "--grl_lambda", "1.0",
        "--lambda_orth", "0.02",
        "--lambda_cons", "0.006",
        "--lambda_group_ce", f"{gce:.4g}",
        "--group_ce_mode", "smooth_dro_capped",
        "--group_ce_min_domains", "2",
        "--group_ce_top_frac", "0.20",
        "--groupdro_tau", "0.38",
        "--groupdro_cap", "0.50",
        "--lambda_proto", f"{proto:.4g}",
        "--lambda_supcon_id", f"{sup:.4g}",
        "--lambda_fishr", f"{fishr:.4g}",
        "--fishr_min_domains", "2",
        "--lambda_feature_norm_guard", f"{fn:.4g}",
        "--feature_norm_guard_mode", "l2",
        "--feature_norm_guard_target", "0.0",
        "--use_proto_memory",
        "--swad_start_epoch", "55" if low else "65",
    ]
    if sat:
        args.extend([
            "--use_concat_sat_channel_aug",
            "--no_use_sat_consistency",
            "--lambda_sat_cls", "0.0",
            "--lambda_sat_cons", "0.0",
            "--concat_sat_ce_weight", "0.0",
            "--sat_cons_start_epoch", "999",
            "--sat_view_prob", f"{sat_p:.3f}",
            "--sat_train_scenarios", scenarios,
            "--concat_sat_start_epoch", str(sat_start),
            "--sat_view_schedule", f"1@{sat_p:.3f}:{scenarios}",
        ])
    else:
        args.extend(["--no_use_concat_sat_channel_aug", "--no_use_sat_consistency", "--lambda_sat_cls", "0.0", "--lambda_sat_cons", "0.0", "--concat_sat_ce_weight", "0.0", "--sat_cons_start_epoch", "999"])
    return args


def run_cmd_args(c: Candidate, phase: str) -> tuple[list[str], str, str]:
    if phase == "stage1":
        args = clean_args(c.shots, eval_sat=False)
        run_suffix = "_S1_anchor"
        log_suffix = "_S1_anchor.out"
    elif phase == "main":
        if c.mode == "direct_lowshot":
            args = clean_args(c.shots, eval_sat=True) if c.stage2_kind == "clean" else stage2_args(c.stage2_kind, c.shots)
            args = [("--wisig_max_train_per_combo" if x == "--wisig_max_train_per_combo" else x) for x in args]
            run_suffix = ""
            log_suffix = ".out"
        else:
            args = stage2_args(c.stage2_kind, c.shots)
            run_suffix = ""
            log_suffix = ".out"
    else:
        raise ValueError(phase)
    args = ["--arch_family", c.arch_family, "--seed", str(c.seed), "--wisig_cap_strategy", c.cap_strategy] + args
    return args, c.run_name + run_suffix, c.run_name + log_suffix


def render_candidate(c: Candidate) -> str:
    direct = "1" if c.mode != "staged" else "0"
    s1_args, s1_run, s1_log = run_cmd_args(c, "stage1")
    main_args, main_run, main_log = run_cmd_args(c, "main")
    return f"""
launch_{c.cid}() {{
  local candidate_id={q(c.cid)}
  local run_name={q(c.run_name)}
  local gpu={c.gpu}
  local direct={direct}
  if should_skip "${{candidate_id}}" "${{run_name}}"; then return 0; fi
  if ! reserve_gpu "${{candidate_id}}" "${{run_name}}" "${{gpu}}"; then return 0; fi
  if [[ "${{DRY_RUN}}" == "1" ]]; then return 0; fi
  local driver_log="${{LOG_ROOT}}/${{run_name}}.driver.out"
  (
    set -euo pipefail
    cd "${{ROOT}}"
    local init_ckpt=""
    if [[ "${{direct}}" == "0" ]]; then
      local s1_dir="${{RUNS_ROOT}}/{s1_run}"
      local s1_log="${{LOG_ROOT}}/{s1_log}"
      mkdir -p "${{s1_dir}}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${{gpu}}" "PYTHONPATH=${{ROOT}}/code:${{ROOT}}:${{PYTHONPATH:-}}" "${{PYTHON}}" -u "${{TRAIN_SCRIPT}}" "${{COMMON_ARGS[@]}}"
{bash_items(s1_args, "        ").rstrip()}
        --run_name {q(s1_run)}
        --latest_save_path "${{s1_dir}}/latest_model.pth"
        --best_save_path "${{s1_dir}}/best_val_model.pth"
        --best_primary_save_path "${{s1_dir}}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${{s1_dir}}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${{s1_dir}}/best_worst_rx_model.pth"
        --ema_save_path "${{s1_dir}}/ema_model.pth"
        --swa_save_path "${{s1_dir}}/swa_model.pth"
        --swad_save_path "${{s1_dir}}/swad_model.pth")
      print_cmd "${{s1_cmd[@]}}"
      "${{s1_cmd[@]}}" > "${{s1_log}}" 2>&1
      init_ckpt="$(choose_init_ckpt "${{s1_dir}}")"
      echo "[STAGE1-DONE] candidate=${{candidate_id}} init_ckpt=${{init_ckpt}}"
    fi
    local main_dir="${{RUNS_ROOT}}/{main_run}"
    local main_log="${{LOG_ROOT}}/{main_log}"
    mkdir -p "${{main_dir}}"
    local main_cmd=(env "CUDA_VISIBLE_DEVICES=${{gpu}}" "PYTHONPATH=${{ROOT}}/code:${{ROOT}}:${{PYTHONPATH:-}}" "${{PYTHON}}" -u "${{TRAIN_SCRIPT}}" "${{COMMON_ARGS[@]}}"
{bash_items(main_args, "        ").rstrip()}
        --run_name {q(main_run)}
        --latest_save_path "${{main_dir}}/latest_model.pth"
        --best_save_path "${{main_dir}}/best_val_model.pth"
        --best_primary_save_path "${{main_dir}}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${{main_dir}}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${{main_dir}}/best_worst_rx_model.pth"
        --ema_save_path "${{main_dir}}/ema_model.pth"
        --swa_save_path "${{main_dir}}/swa_model.pth"
        --swad_save_path "${{main_dir}}/swad_model.pth")
    if [[ -n "${{init_ckpt}}" ]]; then main_cmd+=(--init_checkpoint "${{init_ckpt}}"); fi
    print_cmd "${{main_cmd[@]}}"
    "${{main_cmd[@]}}" > "${{main_log}}" 2>&1
    echo "[CANDIDATE-DONE] ${{candidate_id}}"
  ) > "${{driver_log}}" 2>&1 &
  local driver_pid="$!"
  printf "%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n" "${{candidate_id}}" "${{run_name}}" "{c.arch_family}" "${{gpu}}" "${{driver_pid}}" "${{driver_log}}" | tee -a "${{LOG_ROOT}}/launch_pids.tsv"
}}
"""


def render_launcher(run_id: str, candidates: list[Candidate]) -> str:
    blocks = "\n".join(render_candidate(c) for c in candidates)
    calls = "\n".join(f"launch_{c.cid}" for c in candidates)
    return f"""#!/usr/bin/env bash
set -euo pipefail

ROOT="${{ROOT:-{REMOTE_ROOT}}}"
PYTHON="${{PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}}"
TRAIN_SCRIPT="${{TRAIN_SCRIPT:-${{ROOT}}/code/train.py}}"
RUN_ID="${{RUN_ID:-{run_id}}}"
LOG_ROOT="${{LOG_ROOT:-${{ROOT}}/logs/${{RUN_ID}}}}"
RUNS_ROOT="${{RUNS_ROOT:-${{ROOT}}/runs/${{RUN_ID}}}}"
MAX_TRAIN_PER_GPU="${{MAX_TRAIN_PER_GPU:-4}}"
DRY_RUN="${{DRY_RUN:-0}}"
ONLY_CANDIDATE="${{ONLY_CANDIDATE:-}}"

for arg in "$@"; do
  case "${{arg}}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY_CANDIDATE="${{arg#--only=}}" ;;
    *) echo "[ERROR] unknown argument: ${{arg}}" >&2; exit 2 ;;
  esac
done

gpu_process_count() {{
  local gpu="$1"
  nvidia-smi --id="${{gpu}}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' | wc -l | tr -d ' '
}}
print_cmd() {{ printf '%q ' "$@"; printf '\\n'; }}
should_skip() {{
  local candidate_id="$1" run_name="$2"
  [[ -n "${{ONLY_CANDIDATE}}" && "${{candidate_id}}" != "${{ONLY_CANDIDATE}}" && "${{run_name}}" != "${{ONLY_CANDIDATE}}" ]]
}}
choose_init_ckpt() {{
  local run_dir="$1"
  for name in best_primary_ood_model.pth best_val_model.pth best_strict_udu_model.pth latest_model.pth; do
    [[ -f "${{run_dir}}/${{name}}" ]] && {{ echo "${{run_dir}}/${{name}}"; return 0; }}
  done
  echo "[ERROR] no checkpoint in ${{run_dir}}" >&2
  return 4
}}
declare -A INITIAL_BY_GPU=()
declare -A LAUNCHED_BY_GPU=()
snapshot_capacity() {{
  local gpu
  for gpu in 0 1 2 3 4 5 6 7; do
    INITIAL_BY_GPU[${{gpu}}]="$(gpu_process_count "${{gpu}}")"
    LAUNCHED_BY_GPU[${{gpu}}]=0
  done
}}
reserve_gpu() {{
  local candidate_id="$1" run_name="$2" gpu="$3"
  local initial_count="${{INITIAL_BY_GPU[${{gpu}}]:-0}}"
  local local_count="${{LAUNCHED_BY_GPU[${{gpu}}]:-0}}"
  if [[ "${{DRY_RUN}}" == "1" ]]; then
    echo "[DRY-RUN] reserve candidate=${{candidate_id}} run=${{run_name}} gpu=${{gpu}} initial=${{initial_count}} local=${{local_count}} max=${{MAX_TRAIN_PER_GPU}}"
    return 0
  fi
  if (( initial_count + local_count >= MAX_TRAIN_PER_GPU )); then
    mkdir -p "${{LOG_ROOT}}"
    printf "%s\\t%s\\t%s\\tgpu=%s initial_count=%s local_count=%s max=%s\\n" "${{candidate_id}}" "${{run_name}}" "BLOCKED_CAPACITY" "${{gpu}}" "${{initial_count}}" "${{local_count}}" "${{MAX_TRAIN_PER_GPU}}" | tee -a "${{LOG_ROOT}}/blocked.tsv"
    return 1
  fi
  LAUNCHED_BY_GPU[${{gpu}}]=$(( local_count + 1 ))
  return 0
}}

COMMON_ARGS=(
{bash_items(common_args()).rstrip()}
)

cd "${{ROOT}}"
mkdir -p "${{LOG_ROOT}}" "${{RUNS_ROOT}}"
snapshot_capacity
echo "[SINC-CVCNN-COMPARE] run_id=${{RUN_ID}} dry_run=${{DRY_RUN}} max_train_per_gpu=${{MAX_TRAIN_PER_GPU}} initial_gpu_counts=${{INITIAL_BY_GPU[*]}}"
[[ "${{DRY_RUN}}" == "1" || -f "${{TRAIN_SCRIPT}}" ]] || {{ echo "[ERROR] missing train script: ${{TRAIN_SCRIPT}}" >&2; exit 2; }}

{blocks}

{calls}
echo "[SINC-CVCNN-COMPARE] launch submissions complete"
"""


def write_report(run_id: str, candidates: list[Candidate], script_path: Path, matrix_path: Path, report_path: Path) -> None:
    rows = [
        "| ID | arch | mode | GPU | seed | shots | kind | hypothesis |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for c in candidates:
        rows.append(f"| `{c.cid}` | {c.arch_family} | {c.mode} | {c.gpu} | {c.seed} | {c.shots} | {c.stage2_kind} | {c.hypothesis} |")
    report = f"""# {run_id}

## Objective

Add a low-shot architectural control for the current CVS/CEN51 few-shot work: a CVCNN whose first layer is SincConv (`--arch_family sinc_cvcnn`). This tests whether the current physics-aware CVS backbone is over-constrained or too high-capacity for low-shot adaptation.

## Hypotheses

- If `sinc_cvcnn` beats CVS at K5/K10 but loses at 0.1/K50, the issue is low-shot capacity/regularization, not the overall CVS direction.
- If `sinc_cvcnn` beats matched `cvcnn`, the SincConv first-layer inductive bias is useful even without the full physical branch stack.
- If staged `sinc_cvcnn` improves over direct `sinc_cvcnn` 0.1, then the low-shot anchor idea is architecture-agnostic.
- If `sinc_cvcnn` is stable but lower ceiling, it can serve as a low-shot teacher/selector/probe rather than a final backbone.

## Launch Status

Not launched in this turn. N607 is currently occupied by the 31-candidate staged CVS matrix; launching this comparison now would exceed the intended per-GPU capacity. This matrix is prepared for the next free-capacity window.

## Artifacts

- Launcher: `{script_path.as_posix()}`
- Matrix JSON: `{matrix_path.as_posix()}`
- Report: `{report_path.as_posix()}`

## Candidate Matrix

{chr(10).join(rows)}

## Verification Plan

- `conda activate ssr-gpu; python -m py_compile tools/cen51_sinc_cvcnn_lowshot_compare_matrix.py code/train.py`
- local smoke: instantiate `cvcnn` and `sinc_cvcnn` through `build_dual_model`, run forward/backward.
- `conda activate ssr-gpu; python tools/cen51_sinc_cvcnn_lowshot_compare_matrix.py --run-id {run_id}`
- `bash -n code/scripts/launch_{run_id}.sh`
- remote dry-run only after current GPU capacity is free.

## Metrics To Compare

- K5/K10 validation TX, clean strict UDU, worst-rx, and satellite strict UDU.
- `sinc_cvcnn` vs `cvcnn` at matched K to isolate the SincConv stem.
- staged `sinc_cvcnn` vs direct `sinc_cvcnn` 0.1 to test whether the anchor-expansion mechanism survives a simpler backbone.
- CVS staged matrix vs `sinc_cvcnn` matrix to decide whether CVS low-shot weakness is architecture capacity, physical-branch regularization, or sample selection.
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()
    run_id = str(args.run_id)
    candidates = make_candidates()
    if len(candidates) != 16:
        raise SystemExit(f"expected 16 candidates, got {len(candidates)}")
    script_path = REPO_ROOT / "code" / "scripts" / f"launch_{run_id}.sh"
    report_dir = REPO_ROOT / "automation_reports" / "CV-SincNet" / run_id
    matrix_path = report_dir / "matrix.json"
    report_path = report_dir / "report.md"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    script_path.write_text(render_launcher(run_id, candidates), encoding="utf-8", newline="\n")
    matrix_path.write_text(json.dumps([asdict(c) for c in candidates], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(run_id, candidates, script_path, matrix_path, report_path)
    print(json.dumps({"run_id": run_id, "candidates": len(candidates), "launcher": str(script_path), "matrix": str(matrix_path), "report": str(report_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
