#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/szu2070436088/2510044040/CV-SincNet"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
CONFIG="paper_reproduction/configs/mitigating_receiver_impact_da_manysig_paper_faithful.json"
DATA="Dataset_WigSig/ManySig.pkl"
GROUP="mitigating_da_arch_ablation_20260710_115000"
LOG_ROOT="paper_reproduction/logs/${GROUP}"
RUN_ROOT="paper_reproduction/runs/${GROUP}"
SEED=20260710

cd "${ROOT}"
mkdir -p "${LOG_ROOT}" "${RUN_ROOT}"
manifest="${LOG_ROOT}/launch_manifest.tsv"
printf 'run_id\tcandidate\ttask\tgpu\tpid\tlog\tresult\n' > "${manifest}"

launch_run() {
  local candidate="$1"
  local task="$2"
  local gpu="$3"
  shift 3
  local slug="${task//->/_to_}"
  local run_id="${GROUP}_${candidate}_${slug}_b128_s${SEED}"
  local run_dir="${RUN_ROOT}/${run_id}"
  local log="${LOG_ROOT}/${run_id}.out"
  local result="${run_dir}/results.json"
  mkdir -p "${run_dir}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
    paper_reproduction.mitigating_receiver_impact_da.train \
    --config "${CONFIG}" \
    --run-table2 \
    --manysig-pkl "${DATA}" \
    --methods proposed \
    --tasks "${task}" \
    --batch-size 128 \
    --learning-rate 0.0006 \
    --estimate-steps 7 \
    --base-tau 0.7 \
    --mu 0.5 \
    --kl-weight 0.005 \
    --target-model-selection final \
    --seed "${SEED}" \
    --device cuda:0 \
    --checkpoint-dir "${run_dir}" \
    --output "${result}" \
    "$@" > "${log}" 2>&1 &
  local pid=$!
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_id}" "${candidate}" "${task}" "${gpu}" "${pid}" "${log}" "${result}" >> "${manifest}"
}

paper_args=(
  --epochs 20
  --source-pretrain-epochs 20
  --class-prior-mode uniform
  --kl-estimator-mode dvkl
  --pseudo-threshold-mode paper
  --pseudo-score-mode probability
  --class-weight-timing previous
  --pseudo-state-scope epoch
  --batch-pairing zip_min
  --weighted-ce-reduction paper_sample_mean
)

template_args=(
  "${paper_args[@]}"
  --model-profile pytorch_template_resnet18_hypothesis_v1
)

launch_run template_hypothesis_v1 'd01->d23' 0 "${template_args[@]}"
launch_run template_hypothesis_v1 '14-7->3-19' 1 "${template_args[@]}"
launch_run template_hypothesis_v1 '1-1->1-19' 2 "${template_args[@]}"
launch_run template_hypothesis_v1 '1-1->8-8' 3 "${template_args[@]}"
launch_run template_hypothesis_v1 '7-7->8-8' 4 "${template_args[@]}"

launch_run standard_da_only '14-7->3-19' 5 "${paper_args[@]}" --no-pseudo --no-class-weight
launch_run standard_da_cw '14-7->3-19' 6 "${paper_args[@]}" --no-pseudo
launch_run standard_cpl_cw '14-7->3-19' 7 "${paper_args[@]}" --no-domain-alignment

cat "${manifest}"
