#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --wave 1|2|3" >&2
  exit 2
}

die() {
  echo "NO-GO: $*" >&2
  exit 1
}

[[ $# -eq 2 && "$1" == "--wave" ]] || usage
WAVE="$2"
[[ "${WAVE}" =~ ^[123]$ ]] || usage

ROOT="/home/szu2070436088/2510044040/CV-SincNet"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
CONFIG="paper_reproduction/configs/mitigating_receiver_impact_da_manysig_paper_faithful.json"
DATA="Dataset_WigSig/ManySig.pkl"
BASE_GROUP="mitigating_da_full_multiseed_validation_20260713"
GROUP="${BASE_GROUP}_wave${WAVE}"
CONTROL_ROOT="paper_reproduction/logs/${BASE_GROUP}"
EXPECTED_MATRIX="${CONTROL_ROOT}/expected_matrix.tsv"
LAUNCH_LOCK="${CONTROL_ROOT}/launch.lock"
LOG_ROOT="paper_reproduction/logs/${GROUP}"
RUN_ROOT="paper_reproduction/runs/${GROUP}"
MANIFEST="${LOG_ROOT}/launch_manifest.tsv"

matrix_run_id() {
  local wave="$1"
  local profile="$2"
  local task="$3"
  local seed="$4"
  local slug="${task//->/_to_}"
  printf '%s_wave%s_%s_%s_b128_s%s' \
    "${BASE_GROUP}" "${wave}" "${profile}" "${slug}" "${seed}"
}

emit_expected_matrix() {
  local wave profile task seed gpu
  printf 'wave\tprofile\ttask\tseed\tgpu\trun_id\n'
  while IFS=$'\t' read -r wave profile task seed gpu; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${wave}" "${profile}" "${task}" "${seed}" "${gpu}" \
      "$(matrix_run_id "${wave}" "${profile}" "${task}" "${seed}")"
  done <<'MATRIX'
1	standard_resnet18	14-7->3-19	20260711	0
1	standard_resnet18	1-1->1-19	20260711	1
1	standard_resnet18	1-1->8-8	20260711	2
1	standard_resnet18	7-7->8-8	20260711	3
1	pytorch_template_resnet18_hypothesis_v1	14-7->3-19	20260711	4
1	pytorch_template_resnet18_hypothesis_v1	1-1->1-19	20260711	5
1	pytorch_template_resnet18_hypothesis_v1	1-1->8-8	20260711	6
1	pytorch_template_resnet18_hypothesis_v1	7-7->8-8	20260711	7
2	standard_resnet18	d01->d23	20260711	0
2	pytorch_template_resnet18_hypothesis_v1	d01->d23	20260711	1
2	standard_resnet18	14-7->3-19	20260712	2
2	standard_resnet18	1-1->1-19	20260712	3
2	standard_resnet18	1-1->8-8	20260712	4
2	standard_resnet18	7-7->8-8	20260712	5
2	pytorch_template_resnet18_hypothesis_v1	14-7->3-19	20260712	6
2	pytorch_template_resnet18_hypothesis_v1	1-1->1-19	20260712	7
3	standard_resnet18	d01->d23	20260712	0
3	pytorch_template_resnet18_hypothesis_v1	d01->d23	20260712	1
3	pytorch_template_resnet18_hypothesis_v1	1-1->8-8	20260712	2
3	pytorch_template_resnet18_hypothesis_v1	7-7->8-8	20260712	3
MATRIX
}

validate_expected_matrix() {
  local matrix="$1"
  awk -F '\t' '
    NR == 1 {
      if ($0 != "wave\tprofile\ttask\tseed\tgpu\trun_id") exit 10
      next
    }
    NF != 6 { exit 11 }
    {
      rows++
      wave_count[$1]++
      combination = $2 FS $3 FS $4
      if (seen_combination[combination]++) exit 12
      if (seen_run_id[$6]++) exit 13
    }
    END {
      if (rows != 20) exit 14
      if (wave_count[1] != 8 || wave_count[2] != 8 || wave_count[3] != 4) exit 15
    }
  ' "${matrix}" || die "invalid expected matrix: ${matrix}"
}

install_or_validate_expected_matrix() {
  mkdir -p "${CONTROL_ROOT}"
  local candidate
  candidate="$(mktemp "${CONTROL_ROOT}/.expected_matrix.XXXXXX")"
  trap "rm -f -- '${candidate}'" EXIT
  emit_expected_matrix > "${candidate}"
  validate_expected_matrix "${candidate}"

  if [[ -e "${EXPECTED_MATRIX}" ]]; then
    validate_expected_matrix "${EXPECTED_MATRIX}"
    cmp -s "${candidate}" "${EXPECTED_MATRIX}" || \
      die "existing expected matrix differs from the required 20-row matrix: ${EXPECTED_MATRIX}"
  elif ! (set -o noclobber; cat "${candidate}" > "${EXPECTED_MATRIX}") 2>/dev/null; then
    [[ -e "${EXPECTED_MATRIX}" ]] || die "could not create expected matrix"
    validate_expected_matrix "${EXPECTED_MATRIX}"
    cmp -s "${candidate}" "${EXPECTED_MATRIX}" || \
      die "concurrently created expected matrix differs: ${EXPECTED_MATRIX}"
  fi

  trap - EXIT
  rm -f "${candidate}"
}

assert_wave_outputs_absent() {
  [[ ! -e "${MANIFEST}" ]] || die "manifest already exists: ${MANIFEST}"
  [[ ! -e "${LOG_ROOT}" ]] || die "wave log directory already exists: ${LOG_ROOT}"
  [[ ! -e "${RUN_ROOT}" ]] || die "wave run directory already exists: ${RUN_ROOT}"

  local row wave profile task seed gpu run_id run_dir log result
  for row in "${WAVE_ROWS[@]}"; do
    IFS=$'\t' read -r wave profile task seed gpu run_id <<< "${row}"
    run_dir="${RUN_ROOT}/${run_id}"
    log="${LOG_ROOT}/${run_id}.out"
    result="${run_dir}/results.json"
    [[ ! -e "${run_dir}" ]] || die "run/checkpoint directory already exists: ${run_dir}"
    [[ ! -e "${log}" ]] || die "log already exists: ${log}"
    [[ ! -e "${result}" ]] || die "result already exists: ${result}"
  done
}

assert_other_waves_inactive() {
  local other other_group other_manifest pid process_line
  for other in 1 2 3; do
    [[ "${other}" == "${WAVE}" ]] && continue
    other_group="${BASE_GROUP}_wave${other}"
    other_manifest="paper_reproduction/logs/${other_group}/launch_manifest.tsv"

    if [[ -f "${other_manifest}" ]]; then
      while IFS= read -r pid; do
        [[ "${pid}" =~ ^[0-9]+$ ]] || continue
        if kill -0 "${pid}" 2>/dev/null; then
          die "wave ${other} manifest PID ${pid} is still active"
        fi
      done < <(awk -F '\t' '
        NR == 1 { for (i = 1; i <= NF; i++) if ($i == "pid") pid_col = i; next }
        pid_col && $pid_col != "" { print $pid_col }
      ' "${other_manifest}")
    fi

    while IFS= read -r process_line; do
      [[ "${process_line}" == *"${other_group}"* ]] || continue
      die "related training process for wave ${other} is still active: ${process_line}"
    done < <(pgrep -af 'paper_reproduction.mitigating_receiver_impact_da.train' || true)
  done
}

assert_gpu_capacity() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"

  local gpu_inventory compute_inventory row wave profile task seed gpu run_id
  local gpu_uuid process_count
  gpu_inventory="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)" || \
    die "failed to query GPU inventory"
  compute_inventory="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits)" || \
    die "failed to query GPU compute processes"

  declare -A checked_gpus=()
  for row in "${WAVE_ROWS[@]}"; do
    IFS=$'\t' read -r wave profile task seed gpu run_id <<< "${row}"
    [[ -z "${checked_gpus[${gpu}]:-}" ]] || continue
    checked_gpus[${gpu}]=1

    gpu_uuid="$(awk -F ',' -v target="${gpu}" '
      { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2) }
      $1 == target { print $2 }
    ' <<< "${gpu_inventory}")"
    [[ -n "${gpu_uuid}" ]] || die "assigned GPU ${gpu} was not found"

    process_count="$(awk -F ',' -v target="${gpu_uuid}" '
      { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1) }
      $1 == target { count++ }
      END { print count + 0 }
    ' <<< "${compute_inventory}")"
    (( process_count < 2 )) || \
      die "GPU ${gpu} already has ${process_count} compute processes; launch would exceed two"
  done
}

launch_run() {
  local wave="$1"
  local profile="$2"
  local task="$3"
  local seed="$4"
  local gpu="$5"
  local run_id="$6"
  local run_dir="${RUN_ROOT}/${run_id}"
  local log="${LOG_ROOT}/${run_id}.out"
  local result="${run_dir}/results.json"
  local pid status

  [[ ! -e "${run_dir}" ]] || die "run/checkpoint directory appeared before launch: ${run_dir}"
  [[ ! -e "${log}" ]] || die "log appeared before launch: ${log}"
  [[ ! -e "${result}" ]] || die "result appeared before launch: ${result}"
  mkdir "${run_dir}"

  (
    set -o noclobber
    exec nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u -m \
      paper_reproduction.mitigating_receiver_impact_da.train \
      --config "${CONFIG}" \
      --run-table2 \
      --manysig-pkl "${DATA}" \
      --methods proposed \
      --tasks "${task}" \
      --model-profile "${profile}" \
      --batch-size 128 \
      --learning-rate 0.0006 \
      --base-tau 0.7 \
      --estimate-steps 7 \
      --kl-weight 0.005 \
      --mu 0.5 \
      --epochs 20 \
      --source-pretrain-epochs 20 \
      --class-prior-mode uniform \
      --kl-estimator-mode dvkl \
      --pseudo-threshold-mode paper \
      --pseudo-score-mode probability \
      --pseudo-threshold-floor 0 \
      --pseudo-quota-mode none \
      --class-weight-smoothing 0 \
      --label-smoothing 0 \
      --class-weight-timing previous \
      --pseudo-state-scope epoch \
      --batch-pairing zip_min \
      --weighted-ce-reduction paper_sample_mean \
      --target-model-selection final \
      --seed "${seed}" \
      --device cuda:0 \
      --checkpoint-dir "${run_dir}" \
      --output "${result}" \
      > "${log}" 2>&1
  ) &
  pid=$!

  sleep 3
  if kill -0 "${pid}" 2>/dev/null; then
    status="startup_alive"
  else
    status="startup_failed"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${wave}" "${run_id}" "${profile}" "${task}" "${seed}" \
    "${gpu}" "${pid}" "${log}" "${result}" "${status}" >> "${MANIFEST}"
  [[ "${status}" == "startup_alive" ]] || die "startup check failed for ${run_id}; see ${log}"
}

cd "${ROOT}"
mkdir -p "${CONTROL_ROOT}"
command -v flock >/dev/null 2>&1 || die "flock is unavailable"
exec 9>> "${LAUNCH_LOCK}"
flock -n 9 || die "another ${BASE_GROUP} launcher is in preflight/startup"
install_or_validate_expected_matrix
mapfile -t WAVE_ROWS < <(awk -F '\t' -v wave="${WAVE}" 'NR > 1 && $1 == wave' "${EXPECTED_MATRIX}")
expected_wave_rows=8
[[ "${WAVE}" == "3" ]] && expected_wave_rows=4
[[ "${#WAVE_ROWS[@]}" -eq "${expected_wave_rows}" ]] || \
  die "wave ${WAVE} has ${#WAVE_ROWS[@]} rows; expected ${expected_wave_rows}"

assert_wave_outputs_absent
assert_other_waves_inactive
assert_gpu_capacity

mkdir "${LOG_ROOT}" "${RUN_ROOT}"
if ! (set -o noclobber; printf 'wave\trun_id\tprofile\ttask\tseed\tgpu\tpid\tlog\tresult\tstatus\n' > "${MANIFEST}"); then
  die "refusing to overwrite manifest: ${MANIFEST}"
fi

for row in "${WAVE_ROWS[@]}"; do
  IFS=$'\t' read -r wave profile task seed gpu run_id <<< "${row}"
  launch_run "${wave}" "${profile}" "${task}" "${seed}" "${gpu}" "${run_id}"
done

cat "${MANIFEST}"
