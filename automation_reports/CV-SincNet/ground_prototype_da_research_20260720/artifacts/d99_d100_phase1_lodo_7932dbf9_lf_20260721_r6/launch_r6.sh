#!/usr/bin/env bash
set -euo pipefail
set -o noclobber

run_dir=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6
log_dir=/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6
source_dir=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/source_7932dbf9
config_path=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6/input/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6.json
stdout_path=/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6/runner.out
wrapper_path=/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6/wrapper.out
pid_path=/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6/runner.pid
exit_path=/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6/runner.exit
start_path=/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6/runner.start
end_path=/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6/runner.end
command_path=/home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6/child_command.txt

test -d "$run_dir/input"
test -d "$log_dir"
test ! -e "$run_dir/output"
for path in "$stdout_path" "$wrapper_path" "$pid_path" "$exit_path" "$start_path" "$end_path" "$command_path"; do
  test ! -e "$path"
done

printf '%s\n' 'env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/source_7932dbf9/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/source_7932dbf9/code/scripts/run_d99_d100_phase1_lodo.py --config /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6/input/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6.json --config-sha256 8a63aaab10dceb05811af8f0aa82bfc76e45468e2603beffd419092eebe949f6' > "$command_path"

nohup bash -c '
set -u
cd /home/szu2070436088/2510044040/CV-SincNet
date "+%Y-%m-%d %H:%M:%S %Z" > /home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6/runner.start
set +e
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/source_7932dbf9/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/source_7932dbf9/code/scripts/run_d99_d100_phase1_lodo.py --config /home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6/input/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6.json --config-sha256 8a63aaab10dceb05811af8f0aa82bfc76e45468e2603beffd419092eebe949f6 > /home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6/runner.out 2>&1
rc=$?
printf "%s\n" "$rc" > /home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6/runner.exit
date "+%Y-%m-%d %H:%M:%S %Z" > /home/szu2070436088/2510044040/CV-SincNet/logs/d99_d100_phase1_lodo_7932dbf9_lf_20260721_r6/runner.end
exit "$rc"
' </dev/null > "$wrapper_path" 2>&1 &
runner_pid=$!
printf '%s\n' "$runner_pid" > "$pid_path"
printf 'LANDED pid=%s gpu=5 cwd=%s\n' "$runner_pid" /home/szu2070436088/2510044040/CV-SincNet
