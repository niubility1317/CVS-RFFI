#!/usr/bin/env bash
set -euo pipefail

# D92 E0 FULL CSOAS K10 G0机械预注册骨架。
# 当前状态：WAITING_FOR_SCIENTIFIC_COMMIT。
# 主代理必须在科学commit、runtime archive、入口闭包和本地复核完成后填实下列占位，
# 再由sole runner执行唯一detached command；本文件当前不可用于launch。

project=/home/szu2070436088/2510044040/CV-SincNet
python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
run_id=d92_e0_full_csoas_g0_k10_20260812_v1
scientific_commit=0000000000000000000000000000000000000000
source_root="$project/runs/d92_csoas_g0_source_${scientific_commit}_20260812_v1"
code_root="$source_root/code"
archive="$source_root/d92_csoas_g0_runtime_${scientific_commit}.tar.gz"
output="$project/runs/$run_id"
logs="$project/logs/$run_id"
job="$project/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5"
ground="$project/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component"
ground_sha=15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c

if [[ "$scientific_commit" == "0000000000000000000000000000000000000000" ]]; then
  printf '%s\n' 'WAITING_FOR_SCIENTIFIC_COMMIT' >&2
  exit 78
fi

# 唯一detached command（仅供外层runner登记；当前不执行）：
# cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_g0_source_<scientific_commit>_20260812_v1 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &

# 该占位入口故意fail-closed。禁止在科学commit落地前解包、创建output/log、
# 访问query/scorer或复用TCRA runtime archive。
printf '%s\n' 'WAITING_FOR_SCIENTIFIC_COMMIT' >&2
exit 78
