#!/usr/bin/env bash
set -euo pipefail

test -f /home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1.tar.gz
test "$(sha256sum /home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1.tar.gz | awk '{print $1}')" = "af9bcc8d75e12010ff8fa18ea4aad75d40ee2886744384bafe4ca23738e0f538"
test "$(stat -c '%s' /home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1.tar.gz)" = "215746"
test "$(tar -tzf /home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1.tar.gz | wc -l | tr -d ' ')" = "38"
if tar -tzf /home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1.tar.gz | grep -Eq '(^/|(^|/)\.\.(\/|$)|code/code)'; then
  printf '%s\n' 'unsafe archive member' >&2
  exit 65
fi
test ! -e /home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1
test ! -e /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260813_v1
test ! -e /home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1

tar -xzf /home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1.tar.gz -C /home/szu2070436088/2510044040/CV-SincNet/runs
test -f /home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1/code/cvsrffi/stage2_d92_ccoc_g0.py
test -f /home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1/code/cvsrffi/stage2_d92_e0d_query_evaluation.py
test -f /home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1/code/scripts/run_d92_ccoc_g0.py
test -f /home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1/code/CCOC_G0_SOURCE_MANIFEST.sha256

mkdir -p /home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1
(cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1 && sha256sum -c code/CCOC_G0_SOURCE_MANIFEST.sha256) \
  > /home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1/source_manifest_check.out \
  2> /home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1/source_manifest_check.err
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile \
  /home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1/code/cvsrffi/stage2_d92_ccoc_g0.py \
  /home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1/code/scripts/run_d92_ccoc_g0.py \
  > /home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1/import_closure.out \
  2> /home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1/import_closure.err
PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1/code:/home/szu2070436088/2510044040/CV-SincNet \
  /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -c 'import importlib, pathlib; root=pathlib.Path("/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1/code").resolve(); names=("cvsrffi.stage2_d92_ccoc_g0","cvsrffi.stage2_d92_e0d_query_evaluation","scripts.run_d92_ccoc_g0"); paths=[pathlib.Path(importlib.import_module(name).__file__).resolve() for name in names]; assert all(root in path.parents for path in paths), paths' \
  >> /home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1/import_closure.out \
  2>> /home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1/import_closure.err

env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1/code:/home/szu2070436088/2510044040/CV-SincNet \
  /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u \
  /home/szu2070436088/2510044040/CV-SincNet/runs/d92_ccoc_g0_source_df17c06e_20260813_v1/code/scripts/run_d92_ccoc_g0.py \
  --outer-key rx_7_7__seed_713106__k_10__new_5 \
  --reference-arm E0_FULL_ONLY \
  --candidate-arm E0_FULL_CROSS_CLASS_OFFBLOCK_CONSENSUS \
  --before-enrollment-package-root /home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/offline/predictor/before/enrollment_only \
  --before-enrollment-seal-path /home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/offline/seals/before_enrollment.seal.json \
  --before-enrollment-seal-sha256 e3da38668a1e6ec4053e65e669e2a1845bb43198891644d830950e4550b5cea9 \
  --before-apply-package-root /home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/offline/predictor/before/apply_only_staging \
  --before-apply-seal-path /home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/apply_seals/before_apply.seal.json \
  --before-apply-seal-sha256 736852188c32255647b8105bc7a68d4cc92ca73615e4734d0ed5f4bdd0f04473 \
  --after-enrollment-package-root /home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/offline/predictor/after/enrollment_only \
  --after-enrollment-seal-path /home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/offline/seals/after_enrollment.seal.json \
  --after-enrollment-seal-sha256 2600a21ee9a2f95a8d17fa1f4d2263b0e04d243424e3257474502953ed6d9286 \
  --after-apply-package-root /home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/offline/predictor/after/apply_only_staging \
  --after-apply-seal-path /home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/apply_seals/after_apply.seal.json \
  --after-apply-seal-sha256 afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a \
  --ground-component-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component \
  --ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c \
  --reference-output-root /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260813_v1/reference_e0 \
  --candidate-output-root /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260813_v1/candidate_ccoc \
  --g0-validation-path /home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260813_v1/g0_validation.json \
  --device cuda:0 \
  > /home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1/g0_driver.out \
  2> /home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1/g0_driver.err

/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -c 'import json, pathlib; value=json.loads(pathlib.Path("/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_ccoc_g0_k10_20260813_v1/g0_validation.json").read_text(encoding="utf-8")); assert value.get("marker") == "D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS"; assert value.get("validation", {}).get("pass") is True' \
  > /home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1/marker_check.out \
  2> /home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_ccoc_g0_k10_20260813_v1/marker_check.err

printf '%s\n' 'D92_CCOC_G0_ACTIVE_QUANTUM_RESOURCE_PASS'
