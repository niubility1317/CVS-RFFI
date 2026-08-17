#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/home/szu2070436088/2510044040/CV-SincNet
RUNS="$REMOTE_ROOT/runs"
RUN_ID=d92_e0_full_d42_afcp_g0_k10_20260817_v1
SOURCE_BASENAME=d92_afcp_g0_source_ae4a2a2a_20260817_v1
SOURCE_ROOT="$RUNS/$SOURCE_BASENAME"
SOURCE_ARCHIVE="$RUNS/$SOURCE_BASENAME.tar.gz"
RUN_ROOT="$RUNS/$RUN_ID"
LOG_ROOT="$REMOTE_ROOT/logs/$RUN_ID"
LAUNCH_BASENAME=d92_afcp_g0_launch_ae4a2a2a_20260817_v1
PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python

test -f "$SOURCE_ARCHIVE"
test "$(sha256sum "$SOURCE_ARCHIVE" | awk '{print $1}')" = "10791446976987f57a058ca87d0782154e66d2ee16887a36d2070654aefb5e95"
test "$(stat -c '%s' "$SOURCE_ARCHIVE")" = "244907"
test "$(tar -tzf "$SOURCE_ARCHIVE" | wc -l | tr -d ' ')" = "41"
if tar -tzf "$SOURCE_ARCHIVE" | grep -Eq '(^/|(^|/)\.\.(\/|$)|code/code)'; then
  printf '%s\n' 'unsafe archive member' >&2
  exit 65
fi

test -f "$RUNS/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/offline/seals/before_enrollment.seal.json"
test "$(sha256sum "$RUNS/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/offline/seals/before_enrollment.seal.json" | awk '{print $1}')" = "e3da38668a1e6ec4053e65e669e2a1845bb43198891644d830950e4550b5cea9"
test "$(sha256sum "$RUNS/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/apply_seals/before_apply.seal.json" | awk '{print $1}')" = "736852188c32255647b8105bc7a68d4cc92ca73615e4734d0ed5f4bdd0f04473"
test "$(sha256sum "$RUNS/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/offline/seals/after_enrollment.seal.json" | awk '{print $1}')" = "2600a21ee9a2f95a8d17fa1f4d2263b0e04d243424e3257474502953ed6d9286"
test "$(sha256sum "$RUNS/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/apply_seals/after_apply.seal.json" | awk '{print $1}')" = "afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a"

test ! -e "$SOURCE_ROOT"
test ! -e "$RUN_ROOT"
test ! -e "$LOG_ROOT"

tar -xzf "$SOURCE_ARCHIVE" -C "$RUNS"
for required in \
  code/cvsrffi/__init__.py \
  code/cvsrffi/stage2_d42_unified_shrinkage_lda.py \
  code/cvsrffi/stage2_d92_d42_allclass_fold_consensus_plane.py \
  code/cvsrffi/stage2_d92_e0d_slim.py \
  code/cvsrffi/stage2_d92_e0d_query_evaluation.py \
  code/scripts/run_d92_afcp_g0.py; do
  test -f "$SOURCE_ROOT/$required"
done

mkdir -p "$LOG_ROOT"
"$PYTHON" -m py_compile \
  "$SOURCE_ROOT/code/cvsrffi/stage2_d42_unified_shrinkage_lda.py" \
  "$SOURCE_ROOT/code/cvsrffi/stage2_d92_d42_allclass_fold_consensus_plane.py" \
  "$SOURCE_ROOT/code/cvsrffi/stage2_d92_e0d_slim.py" \
  "$SOURCE_ROOT/code/cvsrffi/stage2_d92_e0d_query_evaluation.py" \
  "$SOURCE_ROOT/code/scripts/run_d92_afcp_g0.py" \
  >"$LOG_ROOT/import_closure.out" 2>"$LOG_ROOT/import_closure.err"
PYTHONPATH="$SOURCE_ROOT/code:$REMOTE_ROOT" "$PYTHON" -c 'import importlib, pathlib; root=pathlib.Path("'"$SOURCE_ROOT"'/code").resolve(); names=("cvsrffi.stage2_d92_d42_allclass_fold_consensus_plane", "cvsrffi.stage2_d92_e0d_slim", "cvsrffi.stage2_d92_e0d_query_evaluation", "scripts.run_d92_afcp_g0"); paths=[pathlib.Path(importlib.import_module(name).__file__).resolve() for name in names]; assert all(root in path.parents for path in paths), paths' \
  >"$LOG_ROOT/import_closure.out" 2>"$LOG_ROOT/import_closure.err"

env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH="$SOURCE_ROOT/code:$REMOTE_ROOT" \
  "$PYTHON" -u "$SOURCE_ROOT/code/scripts/run_d92_afcp_g0.py" \
  --outer-key rx_7_7__seed_713106__k_10__new_5 \
  --reference-arm E0_FULL_ONLY \
  --candidate-arm E0_FULL_D42_ALLCLASS_FOLD_CONSENSUS_PLANE \
  --before-enrollment-package-root "$RUNS/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/offline/predictor/before/enrollment_only" \
  --before-enrollment-seal-path "$RUNS/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/offline/seals/before_enrollment.seal.json" \
  --before-enrollment-seal-sha256 e3da38668a1e6ec4053e65e669e2a1845bb43198891644d830950e4550b5cea9 \
  --before-apply-package-root "$RUNS/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/offline/predictor/before/apply_only_staging" \
  --before-apply-seal-path "$RUNS/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/apply_seals/before_apply.seal.json" \
  --before-apply-seal-sha256 736852188c32255647b8105bc7a68d4cc92ca73615e4734d0ed5f4bdd0f04473 \
  --after-enrollment-package-root "$RUNS/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/offline/predictor/after/enrollment_only" \
  --after-enrollment-seal-path "$RUNS/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/offline/seals/after_enrollment.seal.json" \
  --after-enrollment-seal-sha256 2600a21ee9a2f95a8d17fa1f4d2263b0e04d243424e3257474502953ed6d9286 \
  --after-apply-package-root "$RUNS/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/offline/predictor/after/apply_only_staging" \
  --after-apply-seal-path "$RUNS/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5/apply_seals/after_apply.seal.json" \
  --after-apply-seal-sha256 afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a \
  --ground-component-dir "$RUNS/d19_ciaf_int8_proto_20260717_1039/input/int8_component" \
  --ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c \
  --reference-output-root "$RUN_ROOT/reference_e0" \
  --candidate-output-root "$RUN_ROOT/candidate_afcp" \
  --g0-validation-path "$RUN_ROOT/g0_validation.json" \
  --device cuda:0 \
  >"$LOG_ROOT/g0_driver.out" 2>"$LOG_ROOT/g0_driver.err"

"$PYTHON" -c 'import json, pathlib; value=json.loads(pathlib.Path("'"$RUN_ROOT"'/g0_validation.json").read_text(encoding="utf-8")); validation=value.get("validation", {}); scenes={"leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"}; assert value.get("status") == "D92_AFCP_G0_ACTIVE_RESOURCE_PASS"; assert validation.get("marker") == "D92_AFCP_G0_ACTIVE_RESOURCE_PASS"; assert validation.get("pass") is True; assert set(validation.get("scenes", {})) == scenes; scene_gates=validation.get("scene_gates", {}); assert set(scene_gates) == scenes; assert all(scene_gates.values()); binding=validation.get("outer_binding", {}); assert binding.get("pass") is True' \
  >"$LOG_ROOT/marker_check.out" 2>"$LOG_ROOT/marker_check.err"

printf '%s\n' 'D92_AFCP_G0_ACTIVE_RESOURCE_PASS'
