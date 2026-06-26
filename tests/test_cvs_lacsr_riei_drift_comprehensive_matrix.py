import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tools import cvs_lacsr_riei_drift_comprehensive_matrix as matrix


def test_matrix_uses_one_cvs_lowshot_algorithm_across_all_shots():
    jobs = matrix.make_jobs()

    assert len(jobs) == 200
    assert {job.shots for job in jobs} == {10, 20, 30, 50, 100}
    assert {job.method for job in jobs} == {
        "cvs_lacsr",
        "riei_fixed_nosat",
        "riei_fixed_sat",
        "drift_fixed_nosat",
        "drift_fixed_sat",
    }
    assert {job.cvs_algorithm for job in jobs if job.method == "cvs_lacsr"} == {"CEN51_LACSR"}
    assert all("FULLDG" not in job.run_name and "_LAC_" not in job.run_name for job in jobs)

    profile_ids = {profile.profile_id for profile in matrix.PROFILES}
    assert profile_ids == {
        "rx7_all_d01",
        "rx7_all_d0",
        "rx3_sp_d01",
        "rx5_sp_d01",
        "rx3_lo_d01",
        "rx3_hi_d01",
        "rx5_lo_d01",
        "rx5_hi_d01",
    }
    combo_counts = Counter((job.method, job.shots, job.profile_id) for job in jobs)
    assert set(combo_counts.values()) == {1}
    assert len(combo_counts) == 5 * 5 * 8


def test_command_args_lock_exact_fewshot_caps_and_sat_pairs():
    jobs = matrix.make_jobs()
    cvs_job = next(job for job in jobs if job.method == "cvs_lacsr" and job.shots == 30)
    cvs_args = cvs_job.command_args()

    assert cvs_job.sat_train is True
    assert "--wisig_train_ratio" in cvs_args
    assert cvs_args[cvs_args.index("--wisig_train_ratio") + 1] == "0.1"
    assert "--wisig_val_ratio" in cvs_args
    assert cvs_args[cvs_args.index("--wisig_val_ratio") + 1] == "-1.0"
    assert "--wisig_max_train_per_combo" in cvs_args
    assert cvs_args[cvs_args.index("--wisig_max_train_per_combo") + 1] == "30"
    assert "--use_concat_sat_channel_aug" in cvs_args
    assert "--sat_view_schedule" in cvs_args

    riei_nosat = next(job for job in jobs if job.method == "riei_fixed_nosat" and job.shots == 10)
    riei_nosat_args = riei_nosat.command_args()
    assert riei_nosat.module == "baselines.riei_fd.train"
    assert "--lambda_feature_norm" in riei_nosat_args
    assert riei_nosat_args[riei_nosat_args.index("--lambda_feature_norm") + 1] == "0.0001"
    assert "--use_sat_channel_view_aug" not in riei_nosat_args
    assert "--eval_sat_channel" in riei_nosat_args
    assert riei_nosat_args[riei_nosat_args.index("--wisig_max_train_per_combo") + 1] == "10"

    drift_sat = next(job for job in jobs if job.method == "drift_fixed_sat" and job.shots == 100)
    drift_sat_args = drift_sat.command_args()
    assert drift_sat.module == "baselines.drift.train"
    assert "--mse_cap" in drift_sat_args
    assert drift_sat_args[drift_sat_args.index("--mse_cap") + 1] == "4000"
    assert "--use_sat_channel_view_aug" in drift_sat_args
    assert drift_sat_args[drift_sat_args.index("--wisig_max_train_per_combo") + 1] == "100"


def test_gpu_assignment_and_refill_launcher_contract():
    jobs = matrix.make_jobs()
    per_gpu = Counter(job.gpu for job in jobs)
    assert per_gpu == {gpu: 25 for gpu in range(8)}
    resume_jobs = matrix.make_jobs(exclude_job_ids={"cvs_lacsr_fs010_rx7_all_d01_seed1337"})
    assert len(resume_jobs) == 199
    assert all(job.job_id != "cvs_lacsr_fs010_rx7_all_d01_seed1337" for job in resume_jobs)

    launcher = matrix.render_launcher("unit_cvs_lacsr_riei_drift", jobs)
    assert 'MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-3}"' in launcher
    assert "wait -n" in launcher
    assert "QUEUE_DIR=" in launcher
    assert "queue_events" in launcher
    assert "launch_pids.tsv" in launcher
    assert "manifest_${STAMP}.tsv" in launcher
    assert "cat <<'EOF'" in launcher
    assert 'cat > "${queue_file}" <<EOF' not in launcher
    assert "TARGET_GPU_UUID" in launcher
    assert "--query-compute-apps=gpu_uuid,pid" in launcher
    assert "baselines.riei_fd.train" in launcher
    assert "baselines.drift.train" in launcher
    assert "CEN51_LACSR" in launcher
    assert "FULLDG" not in launcher
    assert "RXGUARD" not in launcher

    first_out_dir = launcher.index('out_dir="${RUN_ROOT}/cvs_lacsr_fs010_rx7_all_d01_seed1337"')
    first_cmd = launcher.index("CMD=(", first_out_dir)
    assert first_out_dir < first_cmd
