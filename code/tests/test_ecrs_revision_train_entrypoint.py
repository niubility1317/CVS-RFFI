"""Real one-step source-only train.py CLI integration on synthetic received IQ.

This tests executable wiring and artifact closure, not scientific performance.
The compact file contains no target observations; all source RX/day slots remain.
"""
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE_RX = (1, 3, 4, 6, 8)
SOURCE_DAY = (1, 2, 3)


@pytest.mark.parametrize('mode', ['disabled_u', 'clean_duplicate'])
def test_v2_u_sampling_and_applied_view_contract(tmp_path, mode):
    argv = command(tmp_path)
    if mode == 'disabled_u':
        argv += ['--no_ecrs_u_pair_enabled']
    else:
        argv += ['--sat_view_schedule', '1@0.0:leo_clear_weak']
    run = subprocess.run(argv, cwd=ROOT/'code',
        env=dict(os.environ, PYTHONPATH=str(ROOT/'code'), PYTHONIOENCODING='utf-8'),
        encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    (tmp_path/'cli_output.txt').write_text(run.stdout, encoding='utf-8')
    assert run.returncode == 0, run.stdout[-12000:]
    record = json.loads((tmp_path/'ecrs_training.jsonl').read_text(encoding='utf-8').splitlines()[0])
    assert record['successful_step']
    assert record['losses']['u_pair']['executed'] is False
    assert record['losses']['u_pair']['valid_count'] == 0
    assert record['u_coverage']['draw_count'] == (0 if mode == 'disabled_u' else 4)
    assert record['u_coverage_scope'] == 'sampled_indices_not_successful_leo_updates'


@pytest.mark.parametrize("row", ["B0", "B1", "B2", "B2-V1"])
def test_legacy_and_zero_effect_source_entrypoints(tmp_path, row):
    version = "v1" if row == "B2-V1" else "v1r"
    argv = command(tmp_path, version) + ["--ecrs_fusion_mode", "off", "--ecrs_lr_clock", "epoch",
                                       "--no_ecrs_gate_calibration_enabled"]
    if row == "B0":
        argv.remove("--use_ecrs")
        argv += ["--no_ecrs_resp_ce_enabled"]
    if row == "B1":
        argv += ["--ecrs_compute_only", "--no_ecrs_resp_ce_enabled"]
    env = dict(os.environ, PYTHONPATH=str(ROOT/'code'), PYTHONIOENCODING='utf-8')
    run = subprocess.run(argv, cwd=ROOT/'code', env=env, encoding='utf-8',
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    (tmp_path/'cli_output.txt').write_text(run.stdout, encoding='utf-8')
    assert run.returncode == 0, run.stdout[-12000:]
    assert (tmp_path/'best.pth').is_file()
    summary = json.loads((tmp_path/'ecrs_final_source_evaluation.json').read_text(encoding='utf-8'))
    assert summary['full_validation'] is True
    assert summary['scenarios']['clean']['count'] > 4
    assert all(Path(value['output_path']).is_file() for value in summary['scenarios'].values())


def write_synthetic_source(path, samples=20, length=64):
    rng = np.random.default_rng(20260906)
    data = []
    for tx in range(3):
        receivers = []
        for rx in range(12):
            days = []
            for day in range(4):
                count = samples if rx in SOURCE_RX and day in SOURCE_DAY else 0
                iq = rng.normal(0, .4, (count, length, 2)).astype(np.float32)
                iq += np.float32(tx*.03 + rx*.002 + day*.001)
                days.append([iq])
            receivers.append(days)
        data.append(receivers)
    with path.open('wb') as handle:
        pickle.dump({'data': data, 'tx_list': [f'tx{i}' for i in range(3)],
                     'rx_list': [f'rx{i}' for i in range(12)],
                     'capture_date_list': [f'day{i}' for i in range(4)],
                     'equalized_list': [1]}, handle)


def command(tmp_path, version='v2'):
    path = tmp_path/'synthetic_source.pkl'
    write_synthetic_source(path)
    cmd = [sys.executable, '-X', 'utf8', str(ROOT/'code/train.py'),
        '--dataset', 'wisig', '--wisig_pkl', str(path),
        '--wisig_train_days', '1,2,3', '--wisig_test_days', '0,1,2,3',
        '--wisig_train_rxs', '1,3,4,6,8', '--wisig_test_rxs', '0,2,5,7,9,10,11',
        '--wisig_target_receiver_only_eval', '--wisig_equalized', '1',
        '--wisig_out_len', '64', '--wisig_domain', 'rx_day',
        '--use_meta_ssl_cvs', '--ssl_labeled_ratio', '.07',
        '--ssl_unlabeled_ratio', '.63', '--ssl_val_ratio', '.30',
        '--num_classes', '3', '--model_size', 'M', '--model_variant', 'lite_d',
        '--branch_ablation', 'no_dac', '--domain_branch_ablation', 'no_stats',
        '--batch_size', '4', '--eval_batch_size', '4', '--train_steps_per_epoch', '1',
        '--eval_max_batches', '1', '--no_eval_sat_channel', '--no_use_aug',
        '--use_concat_sat_channel_aug', '--concat_sat_ce_only',
        '--concat_sat_start_epoch', '1', '--concat_sat_ce_start_epoch', '1',
        '--concat_sat_ce_weight', '.1', '--sat_view_schedule', '1@1.0:leo_clear_weak',
        '--device', 'cpu', '--num_workers', '0',
        '--cpu_threads', '2', '--cpu_interop_threads', '1', '--epochs', '1',
        '--use_ecrs', '--ecrs_version', version, '--ecrs_source_screen_only',
        '--ecrs_resp_ce_enabled', '--ecrs_fusion_mode', 'fixed',
        '--ecrs_fixed_rho', '.05', '--ecrs_fusion_start_epoch', '1',
        '--lambda_ecrs_fusion', '.2', '--ecrs_alpha_resp', '.15',
        '--ecrs_lr_clock', 'effective_step', '--ecrs_rung', 'R7',
        '--log_dir', str(tmp_path/'logs'), '--output_dir', str(tmp_path),
        '--best_save_path', str(tmp_path/'best.pth'),
        '--latest_save_path', str(tmp_path/'latest.pth'),
        '--run_name', f'synthetic_{version}_entrypoint', '--seed', '7']
    if version == 'v2':
        cmd += ['--ecrs_u_pair_enabled', '--lambda_ecrs_u_pair', '.03']
    return cmd


def test_v2_real_train_cli_source_only_ce_u_fusion_and_artifacts(tmp_path):
    argv = command(tmp_path)
    env = dict(os.environ, PYTHONPATH=str(ROOT/'code'), PYTHONIOENCODING='utf-8')
    completed = subprocess.run(argv, cwd=ROOT/'code', env=env, encoding='utf-8',
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    (tmp_path/'cli_output.txt').write_text(completed.stdout, encoding='utf-8')
    assert completed.returncode == 0, completed.stdout[-14000:]
    assert (tmp_path/'latest.pth').is_file()
    screen = json.loads((tmp_path/'ecrs_source_screen_result.json').read_text(encoding='utf-8'))
    assert screen['target_used_for_selection'] is False
    assert screen['status'] == 'SOURCE_SCREEN_COMPLETE'
    # Random synthetic data need not satisfy the source fusion promotion gate.
    assert (screen['selected_checkpoint'] is None) == (not (tmp_path/'best.pth').is_file())
    assert (tmp_path/'logs/logs.jsonl').is_file()
    telemetry = [json.loads(line) for line in (tmp_path/'ecrs_training.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(telemetry) == 1
    payload = torch.load(tmp_path/'latest.pth', map_location='cpu', weights_only=False)
    assert payload['epoch'] == 1
    assert payload['feature_schema'].endswith(':v2')
    assert payload['ecrs_runtime']['batch_index'] == 1
    assert payload['model']['ecrs.resp_to_id.weight'].abs().sum() > 0
    # JSON key layout is trainer-owned; recurse only through named objective maps.
    def find_objective(obj, name):
        if isinstance(obj, dict):
            if name in obj and isinstance(obj[name], dict) and 'executed' in obj[name]:
                return obj[name]
            for value in obj.values():
                found = find_objective(value, name)
                if found is not None:
                    return found
        return None
    for name in ('resp_ce', 'u_pair', 'fused_ce'):
        objective = find_objective(telemetry[0], name)
        assert objective is not None, telemetry
        assert objective['executed'] and objective['valid_count'] > 0, objective
        assert np.isfinite(objective['raw_loss']), objective
    fused = find_objective(telemetry[0], 'fused_ce')
    assert fused['weighted_loss'] == pytest.approx(.2*fused['raw_loss'], rel=1e-5)
    # The fixture physically has zero target samples; no performance claim follows.
    assert payload['args']['ecrs_source_screen_only'] is True
    assert payload['args']['wisig_train_rxs'] == '1,3,4,6,8'


def test_v2_nonfusion_source_selection_saves_best_checkpoint(tmp_path):
    argv = command(tmp_path) + ['--ecrs_fusion_mode', 'off']
    env = dict(os.environ, PYTHONPATH=str(ROOT/'code'), PYTHONIOENCODING='utf-8')
    completed = subprocess.run(argv, cwd=ROOT/'code', env=env, encoding='utf-8',
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    (tmp_path/'cli_output.txt').write_text(completed.stdout, encoding='utf-8')
    assert completed.returncode == 0, completed.stdout[-14000:]
    screen = json.loads((tmp_path/'ecrs_source_screen_result.json').read_text(encoding='utf-8'))
    assert screen['target_used_for_selection'] is False
    assert (tmp_path/'best.pth').is_file()
    assert Path(screen['selected_checkpoint']) == tmp_path/'best.pth'
    checkpoint = torch.load(tmp_path/'best.pth', map_location='cpu', weights_only=False)
    assert checkpoint['feature_schema'].endswith(':v2')
    assert checkpoint['args']['ecrs_fusion_mode'] == 'off'
    assert checkpoint['ecrs_runtime']['epoch'] == 1
    telemetry_before = (tmp_path/'ecrs_training.jsonl').read_bytes()
    resume_argv = argv + ['--ecrs_resume', str(tmp_path/'latest.pth')]
    resumed = subprocess.run(resume_argv, cwd=ROOT/'code', env=env, encoding='utf-8',
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    (tmp_path/'resume_output.txt').write_text(resumed.stdout, encoding='utf-8')
    assert resumed.returncode == 0, resumed.stdout[-14000:]
    assert (tmp_path/'ecrs_training.jsonl').read_bytes() == telemetry_before
    changed = subprocess.run(resume_argv + ['--no_ecrs_u_pair_enabled'], cwd=ROOT/'code',
        env=env, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    (tmp_path/'rejected_resume_output.txt').write_text(changed.stdout, encoding='utf-8')
    assert changed.returncode != 0
    assert 'Resume changes training configuration' in changed.stdout
