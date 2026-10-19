import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.pamf_formal_f50 import runner as r


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def inputs(tmp_path):
    ids = np.array(['record1', 'record2'])
    coarse, residual = tmp_path/r.COARSE_RUN, tmp_path/r.RESIDUAL_RUN
    for root, commit, candidate in ((coarse, r.COARSE_COMMIT, 'tf_unet_mixed'),
                                    (residual, r.RESIDUAL_COMMIT, 'residual_deterministic')):
        write(root/'complete.json', dict(status='ANALYZED', count=2, training_updates=0,
              adaptation_updates=0, truth_last=True))
        write(root/'launch.json', dict(commit=commit))
        folder = root/candidate/'prediction'
        write(folder/'complete.json', dict(status='PREDICTIONS_COMPLETE', candidate=candidate, count=2,
              views=list(r.VIEWS), TX_truth_reads=0, state_updates=0))
        np.save(folder/'physical_ids.npy', ids)
        for view in r.VIEWS:
            np.save(folder/(view+'.npy'), np.ones((2, 2, 256), np.float32))
    folder = residual/'reference'
    write(folder/'manifest.json', dict(status='FORMAL_REFERENCE_READY', count=2, views=list(r.VIEWS),
          original_IQ_bytes_reused=True, TX_truth_reads=0))
    np.save(folder/'physical_ids.npy', ids)
    for view in r.VIEWS:
        np.save(folder/(view+'.npy'), np.ones((2, 2, 256), np.float32))
    return coarse, residual, ids


def test_completed_IQ_reuses_fixed_seven_views_and_original_IDs(tmp_path):
    coarse, residual, expected = inputs(tmp_path)
    ids, paths = r.completed_inputs(coarse, residual, expected_count=2)
    np.testing.assert_array_equal(ids, expected)
    assert set(paths) == set(r.MODES)
    np.save(paths['coarse']/'physical_ids.npy', expected[::-1])
    with pytest.raises(ValueError):
        r.completed_inputs(coarse, residual, expected_count=2)


@pytest.mark.parametrize('mutation', ['commit', 'state_updates', 'views'])
def test_completed_IQ_rejects_wrong_code_or_unfrozen_prediction(tmp_path, mutation):
    coarse, residual, _ = inputs(tmp_path)
    path = coarse/'launch.json' if mutation == 'commit' else coarse/'tf_unet_mixed/prediction/complete.json'
    value = json.loads(path.read_text())
    value[mutation] = {'commit': 'unverified', 'state_updates': 1, 'views': ['clean']}[mutation]
    write(path, value)
    with pytest.raises(ValueError):
        r.completed_inputs(coarse, residual, expected_count=2)


def test_incomplete_readout_cannot_open_truth(tmp_path):
    with pytest.raises(FileNotFoundError):
        r.prediction_arrays(tmp_path, 2)
    write(tmp_path/'complete.json', dict(status='PREDICTIONS_COMPLETE', count=2, views=['clean']))
    with pytest.raises(ValueError):
        r.prediction_arrays(tmp_path, 2)


def test_all_readout_predictions_and_IDs_validate_before_truth(tmp_path):
    ids = np.array(['a', 'b'])
    write(tmp_path/'complete.json', dict(status='PREDICTIONS_COMPLETE', count=2, views=list(r.VIEWS),
        readouts=list(r.MODES), TX_truth_reads=0, state_updates=0, original_default_head=True,
        F_run=r.REFERENCE_RUN, F_seed=2026092701, restored_IQ_reused=True))
    np.save(tmp_path/'physical_ids.npy', ids)
    for view in r.VIEWS:
        for mode in r.MODES:
            np.save(tmp_path/(view+'.'+mode+'.logits.npy'), np.zeros((2, 6), np.float32))
            np.save(tmp_path/(view+'.'+mode+'.features.npy'), np.zeros((2, 160), np.float32))
    r.prediction_arrays(tmp_path, 2, ids)
    with pytest.raises(ValueError):
        r.prediction_arrays(tmp_path, 2, ids[::-1])
    path = tmp_path/'practical_low_suburban.residual.logits.npy'
    np.save(path, np.full((2, 6), np.nan, np.float32))
    with pytest.raises(ValueError):
        r.prediction_arrays(tmp_path, 2, ids)


def test_entry_and_original_F50_three_way_readout_fixed_without_R07():
    value = subprocess.run([sys.executable, '-m', 'experiments.pamf_formal_f50.runner', '--help'], cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=''), capture_output=True, text=True, timeout=30)
    assert value.returncode == 0, value.stderr
    config = r.configuration()
    assert config['count'] == 168000 and config['F_epochs'] == 50 and config['F_updates'] == 11100
    assert config['new_training_updates'] == config['adaptation_updates'] == config['head_fit_updates'] == config['channel_generation_calls'] == 0
    assert config['no_R07_result_or_checkpoint_use'] and config['truth_last']
    assert config['GPU'] == 4 and config['capacity_lock'] == 'receiver_residual_capacity4.lock'


def test_unstarted_resource_replacement_keeps_the_scientific_matrix():
    old, new = r.configuration(), r.configuration('R09_gpu5')
    assert new['GPU'] == 5 and new['run'] == r.REPLACEMENT_RUN
    assert new['replaces_unstarted_run'] == r.RUN
    changed = {key for key in old if old[key] != new[key]}
    assert changed == {'run', 'GPU', 'registered_plan', 'replaces_unstarted_run'}
    with pytest.raises(ValueError):
        r.plan_settings('arbitrary_GPU')
