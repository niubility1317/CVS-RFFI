import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_core90_cross_response_experiment import choose_gpu, inspect_completion, failure_fingerprint, SCENES
from scripts.run_core90_cross_response_experiment import training_command, detached_command, CODE, source_cell_counts, source_block_feasibility


def test_commands_bind_immutable_release_not_project_root():
    argv, cwd = training_command('/python', ['--output_dir','/project/runs/new/U0'])
    assert argv[1] == str(CODE/'SSDG/train_ssdg.py')
    assert cwd == CODE
    argv, cwd = detached_command('/python', ['--project-root','/old/project','--detach'])
    assert Path(argv[1]).parent == CODE/'scripts'
    assert cwd == CODE.parent and '--detach' not in argv


def test_k_cannot_aggregate_across_day_or_condition():
    rows = [SimpleNamespace(tx_i=0,rx_i=1,day_i=day,eq_i=eq) for day in (1,2) for eq in (0,1)]
    with pytest.raises(ValueError,match='day/condition'):
        source_cell_counts(SimpleNamespace(index=rows),range(4),2,'V_s')


def test_validation_checks_complete_blocks_at_frozen_budget():
    from cvsrffi.cross_response.schema import SampleRecord
    records = [SampleRecord(i,t,r,1,0,(t,r,k)) for i,(t,r,k) in enumerate((t,r,k) for t in range(4) for r in range(4) for k in range(2))]
    config = dict(P=4,Q=4,K=2,data_seed=392005,source_eval_max_blocks=16,scheduler_candidate_limit=512)
    result = source_block_feasibility(records,config,validation=True)
    assert result['candidate_count'] == 1 and result['valid_blocks'] == 16
    with pytest.raises(ValueError,match='16 complete blocks'):
        source_block_feasibility(records[:-1],config,validation=True)


def test_gpu_reserves_unregistered_new_jobs_and_ignores_existing_total():
    jobs = [dict(gpu='GPU-a', pid=51)]
    assert choose_gpu(['GPU-a'], jobs, {'GPU-a':12000}, {30:'GPU-a',31:'GPU-a'}, minimum=6500, limit=2) is None
    assert choose_gpu(['GPU-a'], jobs, {'GPU-a':12000}, {30:'GPU-a',31:'GPU-a',51:'GPU-a'}, minimum=6500, limit=2) == 'GPU-a'
    jobs.append(dict(gpu='GPU-a', pid=52))
    assert choose_gpu(['GPU-a'], jobs, {'GPU-a':20000}, {51:'GPU-a',52:'GPU-a'}, minimum=6500, limit=2) is None


def make_artifacts(tmp_path):
    pred = tmp_path/'final_predictions'
    pred.mkdir()
    (tmp_path/'final_ssdg.pth').write_bytes(b'fixture')
    (pred/'prediction_manifest.json').write_text(json.dumps(dict(state='PREDICTIONS_FIXED', scenarios=list(SCENES), record_count=2, rows_written=8)))
    (pred/'independent_scores.json').write_text(json.dumps(dict(status='COMPLETE', truth_last=True, prediction_rows=8)))
    (tmp_path/'cross_response_activation.json').write_text(json.dumps(dict(status='ACTIVE_VERIFIED', missing=[])))
    for scene in SCENES:
        (pred/f'predictions.{scene}.jsonl').write_text('{}\n{}\n')
    return pred


def test_completion_requires_all_artifacts_and_full_scene_counts(tmp_path):
    assert inspect_completion(tmp_path,0)['status'] == 'FAILED'
    pred = make_artifacts(tmp_path)
    assert inspect_completion(tmp_path,0)['status'] == 'COMPLETE'
    assert inspect_completion(tmp_path,12)['status'] == 'FAILED'
    (pred/'predictions.clean.jsonl').write_text('{}\n')
    assert inspect_completion(tmp_path,0)['reason'] == 'ARTIFACT_VALIDATION_FAILED'


def test_activation_failure_not_claimed_complete(tmp_path):
    make_artifacts(tmp_path)
    (tmp_path/'cross_response_activation.json').write_text(json.dumps(dict(status='INACTIVE', missing=['domain_gradient_steps'])))
    assert inspect_completion(tmp_path,12)['reason'] == 'CROSS_RESPONSE_ACTIVATION_INCOMPLETE'


def test_fingerprint_normalizes_volatile_numbers(tmp_path):
    a,b = tmp_path/'a.log',tmp_path/'b.log'
    a.write_text('RuntimeError: CUDA out of memory. Tried to allocate 123 MiB at 0xabc\n')
    b.write_text('RuntimeError: CUDA out of memory. Tried to allocate 456 MiB at 0xdef\n')
    assert failure_fingerprint(a) == failure_fingerprint(b)
