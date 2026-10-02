"""Only repaired terminal closure/truth-last orchestration, using literal inputs."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code')]
import score_d92_support_metric_fixed_outputs_after_scalar_failure as repair


def literal_case(tmp_path,monkeypatch,*,all_failed=False):
    # Reuse the existing hand-built full 2400-parent fixture only. Its row loader
    # is a fixed-input oracle: no producer/model/fitter is run by these tests.
    path=ROOT/'tests/test_d92_support_metric_joint_benchmark_scoring.py'
    spec=importlib.util.spec_from_file_location('literal_support_metric_fixed_fixture',path)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    fixture.score=repair.fixed_score
    case=fixture.full_virtual_case(tmp_path,monkeypatch)
    terminal=fixture.score.read(case['root']/'complete.json')
    rows=case['spec']['rows'];completed=0;episodes=0
    for index,row in enumerate(rows):
        lane=terminal['rows'][row['row_id']];pid=700+index
        value=case['fixed'][Path(lane['prediction_output']).resolve()]
        value['complete']['pid']=pid
        value['complete']['state_manifest']='state_manifest.json'
        value['complete']['row_basis_construction_count']=1
        lane.update(pid=pid,process_returncode=0)
        if all_failed or index==0:
            lane.pop('marker');lane.update(status='FAILED',phase='PREDICTION',error_type='ValueError',
                error=repair.KNOWN_ERROR,preserved_prediction_output=lane['prediction_output'])
        else:
            lane['marker']=deepcopy(value['complete']);completed+=1
            episodes+=case['spec']['benchmark']['cohorts'][row['cohort']]['expected_split_count']
    terminal.update(status=repair.GLOBAL_FAILED,all_predictions_fixed=False,
        completed_row_count=completed,completed_episode_count=episodes)
    fixture.dump(case['root']/'complete.json',terminal)
    fixture.dump(case['root']/'state.json',terminal['rows'])
    case.update(dump=fixture.dump,terminal=terminal,
        original_bytes={name:(case['root']/name).read_bytes() for name in ('startup.json','state.json','complete.json')})
    return case


def guard_truth(case,monkeypatch,*,allowed):
    read=repair.fixed_score.read;calls=[]
    def guarded(path):
        if str(path) in case['truth_paths']:
            assert allowed,'Truth must remain inaccessible on any closure failure'
            assert len(case['validations'])==8,'All four rows and independent rereads must precede truth'
            calls.append(str(path))
        return read(path)
    monkeypatch.setattr(repair.fixed_score,'read',guarded)
    return calls


@pytest.mark.parametrize('all_failed',[False,True])
def test_complete_fixed_outputs_scored_only_after_all_closure_preserving_original_failed_state(tmp_path,monkeypatch,all_failed):
    case=literal_case(tmp_path,monkeypatch,all_failed=all_failed);output=tmp_path/'derived'
    truth=guard_truth(case,monkeypatch,allowed=True)
    result=repair.score_fixed_outputs(spec=case['spec'],output=output)
    assert result['row_count']==4 and result['parent_count']==2400 and len(truth)==2
    assert result['status']==repair.fixed_score.STATUS and result['method']==repair.fixed_score.METHOD
    assert result['original_supervisor_status']==repair.GLOBAL_FAILED and result['original_status_unchanged']
    closure=repair.fixed_score.read(output/'closure.json')
    assert closure['status']==repair.STATUS and closure['parent_count']==2400 and closure['row_count']==4
    assert closure['truth_read'] is False and closure['second_readback_complete']
    assert closure['original_completed_row_count']==(0 if all_failed else 3)
    assert all(p['metrics']['A_old_accuracy']==.5 for p in result['parents'])
    assert all(p['metrics']['R0_C_old_accuracy']==1/6 for p in result['parents'])
    assert result['statistics']['by_k_new_count'] and result['statistics']['model_seed_mean_sd']
    assert result['resources']['C_public_predict_internal_work'] is None
    assert result['resources']['actual_work_aggregation']=={
        k:'MAX' if k in repair.fixed_score.PEAK_COUNTERS else 'SUM' for k in repair.fixed_score.WORK_KEYS}
    assert repair.fixed_score.read(output/'summary.json')==result
    for name,data in case['original_bytes'].items():assert (case['root']/name).read_bytes()==data


@pytest.mark.parametrize('fault',['different_error','nonzero_exit','missing_row','incomplete_2400',
    'second_read_changed','wrong_PID','wrong_complete_marker','nonterminal','state_changed'])
def test_any_incomplete_or_different_failure_blocks_all_truth(tmp_path,monkeypatch,fault):
    case=literal_case(tmp_path,monkeypatch);terminal=deepcopy(case['terminal'])
    first=next(iter(terminal['rows']));lane=terminal['rows'][first]
    if fault=='different_error':lane['error']='Some other numerical error'
    elif fault=='nonzero_exit':lane['process_returncode']=1
    elif fault=='missing_row':terminal['rows'].pop(first)
    elif fault=='wrong_PID':lane['pid']+=1
    elif fault=='wrong_complete_marker':
        completed=next(v for v in terminal['rows'].values() if v['status']=='COMPLETE')
        completed['marker']=dict(completed['marker'],pid=99999)
    elif fault=='nonterminal':terminal['status']='SUPPORT_METRIC_QUERY_BENCHMARK_STARTED'
    elif fault=='incomplete_2400':case['fixed'][list(case['fixed'])[-1]]['splits'].popitem()
    elif fault=='second_read_changed':
        loader=repair.fixed_score.load_fixed_predictions
        def changed(**kwargs):
            value=loader(**kwargs)
            if len(case['validations'])==8:value['streams']['B'][next(iter(value['splits']))][0]['prediction']='different'
            return value
        monkeypatch.setattr(repair.fixed_score,'load_fixed_predictions',changed)
    case['dump'](case['root']/'complete.json',terminal)
    states=deepcopy(terminal['rows'])
    if fault=='state_changed':states[first]['error']='changed'
    case['dump'](case['root']/'state.json',states)
    truth=guard_truth(case,monkeypatch,allowed=False)
    output=tmp_path/'rejected'
    with pytest.raises(ValueError):repair.score_fixed_outputs(spec=case['spec'],output=output)
    assert truth==[] and not (output/'summary.json').exists()
    assert repair.fixed_score.read(output/'failure.json')['status']=='TECHNICAL_FAILURE'


def test_exclusive_output_and_original_run_write_boundary(tmp_path):
    run=tmp_path/'original';run.mkdir();output=tmp_path/'existing';output.mkdir()
    spec=dict(execution=dict(remote_run_root=str(run)))
    with pytest.raises(ValueError,match='Exclusive'):repair.score_fixed_outputs(spec=spec,output=output)
    with pytest.raises(ValueError,match='outside original'):repair.score_fixed_outputs(spec=spec,output=run/'new')
    assert not (run/'new').exists()
