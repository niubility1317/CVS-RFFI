from copy import deepcopy
import json
from pathlib import Path
import statistics
import tempfile
import pytest
from experiments.pamf_ssl_research.matched_report import (
    DATA_KEYS,HARD_RUN,PAIRS,READOUTS,SEEDS,VIEWS,compare,complete_report)


def fixture(run=next(iter(PAIRS))):
    method,control,_=PAIRS[run];summaries=[];specs=[];completions=[]
    for candidate,name,rid in ((True,method,run),(False,control,HARD_RUN)):
        rows=[]
        for readout in READOUTS:
            for view in VIEWS:
                multiplier=6 if view=='six_mean' else 1;denom=168000*multiplier
                values=[multiplier*(100000+10*i+(4000 if readout=='DURA-LU' else 0)+(10*(i+1) if candidate else 0)) for i in range(3)]
                rows.append(dict(method=name,readout=readout,view=view,dimension='overall',stratum='ALL',
                    count_per_seed=denom,correct_total=sum(values),mean_accuracy_percent=100.*sum(values)/(3*denom),
                    **{'correct_s'+str(seed):value for seed,value in zip(SEEDS,values)}))
        summaries.append(dict(run_id=rid,model_seeds=list(SEEDS),class_order=list(range(6)),summary=rows))
        specs.append(dict(run_id=rid,data={key:'same:'+key for key in DATA_KEYS}))
        completions.append(dict(run_id=rid,status='VERIFIED_COMPLETE',independent_truth_last=True,
            readouts=list(READOUTS),views_per_readout=7,count_per_view=168000,
            completed_F_models=3 if candidate else 6,completed_readout_view_cells=42 if candidate else 84))
    return summaries,specs,completions


@pytest.mark.parametrize('run',tuple(PAIRS))
def test_exact_three_seed_paired_delta_SD_and_both_full_readouts(run):
    summaries,specs,done=fixture(run);result=compare(*summaries,*specs,*done)
    assert len(result['rows'])==16 and result['reference_run']==HARD_RUN
    for row in result['rows']:
        expected=[100.*(10*(i+1))/168000 for i in range(3)]
        assert row['paired_delta_pp_mean']==pytest.approx(statistics.mean(expected))
        assert row['paired_delta_pp_sample_sd']==pytest.approx(statistics.stdev(expected))
        assert row['positive_seeds']==3 and [r['model_seed'] for r in row['per_seed']]==list(SEEDS)
        assert row['net_correct_total']==(360 if row['view']=='six_mean' else 60)
    assert 'F20/F21' in result['primary_goal_unchanged'] and 'goal_met' not in result


def test_incomplete_matrix_returns_wait_before_opening_any_score_summary():
    run=next(iter(PAIRS))
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);folder=root/run;folder.mkdir()
        (folder/'evaluation_completion.json').write_text(json.dumps(dict(status='VERIFIED_COMPLETE')),encoding='utf-8')
        result=complete_report(root,run)
        assert result==dict(status='WAIT_COMPLETE_MATCHED_REPORTS',missing_run=HARD_RUN,scores_read=False)
        assert not (folder/'matched_comparison.json').exists()


def test_mismatched_target_contract_and_missing_seed_are_rejected():
    summaries,specs,done=fixture();bad=deepcopy(specs);bad[0]['data']['physical_ids_ref']='different'
    with pytest.raises(ValueError,match='physical data contract'):compare(*summaries,*bad,*done)
    bad=deepcopy(summaries);bad[0]['model_seeds']=list(SEEDS[:2])
    with pytest.raises(ValueError,match='full target matrices'):compare(*bad,*specs,*done)


def test_six_scene_aggregate_cannot_hide_missing_scene_counts():
    summaries,specs,done=fixture();row=next(r for r in summaries[0]['summary'] if r['readout']=='raw' and r['view']=='six_mean')
    row['correct_s'+str(SEEDS[0])]+=1;row['correct_total']+=1
    row['mean_accuracy_percent']=100.*row['correct_total']/(3*row['count_per_seed'])
    with pytest.raises(ValueError,match='sum of all six'):compare(*summaries,*specs,*done)


def test_complete_report_writes_both_artifacts_and_preserves_existing_output():
    summaries,specs,done=fixture();run=summaries[0]['run_id']
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        for summary,spec,completion in zip(summaries,specs,done):
            folder=root/summary['run_id'];folder.mkdir()
            for name,value in (('target_summary.json',summary),('experiment.json',spec),('evaluation_completion.json',completion)):
                (folder/name).write_text(json.dumps(value),encoding='utf-8')
        result=complete_report(root,run)
        assert result['status']=='VERIFIED_PAIRED_MATCHED_COMPARISON'
        report=(root/run/'matched_comparison.md').read_text(encoding='utf-8')
        assert 'DURA-LU' in report and 'raw' in report and 'F20' in report
        with pytest.raises(FileExistsError):complete_report(root,run)
