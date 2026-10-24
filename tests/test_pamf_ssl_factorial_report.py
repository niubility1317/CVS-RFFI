from copy import deepcopy
import json
from pathlib import Path
import pytest
from experiments.pamf_ssl_research.factorial_report import MATRICES,analyze,complete_report
from experiments.pamf_ssl_research.matched_report import SEEDS,VIEWS,READOUTS,DATA_KEYS


def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v),encoding='utf-8')


def fixture(root):
    bases={'joint':[140000,140100,140200],'path':[141000,141100,141200],
           'As':[141500,141600,141700],'combined':[143000,143300,143200]}
    for key,(run,method) in MATRICES.items():
        folder=root/run
        save(folder/'evaluation_completion.json',dict(status='VERIFIED_COMPLETE',run_id=run,training_execution_commit='a'*40,
            independent_truth_last=True,completed_F_models=3,readouts=list(READOUTS),views_per_readout=7,count_per_view=168000,
            completed_readout_view_cells=42,fixed_G_version='dura_v1'))
        save(folder/'experiment.json',dict(run_id=run,data={k:'same' for k in DATA_KEYS}))
        save(folder/'source_training_complete.json',dict(cumulative_updates_per_model=11100,model_seeds=list(SEEDS),
            rows=[dict(method=method,source_epoch=50,source_updates=11100,actual_adapter_SSL=dict(default_head_joint=True,U_truth_read=False)) for _ in SEEDS]))
        cells=[]
        for readout in READOUTS:
            for view in VIEWS:
                factor=6 if view=='six_mean' else 1;counts=[factor*(v+(100 if readout=='DURA-LU' else 0)) for v in bases[key]]
                cells.append(dict(method=method,readout=readout,view=view,dimension='overall',stratum='ALL',count_per_seed=168000*factor,
                    correct_total=sum(counts),mean_accuracy_percent=100*sum(counts)/(3*168000*factor),**{'correct_s'+str(s):v for s,v in zip(SEEDS,counts)}))
        save(folder/'target_summary.json',dict(run_id=run,model_seeds=list(SEEDS),class_order=list(range(6)),summary=cells))
    return bases


def test_waits_for_all_four_completion_records_before_reading_any_score_summary(tmp_path):
    fixture(tmp_path);(tmp_path/MATRICES['As'][0]/'evaluation_completion.json').unlink()
    (tmp_path/MATRICES['joint'][0]/'target_summary.json').write_text('scores must not be read',encoding='utf-8')
    result=analyze(tmp_path)
    assert result['status']=='WAIT_COMPLETE_FACTORIAL' and result['scores_read'] is False
    assert complete_report(tmp_path)['status']=='WAIT_COMPLETE_FACTORIAL'
    assert not (tmp_path/MATRICES['combined'][0]/'factorial_comparison.json').exists()


def test_exact_three_neighbor_differences_and_nonadditive_recipe_interaction(tmp_path):
    fixture(tmp_path);r=analyze(tmp_path)
    assert r['status']=='VERIFIED_COMPLETE_FACTORIAL' and len(r['neighbors'])==48 and len(r['interaction'])==16
    v=next(x for x in r['neighbors'] if (x['reference'],x['readout'],x['view'])==('path','raw','clean'))
    assert [x['net_correct'] for x in v['per_seed']]==[2000,2200,2000]
    assert v['net_correct_total']==6200 and v['paired_delta_pp_mean']==pytest.approx(100*6200/504000)
    i=next(x for x in r['interaction'] if (x['readout'],x['view'])==('raw','clean'))
    assert [x['interaction_correct'] for x in i['per_seed']]==[500,700,500]
    assert i['interaction_pp_mean']==pytest.approx(100*1700/504000)
    assert i['interaction_pp_sample_sd']>0 and 'F20/F21' in r['primary_goal_unchanged']
    out=complete_report(tmp_path);assert out['status']=='VERIFIED_COMPLETE_FACTORIAL'
    assert (tmp_path/MATRICES['combined'][0]/'factorial_comparison.md').is_file()
    with pytest.raises(FileExistsError):complete_report(tmp_path)


@pytest.mark.parametrize('failure',['contract','view_count','six_sum','seed'])
def test_rejects_incomparable_or_incomplete_matrices(tmp_path,failure):
    fixture(tmp_path);folder=tmp_path/MATRICES['path'][0]
    name='experiment.json' if failure=='contract' else 'evaluation_completion.json' if failure=='view_count' else 'target_summary.json'
    p=folder/name;v=json.loads(p.read_text())
    if failure=='contract':v['data']['split_id']='other'
    elif failure=='view_count':v['completed_readout_view_cells']=28
    elif failure=='seed':v['model_seeds']=list(SEEDS[:2])
    else:
        c=next(x for x in v['summary'] if x['readout']=='raw' and x['view']=='six_mean');c['correct_s'+str(SEEDS[0])]+=1
        c['correct_total']+=1;c['mean_accuracy_percent']=100*c['correct_total']/(3*c['count_per_seed'])
    save(p,v)
    with pytest.raises(ValueError):analyze(tmp_path)
