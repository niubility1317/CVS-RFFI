"""Literal complete query-score reporting; no external artifacts or model calls."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import report_d92_margin_joint_benchmark as report
import score_d92_margin_joint_benchmark as scorer
from test_score_d92_margin_joint_benchmark import synthetic_bundle,COMMIT,CHECKPOINT


@pytest.fixture
def scored(tmp_path):
    case=synthetic_bundle(tmp_path)
    return scorer.score_benchmark(spec=case['spec'],output=case['output'])


def test_complete_report_percent_pp_full_matrix_K1_and_new0(scored):
    text,value=report.report_benchmark(score=scored)
    assert value['status']==report.REPORT_STATUS and len(value['tables']['by_k_new_count'])==20
    assert value['automatic_promotion'] is value['performance_improvement_claimed'] is False
    assert '（%）' in text and '（pp）' in text and 'N/A' in text
    assert '新的独立确认' in text and 'K1 有独立 query' in text
    assert value['release_commit']!=value['code_commit']
    for row in value['tables']['by_k_new_count']:
        assert row['values']['A_old_accuracy']==.5
        if row['new_count']==0:
            assert row['values']['C_new_accuracy'] is row['values']['C_h'] is row['values']['C_abs_new_old_gap'] is None


def test_parent_first_H_and_absolute_gap_are_preserved():
    metrics=dict.fromkeys(scorer.METRICS,0.)
    parents=[]
    for old,new in ((1/3,1.),(1.,1/3)):
        parents.append(dict(model_seed=7,row_id='synthetic',receiver='rx',scenario='scene',support_seed=11,k=5,new_count=2,
            metrics=dict(metrics,C_old_accuracy=old,C_new_accuracy=new,C_h=2*old*new/(old+new),C_abs_new_old_gap=abs(old-new))))
    cells=report._cells(scorer._statistics(parents)['by_k_new_count'],('k','new_count'))
    assert cells[0]['values']['C_h']==pytest.approx(.5)
    assert cells[0]['values']['C_h']!=pytest.approx(2/3)
    assert cells[0]['values']['C_abs_new_old_gap']==pytest.approx(2/3)


@pytest.mark.parametrize('kind',['status','runtime','matrix','gain','H','new0','K1','source','truth_last','aggregate'])
def test_report_refuses_incomplete_unpaired_or_fabricated_scores(scored,kind):
    bad=deepcopy(scored);p=next(v for v in bad['parents'] if v['new_count']>0)
    if kind=='status':bad['status']='PARTIAL'
    elif kind=='runtime':bad['release_commit']='c'*40
    elif kind=='matrix':bad['parents'].pop();bad['parent_count']-=1
    elif kind=='gain':p['metrics']['adaptation_gain_B_minus_A']+=.01
    elif kind=='H':p['metrics']['C_h']+=.01
    elif kind=='new0':next(v for v in bad['parents'] if not v['new_count'])['metrics']['C_h']=0.
    elif kind=='K1':next(v for v in bad['parents'] if v['k']==1)['metrics']['A_old_accuracy']=None
    elif kind=='source':p['checkpoint_sha256']='2'*64
    elif kind=='truth_last':bad['prediction_validation_complete_before_truth']=False
    else:bad['statistics']['overall'][0]['mean']=.123
    with pytest.raises(ValueError):report.report_benchmark(score=bad)


def legacy_fixture(current):
    method='D92-BranchLocalRidge-v1';row=current['current_metadata']['spec']['rows'][0]
    records=[]
    for p in current['parents']:
        records.append(dict(row_id='original_model_row',model_seed=p['model_seed'],method=method,
            **{k:p[k] for k in ('split_id','receiver','scenario','k','new_count','support_seed','query_count')},
            classes=p['old_classes']+[v for v in p['registered_classes'] if v not in p['old_classes']],
            class_count=p['registered_class_count'],old_accuracy=.5,new_accuracy=.5 if p['new_count'] else None,
            harmonic_mean=.5 if p['new_count'] else None))
    spec=dict(run_id='explicit_completed_baseline',rows=[dict(row_id='original_model_row',seeds=dict(model=row['expected_model_seed']),
        expected_checkpoint_sha256=CHECKPOINT,reuse_row_root=row['row_root'])],
        data=dict(capsule_id='synthetic_capsule',target_receivers=['synthetic_rx'],scenarios=['synthetic_scene'],
            k=list(scorer.KS),new_class_counts=list(scorer.NEWS),support_seeds=[11]),
        confirmation=dict(candidate_method=method,reuse_validated_capsule_id='synthetic_capsule'))
    scored=dict(status='SCORED',results=records,selection_feedback_forbidden=True)
    meta=dict(method=method,spec=spec,startup=dict(spec=spec,commit='d'*40),
        complete=dict(status='SCORED',commit='d'*40,model_rows=1,records=len(records),selection_feedback_forbidden=True))
    return scored,meta


def test_explicit_legacy_baseline_pairs_C_only_and_missing_stage_NA(scored):
    prior,meta=legacy_fixture(scored)
    text,value=report.report_benchmark(score=scored,baseline_score=prior,baseline_metadata=meta)
    assert len(value['baseline_comparisons'])==20
    assert 'Legacy SCORED/results' in text
    for row in value['baseline_comparisons']:
        for name in ('A_old_accuracy','B_old_accuracy','adaptation_gain_B_minus_A','total_old_accuracy_drop'):
            assert row['baseline_metrics'][name] is row['differences'][name] is None
        assert row['baseline_runtime_commit']=='d'*40
        assert row['differences']['C_old_accuracy'] is not None


@pytest.mark.parametrize('kind',['incomplete','method','row','capsule','checkpoint','split','classes','count','source','query_ids'])
def test_legacy_baseline_identity_is_checked_without_guessing(scored,kind):
    prior,meta=legacy_fixture(scored)
    if kind=='incomplete':meta['complete']['status']='PREDICTIONS_COMPLETE'
    elif kind=='method':meta['method']='other'
    elif kind=='row':prior['results'][0]['row_id']='other'
    elif kind=='capsule':meta['spec']['data']['capsule_id']='other'
    elif kind=='checkpoint':meta['spec']['rows'][0]['expected_checkpoint_sha256']='2'*64
    elif kind=='split':prior['results'][0]['split_id']='other'
    elif kind=='classes':prior['results'][0]['classes'][0]='other'
    elif kind=='count':prior['results'][0]['query_count']+=1
    elif kind=='source':meta['spec']['rows'][0]['reuse_row_root']='other'
    else:prior['results'][0]['query_ids']=['other']
    with pytest.raises(ValueError):report.report_benchmark(score=scored,baseline_score=prior,baseline_metadata=meta)


def test_baseline_requires_explicit_metadata_and_current_metadata_match(scored):
    prior,_=legacy_fixture(scored)
    with pytest.raises(ValueError):report.report_benchmark(score=scored,baseline_score=prior)
    with pytest.raises(ValueError):report.report_benchmark(score=scored,current_metadata={})
    # No automatic search: omitting a baseline produces no comparison.
    assert report.report_benchmark(score=scored)[1]['baseline_comparisons'] is None
