import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code')]
import evaluate_d92_branch_metric_probe as entry
import summarize_d92_branch_metric_probe as summary
from test_evaluate_d92_branch_metric_probe import fixture


def write(path,value):
    path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')


def complete_fixture(root,ks=(1,2)):
    run=root/'run';run.mkdir();rows=[];cohorts={}
    for cohort in ('rx1','rx3'):
        for seed in range(2026092701,2026092705):
            rid=f'{cohort}-{seed}';lane=run/rid;lane.mkdir()
            args=fixture(lane,ks=ks);args['output']=lane/'probe';args['expected_model_seed']=seed
            for name in ('features_complete.json','startup.json','checkpoint_provenance.json'):
                path=args['support_features']/name;value=entry.read(path);value['model_seed']=seed
                if 'provenance' in value:value['provenance']['model_seed']=seed
                write(path,value)
            with contextlib.redirect_stdout(io.StringIO()):entry.evaluate(**args)
            rows.append(dict(row_id=rid,cohort=cohort,seeds=dict(model=seed),expected_checkpoint_sha256=args['expected_checkpoint_sha256']))
            cohorts[cohort]=dict(matrix=args['config']['matrix'],capsule_id=args['expected_capsule_id'],expected_split_count=len(ks))
    spec=dict(run_id='synthetic',execution=dict(remote_run_root=str(run)),rows=rows,probe=dict(cohorts=cohorts,total_episodes=8*len(ks)))
    write(run/'startup.json',dict(spec=spec,query_access=False,source_sample_access=False,commit='abc',started=1.))
    write(run/'complete.json',dict(status='SUPPORT_PROBE_COMPLETE',completed_rows=8,model_rows=8,episodes=8*len(ks),query_access=False,source_sample_access=False,commit='abc',finished=2.))
    write(run/'state.json',{r['row_id']:dict(status='SUPPORT_PROBE_COMPLETE',episodes=len(ks)) for r in rows})
    return spec,run


def test_complete_aggregation_reads_only_new_evidence_and_parent_weighted(tmp_path,monkeypatch):
    spec,run=complete_fixture(tmp_path,ks=(1,2,5))
    original=Path.open;opened=[]
    def guarded(path,*args,**kwargs):
        mode=args[0] if args else kwargs.get('mode','r')
        if 'r' in mode:
            opened.append(path)
            assert path.name in {'startup.json','complete.json','state.json','probe_complete.json','features_complete.json','fit_trace.jsonl','compact.jsonl'}
        return original(path,*args,**kwargs)
    with monkeypatch.context() as context:
        context.setattr(Path,'open',guarded)
        result=summary.summarize(spec=spec,output=tmp_path/'summary')
    assert opened
    assert result['coverage'] == dict(episodes=24,parent_episodes=24,k1_numerical_only=8,standard_oof_parents=16,proxy_parents=16,
        proxy_anchors=56,standard_factorizations=64,proxy_factorizations=56,actual_factorizations=120,proxy_held_occurrences_per_arm=352)
    proxy=next(r for r in result['statistics'] if r['diagnostic']=='support_oneshot_proxy' and r['metric']=='within_metric.h')
    assert proxy['count']==16  # Parent count, not 56 anchors or 352 held occurrences.
    expected=[]
    for row in spec['rows']:
        traces=list(summary.jsonlines(run/row['row_id']/'probe/fit_trace.jsonl'))
        expected.extend(r['oneshot_proxy']['parent_mean_metrics']['within_metric']['h'] for r in traces if r['k']>1)
    assert proxy['mean']==pytest.approx(sum(expected)/len(expected))
    assert result['candidate_assessment']['proxy_candidate_ncm_equivalent'] is True
    assert result['candidate_assessment']['standard']['passes'] is None  # Small fixture, no formal screen.
    assert not result['automatic_promotion'] and result['selected_arm'] is None
    assert (tmp_path/'summary/by_parent_k_newcount.csv').is_file()
    with pytest.raises(FileExistsError):summary.summarize(spec=spec,output=tmp_path/'summary')


@pytest.mark.parametrize('fault',['anchor','train_held','confusion','nll_count','paired_count','parent_mean','constant_factor','k1_proxy','equivalence','missing_parent'])
def test_reject_corrupt_or_missing_proxy_evidence(tmp_path,fault):
    spec,run=complete_fixture(tmp_path)
    path=run/spec['rows'][0]['row_id']/'probe/fit_trace.jsonl'
    records=list(summary.jsonlines(path));r=records[1];trial=r['oneshot_proxy']['trials'][0]
    if fault=='anchor':trial['trial']=1
    elif fault=='train_held':trial['training_ids'][0]=trial['held_ids'][0]
    elif fault=='confusion':trial['oof']['within_metric']['confusion'][0][0]+=1
    elif fault=='nll_count':trial['oof']['within_metric']['classwise_count'][0]+=1
    elif fault=='paired_count':trial['paired']['within_metric_minus_kernel_ncm']['counts']['both_wrong']+=1
    elif fault=='parent_mean':r['oneshot_proxy']['parent_mean_metrics']['within_metric']['h']+=.1
    elif fault=='constant_factor':trial['stages'][-1]['factorization_calls']=1
    elif fault=='k1_proxy':records[0]['oneshot_proxy']=deepcopy(r['oneshot_proxy'])
    elif fault=='equivalence':trial['ncm_equivalence']['tolerance']=1.
    elif fault=='missing_parent':records.pop()
    path.write_text('\n'.join(json.dumps(v) for v in records)+'\n',encoding='utf-8')
    with pytest.raises(ValueError):summary.summarize(spec=spec,output=tmp_path/'summary')
    assert not (tmp_path/'summary').exists()


def screening_stats():
    values={}
    for diagnostic,controls in [('physical_oof',('interaction_ridge','kernel_ncm')),('support_oneshot_proxy',('interaction_ridge',))]:
        for k in (5,10,20):
            for control in controls:
                for field in ('h','new_accuracy','old_accuracy'):
                    summary.add(values,(diagnostic,'new_present',k,'paired','within_metric_minus_'+control+'.'+field),.01)
    return values


@pytest.mark.parametrize('diagnostic,control',[('physical_oof','interaction_ridge'),('physical_oof','kernel_ncm'),('support_oneshot_proxy','interaction_ridge')])
@pytest.mark.parametrize('k',[5,10,20])
@pytest.mark.parametrize('metric,bad',[('h',0.),('new_accuracy',-.001),('old_accuracy',-.011)])
def test_separate_screen_cannot_hide_one_parent_k_failure(diagnostic,control,k,metric,bad):
    values=screening_stats()
    key=(diagnostic,'new_present',k,'paired','within_metric_minus_'+control+'.'+metric)
    values.pop(key);summary.add(values,key,bad)
    result=summary.assessment(values,True,True)
    failed='standard' if diagnostic=='physical_oof' else 'oneshot_proxy'
    other='oneshot_proxy' if failed=='standard' else 'standard'
    assert result[failed]['passes'] is False
    assert result[other]['passes'] is True


def test_proxy_requires_ncm_equality_instead_of_strict_improvement():
    values=screening_stats()
    assert summary.assessment(values,True,True)['oneshot_proxy']['passes'] is True
    assert summary.assessment(values,False,True)['oneshot_proxy']['passes'] is False
    assert summary.assessment(values,True,False)['oneshot_proxy']['passes'] is None


def test_true_k1_and_old_only_have_no_fabricated_metrics(tmp_path):
    args=fixture(tmp_path,ks=(1,2))
    arrays,tasks,old,*_=entry.load_support(support_features=args['support_features'],capsule=args['capsule'],
        expected_capsule_id=args['expected_capsule_id'],expected_checkpoint_sha256=args['expected_checkpoint_sha256'],
        expected_model_seed=args['expected_model_seed'],config=dict(algorithm=entry.CACHE_VALIDATION_CONFIG,matrix=args['config']['matrix']))
    for split,positions,labels in tasks:
        record=entry.probe_branch_metric(**{k:v[positions] for k,v in arrays.items()},support_labels=labels,
            support_ids=split['support_ids'],classes=split['registered_classes'],old_classes=split['registered_classes'])
        record=entry.compact_proxy(record,split['split_id'])
        record.update(split_id=split['split_id'],new_count=0,query_rows_used=0,source_rows_used=0)
        verified=summary.verify_record(record)
        if split['k']==1:assert verified['standard']==verified['proxy']=={}
        else:
            for population in ('standard','proxy'):
                for values in verified[population].values():assert values['new_accuracy'] is None and values['h'] is None
