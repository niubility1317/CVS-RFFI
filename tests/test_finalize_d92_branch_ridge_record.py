"""Only full synthetic matrices and metadata, never actual target artifacts."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import finalize_d92_branch_ridge_record as target
from summarize_d92_confirmation import summarize
from summarize_d92_repeated_benchmark import combine

# Reuse only the synthetic score factory in an isolated module namespace.
_spec=importlib.util.spec_from_file_location('_branch_archive_test_factory',ROOT/'tests/test_finalize_d92_mvridge_record.py')
factory=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(factory)
factory.target=SimpleNamespace(KEY=target.KEY,METHODS=target.METHODS,METRICS=target.METRICS)


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')


def spec(cohort):
    return dict(run_id=target.RUNS[cohort],permissions=dict(claim_scope='REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS'),
        rows=[dict(row_id=str(seed),seeds=dict(model=seed),source_root='/source/'+str(seed),reuse_row_root='/original/'+cohort+'/'+str(seed),
            expected_checkpoint_sha256=f'{seed:064x}') for seed in range(2026092701,2026092705)],
        data=dict(capsule_id=cohort,target_receivers=['a','b','c'] if cohort=='rx3' else ['d'],scenarios=['x','y','z'],
            k=[1,5,10,20],new_class_counts=[0,2,5,10,20],support_seeds=[11,12,13,14,15]),
        confirmation=dict(candidate_method=target.METHODS[1],candidate_folder='branch_ridge',candidate=copy.deepcopy(target.FROZEN_CONFIG),
            candidate_predictor='evaluate_d92_branch_ridge.py',candidate_mode='d92_branch_ridge_registration',capsule='/capsule/'+cohort,reuse_validated_capsule_id=cohort),
        metrics_plan=dict(acceptance=dict(old_max_drop=.01,require_new_improvement=True)),
        joint_benchmark=dict(old_accuracy_scope='All paired cells at this K, including old-only registration'))


@pytest.fixture(scope='module')
def synthetic():
    specs=[spec(c) for c in target.RUNS];scores=[factory.score_rows(s) for s in specs]
    inputs={c:dict(scores=d,summary=summarize(d,s)) for c,s,d in zip(target.RUNS,specs,scores)}
    return inputs,combine(specs,scores),specs


def stat(value,count,low=None,high=None):
    return dict(min=value if low is None else low,mean=value,max=value if high is None else high,total=value*count)


def fit_audit():
    runs=[];all_models=[]
    for cohort in target.RUNS:
        s=spec(cohort);n=900 if cohort=='rx3' else 300;models=[]
        for row in s['rows']:
            count=100 if cohort=='rx3' else 50
            resource=dict(checkpoint_sha256=row['expected_checkpoint_sha256'],existing_model_file_bytes=15992872+row['seeds']['model']%4,
                feature_array_bytes=2944*count,feature_file_bytes=3000*count,registry_array_bytes=80*count,count=count,
                native_physical_forward_count=count,native_batch_calls=count,smoke_forward_count=1,smoke_batch_calls=1,
                native_total_physical_forward_count=count+1,native_total_batch_calls=count+1,support_cache_reuse_count=0,
                extraction_timing=dict(total_seconds=2.,native_forward_seconds=1.),export_peak_rss_bytes=1000000,
                model_incremental_transfer_bytes=None,new_source_payload_bytes=0,new_ground_statistics_bytes=0)
            models.append(dict(row_id=row['row_id'],model_seed=row['seeds']['model'],fits=n,k1_fits=n//4,full_fit_calls=n,factorizations=n,
                solver_counts={'dual':n},status_counts={'CLOSED_FORM_SOLVED':n},resources=resource,
                fit_measurements={k:stat(v,n) for k,v in dict(loss_data=1.,loss_ridge=.5,loss_total=1.5,gradient_norm=1e-13,
                    normal_equation_residual=1e-13,intercept_gradient_norm=1e-15).items()},
                timing={k:stat(v,n) for k,v in dict(fit_seconds=.1,fit_call_seconds=.11,query_score_seconds=.01,
                    prediction_write_seconds=.001,total_seconds=.13).items()},
                state_bytes={k:stat(5896*13.4,n,5896*6,5896*26) for k in ('head_bytes','persistent_state_bytes')},peak_process_rss_bytes=100000))
        runs.append(dict(status='VERIFIED',run_id=s['run_id'],models=models,commit='synthetic',total_fits=n*4,k1_fits=n,
            run_wall_seconds=20.,run_started=100. if cohort=='rx3' else 105.,run_finished=120. if cohort=='rx3' else 125.))
        all_models.extend(models)
    fit=dict(status='VERIFIED',method=target.METHODS[1],runs=runs,total_fits=4800,k1_fits=1200,full_fit_calls=4800,factorizations=4800,
        fold_fit_calls=0,optimizer_steps=0,oof=None,feature_dim=736,numeric_head_bytes_formula='8*(736+1)*C = 5896*C',
        query_rows_used_for_fit=0,source_rows_used_for_fit=0,new_source_payload_bytes=0,new_ground_statistics_bytes=0,
        selection='fixed_no_selection',raw_traces_preserved=True,parameter_feedback_forbidden=True,model_incremental_transfer_bytes=None,
        summed_process_work_seconds={k:sum(m['timing'][k]['total'] for m in all_models) for k in all_models[0]['timing']},
        summed_extraction_seconds=16.,summed_extraction_stage_seconds=dict(total_seconds=16.,native_forward_seconds=8.),
        joint_wall_span_seconds=25.,summed_cohort_wall_seconds=40.,max_observed_process_rss_bytes=1000000,
        existing_unique_model_file_bytes=sum(m['resources']['existing_model_file_bytes'] for m in runs[0]['models']),unique_model_count=4)
    for field,key in [('received_feature_array_bytes','feature_array_bytes'),('received_feature_file_bytes','feature_file_bytes'),
        ('received_registry_array_bytes','registry_array_bytes'),('received_native_forward_count','native_physical_forward_count'),
        ('synthetic_smoke_forward_count','smoke_forward_count'),('actual_total_native_forward_count','native_total_physical_forward_count'),
        ('actual_total_native_batch_calls','native_total_batch_calls')]:fit[field]=sum(m['resources'][key] for m in all_models)
    return fit


def workspace(tmp,synthetic):
    inputs,combined,_=synthetic;base=tmp/'automation_reports/CV-SincNet'
    for cohort,run in target.RUNS.items():
        folder=base/run;folder.mkdir(parents=True);(folder/'report.md').write_bytes(b'Historical RUNNING\r\n')
        result=folder/'results';s=spec(cohort)
        dump(result/'complete.json',dict(status='SCORED',commit='synthetic',records=7236 if cohort=='rx3' else 2412))
        dump(result/'startup.json',dict(commit='synthetic',spec=s));dump(result/'arithmetic_audit.json',dict(status='VERIFIED'))
        dump(result/'scores.json',inputs[cohort]['scores']);dump(result/'summary/summary.json',inputs[cohort]['summary'])
        old=copy.deepcopy(s);old['run_id']=target.BASELINES[cohort];old['data']['capsule_id']=None;old['confirmation'].pop('reuse_validated_capsule_id')
        for row in old['rows']:row['output_root']=row['reuse_row_root']
        original=base/target.BASELINES[cohort]/'results'
        dump(original/'startup.json',dict(commit='historical',spec=old));dump(original/'scores.json',inputs[cohort]['scores'])
    result=base/target.RUNS['rx3']/'results';dump(result/'fit_audit.json',fit_audit());dump(result/'combined_rx4/summary.json',combined)
    (result/'combined_rx4/report.md').write_bytes(b'Joint report history\r\n')
    return tmp


def test_full_matrix_both_scopes_and_isolated_reuse(synthetic):
    import finalize_d92_mvridge_record as old
    inputs,summary,_=synthetic;out=target.interpret(inputs,summary)
    assert old.METHODS[1]=='D92-MVRidge-v1'
    assert out['records']==9648 and out['preregistered_guard_pass'] is True
    assert out['candidate_promoted'] is out['goal_complete'] is False
    assert len(out['per_model_seed_k'])==16 and len(out['complete_k_by_new_count'])==20
    for row in out['per_k']:
        assert row['joint_delta_pp']['old_accuracy']==pytest.approx(-2)
        assert row['old_all_cells']['delta_pp']==pytest.approx(.4)
        assert row['joint_delta_pp']['new_accuracy']==pytest.approx(3.5)
        assert row['joint_cells']==960 and row['old_all_cells']['cells']==1200
    assert out['fit_protocol']['full_fit_calls']==4800 and out['fit_protocol']['oof'] is None


def test_failure_not_hardcoded(synthetic):
    _,_,specs=synthetic;scores=[factory.score_rows(s,adverse=True) for s in specs]
    inputs={c:dict(scores=d,summary=summarize(d,s)) for c,s,d in zip(target.RUNS,specs,scores)}
    assert target.interpret(inputs,combine(specs,scores))['preregistered_guard_pass'] is False


@pytest.mark.parametrize('fault',['missing','pair','guard','weight'])
def test_bad_arithmetic_rejected(synthetic,fault):
    inputs,summary,_=copy.deepcopy(synthetic)
    if fault=='missing':inputs['rx3']['scores']['results'].pop()
    if fault=='pair':inputs['rx3']['scores']['results'][1]['split_id']='wrong'
    if fault=='guard':summary['comparisons'][0]['old_guard_delta']=-.02
    if fault=='weight':summary['tables']['per_k'][0]['new_accuracy']+=.01
    with pytest.raises(ValueError):target.interpret(inputs,summary)


@pytest.mark.parametrize('fault',['fold','dimension','state','source','sha','steps','smoke','nan','objective','time','transfer'])
def test_bad_cost_rejected(fault):
    fit=fit_audit();m=fit['runs'][0]['models'][0]
    if fault=='fold':fit['fold_fit_calls']=1
    if fault=='dimension':fit['feature_dim']=256
    if fault=='state':m['state_bytes']['head_bytes']['max']=53456
    if fault=='source':fit['source_rows_used_for_fit']=1
    if fault=='sha':m['resources']['checkpoint_sha256']='0'*64
    if fault=='steps':fit['optimizer_steps']=64
    if fault=='smoke':m['resources']['smoke_forward_count']=0
    if fault=='nan':m['timing']['fit_seconds']['mean']=float('nan')
    if fault=='objective':m['fit_measurements']['loss_total']=stat(2.,900)
    if fault=='time':fit['summed_process_work_seconds']['fit_seconds']+=1
    if fault=='transfer':fit['model_incremental_transfer_bytes']=0
    with pytest.raises(ValueError):target.costs(fit,{c:dict(commit='synthetic',spec=spec(c)) for c in target.RUNS},{c:dict(commit='synthetic') for c in target.RUNS})


def test_cost_actual_schema():
    data=target.costs(fit_audit(),{c:dict(commit='synthetic',spec=spec(c)) for c in target.RUNS},{c:dict(commit='synthetic') for c in target.RUNS})
    assert sum(r['total_fits'] for r in data.values())==4800
    text=target.cost_text(data['rx3']);assert '5896C' in text and '无运行时 OOF' in text and '部署状态和增量模型传输未知' in text


def test_terminal_barrier_before_scores(tmp_path,synthetic,monkeypatch):
    workspace(tmp_path,synthetic);path=tmp_path/'automation_reports/CV-SincNet'/target.RUNS['rx1']/'results/complete.json'
    dump(path,dict(status='RUNNING'));original=target.read
    def guard(p):
        assert p.name!='scores.json','premature score read'
        return original(p)
    monkeypatch.setattr(target,'read',guard)
    with pytest.raises(ValueError,match='Both cohorts'):target.prepare(tmp_path)


def test_dryrun_write_preserve_and_repeat_refused(tmp_path,synthetic):
    workspace(tmp_path,synthetic);all_files={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    writes,appends=target.prepare(tmp_path)
    assert all(p.read_bytes()==v for p,v in all_files.items())
    assert len(writes)==len(appends)==3
    target.persist(writes,appends)
    appended={p for p,_ in appends}
    assert all(p.read_bytes()==v for p,v in all_files.items() if p not in appended)
    for p,text in appends:
        assert p.read_bytes()==all_files[p]+text.encode('utf-8')
        assert 'MVRidge' not in text and '诊断CV' not in text
    with pytest.raises(FileExistsError):target.prepare(tmp_path)


def test_baseline_entire_record_unchanged(tmp_path,synthetic):
    workspace(tmp_path,synthetic);p=tmp_path/'automation_reports/CV-SincNet'/target.BASELINES['rx3']/'results/scores.json'
    data=target.read(p);data['results'][0]['confusion'][0][0]+=1;dump(p,data)
    with pytest.raises(ValueError,match='Original D92/DG record differs'):target.prepare(tmp_path)
