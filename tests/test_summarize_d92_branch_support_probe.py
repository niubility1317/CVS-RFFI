import copy
import csv
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'code')]
import summarize_d92_branch_support_probe as summary
from cvsrffi.d92_branch_support_probe import probe_branch_support, FROZEN_CONFIG


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def write_records(path, rows):
    path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows), encoding='utf-8')


def fixture(root):
    rng = np.random.default_rng(62)
    matrix = dict(receivers=['receiver'], scenarios=['scene'], ks=[1, 2], new_counts=[0, 1], support_seeds=[17])
    templates = []
    for k in matrix['ks']:
        for new in matrix['new_counts']:
            c = 2+new; n = c*k
            x = {key: rng.normal(size=(n, 96 if key == 'fft' else 160)) for key in ('z_id', 'fft', 't_emb', 'f_emb', 'pa_local')}
            result = probe_branch_support(**x, support_labels=np.repeat(np.arange(c), k),
                support_ids=['id%03d'%i for i in range(n)], classes=['c%d'%i for i in range(c)], old_classes=['c0', 'c1'])
            result.update(split_id=f'k{k}-new{new}', receiver='receiver', scenario='scene', new_count=new,
                support_seed=17, scope=summary.SCOPE, query_rows_used=0, source_rows_used=0,
                registered_classes=['c%d'%i for i in range(c)], fit_call_seconds=result['fit_seconds']+.01)
            templates.append(result)
    rows=[]; cohorts={}; state={}
    for co in ('rx1', 'rx3'):
        cohorts[co] = dict(capsule_id=co+'-capsule', expected_split_count=4, matrix=matrix)
        for seed in range(2026092701, 2026092705):
            rid = co+'-'+str(seed); sha = str(seed-2026092700)*64
            rows.append(dict(row_id=rid, cohort=co, seeds=dict(model=seed), expected_checkpoint_sha256=sha))
            state[rid] = dict(status='SUPPORT_PROBE_COMPLETE', episodes=4)
            lane = root/rid
            common = dict(checkpoint_sha256=sha, capsule_id=co+'-capsule', model_seed=seed)
            feature = dict(common, status='BRANCH_SUPPORT_FEATURES_COMPLETE', schema='d92_branch_support_features_v1',
                view_count_per_observation=1, split_count=4, classes=['c0','c1'], count=6,
                query_iq_access=False, query_used_for_fitting=False, source_data_access=False, truth_read=False,
                adapted_state_inherited=False, encoder_updated=False, query_rows_read=0, new_source_payload_bytes=0,
                new_ground_statistics_bytes=0, identity_check_additional_encoder_forwards=0, native_eval=True,
                native_parameters_unchanged=True, native_buffers_unchanged=True, feature_array_bytes=6*2944,
                feature_file_bytes=20000, model_file_bytes=10000, native_physical_forward_count=6,
                support_iq_rows_read=6, identity_reference_checks=6, native_batch_calls=1,
                timing=dict(native_forward_seconds=.1, support_read_seconds=.05, total_seconds=.2), peak_process_rss_bytes=None,
                index_bytes=48, registry_array_bytes=100, metadata_file_bytes={'startup.json':200,'support_splits.json':200,'checkpoint_provenance.json':200},
                artifact_file_bytes_excluding_completion_marker=20600,
                feature_contract=dict(branch_keys=['z_id','t_emb','f_emb','pa_local'],branch_dim=160,fft_dim=96,
                    view_count=1,identity_feature_key='feat_joint',view='original_received_observation'))
            payload = dict(model_file_bytes=10000, support_feature_array_bytes=6*2944, support_feature_file_bytes=20000,
                native_physical_forward_count=6, native_batch_calls=1, new_source_payload_bytes=0,
                new_ground_statistics_bytes=0, feature_extraction_timing=feature['timing'])
            marker = dict(common, status='SUPPORT_PROBE_COMPLETE', scope=summary.SCOPE, algorithm=FROZEN_CONFIG,
                matrix=matrix, episodes=4,k1_episodes=2,oof_episodes=2,factorization_count=24,
                query_rows_used=0,source_rows_used=0,optimizer_steps=0,persistent_state_bytes=0,truth_read=False,
                payload_audit=payload, peak_process_rss_bytes=None)
            startup = dict(common, config=dict(algorithm=FROZEN_CONFIG,matrix=matrix), query_rows_used=0,source_rows_used=0,
                optimizer_steps=0,query_iq_access=False,truth_read=False,adapted_state_inherited=False,
                cross_row_adapted_state_reuse=False,checkpoint_loaded=False,encoder_updated=False,payload_audit=payload)
            write(lane/'feature_cache/features_complete.json', feature)
            (lane/'feature_cache/support_branch_features.npz').write_bytes(bytes(20000))
            for name in feature['metadata_file_bytes']:
                (lane/'feature_cache'/name).write_bytes(bytes(200))
            write(lane/'probe/probe_complete.json', marker)
            write(lane/'probe/startup.json', startup)
            write_records(lane/'probe/fit_trace.jsonl', templates)
            compact=[]
            for number, r in enumerate(templates, 1):
                small={k:r[k] for k in summary.COORDS+('split_id','support_count','fold_count','factorization_count','optimizer_steps','persistent_state_bytes','fit_seconds','fit_call_seconds')}
                small.update(classes=len(r['classes']),completed=number,total=4,query_rows_used=0,source_rows_used=0,
                    log_write_seconds=.01,total_seconds=r['fit_call_seconds']+.01,peak_process_rss_bytes=None)
                small.update({key:summary.scalar_tree(r[key]) for key in ('numerical','oof','reconstruction','paired')})
                compact.append(small)
            write_records(lane/'probe/compact.jsonl', compact)
    spec=dict(run_id='synthetic-probe',rows=rows,probe=dict(cohorts=cohorts,model_rows=8,total_episodes=32,
        query_access=False,source_payload_bytes=0,full_support_head=False),execution=dict(remote_run_root=str(root)))
    write(root/'complete.json',dict(status='SUPPORT_PROBE_COMPLETE',completed_rows=8,model_rows=8,episodes=32,
        query_access=False,source_sample_access=False,query_performance_claim=False,optimizer_steps=0,commit='synthetic-head',finished=120.))
    write(root/'startup.json',dict(spec=spec,query_access=False,source_sample_access=False,commit='synthetic-head',started=100.))
    write(root/'state.json',state)
    return spec


@pytest.fixture(scope='module')
def template(tmp_path_factory):
    root=tmp_path_factory.mktemp('probe-summary-template')
    return root,fixture(root)


@pytest.fixture
def case(tmp_path, template):
    source,spec=template
    root=tmp_path/'run';shutil.copytree(source,root)
    spec=copy.deepcopy(spec);spec['execution']['remote_run_root']=str(root)
    launch=json.loads((root/'startup.json').read_text(encoding='utf-8'));launch['spec']=spec
    write(root/'startup.json',launch)
    return root,spec,tmp_path/'out'


def test_complete_streaming_summary_and_no_array_access(case,monkeypatch):
    root,spec,out=case
    original=Path.read_text
    def guarded(path,*args,**kwargs):
        assert path.suffix != '.jsonl', 'Raw traces must stream, not read_text'
        assert not any(part in path.name for part in ('scores','results','index','received','npz'))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guarded)
    result=summary.summarize(spec=spec,output=out)
    assert result['coverage']==dict(episodes=32,k1_episodes=16,oof_episodes=16,factorizations=192,physical_oof_records_per_arm=80)
    assert result['selected_arm'] is None and not result['automatic_promotion']
    assert result['unique_model_count']==4 and result['unique_model_package_bytes']==40000
    assert result['run_wall_seconds']==20.
    assert result['resources']['export.model_file_bytes']['sum']==80000
    assert result['resources']['export.native_physical_forward_count']['sum']==48
    assert result['resources']['export.peak_process_rss_bytes']['mean'] is None
    assert result['query_rows_used']==result['source_rows_used']==result['persistent_classifier_bytes']==0
    assert not any('rows' in row for row in result['statistics'])
    assert {r['population'] for r in result['statistics']}=={'old_only','new_present'}
    oldnew=[r for r in result['statistics'] if r['population']=='old_only' and r['kind']=='arm' and r['metric'].endswith('.new_accuracy')]
    assert all(r['count']==0 and r['null_count']==8 and r['mean'] is None for r in oldnew)
    assert set(summary.STRATA)=={p.stem for p in out.glob('by_*.csv')}
    with (out/'by_k_newcount.csv').open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert {r['k'] for r in rows}=={'1','2'}
    assert 'correlated' in original(out/'report.md',encoding='utf-8')
    json.dumps(result,allow_nan=False)
    with pytest.raises(FileExistsError):summary.summarize(spec=spec,output=out)


@pytest.mark.parametrize('mutation', ['missing_lane','incomplete','sha','source','bytes','k1_oof',
    'fold_leak','loss','gradient_nan','duplicate_cell','class_metric','paired','reconstruction','compact','extra_trace',
    'root_binding','actual_file_bytes'])
def test_reject_incomplete_or_corrupt_diagnostics(case,mutation):
    root,spec,out=case;lane=root/spec['rows'][0]['row_id'];trace=lane/'probe/fit_trace.jsonl'
    if mutation=='missing_lane':spec['rows'].pop()
    elif mutation=='root_binding':
        path=root/'startup.json';value=json.loads(path.read_text());value['spec']['run_id']='other';write(path,value)
    elif mutation=='actual_file_bytes':(lane/'feature_cache/support_branch_features.npz').write_bytes(b'broken-size')
    elif mutation=='incomplete':
        value=json.loads((root/'complete.json').read_text());value['status']='FAILED';write(root/'complete.json',value)
    elif mutation in ('sha','source','bytes'):
        path=lane/'feature_cache/features_complete.json';value=json.loads(path.read_text())
        value[{'sha':'checkpoint_sha256','source':'source_data_access','bytes':'feature_array_bytes'}[mutation]]={'sha':'f'*64,'source':True,'bytes':1}[mutation]
        write(path,value)
    else:
        rows=records(trace)
        if mutation=='k1_oof':rows[0]['oof']={}
        if mutation=='fold_leak':rows[2]['folds'][0]['training_ids'].append(rows[2]['folds'][0]['held_ids'][0])
        if mutation=='loss':rows[2]['folds'][0]['stages'][0]['loss_total']+=1
        if mutation=='gradient_nan':
            text=trace.read_text();text=text.replace('"gradient_norm": ', '"gradient_norm": NaN, "ignored": ',1);trace.write_text(text)
        if mutation=='duplicate_cell':rows[1]=copy.deepcopy(rows[0])
        if mutation=='class_metric':rows[2]['oof']['z']['metrics']['classwise'][0]['accuracy']+=.1
        if mutation=='paired':rows[2]['paired']['z']['aux_minus_base']['rows'][0]['correct_delta']=17
        if mutation=='reconstruction':rows[2]['reconstruction']['z']['squared_error_sum']+=1
        if mutation=='compact':
            p=lane/'probe/compact.jsonl';small=records(p);small[2]['oof']['z']['metrics']['macro_accuracy']+=.1;write_records(p,small)
        if mutation=='extra_trace':rows.append(copy.deepcopy(rows[0]))
        if mutation!='gradient_nan':write_records(trace,rows)
    with pytest.raises((ValueError,KeyError)):summary.summarize(spec=spec,output=out)
    assert not out.exists()


def test_null_statistics_and_finite_decode():
    s=summary.Stat();s.add(None);s.add(2);s.add(4)
    assert s.result()==dict(count=2,null_count=1,sum=6.,mean=3.,min=2,max=4)
    for value in ('{"x":NaN}','{"x":1e999}'):
        with pytest.raises(ValueError):summary.decode(value)
