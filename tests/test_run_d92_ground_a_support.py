"""Synthetic metadata and fixed scorer calls; never original experiment inputs."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import Mock,patch

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code')]
import run_d92_ground_a_support as runner


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def fixture(tmp_path,*,margin=False):
    rows=[];output=tmp_path/'supplement'
    schemas=(runner.MARGIN_SCHEMA,) if margin else tuple(sorted(runner.LEGACY_SCHEMAS))
    for schema in schemas:
        method_info=runner.METHODS[schema]
        method,status,scope,*_=method_info;source_run=schema+'_synthetic';source=tmp_path/source_run
        summary=source/'analysis'/'summary.json'
        save(summary,dict(status=runner.SUMMARY_STATUSES[schema],scope=scope,schema=schema,method=method,
            run_id=source_run,release_commit='c'*40,coverage=dict(episodes=160,sequence_paths=1800),
            old_class_count=6,query_rows_used=0,source_rows_used=0,statistics={'DO_NOT_PARSE':[.1,.2]}))
        for index in range(4):
            source_row='source-row-'+str(index);name=schema+'-'+str(index);lane=source/source_row
            model=index//2 if margin else index
            cache=tmp_path/('cache-'+str(index));packet=tmp_path/('packet-'+str(model));cache.mkdir(exist_ok=True);packet.mkdir(exist_ok=True)
            row=dict(row_id=name,source_run_id=source_run,source_row_id=source_row,packet=packet.as_posix(),support_features=cache.as_posix(),
                fit_trace=(lane/'fit_trace.jsonl').as_posix(),source_summary=summary.as_posix(),expected_summary_status=runner.SUMMARY_STATUSES[schema],
                expected_summary_schema=schema,expected_method=method,expected_runtime_commit='c'*40,
                expected_checkpoint_sha256=(str(model+1)*64 if margin else 'd'*64),
                expected_capsule_id=('capsule-'+str(index%2) if margin else 'capsule'),
                expected_model_seed=17+model,output_root=(output/name).as_posix())
            selection=dict(splits=[dict(split_id=f'split-{scene}-{k}-{new}',k=k,new_count=new)
                for scene in range(2) for k in (1,5,10,20) for new in (0,2,5,10,20)])
            common=dict(runner.binding(row),schema=schema,method=method,scope=scope,query_rows_used=0,source_rows_used=0,
                truth_read=False,support_features=cache.as_posix(),episodes=40)
            save(lane/'startup.json',dict(common,config=dict(selection=selection,ignored_algorithm={'DO_NOT_PARSE':[99]})))
            save(lane/'probe_complete.json',dict(common,status=status,sequence_paths=450,selection=selection))
            (lane/'fit_trace.jsonl').write_text('TRACE_MUST_ONLY_BE_OPENED_BY_FROZEN_SCORER\n',encoding='utf-8')
            rows.append(row)
    return dict(run_id='synthetic-ground-a-supplement',spec_path='configs/synthetic-ground-a-support.json',
        code=dict(cwd=(tmp_path/'release').as_posix(),environment=Path(sys.executable).as_posix()),
        execution=dict(remote_run_root=output.as_posix(),launch_owner='root',cpu_lanes=1,blas_threads_per_lane=2),rows=rows)


def fake_score(**kwargs):
    out=Path(kwargs['output']);out.mkdir(exist_ok=False)
    bound=dict(run_id=kwargs['run_id'],row_id=kwargs['row_id'],checkpoint_sha256=kwargs['expected_checkpoint_sha256'],
        capsule_id=kwargs['expected_capsule_id'],model_seed=kwargs['expected_model_seed'])
    marker=dict(schema=runner.ROW_SCHEMA,status=runner.ROW_STATUS,binding=bound,parent_count=40,query_rows_used=0,
        original_summary_modified=False,prediction_status='GROUND_A_SUPPORT_PREDICTIONS_FIXED')
    save(out/'complete.json',marker)
    source=json.loads((Path(kwargs['fit_trace']).parent/'startup.json').read_text(encoding='utf-8'))
    return dict(schema=runner.ROW_SCHEMA,status=runner.ROW_STATUS,scope=runner.SCOPE,binding=bound,parent_count=40,
        method_schema=source['schema'],method=source['method'],adapted_path=runner.METHODS[source['schema']][3],
        query_rows_used=0,source_rows_used=0,training_performed=False,calibration_performed=False,original_summary_modified=False,
        original_trace_modified=False,resources=dict(native_single_record_score_seconds=.01,score_numeric_bytes=120,incremental_transmission_bytes=None))


def test_all_eight_declared_rows_source_bindings_once_and_readonly_inputs(tmp_path):
    spec=fixture(tmp_path);calls=[];before={path:path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    original=runner.read_selected;summary_reads=[]
    def selected(path,fields):
        if Path(path).name=='summary.json':summary_reads.append(path)
        return original(path,fields)
    def score(**kwargs):
        assert len(summary_reads)==2
        calls.append(kwargs);return fake_score(**kwargs)
    with patch.object(runner,'read_selected',side_effect=selected):result=runner.run(spec,'a'*40,score_fn=score)
    assert len(calls)==8 and calls==[runner.score_arguments(row) for row in spec['rows']]
    assert result['status']==runner.STATUS and result['completed_rows']==8 and result['parent_count']==320
    assert all(path.read_bytes()==content for path,content in before.items())
    root=Path(spec['execution']['remote_run_root']);assert json.loads((root/'complete.json').read_text())==result
    startup=json.loads((root/'startup.json').read_text());assert startup['spec']==spec and startup['commit']=='a'*40
    events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    assert [event['actual_call']['kwargs'] for event in events if event['event']=='ROW_START']==calls
    assert (root/'scoring.log').is_file() and (root/'compact.csv').is_file()
    assert result['incremental_network_transfer_bytes'] is result['energy'] is None
    with patch.object(runner,'verify_sources') as verifier,pytest.raises(FileExistsError):runner.run(spec,'a'*40,score_fn=Mock())
    verifier.assert_not_called()


def test_results_and_nonselection_config_are_never_decoded(tmp_path):
    spec=fixture(tmp_path);decoder=runner.json.JSONDecoder
    class RejectScores(decoder):
        def raw_decode(self,text,index=0):
            if text[index:].startswith('{"DO_NOT_PARSE"'):raise AssertionError('Result/algorithm decoded')
            return super().raw_decode(text,index)
    with patch.object(runner.json,'JSONDecoder',RejectScores):
        evidence=runner.verify_sources(spec)
    assert evidence['statistics_deserialized'] is False
    assert len(evidence['source_summaries'])==2 and len(evidence['lanes'])==8
    for bad in ('{"a":1,"a":2}','{"a":1,}','{"a":1} extra'):
        with pytest.raises(ValueError):runner.selected_json(bad,{'a'})


@pytest.mark.parametrize('case',['summary_status','summary_commit','summary_scope','summary_schema','summary_query','summary_source',
    'summary_parents','summary_paths','lane_identity','lane_seed','lane_cache','lane_status','lane_selection','lane_paths','lane_query'])
def test_source_mismatch_stops_before_any_score_and_preserves_failure(tmp_path,case):
    spec=fixture(tmp_path);row=spec['rows'][-1]
    path=Path(row['source_summary']) if case.startswith('summary') else Path(row['fit_trace']).parent/('startup.json' if case in ('lane_cache','lane_identity','lane_seed','lane_query') else 'probe_complete.json')
    data=json.loads(path.read_text())
    if case=='summary_status':data['status']='PARTIAL'
    elif case=='summary_commit':data['release_commit']='b'*40
    elif case=='summary_scope':data['scope']='OTHER'
    elif case=='summary_schema':data['schema']='OTHER'
    elif case=='summary_query':data['query_rows_used']=1
    elif case=='summary_source':data['source_rows_used']=1
    elif case=='summary_parents':data['coverage']['episodes']=159
    elif case=='summary_paths':data['coverage']['sequence_paths']=1799
    elif case=='lane_identity':data['row_id']='other'
    elif case=='lane_seed':data['model_seed']+=1
    elif case=='lane_cache':data['support_features']=(tmp_path/'other-cache').as_posix()
    elif case=='lane_status':data['status']='PARTIAL'
    elif case=='lane_selection':data['selection']['splits'][0]['split_id']='replaced'
    elif case=='lane_paths':data['sequence_paths']=449
    elif case=='lane_query':data['query_rows_used']=1
    save(path,data);score=Mock()
    with pytest.raises(ValueError):runner.run(spec,'a'*40,score_fn=score)
    score.assert_not_called();root=Path(spec['execution']['remote_run_root'])
    assert (root/'failed.json').is_file() and not (root/'complete.json').exists()
    assert all(value['status']=='PENDING' for value in json.loads((root/'state.json').read_text()).values())


def test_failure_preserves_first_rows_and_fixed_prediction_artifact_without_retry(tmp_path):
    spec=fixture(tmp_path);calls=[]
    def score(**kwargs):
        calls.append(kwargs)
        if len(calls)==3:
            path=Path(kwargs['output']);path.mkdir();(path/'fixed_predictions').write_text('synthetic fixed before failed join')
            raise RuntimeError('synthetic independent join failure')
        return fake_score(**kwargs)
    with pytest.raises(RuntimeError):runner.run(spec,'a'*40,score_fn=score)
    root=Path(spec['execution']['remote_run_root']);failure=json.loads((root/'failed.json').read_text())
    assert len(calls)==3 and failure['completed_rows']==2 and failure['automatic_retry'] is False
    assert (Path(spec['rows'][2]['output_root'])/'fixed_predictions').is_file()
    assert all((Path(row['output_root'])/'complete.json').is_file() for row in spec['rows'][:2])


def test_returned_success_without_independent_matching_marker_is_rejected(tmp_path):
    spec=fixture(tmp_path)
    def score(**kwargs):
        value=fake_score(**kwargs);path=Path(kwargs['output'])/'complete.json';marker=json.loads(path.read_text())
        marker['binding']['row_id']='different';save(path,marker);return value
    with pytest.raises(ValueError,match='readback'):runner.run(spec,'a'*40,score_fn=score)
    assert (Path(spec['execution']['remote_run_root'])/'failed.json').is_file()


@pytest.mark.parametrize('case',['row_count','duplicate_row','duplicate_source','output','input_overlap','lanes','owner','schema','source_summary','commit'])
def test_spec_rejections_before_output_creation(tmp_path,case):
    spec=fixture(tmp_path)
    if case=='row_count':spec['rows'].pop()
    elif case=='duplicate_row':spec['rows'][1]['row_id']=spec['rows'][0]['row_id']
    elif case=='duplicate_source':spec['rows'][1]['source_row_id']=spec['rows'][0]['source_row_id']
    elif case=='output':spec['rows'][0]['output_root']=(tmp_path/'outside').as_posix()
    elif case=='input_overlap':spec['rows'][0]['packet']=tmp_path.as_posix()
    elif case=='lanes':spec['execution']['cpu_lanes']=2
    elif case=='owner':spec['execution']['launch_owner']='other'
    elif case=='schema':spec['rows'][0]['expected_summary_schema']='UNKNOWN'
    elif case=='source_summary':spec['rows'][0]['source_summary']=(tmp_path/'different'/'summary.json').as_posix()
    with pytest.raises(ValueError):runner.run(spec,'invalid' if case=='commit' else 'a'*40,score_fn=Mock())
    assert not Path(spec['execution']['remote_run_root']).exists()


@pytest.mark.parametrize('method',['affine','conditional','margin'])
def test_row_adapter_uses_real_fixed_scorer_production_trace_and_true_k1(tmp_path,method):
    from test_score_d92_ground_a_support import _case
    case=_case(tmp_path/'source',method=method,k=1,new=0);bound=case['binding']
    row=dict(row_id='new-analysis-row',source_run_id=bound['run_id'],source_row_id=bound['row_id'],
        packet=str(case['packet']),support_features=str(case['cache']),fit_trace=str(case['lane']/'fit_trace.jsonl'),
        output_root=str(tmp_path/'independent-supplement'),expected_checkpoint_sha256=bound['checkpoint_sha256'],
        expected_capsule_id=bound['capsule_id'],expected_model_seed=bound['model_seed'])
    result=runner.score_row(row,dict(episodes=1))
    assert result['parent_count']==1 and result['status']==runner.ROW_STATUS
    paired=json.loads((Path(row['output_root'])/'pairing.json').read_text(encoding='utf-8'))
    assert paired['row_id']==bound['row_id'] and paired['parents'][0]['oof']['metrics']['A_old_accuracy'] is None
    assert paired['parents'][0]['paired_paths']==[]


def test_margin_single_source_four_rows_two_models_by_two_capsules(tmp_path):
    spec=fixture(tmp_path,margin=True);runner.validate_spec(spec)
    evidence=runner.verify_sources(spec)
    assert len(evidence['source_summaries'])==1 and len(evidence['lanes'])==4
    assert evidence['statistics_deserialized'] is False
    calls=[]
    def score(**kwargs):calls.append(kwargs);return fake_score(**kwargs)
    result=runner.run(spec,'a'*40,score_fn=score)
    assert result['rows']==result['completed_rows']==len(calls)==4 and result['parent_count']==160
    assert calls==[runner.score_arguments(row) for row in spec['rows']]


@pytest.mark.parametrize('case',['mixed_method','split_source','wrong_status','seed_crossing','capsule_crossing',
    'checkpoint_for_same_model','packet_for_same_model','summary_query','lane_seed','summary_partial'])
def test_margin_contract_and_metadata_tampering_stop_before_scores(tmp_path,case):
    spec=fixture(tmp_path,margin=True);row=spec['rows'][0]
    if case=='mixed_method':
        schema='d92_affine_joint_local_ridge_v1';row.update(expected_summary_schema=schema,
            expected_summary_status=runner.SUMMARY_STATUSES[schema],expected_method=runner.METHODS[schema][0])
    elif case=='split_source':row['source_run_id']='another-run'
    elif case=='wrong_status':row['expected_summary_status']='COMPLETE_CONDITIONAL_JOINT_PROBE_VERIFIED'
    elif case=='seed_crossing':row['expected_model_seed']=999
    elif case=='capsule_crossing':row['expected_capsule_id']=spec['rows'][1]['expected_capsule_id']
    elif case=='checkpoint_for_same_model':row['expected_checkpoint_sha256']='e'*64
    elif case=='packet_for_same_model':row['packet']=(tmp_path/'different-packet').as_posix()
    else:
        path=Path(row['fit_trace']).parent/'startup.json' if case=='lane_seed' else Path(row['source_summary'])
        value=json.loads(path.read_text(encoding='utf-8'))
        if case=='lane_seed':value['model_seed']+=1
        elif case=='summary_query':value['query_rows_used']=1
        else:value['status']='PARTIAL'
        save(path,value)
    score=Mock()
    with pytest.raises(ValueError):runner.run(spec,'a'*40,score_fn=score)
    score.assert_not_called()


def test_margin_returned_other_candidate_is_not_verified_by_generic_marker(tmp_path):
    spec=fixture(tmp_path,margin=True)
    def score(**kwargs):return dict(fake_score(**kwargs),adapted_path='R0')
    with pytest.raises(ValueError,match='declared fixed Margin'):runner.run(spec,'a'*40,score_fn=score)
    root=Path(spec['execution']['remote_run_root'])
    assert (root/'failed.json').exists() and not (root/'complete.json').exists()
