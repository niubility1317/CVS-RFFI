"""Sequential packet-only runner with synthetic checkpoint/source documents."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code')]
import run_d92_ground_a_packet_export as runner
from test_export_d92_ground_classifier_a_packet import fixture as checkpoint_fixture


def fixture(tmp_path):
    run_root=tmp_path/'run';rows=[]
    roles=dict(role_ids={'L_s':['source-0','source-1'],'V_s':['source-2'],'T_s':['source-3']},
        source_rxs=['rx-ground'],source_days=['day-source'],ratios=[.6,.2,.2],split_seed=11,num_classes=6)
    expected=tmp_path/'original_source_contract.json'
    expected.write_text(json.dumps(dict(roles,classes=['not','used','to','replace','head','order'])),encoding='utf-8')
    for index in range(2):
        directory=tmp_path/('source-'+str(index));directory.mkdir()
        checkpoint,weight,binding,provenance=checkpoint_fixture(directory,scale=17.25+index)
        actual=directory/'source_contract.json';actual.write_text(json.dumps(dict(roles,classes=binding['ordered_classes'],
            native_role_comparison='EXACT_MATCH',checkpoint_init='scratch_only',target_access_before_freeze=False)),encoding='utf-8')
        initial=directory/'initialization.json';initial.write_text(json.dumps(dict(scratch_only=True,checkpoint_sources=[],
            target_contact=False,target_training_contact=False,source_roles='EXACT_MATCH',ema_origin='this_run_student')),encoding='utf-8')
        terminal=directory/'phase1_terminal_status.json';terminal.write_text(json.dumps(dict(schema='phase1_terminal_status_v2',
            selected_checkpoint=checkpoint.as_posix(),selected_checkpoint_sha256=binding['checkpoint_sha256'],selection_source='training_final_only',
            scores={'DO_NOT_PARSE':[.123,.456]})),encoding='utf-8')
        startup=directory/'startup.json';startup.write_text(json.dumps(dict(selection='final_epoch_200',
            scores={'DO_NOT_PARSE':[.789]})),encoding='utf-8')
        name='ground-a-synthetic-'+str(index)
        rows.append(dict(row_id=name,checkpoint=checkpoint.as_posix(),source_contract_ref=actual.as_posix(),
            expected_source_contract_ref=expected.as_posix(),initialization_ref=initial.as_posix(),selected_checkpoint_ref=terminal.as_posix(),
            training_startup_ref=startup.as_posix(),binding=binding,provenance=provenance,output_root=(run_root/name).as_posix()))
    return dict(run_id='synthetic-ground-a',spec_path='configs/synthetic-ground-a.json',
        code=dict(cwd=(tmp_path/'release').as_posix(),environment=Path(sys.executable).as_posix()),
        execution=dict(remote_run_root=run_root.as_posix(),launch_owner='root',cpu_lanes=1,blas_threads_per_lane=2),rows=rows)


def test_full_sequential_packets_source_order_logs_and_measured_totals(tmp_path):
    spec=fixture(tmp_path);calls=[];inputs={row['checkpoint']:Path(row['checkpoint']).read_bytes() for row in spec['rows']}
    def export(**kwargs):
        calls.append(kwargs['checkpoint']);return runner.export_packet(**kwargs)
    result=runner.run(spec,'a'*40,export_fn=export);root=Path(spec['execution']['remote_run_root'])
    assert calls==[row['checkpoint'] for row in spec['rows']]
    assert result['status']==runner.STATUS and result['rows']==result['completed_rows']==2
    markers=[json.loads((Path(row['output_root'])/'complete.json').read_text(encoding='utf-8')) for row in spec['rows']]
    assert result['packet_total_file_bytes']==sum(m['packet_total_file_bytes'] for m in markers)
    assert result['weight_numeric_bytes']==7680 and result['packet_total_file_bytes']>7680
    startup=json.loads((root/'startup.json').read_text(encoding='utf-8'))
    assert startup['spec']==spec and startup['commit']=='a'*40 and startup['torch_threads']==2
    assert startup['encoder_executed'] is startup['actual_A_evaluated'] is False
    assert all(Path(path).read_bytes()==before for path,before in inputs.items())
    events=[json.loads(row) for row in (root/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['event'] for r in events]==['STARTUP','PACKET_EXPORT_START','PACKET_EXPORT_COMPLETE','PACKET_EXPORT_START','PACKET_EXPORT_COMPLETE','COMPLETE']
    assert [r['scale'] for r in events if r['event']=='PACKET_EXPORT_START']==[17.25,18.25]
    for row in spec['rows']:
        bound=json.loads((root/(row['row_id']+'_source_binding.json')).read_text(encoding='utf-8'))
        assert bound['ordered_classes']==row['binding']['ordered_classes'] and bound['scores_deserialized'] is False
    assert (root/'training.log').is_file() and (root/'compact.csv').is_file()
    with patch.object(runner,'verify_row_sources') as verifier,pytest.raises(FileExistsError):runner.run(spec,'a'*40)
    verifier.assert_not_called()


@pytest.mark.parametrize('case',['scratch','inheritance','target','target_training','native_role','physical_role','classes','selected_sha','selected_path','selection_source','startup_selection','startup_missing_selection'])
def test_source_binding_mismatch_prevents_any_checkpoint_load(tmp_path,case):
    spec=fixture(tmp_path);row=spec['rows'][0]
    if case in ('scratch','inheritance','target','target_training'):
        path=Path(row['initialization_ref']);data=json.loads(path.read_text())
        key,value={'scratch':('scratch_only',False),'inheritance':('checkpoint_sources',['prior']),
            'target':('target_contact',True),'target_training':('target_training_contact',True)}[case];data[key]=value
    elif case in ('native_role','physical_role','classes'):
        path=Path(row['source_contract_ref']);data=json.loads(path.read_text())
        if case=='native_role':data['native_role_comparison']='UNKNOWN'
        elif case=='physical_role':data['role_ids']['L_s']=['different-id']
        else:data['classes'].reverse()
    elif case.startswith('startup_'):
        path=Path(row['training_startup_ref']);data=json.loads(path.read_text())
        if case=='startup_selection':data['selection']='final_epoch_199'
        else:data.pop('selection')
    else:
        path=Path(row['selected_checkpoint_ref']);data=json.loads(path.read_text())
        key,value={'selected_sha':('selected_checkpoint_sha256','b'*64),'selected_path':('selected_checkpoint','/another/checkpoint'),
            'selection_source':('selection_source','metric_selected')}[case];data[key]=value
    path.write_text(json.dumps(data),encoding='utf-8')
    with patch.object(runner,'export_packet') as exporter,pytest.raises(ValueError):runner.run(spec,'a'*40,export_fn=exporter)
    exporter.assert_not_called();root=Path(spec['execution']['remote_run_root'])
    assert (root/'failed.json').exists() and not (root/'complete.json').exists()
    state=json.loads((root/'state.json').read_text())
    assert state[spec['rows'][0]['row_id']]['status']==runner.FAILED and state[spec['rows'][1]['row_id']]['status']=='PENDING'


@pytest.mark.parametrize('epoch',[None,199,201,200.,True,'200'])
def test_final_epoch_provenance_exact_integer_required_before_file_reads(tmp_path,epoch):
    spec=fixture(tmp_path);row=spec['rows'][0]
    if epoch is None:row['provenance'].pop('checkpoint_epoch')
    else:row['provenance']['checkpoint_epoch']=epoch
    with patch.object(runner,'read_selected') as reader:
        with pytest.raises(ValueError,match='final-epoch-200'):runner.validate_spec(spec)
        with pytest.raises(ValueError,match='final-epoch-200'):runner.verify_row_sources(row)
    reader.assert_not_called()
    with pytest.raises(ValueError):runner.run(spec,'a'*40)
    assert not Path(spec['execution']['remote_run_root']).exists()


def test_terminal_score_sections_never_deserialized(tmp_path):
    spec=fixture(tmp_path);base=runner.json.JSONDecoder
    class RejectScoreDecoder(base):
        def raw_decode(self,text,index=0):
            if text[index:].startswith('{"DO_NOT_PARSE"'):raise AssertionError('score section deserialized')
            return super().raw_decode(text,index)
    with patch.object(runner.json,'JSONDecoder',RejectScoreDecoder):
        assert runner.verify_row_sources(spec['rows'][0])['scores_deserialized'] is False
    text=json.dumps({'allowed':1,'unused':[{'a':'brace } comma , quote "'}]})
    assert runner.selected_json(text,{'allowed'})=={'allowed':1}
    for bad in ('{"a":1,"a":2}','{"a":1,}','{"a":1} extra'):
        with pytest.raises(ValueError):runner.selected_json(bad,{'a'})


def test_failed_second_export_preserves_first_packet_and_does_not_retry(tmp_path):
    spec=fixture(tmp_path);calls=[]
    def export(**kwargs):
        calls.append(kwargs['checkpoint'])
        if len(calls)==2:raise RuntimeError('synthetic second export failure')
        return runner.export_packet(**kwargs)
    with pytest.raises(RuntimeError):runner.run(spec,'a'*40,export_fn=export)
    assert len(calls)==2 and runner.load_packet(spec['rows'][0]['output_root']).classes==tuple(spec['rows'][0]['binding']['ordered_classes'])
    failure=json.loads((Path(spec['execution']['remote_run_root'])/'failed.json').read_text())
    assert failure['completed_rows']==1 and failure['automatic_retry'] is False


@pytest.mark.parametrize('case',['lanes','owner','output','relative_input','missing_binding','missing_startup_ref','wrong_startup_parent','bad_commit'])
def test_spec_scope_rejections_before_run_mutation(tmp_path,case):
    spec=fixture(tmp_path)
    if case=='lanes':spec['execution']['cpu_lanes']=2
    elif case=='owner':spec['execution']['launch_owner']='unknown'
    elif case=='output':spec['rows'][0]['output_root']='/outside'
    elif case=='relative_input':spec['rows'][0]['checkpoint']='checkpoint.pth'
    elif case=='missing_binding':spec['rows'][0]['binding'].pop('scale')
    elif case=='missing_startup_ref':spec['rows'][0].pop('training_startup_ref')
    elif case=='wrong_startup_parent':spec['rows'][0]['training_startup_ref']=spec['rows'][1]['training_startup_ref']
    with pytest.raises(ValueError):runner.run(spec,'unknown' if case=='bad_commit' else 'a'*40)
    assert not Path(spec['execution']['remote_run_root']).exists()
