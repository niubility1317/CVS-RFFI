"""Small scalar-r02 publication regression set; all inputs are source literals."""
if __name__=='__main__':
    import ast
    from pathlib import Path
    import sys
    if sys.argv[1:]!=['--static-check']:raise SystemExit('Only --static-check is available as a script')
    roots=[Path('E:/type10-7'),Path('E:/type10-7/code/snapshots/d92_support_upgrade_20260928_wt')]
    for relative in ('tools/publish_d92_support_metric_support_analysis_scalar_r02.py',
                     'tests/test_d92_support_metric_support_analysis_scalar_r02_publication.py'):
        raw=[(root/relative).read_bytes() for root in roots]
        if raw[0]!=raw[1]:raise ValueError('Owned mirrors differ: '+relative)
        ast.parse(raw[0].decode('utf-8',errors='strict'),filename=relative)
        print('STATIC_UTF8_AST_MIRROR_OK',relative,len(raw[0]))
    raise SystemExit(0)

from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import publish_d92_support_metric_support_analysis_scalar_r02 as publisher


def literal_record():
    plan=dict(status='PREREGISTERED_NOT_LAUNCHED',attempt_id='r02',source_commit=None,
        source_preparation_commit=publisher.PREPARATION_COMMIT,fitted_runtime_commit=publisher.FITTED_COMMIT,
        paths=list(publisher.PATHS),release=publisher.RELEASE,output=publisher.OUTPUT,
        log=publisher.RELEASE+'/analysis.log',
        command=['/source-only/python','-u',publisher.RELEASE+'/'+publisher.PATHS[0],
            '--spec',publisher.RELEASE+'/'+publisher.PATHS[1],'--run-root',publisher.ORIGINAL_RUN,
            '--output',publisher.OUTPUT,'--expected-runtime-commit',publisher.FITTED_COMMIT],
        query_read=False,source_sample_read=False,fit=False,parameter_selection=False,
        launch_owner='root',automatic_retry=False,expected_complete_status=publisher.COMPLETE_STATUS,
        expected_summary_status=publisher.SUMMARY_STATUS,coverage=dict(rows=4,parents=160,sequence_paths=1800))
    return dict(run_id=publisher.RUN_ID,status='TRAINING_COMPLETE',execution=dict(
        remote_run_root=publisher.ORIGINAL_RUN,support_analysis=dict(status='FAILED',attempt_id='r01',
            preserved_failure='SOURCE_ONLY_R01_FAILURE'),
        support_analysis_scalar_r02=plan))


def test_r02_reads_only_new_plan_and_exact_two_sources_with_immutable_fitted_runtime():
    record=literal_record();before=deepcopy(record);binding=publisher.validate_record(record)
    assert publisher.PLAN_KEY=='support_analysis_scalar_r02'
    assert binding['attempt_id']=='r02' and binding['release']==publisher.RELEASE
    assert binding['output']==publisher.ORIGINAL_RUN+'-analysis-scalar-r02'
    assert binding['source_preparation_commit']=='f291990c45f14cf322dfb1c6d2fc5a97e23f5a91'
    assert binding['fitted_runtime_commit']=='3c3eb6c513aaf20fe84514dcbeab3584bad1f238'
    assert binding['paths']==['tools/analyze_d92_support_metric_joint_probe_scalar_r02.py',
        'configs/d92_support_metric_joint_support_20261002.json']
    assert record==before and record['execution']['support_analysis']['status']=='FAILED'


@pytest.mark.parametrize('fault',['r01','preparation','dispatched','third_source','old_release','old_output','missing_new_key'])
def test_scalar_r02_rejects_old_or_unregistered_bindings_without_fallback(fault):
    record=literal_record();plan=record['execution'][publisher.PLAN_KEY]
    if fault=='r01':plan['attempt_id']='r01'
    elif fault=='preparation':plan['source_preparation_commit']='20c91c59f78619edb113eedae78c129d45ea0820'
    elif fault=='dispatched':plan['status']='DISPATCHED_AWAITING_INDEPENDENT_READBACK'
    elif fault=='third_source':plan['paths'].append('tools/unregistered_analyzer.py')
    elif fault=='old_release':plan['release']='/home/szu2070436088/2510044040/CV-SincNet/releases/d92_support_metric_support_analysis_20261002_r01'
    elif fault=='old_output':plan['output']=publisher.ORIGINAL_RUN+'-analysis-r01'
    else:record['execution'].pop(publisher.PLAN_KEY)
    with pytest.raises(ValueError):publisher.validate_record(record)


def fake_publication(tmp_path,*,timeout_phase=None):
    record=literal_record();record_path=tmp_path/'literal_record.json'
    record_path.write_text(json.dumps(record),encoding='utf-8');folder=tmp_path/'fresh_scalar_r02'
    source=dict(source_commit='e'*40,record_commit='e'*40,branch='source_only',remote_oid='e'*40)
    binding=publisher.validate_record(record);binding.update(source);calls=[]
    closed=dict(run_id=publisher.RUN_ID,runtime_commit=publisher.FITTED_COMMIT,
        status=publisher.COMPLETE_STATUS,model_rows=4,completed_rows=4,episodes=160,sequence_paths=1800,
        truth_read=False,query_access=False,source_sample_access=False)
    launch=dict(status='DISPATCHED_AWAITING_READBACK',pid=5555,cwd=publisher.RELEASE,
        source_commit=source['source_commit'],fitted_runtime_commit=publisher.FITTED_COMMIT,run_id=publisher.RUN_ID,
        output=publisher.OUTPUT,original_run=publisher.ORIGINAL_RUN,log=publisher.RELEASE+'/analysis.log',
        argv=binding['command'],launch_owner='root',remote_user=publisher.REMOTE_USER,
        environment=binding['environment'],query_read=False,source_sample_read=False,fit=False,
        parameter_selection=False,automatic_retry=False)
    def call(argv,**kwargs):
        calls.append(argv)
        if argv[:2]==['git','archive']:
            path=Path(next(v.split('=',1)[1] for v in argv if v.startswith('--output=')))
            with tarfile.open(path,'w') as stream:
                for relative in publisher.PATHS:
                    item=tarfile.TarInfo(publisher.RELEASE.rsplit('/',1)[-1]+'/'+relative)
                    payload=b'SOURCE_ONLY_SCALAR_R02';item.size=len(payload)
                    stream.addfile(item,io.BytesIO(payload))
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        if argv[0]=='scp':
            if timeout_phase=='TRANSFER':raise subprocess.TimeoutExpired(argv,60,output=b'scalar-r02 transfer bytes',stderr=b'literal timeout')
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        program=kwargs['input']
        if 'PREFLIGHT_COMPLETE_NOT_DISPATCHED' in program:
            value=dict(status='PREFLIGHT_COMPLETE_NOT_DISPATCHED',remote_user=publisher.REMOTE_USER,
                run_id=publisher.RUN_ID,source_commit=source['source_commit'],fitted_runtime_commit=publisher.FITTED_COMMIT,
                release=publisher.RELEASE,output=publisher.OUTPUT,original_run=publisher.ORIGINAL_RUN,
                original_complete=closed,fit=False,query_read=False,source_sample_read=False,
                parameter_selection=False,automatic_retry=False)
        elif 'DISPATCH_METADATA_READBACK' in program:
            value=dict(status='DISPATCH_METADATA_READBACK',run_id=publisher.RUN_ID,metadata=dict(launch=launch),
                metadata_errors={},original_complete=closed,original_run=publisher.ORIGINAL_RUN,
                fit=False,query_read=False,source_sample_read=False)
        else:
            if timeout_phase=='REMOTE_EXTRACT_DISPATCH':raise subprocess.TimeoutExpired(argv,60,output=b'scalar-r02 possible dispatch',stderr=b'literal timeout')
            value=launch
        return SimpleNamespace(returncode=0,stdout=json.dumps(value),stderr='')
    return record_path,folder,calls,call,source


def test_literal_r02_publication_does_not_change_record_or_promote_artifact(tmp_path):
    record,folder,calls,call,source=fake_publication(tmp_path);before=record.read_bytes()
    result=publisher.publish(record,repository='/source-only/repository',artifact_root=folder,
        call=call,source_check=lambda *args:source)
    assert result['status']=='DISPATCHED_AWAITING_INDEPENDENT_READBACK' and result['analysis_verified'] is False
    assert result['binding']['attempt_id']=='r02' and result['binding']['paths']==list(publisher.PATHS)
    assert record.read_bytes()==before and len(calls)==5
    assert result['summary_status'] is None and (folder/'readback.json').exists()


@pytest.mark.parametrize('phase',['TRANSFER','REMOTE_EXTRACT_DISPATCH'])
def test_scalar_r02_timeout_preserves_bytes_unknown_and_blocks_repeat(tmp_path,phase):
    record,folder,calls,call,source=fake_publication(tmp_path,timeout_phase=phase)
    with pytest.raises(subprocess.TimeoutExpired):
        publisher.publish(record,artifact_root=folder,call=call,source_check=lambda *args:source)
    evidence=json.loads((folder/'publication.json').read_text(encoding='utf-8'))
    assert evidence['status']=='UNKNOWN' and evidence['phase']==phase and evidence['automatic_retry'] is False
    assert (folder/(phase+'.stdout')).read_bytes()==(
        b'scalar-r02 transfer bytes' if phase=='TRANSFER' else b'scalar-r02 possible dispatch')
    assert (folder/(phase+'.stderr')).read_bytes()==b'literal timeout'
    count=len(calls)
    with pytest.raises(ValueError,match='Existing publication'):
        publisher.publish(record,artifact_root=folder,call=call,source_check=lambda *args:source)
    assert len(calls)==count
