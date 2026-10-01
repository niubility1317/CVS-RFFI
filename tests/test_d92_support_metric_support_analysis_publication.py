"""Source-only fixtures for the independent SupportMetric analysis publisher.

No existing record, configuration, result, data, model or SSH endpoint is read.
The author runs only --static-check; root owns pytest and all external actions.
"""
if __name__=='__main__':
    import ast
    from pathlib import Path
    import sys
    if sys.argv[1:]!=['--static-check']:raise SystemExit('Only --static-check is available as a script')
    workspace=Path('E:/type10-7');worktree=workspace/'code/snapshots/d92_support_upgrade_20260928_wt'
    for relative in ('tools/publish_d92_support_metric_support_analysis.py',
                     'tests/test_d92_support_metric_support_analysis_publication.py'):
        versions=[(root/relative).read_bytes() for root in (workspace,worktree)]
        if versions[0]!=versions[1]:raise ValueError('Owned mirrors differ: '+relative)
        ast.parse(versions[0].decode('utf-8',errors='strict'),filename=relative)
        print('STATIC_UTF8_AST_MIRROR_OK',relative,len(versions[0]))
    raise SystemExit(0)

from copy import deepcopy
import getpass
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import publish_d92_support_metric_support_analysis as publisher


def literal_record():
    original='/source-only/runs/support_metric_original'
    release='/source-only/releases/support_metric_analysis_r01'
    output='/source-only/runs/support_metric_analysis_r01'
    plan=dict(status='PREREGISTERED_NOT_LAUNCHED',attempt_id='r01',source_commit=None,
        source_preparation_commit=publisher.PREPARATION_COMMIT,
        fitted_runtime_commit=publisher.FITTED_COMMIT,paths=list(publisher.PATHS),
        release=release,output=output,log=release+'/analysis.log',
        command=['/source-only/python','-u',release+'/'+publisher.PATHS[0],
            '--spec',release+'/'+publisher.PATHS[1],'--run-root',original,
            '--output',output,'--expected-runtime-commit',publisher.FITTED_COMMIT],
        query_read=False,source_sample_read=False,fit=False,parameter_selection=False,
        launch_owner='root',automatic_retry=False,expected_complete_status=publisher.COMPLETE_STATUS,
        expected_summary_status=publisher.SUMMARY_STATUS,coverage=dict(rows=4,parents=160,sequence_paths=1800))
    return dict(run_id=publisher.RUN_ID,status='TRAINING_COMPLETE',
        execution=dict(remote_run_root=original,support_analysis=plan))


def literal_complete():
    return dict(schema='d92_support_metric_joint_support_probe_v1',
        status=publisher.COMPLETE_STATUS,run_id=publisher.RUN_ID,runtime_commit=publisher.FITTED_COMMIT,
        model_rows=4,completed_rows=4,episodes=160,sequence_paths=1800,workload_complete=True,
        truth_read=False,query_access=False,source_sample_access=False,automatic_retry=False,
        rows={f'row_{i}':dict(status=publisher.COMPLETE_STATUS,release_commit=publisher.FITTED_COMMIT,
            episodes=40,sequence_paths=450) for i in range(4)})


def test_literal_registered_plan_and_original_metadata_close_without_state_promotion():
    record=literal_record();before=deepcopy(record);binding=publisher.validate_record(record)
    evidence=publisher.validate_original_complete(literal_complete(),binding)
    assert evidence['episodes']==160 and evidence['sequence_paths']==1800
    assert record==before and record['execution']['support_analysis']['source_commit'] is None
    assert binding['environment']['CUDA_VISIBLE_DEVICES']==''
    assert all(binding['environment'][k]=='2' for k in
        ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'))


@pytest.mark.parametrize('fault',['training_incomplete','attempt_dispatched','nonempty_source','wrong_preparation',
    'wrong_runtime','query','source','fit','selection','owner','retry','whitelist','coverage',
    'output_original_child','output_original_parent','release_original_child','argv_extra','argv_runtime','log_elsewhere'])
def test_plan_rejects_direct_permission_status_and_path_errors(fault):
    record=literal_record();plan=record['execution']['support_analysis']
    if fault=='training_incomplete':record['status']='RUNNING'
    elif fault=='attempt_dispatched':plan['status']='RUNNING'
    elif fault=='nonempty_source':plan['source_commit']='d'*40
    elif fault=='wrong_preparation':plan['source_preparation_commit']='a'*40
    elif fault=='wrong_runtime':plan['fitted_runtime_commit']='a'*40
    elif fault in ('query','source','fit','selection'):
        plan[dict(query='query_read',source='source_sample_read',fit='fit',selection='parameter_selection')[fault]]=True
    elif fault=='owner':plan['launch_owner']='worker'
    elif fault=='retry':plan['automatic_retry']=True
    elif fault=='whitelist':plan['paths'].append('code/cvsrffi/unwanted_fitter.py')
    elif fault=='coverage':plan['coverage']['parents']=159
    elif fault=='output_original_child':plan['output']=record['execution']['remote_run_root']+'/analysis'
    elif fault=='output_original_parent':plan['output']='/source-only/runs'
    elif fault=='release_original_child':plan['release']=record['execution']['remote_run_root']+'/release'
    elif fault=='argv_extra':plan['command'].append('--retune')
    elif fault=='argv_runtime':plan['command'][-1]='a'*40
    else:plan['log']='/source-only/elsewhere.log'
    with pytest.raises(ValueError):publisher.validate_record(record)


@pytest.mark.parametrize('fault',['rows','parents','paths','runtime','truth','query','source','row_status','row_runtime','row_coverage'])
def test_original_complete_rejects_incomplete_or_cross_runtime_metadata(fault):
    complete=literal_complete()
    if fault=='rows':complete['completed_rows']=3
    elif fault=='parents':complete['episodes']=159
    elif fault=='paths':complete['sequence_paths']=1799
    elif fault=='runtime':complete['runtime_commit']='a'*40
    elif fault in ('truth','query','source'):
        complete[dict(truth='truth_read',query='query_access',source='source_sample_access')[fault]]=True
    elif fault=='row_status':complete['rows']['row_0']['status']='FAILED'
    elif fault=='row_runtime':complete['rows']['row_0']['release_commit']='a'*40
    else:complete['rows']['row_0']['sequence_paths']=449
    with pytest.raises(ValueError):
        publisher.validate_original_complete(complete,publisher.validate_record(literal_record()))


def archive_members(binding):
    return [tarfile.TarInfo(PurePathName(binding['release'])+'/'+relative) for relative in publisher.PATHS]


def PurePathName(value):
    return value.rsplit('/',1)[-1]


def test_exact_two_file_archive_is_accepted_without_models_or_results():
    binding=publisher.validate_record(literal_record())
    assert len(publisher.validate_archive_members(archive_members(binding),binding['release'],list(publisher.PATHS)))==2


@pytest.mark.parametrize('fault',['traversal','absolute','windows','symlink','hardlink','device',
    'extra_file','duplicate','missing','unlisted_directory'])
def test_archive_rejects_unsafe_or_nonwhitelisted_members(fault):
    binding=publisher.validate_record(literal_record());members=archive_members(binding)
    if fault=='traversal':members[0].name=PurePathName(binding['release'])+'/../evil.py'
    elif fault=='absolute':members[0].name='/tmp/evil.py'
    elif fault=='windows':members[0].name=PurePathName(binding['release'])+'\\tools\\evil.py'
    elif fault in ('symlink','hardlink','device'):
        members[0].type=dict(symlink=tarfile.SYMTYPE,hardlink=tarfile.LNKTYPE,device=tarfile.CHRTYPE)[fault]
    elif fault=='extra_file':members.append(tarfile.TarInfo(PurePathName(binding['release'])+'/result.json'))
    elif fault=='duplicate':members.append(deepcopy(members[0]))
    elif fault=='missing':members.pop()
    else:
        item=tarfile.TarInfo(PurePathName(binding['release'])+'/unlisted');item.type=tarfile.DIRTYPE;members.append(item)
    with pytest.raises(ValueError):
        publisher.validate_archive_members(members,binding['release'],list(publisher.PATHS))


def test_remote_programs_compile_and_user_boundary_precedes_original_reads(monkeypatch):
    binding=publisher.validate_record(literal_record());binding.update(source_commit='d'*40,archive_sha256='e'*64)
    programs=[publisher.preflight_program(binding),publisher.dispatch_program(binding),publisher.readback_program(binding)]
    monkeypatch.setattr(getpass,'getuser',lambda:'root')
    for program in programs:
        compile(program,'source-only-unexecuted-remote','exec')
        with pytest.raises(PermissionError,match='Ordinary N607 user'):
            exec(program,{})
        assert 'np.load' not in program and 'torch.load' not in program
        assert 'summary.json' not in program
    compile(publisher.WRAPPER,'source-only-unexecuted-wrapper','exec')
    assert 'PROCESS_EXITED_AWAITING_ARTIFACT_READBACK' in publisher.WRAPPER
    assert "'VERIFIED'" not in publisher.WRAPPER


@pytest.mark.parametrize('fault',['remote_oid','dirty_whitelist'])
def test_git_binding_requires_independent_head_and_clean_exact_whitelist(fault):
    head='d'*40;branch='source_only_branch'
    def fake(argv,**kwargs):
        operation=argv[1:]
        if operation==['rev-parse','HEAD']:return head+'\n'
        if operation==['branch','--show-current']:return branch+'\n'
        if operation[:2]==['ls-remote','origin']:
            return ('a'*40 if fault=='remote_oid' else head)+'\trefs/heads/'+branch+'\n'
        assert operation[:3]==['status','--porcelain','--']
        assert operation[3:]==list(publisher.PATHS)
        return ' M '+publisher.PATHS[0]+'\n'
    with pytest.raises(ValueError):
        publisher.git_binding('/source-only/repository',list(publisher.PATHS),check_output=fake)


def fake_publication(tmp_path,*,failure=None):
    record=literal_record();record_path=tmp_path/'literal_record.json'
    record_path.write_text(json.dumps(record),encoding='utf-8')
    folder=tmp_path/'fresh_publication';calls=[]
    source=dict(source_commit='d'*40,record_commit='d'*40,branch='source_only',remote_oid='d'*40)
    binding=publisher.validate_record(record);binding.update(source)
    launch=dict(status='DISPATCHED_AWAITING_READBACK',pid=12345,cwd=binding['release'],
        source_commit='d'*40,fitted_runtime_commit=publisher.FITTED_COMMIT,run_id=publisher.RUN_ID,
        output=binding['output'],original_run=binding['original_run'],log=binding['log'],
        argv=binding['command'],launch_owner='root',remote_user=publisher.REMOTE_USER,
        environment=binding['environment'],query_read=False,source_sample_read=False,
        fit=False,parameter_selection=False,automatic_retry=False)
    def call(argv,**kwargs):
        calls.append((argv,kwargs))
        if argv[:2]==['git','archive']:
            archive=Path(next(v.split('=',1)[1] for v in argv if v.startswith('--output=')))
            with tarfile.open(archive,'w') as stream:
                for member in archive_members(binding):
                    payload=b'SOURCE_ONLY_SYNTHETIC_SOURCE';member.size=len(payload)
                    stream.addfile(member,io.BytesIO(payload))
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        if argv[0]=='scp':
            if failure=='transfer_timeout':raise subprocess.TimeoutExpired(argv,60,output='partial transfer',stderr='synthetic timeout')
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        program=kwargs['input']
        if 'PREFLIGHT_COMPLETE_NOT_DISPATCHED' in program:
            value=dict(status='PREFLIGHT_COMPLETE_NOT_DISPATCHED',remote_user=publisher.REMOTE_USER,
                run_id=publisher.RUN_ID,source_commit='d'*40,fitted_runtime_commit=publisher.FITTED_COMMIT,
                release=binding['release'],output=binding['output'],original_run=binding['original_run'],
                original_complete=publisher.validate_original_complete(literal_complete(),binding),
                fit=False,query_read=False,source_sample_read=False,parameter_selection=False,automatic_retry=False)
        elif 'DISPATCH_METADATA_READBACK' in program:
            value=dict(status='DISPATCH_METADATA_READBACK',run_id=publisher.RUN_ID,metadata=dict(launch=deepcopy(launch)),
                metadata_errors={},original_complete=publisher.validate_original_complete(literal_complete(),binding),
                original_run=binding['original_run'],fit=False,query_read=False,source_sample_read=False)
            if failure=='readback_mixed':value['metadata']['launch']['fitted_runtime_commit']='a'*40
        else:
            if failure=='dispatch_timeout':raise subprocess.TimeoutExpired(argv,60,output='possible dispatch',stderr='synthetic timeout')
            value=launch
        return SimpleNamespace(returncode=0,stdout=json.dumps(value),stderr='')
    return record_path,folder,calls,call,source


def test_fake_publication_observes_metadata_only_and_never_promotes_record(tmp_path):
    record,folder,calls,call,source=fake_publication(tmp_path);before=record.read_bytes()
    result=publisher.publish(record,repository='/source-only/repository',artifact_root=folder,
        call=call,source_check=lambda *args:source)
    assert result['status']=='DISPATCHED_AWAITING_INDEPENDENT_READBACK'
    assert result['analysis_verified'] is False and result['summary_status'] is None
    assert result['record_modified'] is False and record.read_bytes()==before
    assert len(calls)==5 and (folder/'readback.json').is_file()
    assert json.loads(record.read_text(encoding='utf-8'))['execution']['support_analysis']['status']=='PREREGISTERED_NOT_LAUNCHED'
    with pytest.raises(ValueError,match='Existing publication'):
        publisher.publish(record,artifact_root=folder,call=call,source_check=lambda *args:source)
    assert len(calls)==5


def test_cli_passes_record_as_explicit_positional_argument_without_external_actions(monkeypatch,capsys):
    seen=[]
    def fake(record,**kwargs):
        seen.append((record,kwargs));return dict(status='DISPATCHED_AWAITING_INDEPENDENT_READBACK')
    monkeypatch.setattr(publisher,'publish',fake)
    monkeypatch.setattr(sys,'argv',['source-only-publisher','--record','literal_record.json',
        '--repository','literal_repository','--artifact-root','fresh_literal_folder','--ssh-config','literal_ssh_config'])
    publisher.main()
    assert seen==[(Path('literal_record.json'),dict(repository=Path('literal_repository'),
        artifact_root=Path('fresh_literal_folder'),ssh_config='literal_ssh_config'))]
    assert json.loads(capsys.readouterr().out)['status']=='DISPATCHED_AWAITING_INDEPENDENT_READBACK'


@pytest.mark.parametrize('failure',['transfer_timeout','dispatch_timeout','readback_mixed'])
def test_transport_ambiguity_is_unknown_preserved_and_never_retried(tmp_path,failure):
    record,folder,calls,call,source=fake_publication(tmp_path,failure=failure)
    with pytest.raises((subprocess.TimeoutExpired,ValueError)):
        publisher.publish(record,artifact_root=folder,call=call,source_check=lambda *args:source)
    evidence=json.loads((folder/'publication.json').read_text(encoding='utf-8'))
    assert evidence['status']=='UNKNOWN' and evidence['automatic_retry'] is False
    assert evidence['analysis_verified'] is False and evidence['summary_status'] is None
    assert evidence['dispatch_response_observed'] is (failure=='readback_mixed')
    count=len(calls)
    with pytest.raises(ValueError,match='Existing publication'):
        publisher.publish(record,artifact_root=folder,call=call,source_check=lambda *args:source)
    assert len(calls)==count
    if failure.endswith('timeout'):
        assert (folder/(evidence['phase']+'.stdout')).read_text(encoding='utf-8') in ('partial transfer','possible dispatch')
