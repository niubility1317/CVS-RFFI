"""Literal offline fixtures: never invoke the original scorer or read real outputs."""
if __name__=='__main__':
    import ast
    from pathlib import Path
    import sys
    if sys.argv[1:]!=['--static-check']:raise SystemExit('Only --static-check is available as a script')
    roots=[Path('E:/type10-7'),Path('E:/type10-7/code/snapshots/d92_support_upgrade_20260928_wt')]
    for relative in ('tools/execute_frozen_d92_group_query_score_contract_r02.py','tests/test_execute_frozen_d92_group_query_score_contract_r02.py'):
        raw=[(root/relative).read_bytes() for root in roots]
        if raw[0]!=raw[1]:raise ValueError('Owned mirrors differ: '+relative)
        ast.parse(raw[0].decode('utf-8',errors='strict'),filename=relative)
        print('STATIC_UTF8_AST_MIRROR_OK',relative,len(raw[0]))
    raise SystemExit(0)

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import execute_frozen_d92_group_query_score_contract_r02 as wrapper


def literal_plan():
    return dict(schema=wrapper.SCHEMA,run_id=wrapper.RUN_ID,attempt_id='r02',
        fitted_runtime_commit=wrapper.RUNTIME,launch_owner='root',automatic_retry=False,
        training=False,fit=False,parameter_selection=False,
        command=[wrapper.PYTHON,'-u',wrapper.SCORER,'--spec',wrapper.SPEC,
            '--run-root',wrapper.ORIGINAL_RUN,'--output',wrapper.OUTPUT],
        original_release=wrapper.RELEASE,original_run=wrapper.ORIGINAL_RUN,scorer_release=wrapper.SCORER_RELEASE,
        spec_path=wrapper.SPEC,
        output_directory=wrapper.OUTPUT_DIRECTORY,output=wrapper.OUTPUT,
        expected_score_status=wrapper.SCORE_STATUS,expected_rows=4,expected_parents=2400,
        environment=dict(wrapper.CPU_ENV),cpu_lanes=2,blas_threads=2,
        source_preparation_commit='c7c46558d5eb8f34bb57fb09c12adc7ac2c5109f')


def literal_metadata():
    rows=[dict(row_id=co+'_model_'+str(seed),cohort=co,expected_model_seed=seed,
        expected_checkpoint_sha256=('b' if seed==2026092701 else 'c')*64,
        output_root=wrapper.ORIGINAL_RUN+'/'+co+'_model_'+str(seed))
        for co in ('rx3','rx1') for seed in (2026092701,2026092702)]
    spec=dict(schema=wrapper.PREDICTION_SCHEMA,run_id=wrapper.RUN_ID,group_id='literal_original_group',
        code=dict(cwd=wrapper.RELEASE,environment=wrapper.PYTHON,commit='a'*40),
        execution=dict(remote_run_root=wrapper.ORIGINAL_RUN),
        benchmark=dict(cohorts=dict(rx3=dict(expected_split_count=900),rx1=dict(expected_split_count=300))),
        rows=rows)
    startup=dict(schema=wrapper.PREDICTION_SCHEMA,status='GROUP_BARRIER_QUERY_SUPERVISOR_STARTED',
        resolved_spec=deepcopy(spec),run_id=wrapper.RUN_ID,group_id=spec['group_id'],
        runtime_commit=wrapper.RUNTIME,code_commit=spec['code']['commit'],
        rows={r['row_id']:dict(status='PENDING') for r in rows},
        truth_read=False,scorer_invoked=False,automatic_retry=False)
    complete=dict(startup,status='GROUP_BARRIER_QUERY_BENCHMARK_PREDICTIONS_COMPLETE',
        row_count=4,completed_row_count=4,all_predictions_fixed=True)
    complete['rows']={}
    for row in rows:
        count=900 if row['cohort']=='rx3' else 300
        complete['rows'][row['row_id']]=dict(row,status='COMPLETE',release_commit=wrapper.RUNTIME,
            marker=dict(status='COMPLETE',split_count=count,completed_split_count=count,release_commit=wrapper.RUNTIME))
    return spec,startup,complete


def test_literal_plan_and_original_900_900_300_300_metadata_close():
    plan=literal_plan();before=deepcopy(plan)
    assert wrapper.validate_plan(plan,'d'*40)==plan
    closure=wrapper.validate_original_metadata(*literal_metadata())
    assert closure['rows']==4 and closure['parents']==2400
    assert closure['runtime_commit']==wrapper.RUNTIME and plan==before
    assert plan['attempt_id']=='r02' and plan['scorer_release']==wrapper.SCORER_RELEASE
    assert plan['original_release']==wrapper.RELEASE and wrapper.SCORER_RELEASE!=wrapper.RELEASE
    assert plan['command'][2]==wrapper.SCORER and plan['output_directory'].endswith('-score-contract-r02')


@pytest.mark.parametrize('fault',['runtime','output_child','output_original','release','argv_scorer','argv_extra',
    'argv_python','fit','training','selection','owner','retry','environment','lanes','publication_oid',
    'scorer_release','old_attempt'])
def test_plan_rejects_paths_method_invocation_permissions_or_source_identity(fault):
    plan=literal_plan();source='d'*40
    if fault=='runtime':plan['fitted_runtime_commit']='e'*40
    elif fault=='output_child':plan['output_directory']=wrapper.ORIGINAL_RUN+'/score'
    elif fault=='output_original':plan['output']=wrapper.ORIGINAL_RUN+'/summary.json'
    elif fault=='release':plan['original_release']='/home/szu2070436088/other_release'
    elif fault=='argv_scorer':plan['command'][2]='/home/szu2070436088/other_scorer.py'
    elif fault=='argv_extra':plan['command'].append('--retune')
    elif fault=='argv_python':plan['command'][0]='/other/python'
    elif fault in ('fit','training','selection','retry'):
        plan[dict(fit='fit',training='training',selection='parameter_selection',retry='automatic_retry')[fault]]=True
    elif fault=='owner':plan['launch_owner']='worker'
    elif fault=='environment':plan['environment']['CUDA_VISIBLE_DEVICES']='0'
    elif fault=='lanes':plan['cpu_lanes']=3
    elif fault=='scorer_release':plan['scorer_release']=wrapper.RELEASE
    elif fault=='old_attempt':plan['attempt_id']='r01'
    else:source='not_an_oid'
    with pytest.raises(ValueError):wrapper.validate_plan(plan,source)


@pytest.mark.parametrize('fault',['incomplete','not_fixed','truth','scorer','retry','runtime','row_status',
    'row_marker_count','row_runtime','missing_row','wrong_cohort_count','different_resolved_spec'])
def test_original_metadata_rejects_partial_or_changed_prediction_closure(fault):
    spec,startup,complete=literal_metadata()
    if fault=='incomplete':complete['completed_row_count']=3
    elif fault=='not_fixed':complete['all_predictions_fixed']=False
    elif fault in ('truth','scorer','retry'):
        complete[dict(truth='truth_read',scorer='scorer_invoked',retry='automatic_retry')[fault]]=True
    elif fault=='runtime':complete['runtime_commit']='e'*40
    elif fault=='row_status':next(iter(complete['rows'].values()))['status']='FAILED'
    elif fault=='row_marker_count':next(iter(complete['rows'].values()))['marker']['completed_split_count']=899
    elif fault=='row_runtime':next(iter(complete['rows'].values()))['release_commit']='e'*40
    elif fault=='missing_row':complete['rows'].pop(next(iter(complete['rows'])))
    elif fault=='wrong_cohort_count':spec['benchmark']['cohorts']['rx3']['expected_split_count']=899
    else:complete['resolved_spec']['code']['commit']='f'*40
    with pytest.raises(ValueError):wrapper.validate_original_metadata(spec,startup,complete)


def fake_execution(tmp_path,*,returncode=0,metadata_fault=False):
    plan=literal_plan();spec,startup,complete=literal_metadata();calls=[];reads=[]
    if metadata_fault:complete['all_predictions_fixed']=False
    values={wrapper.SPEC:spec,wrapper.ORIGINAL_RUN+'/startup.json':startup,
        wrapper.ORIGINAL_RUN+'/complete.json':complete}
    def reader(path):
        reads.append(path);assert path in values
        return deepcopy(values[path])
    def popen(argv,**kwargs):
        calls.append((argv,kwargs));return SimpleNamespace(pid=12345,returncode=None)
    def waiter(child):
        assert child.pid==12345
        return dict(returncode=returncode,child_peak_rss_bytes=None,child_peak_rss_scope='N/A; LITERAL_OFFLINE_FIXTURE')
    out=tmp_path/'fresh_score_contract_r02'
    def path_factory(path):
        assert path==wrapper.OUTPUT_DIRECTORY
        return out
    hooks=dict(read_metadata=reader,popen=popen,waiter=waiter,
        user_fn=lambda:wrapper.REMOTE_USER,path_factory=path_factory)
    return plan,out,calls,reads,hooks


@pytest.mark.parametrize('returncode',[0,7])
def test_execute_fake_child_records_actual_binding_but_never_reads_summary_or_promotes(tmp_path,returncode):
    plan,out,calls,reads,hooks=fake_execution(tmp_path,returncode=returncode)
    before=deepcopy(plan);result=wrapper.execute(plan,'d'*40,**hooks)
    assert len(calls)==1 and reads==[wrapper.SPEC,wrapper.ORIGINAL_RUN+'/startup.json',wrapper.ORIGINAL_RUN+'/complete.json']
    assert calls[0][0]==plan['command'] and calls[0][1]['cwd']==wrapper.SCORER_RELEASE
    assert calls[0][1]['env']['CUDA_VISIBLE_DEVICES']==''
    assert result['status']==('PROCESS_EXITED_AWAITING_ARTIFACT_READBACK' if returncode==0 else 'FAILED')
    assert result['publication_source_commit']=='d'*40 and result['fitted_runtime_commit']==wrapper.RUNTIME
    assert result['scorer_release']==wrapper.SCORER_RELEASE and result['original_release']==wrapper.RELEASE
    assert result['cwd']==wrapper.SCORER_RELEASE and result['attempt_id']=='r02'
    assert result['returncode']==returncode and result['pid']==12345 and result['wall_seconds']>=0
    assert result['score_artifact_verified'] is False and result['summary_read'] is False
    assert result['child_peak_rss_bytes'] is None and plan==before
    assert (out/'score_startup.json').is_file() and (out/'score_execution.json').is_file() and (out/'score.log').is_file()
    assert not (out/'summary.json').exists()
    with pytest.raises(ValueError,match='Existing scoring attempt'):wrapper.execute(plan,'d'*40,**hooks)
    assert len(calls)==1


@pytest.mark.parametrize('fault',['ordinary_user','metadata'])
def test_execution_rejects_before_output_creation_or_original_scorer_call(tmp_path,fault):
    plan,out,calls,reads,hooks=fake_execution(tmp_path,metadata_fault=fault=='metadata')
    if fault=='ordinary_user':hooks['user_fn']=lambda:'root'
    with pytest.raises(ValueError):wrapper.execute(plan,'d'*40,**hooks)
    assert not out.exists() and calls==[]
    if fault=='ordinary_user':assert reads==[]


def test_failed_spawn_is_preserved_without_second_process_or_artifact_claim(tmp_path):
    plan,out,calls,reads,hooks=fake_execution(tmp_path)
    def failed(argv,**kwargs):calls.append(argv);raise OSError('literal spawn failure')
    hooks['popen']=failed
    with pytest.raises(OSError):wrapper.execute(plan,'d'*40,**hooks)
    result=json.loads((out/'score_execution.json').read_text(encoding='utf-8'))
    assert result['status']=='FAILED' and result['returncode'] is None
    assert result['score_artifact_verified'] is False and result['automatic_retry'] is False
    assert (out/'score.log').exists() and len(calls)==1


def test_linux_child_wait4_rss_is_per_child_and_nonlinux_measurement_is_na():
    child=SimpleNamespace(pid=77,returncode=None)
    seen=[]
    def wait4(pid,options):seen.append((pid,options));return pid,0,SimpleNamespace(ru_maxrss=123)
    measured=wrapper.wait_for_child(child,platform_name='linux',wait4=wait4)
    assert seen==[(77,0)] and measured['returncode']==0 and child.returncode==0
    assert measured['child_peak_rss_bytes']==123*1024
    child=SimpleNamespace(pid=88,wait=lambda:0)
    measured=wrapper.wait_for_child(child,platform_name='win32',wait4=wait4)
    assert measured['child_peak_rss_bytes'] is None and len(seen)==1


def test_cli_binds_plan_and_publication_source_without_invoking_scorer(monkeypatch,capsys):
    seen=[]
    monkeypatch.setattr(wrapper,'read',lambda path:literal_plan())
    def fake(plan,commit):seen.append((plan,commit));return dict(status='PROCESS_EXITED_AWAITING_ARTIFACT_READBACK',pid=123,returncode=0)
    monkeypatch.setattr(wrapper,'execute',fake)
    monkeypatch.setattr(sys,'argv',['wrapper','--plan','literal_plan.json','--publication-source-commit','d'*40])
    wrapper.main()
    value=json.loads(capsys.readouterr().out)
    assert seen[0][1]=='d'*40 and value['summary_read'] is False and value['score_artifact_verified'] is False
