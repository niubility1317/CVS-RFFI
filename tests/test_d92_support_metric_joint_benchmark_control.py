"""Literal SupportMetric query control fixtures; no existing inputs/specs/results.

Only --static-check is run by the author. Root owns pytest and isolated imports.
"""
if __name__=='__main__':
    import ast
    from pathlib import Path
    import sys
    if sys.argv[1:]!=['--static-check']:raise SystemExit('Only --static-check is available as a script')
    roots=[Path('E:/type10-7'),Path('E:/type10-7/code/snapshots/d92_support_upgrade_20260928_wt')]
    for relative in ('tools/run_d92_support_metric_joint_benchmark.py',
                     'tools/publish_d92_support_metric_joint_benchmark.py',
                     'tests/test_d92_support_metric_joint_benchmark_control.py'):
        raw=[(root/relative).read_bytes() for root in roots]
        if raw[0]!=raw[1]:raise ValueError('Owned mirrors differ: '+relative)
        ast.parse(raw[0].decode('utf-8',errors='strict'),filename=relative)
        print('STATIC_UTF8_AST_MIRROR_OK',relative,len(raw[0]))
    raise SystemExit(0)

from copy import deepcopy
import getpass
import itertools
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'code'),str(Path(__file__).resolve().parents[1]/'tools')]
from cvsrffi.d92_support_metric_joint_local_ridge import FROZEN_CONFIG
import run_d92_support_metric_joint_benchmark as control
import publish_d92_support_metric_joint_benchmark as publisher


def literal_spec(run_root='/source-only/runs/support_metric_query'):
    spec=dict(schema=control.SCHEMA,run_id='SOURCE_ONLY_SUPPORT_METRIC_QUERY',
        group_id='source-only-support-metric-query',spec_path='configs/SOURCE_ONLY_SUPPORT_METRIC_QUERY.json',
        code=dict(cwd='/source-only/releases/support_metric_query',environment='/source-only/python',commit='a'*40),
        execution=dict(remote_run_root=run_root,launch_owner='root',cpu_lanes=2,blas_threads=2),
        benchmark=dict(config=dict(algorithm=deepcopy(FROZEN_CONFIG),support_metric_resources=dict(control.RESOURCES)),
            cohorts={}),rows=[])
    for cohort in ('rx3','rx1'):
        spec['benchmark']['cohorts'][cohort]=dict(capsule='/source-only/inputs/'+cohort,
            truth='/source-only/truth/'+cohort,expected_capsule_id='SOURCE_ONLY_RESIDUAL_'+cohort,
            matrix=dict(control.MATRIX_BASE,receivers=control.RECEIVERS[cohort]),
            expected_split_count=900 if cohort=='rx3' else 300)
        for index,seed in enumerate((2026092701,2026092702)):
            name=cohort+'_model_'+str(seed)
            spec['rows'].append(dict(row_id=name,cohort=cohort,expected_model_seed=seed,
                expected_checkpoint_sha256=('b' if index else 'c')*64,
                row_root='/source-only/inputs/native_'+name,branch_features='/source-only/inputs/branches_'+name,
                ground_packet='/source-only/inputs/native_A_'+str(seed),
                ground_summary='/source-only/inputs/centers_'+str(seed),ground_summary_already_deployed=False,
                output_root=run_root+'/'+name))
    return spec


def literal_preflight(spec,row,commit='d'*40):
    args=control.predict_arguments(spec,row,commit);co=spec['benchmark']['cohorts'][row['cohort']]
    matrix=co['matrix'];splits=[]
    for rx,scenario,k,new,seed in itertools.product(*(matrix[key] for key in
        ('receivers','scenarios','ks','new_counts','support_seeds'))):
        sid='_'.join(map(str,(row['cohort'],rx,scenario,k,new,seed)))
        old_ids=[sid+'_old_'+str(i) for i in range(6*k)]
        new_ids=[sid+'_new_'+str(i) for i in range(new*k)]
        query=[sid+'_query_0',sid+'_query_1']
        splits.append(dict(split_id=sid,receiver=rx,scenario=scenario,k=k,new_count=new,support_seed=seed,
            support_ids=old_ids+new_ids,old_support_ids=old_ids,new_support_ids=new_ids,
            query_ids=query,query_count=len(query)))
    identity=dict(checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['expected_model_seed'])
    return dict(status='SUPPORT_METRIC_QUERY_PREFLIGHT_COMPLETE',
        schema='d92_support_metric_joint_query_predictions_v1',method=FROZEN_CONFIG['method'],
        run_id=spec['run_id'],row_id=row['row_id'],release_commit=commit,capsule_id=co['expected_capsule_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'],model_seed=row['expected_model_seed'],
        algorithm=deepcopy(FROZEN_CONFIG),support_metric_resources=dict(control.RESOURCES),
        query_fit_access=False,source_fit_access=False,truth_read=False,checkpoint_loaded=False,encoder_called=False,
        cross_split_adapted_state_reuse=False,technical_query_chunk_size=1,
        ground_summary_already_deployed=row['ground_summary_already_deployed'],
        C_decision_policy=control.DECISION_POLICY,C_decision_certificate_schema=control.DECISION_SCHEMA,
        run_binding=dict(prediction_output_root=args['output'],**{key:args[key] for key in
            ('row_root','capsule','branch_features','ground_packet','ground_summary')}),
        source_identity=dict(identity),ground_packet_identity=dict(identity),ground_geometry_identity=dict(identity),
        splits=splits,split_count=len(splits),query_record_count=sum(len(s['query_ids']) for s in splits))


def test_literal_full_four_row_2400_matrix_and_six_fixed_resources():
    spec=control.validate_spec(literal_spec())
    assert sum(spec['benchmark']['cohorts'][r['cohort']]['expected_split_count'] for r in spec['rows'])==2400
    assert spec['benchmark']['config']['support_metric_resources']==dict(max_newton_iterations=100,
        max_line_search_trials=64,max_factor_buffer_bytes=167772160,max_integer_bits=65536,
        max_fraction_operations=65536,max_secular_iterations=128)
    for row in spec['rows']:control.validate_preflight(literal_preflight(spec,row),spec,row,'d'*40)


@pytest.mark.parametrize('fault',['algorithm','missing_resource','bool_resource','changed_resource','secular129',
    'owner','lanes','blas','duplicate_model','missing_row','matrix_partial','split_count','source_overlap','same_checkpoint'])
def test_literal_spec_rejects_method_resource_matrix_and_ownership_errors(fault):
    spec=literal_spec()
    if fault=='algorithm':spec['benchmark']['config']['algorithm']['damping']='I5'
    elif fault=='missing_resource':spec['benchmark']['config']['support_metric_resources'].pop('max_integer_bits')
    elif fault=='bool_resource':spec['benchmark']['config']['support_metric_resources']['max_secular_iterations']=True
    elif fault=='changed_resource':spec['benchmark']['config']['support_metric_resources']['max_fraction_operations']+=1
    elif fault=='secular129':spec['benchmark']['config']['support_metric_resources']['max_secular_iterations']=129
    elif fault=='owner':spec['execution']['launch_owner']='worker'
    elif fault=='lanes':spec['execution']['cpu_lanes']=3
    elif fault=='blas':spec['execution']['blas_threads']=3
    elif fault=='duplicate_model':spec['rows'][1]['expected_model_seed']=spec['rows'][0]['expected_model_seed']
    elif fault=='missing_row':spec['rows'].pop()
    elif fault=='matrix_partial':spec['benchmark']['cohorts']['rx3']['matrix']['support_seeds']=[2026092711]
    elif fault=='split_count':spec['benchmark']['cohorts']['rx3']['expected_split_count']=899
    elif fault=='source_overlap':spec['rows'][0]['ground_summary']=spec['execution']['remote_run_root']+'/centers'
    else:
        for row in spec['rows']:row['expected_checkpoint_sha256']='b'*64
    with pytest.raises((ValueError,KeyError)):control.validate_spec(spec)


@pytest.mark.parametrize('fault',['query_fit','source_fit','truth','encoder','chunk','resource','ground_path',
    'model_binding','missing_split','duplicate_split','query_support_overlap','query_count','decision_schema','decision_policy'])
def test_preflight_rejects_direct_access_physical_and_method_metadata_errors(fault):
    spec=literal_spec();row=spec['rows'][0];value=literal_preflight(spec,row)
    if fault in ('query_fit','source_fit','truth','encoder'):
        value[dict(query_fit='query_fit_access',source_fit='source_fit_access',truth='truth_read',encoder='encoder_called')[fault]]=True
    elif fault=='chunk':value['technical_query_chunk_size']=2
    elif fault=='resource':value['support_metric_resources']['max_integer_bits']+=1
    elif fault=='ground_path':value['run_binding']['ground_summary']='/source-only/other_centers'
    elif fault=='model_binding':value['ground_geometry_identity']['model_seed']+=1
    elif fault=='missing_split':value['splits'].pop();value['split_count']-=1
    elif fault=='duplicate_split':value['splits'][-1]=deepcopy(value['splits'][0])
    elif fault=='query_support_overlap':value['splits'][0]['query_ids'][0]=value['splits'][0]['support_ids'][0]
    elif fault=='query_count':value['query_record_count']+=1
    elif fault=='decision_schema':value['C_decision_certificate_schema']='old_or_unknown'
    else:value['C_decision_policy']='ARGMAX_DISPLAY_ROUNDED_SCORE'
    with pytest.raises(ValueError):control.validate_preflight(value,spec,row,'d'*40)


def test_command_uses_new_entry_preflight_same_config_and_never_passes_truth():
    spec=literal_spec();row=spec['rows'][0]
    argv=control.command(spec,row,'d'*40,'/source-only/resolved_config.json',preflight=True)
    assert argv[2].endswith('/tools/evaluate_d92_support_metric_joint_benchmark.py')
    assert argv[-1]=='--preflight-only' and '--truth' not in argv and '--role' not in argv
    assert argv[argv.index('--ground-summary-already-deployed')+1]=='false'
    args=control.predict_arguments(spec,row,'d'*40)
    args['config']['algorithm']['radius']=9
    assert spec['benchmark']['config']['algorithm']['radius']==.5


def test_marker_delegates_only_to_new_standalone_zero_truth_validator(monkeypatch):
    import score_d92_support_metric_joint_benchmark as scorer
    seen=[]
    def validate(*args):seen.append(args);return dict(status='COMPLETE')
    monkeypatch.setattr(scorer,'validate_row_output',validate)
    spec=literal_spec();row=spec['rows'][0];preflight=literal_preflight(spec,row)
    assert control.verify_marker('source-only-output',spec,row,'d'*40,preflight)==dict(status='COMPLETE')
    assert seen==[('source-only-output',spec,row,'d'*40,preflight)]


def fake_control_run(tmp_path,*,bad_preflight=False,bad_exit=False):
    spec=literal_spec((tmp_path/'exclusive_run').as_posix());order=[]
    def preflight(argv,*args):
        row=next(r for r in spec['rows'] if r['row_id']==argv[argv.index('--row-id')+1])
        order.append(('preflight',row['row_id']))
        value=literal_preflight(spec,row)
        if bad_preflight and row is spec['rows'][0]:value['truth_read']=True
        return value
    def launch(argv,log,cwd,environment,started):
        row=next(r for r in spec['rows'] if r['row_id']==argv[argv.index('--row-id')+1])
        order.append(('launch',row['row_id']));pid=10000+spec['rows'].index(row);started(pid)
        return dict(pid=pid,returncode=1 if bad_exit and row is spec['rows'][0] else 0)
    def marker(directory,actual,row,commit,preflight):
        return dict(status='COMPLETE',pid=10000+spec['rows'].index(row),split_count=preflight['split_count'])
    return spec,order,preflight,launch,marker


def test_supervisor_prefights_every_row_before_launch_and_keeps_full_closed_counts(tmp_path):
    spec,order,preflight,launch,marker=fake_control_run(tmp_path)
    complete=control.run(spec,'d'*40,preflight_fn=preflight,launch_fn=launch,marker_fn=marker)
    assert [item[0] for item in order[:4]]==['preflight']*4
    assert complete['status']==control.STATUS and complete['completed_row_count']==4
    assert complete['declared_episode_count']==complete['completed_episode_count']==2400
    assert complete['truth_read'] is False and complete['scorer_invoked'] is False
    assert complete['automatic_retry'] is False
    assert (Path(spec['execution']['remote_run_root'])/'startup.json').is_file()
    with pytest.raises(FileExistsError):control.run(spec,'d'*40,preflight_fn=preflight,launch_fn=launch,marker_fn=marker)
    assert len(order)==8


@pytest.mark.parametrize('failure',['preflight','prediction'])
def test_failed_row_preserves_other_rows_and_never_retries_or_claims_full_completion(tmp_path,failure):
    spec,order,preflight,launch,marker=fake_control_run(tmp_path,
        bad_preflight=failure=='preflight',bad_exit=failure=='prediction')
    with pytest.raises(RuntimeError):control.run(spec,'d'*40,preflight_fn=preflight,launch_fn=launch,marker_fn=marker)
    complete=control.read(Path(spec['execution']['remote_run_root'])/'complete.json')
    assert complete['status']==control.FAILED and complete['completed_row_count']==3
    assert complete['all_predictions_fixed'] is False and complete['automatic_retry'] is False
    assert complete['rows'][spec['rows'][0]['row_id']]['status']=='FAILED'
    assert len([v for v in order if v[0]=='launch'])==(3 if failure=='preflight' else 4)


def test_release_whitelist_covers_metric_modules_new_tools_and_no_states():
    paths=publisher.release_paths(literal_spec())
    assert len(paths)==len(set(paths))
    for name in ('d92_support_metric_basis','d92_support_metric_step','d92_support_metric_joint_local_ridge'):
        assert 'code/cvsrffi/'+name+'.py' in paths
    for name in ('evaluate_d92_support_metric_joint_benchmark','run_d92_support_metric_joint_benchmark',
                 'score_d92_support_metric_joint_benchmark','publish_d92_support_metric_joint_benchmark'):
        assert 'tools/'+name+'.py' in paths
    assert not any(p.startswith(('automation_reports/','experiment_registry/')) or p.endswith(('.npz','.pth','.bin')) for p in paths)
    compile(publisher.REMOTE,'source-only-unexecuted-remote-dispatch','exec')
    compile(publisher.IMPORT_PROGRAM,'source-only-unexecuted-isolated-import','exec')
    assert 'Ordinary N607 user required' in publisher.REMOTE
    assert publisher.EXPECTED_REMOTE_USER=='szu2070436088'


@pytest.mark.parametrize('remote_user',['root','szu2310433034'])
def test_remote_dispatch_namespace_imports_user_checker_and_rejects_before_process_or_paths(monkeypatch,remote_user):
    calls=[]
    monkeypatch.setattr(getpass,'getuser',lambda:remote_user)
    def forbidden(*args,**kwargs):
        calls.append((args,kwargs));raise AssertionError('Child process must not start for a forbidden user')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    # The remote process has its own namespace. Deliberately supply only the
    # user field: rejection must precede even reading release/archive paths.
    namespace={}
    program=publisher.REMOTE.replace('CONFIG',repr(dict(expected_remote_user=publisher.EXPECTED_REMOTE_USER)))
    with pytest.raises(PermissionError,match='Ordinary N607 user required'):
        exec(compile(program,'literal-unexecuted-remote-user-boundary','exec'),namespace)
    assert namespace['getpass'] is getpass and calls==[]


@pytest.mark.parametrize('fault',['user','pid','source','startup','metadata_error'])
def test_independent_landing_readback_rejects_wrong_user_pid_or_binding(fault):
    spec=literal_spec();binding=dict(runtime_commit='d'*40)
    launch=dict(status='DISPATCHED_AWAITING_READBACK',pid=10001,remote_user=publisher.EXPECTED_REMOTE_USER,
        runtime_commit='d'*40,cwd=spec['code']['cwd'],run_root=spec['execution']['remote_run_root'],launch_owner='root')
    startup=dict(runtime_commit='d'*40,code_commit=spec['code']['commit'],resolved_spec=spec,
        run_id=spec['run_id'],group_id=spec['group_id'],schema=control.SCHEMA,status=control.STARTED,pid=10001,
        truth_read=False,scorer_invoked=False)
    value=dict(metadata=dict(launch=launch,startup=startup),metadata_errors={})
    assert publisher.verify_landing(value,spec,binding)['scope']=='LAUNCH_AND_SUPERVISOR_STARTUP_NOT_PREDICTION_COMPLETION'
    if fault=='user':launch['remote_user']='root'
    elif fault=='pid':launch['pid']=None;startup['pid']=None
    elif fault=='source':launch['runtime_commit']='e'*40
    elif fault=='startup':startup['status']='COMPLETE'
    else:value['metadata_errors']['startup']='truncated'
    with pytest.raises(ValueError):publisher.verify_landing(value,spec,binding)


def test_transfer_timeout_preserves_unknown_layer_without_dispatch_or_retry(tmp_path):
    spec=literal_spec();root=tmp_path/'source_only_repository';root.mkdir()
    for relative in publisher.release_paths(spec):
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if relative==spec['spec_path']:path.write_text(json.dumps(spec),encoding='utf-8')
        else:path.write_text('# source-only transport fixture\n',encoding='utf-8')
    calls=[]
    def fake_call(argv,**kwargs):
        calls.append(argv)
        if argv[0]=='scp':raise subprocess.TimeoutExpired(argv,60,output=b'partial transfer',stderr=b'synthetic timeout')
        if argv[:2]==['git','archive']:
            Path(next(v.split('=',1)[1] for v in argv if v.startswith('--output='))).write_bytes(b'SOURCE_ONLY_ARCHIVE_FIXTURE')
        return None
    binding=dict(runtime_commit='d'*40,preparation_commit='a'*40,branch='source_only',remote_oid='d'*40)
    artifact=tmp_path/'publication'
    with pytest.raises(subprocess.TimeoutExpired):
        publisher.publish(spec['spec_path'],root=root,artifact_root=artifact,call=fake_call,
            import_check=lambda root:dict(status='VERIFIED'),git_check=lambda *args:binding)
    folder=artifact/Path(spec['code']['cwd']).name;value=control.read(folder/'publication.json')
    assert value['status']=='UNKNOWN' and value['phase']=='TRANSFER' and value['automatic_retry'] is False
    assert (folder/'TRANSFER.timeout.stdout').read_bytes()==b'partial transfer'
    assert (folder/'TRANSFER.timeout.stderr').read_bytes()==b'synthetic timeout'
    assert value['timeout_output_files']==dict(stdout='TRANSFER.timeout.stdout',stderr='TRANSFER.timeout.stderr')
    assert len(calls)==3
    with pytest.raises(FileExistsError):
        publisher.publish(spec['spec_path'],root=root,artifact_root=artifact,call=fake_call,
            import_check=lambda root:dict(status='VERIFIED'),git_check=lambda *args:binding)
    assert len(calls)==3
