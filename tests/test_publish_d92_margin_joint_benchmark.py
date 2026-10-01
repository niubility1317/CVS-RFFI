"""Publication contract tests use fake Git/SSH; isolated import opens no data."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import publish_d92_margin_joint_benchmark as pub
from test_run_d92_margin_joint_benchmark import spec, COMMIT


def test_precise_whitelist_no_native_or_data_directories(tmp_path):
    s=spec(tmp_path);paths=pub.release_paths(s)
    assert len(paths)==len(set(paths))
    assert all(v.endswith(('.py','.md','.json')) for v in paths)
    assert 'code' not in paths and 'experiment_registry' not in paths
    assert not any('checkpoint_loading' in v or 'identity_only_forward' in v or 'handoff' in v for v in paths)
    assert {'tools/evaluate_d92_margin_joint_benchmark.py','tools/score_d92_margin_joint_benchmark.py',
            'tools/report_d92_margin_joint_benchmark.py','tools/export_d92_branch_features.py',
            'tools/d92_orbit_feature_cache.py'}<=set(paths)
    assert paths[-1]==s['spec_path']


def test_readiness_cannot_claim_missing_source_available(tmp_path):
    s=spec(tmp_path);value=pub.readiness(s,tmp_path)
    assert value['status']=='INCOMPLETE_SOURCE_BUNDLE' and value['missing'] and value['launched'] is False


@pytest.mark.parametrize('changed', ['unpushed','dirty','detached'])
def test_git_binding_requires_independent_oid_and_clean_whitelist(tmp_path,changed):
    s=spec(tmp_path);calls=[]
    def output(argv,**kwargs):
        calls.append(argv)
        if argv[1:3]==['rev-parse','HEAD']:return COMMIT+'\n'
        if argv[1:3]==['branch','--show-current']:return '' if changed=='detached' else 'topic\n'
        if argv[1]=='ls-remote':return ('c'*40 if changed=='unpushed' else COMMIT)+'\trefs/heads/topic\n'
        if argv[1]=='status':return ' M tools/run_d92_margin_joint_benchmark.py' if changed=='dirty' else ''
        raise AssertionError(argv)
    with pytest.raises(ValueError):pub.git_binding(s,tmp_path,check_output=output,check_call=lambda *a,**k:None)


def test_git_parent_and_actual_runtime_are_distinct(tmp_path):
    s=spec(tmp_path);calls=[]
    def output(argv,**kwargs):
        if argv[1]=='rev-parse':return COMMIT
        if argv[1]=='branch':return 'topic'
        if argv[1]=='ls-remote':return COMMIT+'\trefs/heads/topic'
        return ''
    def check(argv,**kwargs):calls.append(argv)
    value=pub.git_binding(s,tmp_path,check_output=output,check_call=check)
    assert value['runtime_commit']==COMMIT and value['preparation_commit']=='a'*40
    assert calls[0]==['git','merge-base','--is-ancestor','a'*40,COMMIT]
    assert calls[1][:5]==['git','diff','--exit-code','a'*40,COMMIT]


def test_remote_program_safe_members_cpu_only_prediction_owner(tmp_path):
    s=spec(tmp_path);binding=dict(runtime_commit=COMMIT)
    cfg=pub.remote_config(s,binding,'/archive/source.tar','d'*64)
    program=pub.REMOTE.replace('CONFIG',repr(cfg));compile(program,'synthetic remote','exec')
    assert 'member.isfile()' in program and 'member.isdir()' in program
    assert 'files!=expected' in program and "'..' in name.parts" in program
    assert 'start_new_session=True' in program and "'--commit'" in program
    assert 'nvidia-smi' not in program and 'kill(' not in program
    assert "'--score'" not in program and "'tools/run_d92_margin_joint_benchmark.py'" in program
    assert cfg['environment']['CUDA_VISIBLE_DEVICES']==''
    assert cfg['environment']['OPENBLAS_NUM_THREADS']=='2'


def test_reconcile_is_read_only_metadata(tmp_path):
    program=pub.reconcile_program(spec(tmp_path));compile(program,'reconcile','exec')
    assert 'startup.json' in program and 'complete.json' in program and 'archive_sha256' in program
    assert 'Popen' not in program and 'truth.json' not in program and 'np.load' not in program
    assert 'os.kill(pid,0)' in program


@pytest.mark.parametrize('change', ['commit','pid','spec','truth'])
def test_exit_or_launch_json_is_not_post_state_verification(tmp_path,change):
    s=spec(tmp_path);binding=dict(runtime_commit=COMMIT)
    startup=dict(runtime_commit=COMMIT,code_commit='a'*40,resolved_spec=s,run_id=s['run_id'],pid=42,
                 group_id=s['group_id'],schema=pub.SCHEMA,status=pub.STARTED,truth_read=False,scorer_invoked=False)
    launch=dict(runtime_commit=COMMIT,cwd=s['code']['cwd'],run_root=s['execution']['remote_run_root'],launch_owner='root',pid=42)
    if change=='commit':startup['runtime_commit']='c'*40
    elif change=='pid':startup['pid']=43
    elif change=='spec':startup['resolved_spec']={}
    else:startup['truth_read']=True
    with pytest.raises(ValueError):pub.verify_landing(dict(metadata=dict(startup=startup,launch=launch)),s,binding)


def test_exact_runtime_bundle_imports_without_data_or_encoder():
    value=pub.verify_bundle_imports()
    assert value['status']=='VERIFIED' and value['data_access'] is False and value['native_encoder_loaded'] is False
    assert all(v in pub.RUNTIME_PATHS for v in value['module_files'])


def test_unknown_remote_landing_is_preserved_without_retry(tmp_path,monkeypatch):
    s=spec(tmp_path);s['code']['cwd']='/exclusive-release';s['execution']['remote_run_root']='/exclusive-run'
    for row in s['rows']:row['output_root']='/exclusive-run/'+row['row_id']
    root=tmp_path/'source-repo';root.mkdir()
    for name in pub.release_paths(s):
        file=root/name;file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text(json.dumps(s) if name==s['spec_path'] else 'synthetic source',encoding='utf-8')
    calls=[]
    def call(argv,**kwargs):
        calls.append(argv)
        if argv[:2]==['git','archive']:
            file=Path(next(v.split('=',1)[1] for v in argv if v.startswith('--output=')));file.write_bytes(b'synthetic tar')
        if argv[0]=='ssh' and kwargs.get('capture_output'):
            return SimpleNamespace(returncode=1,stdout=b'partial landing',stderr=b'unknown launch boundary')
        return SimpleNamespace(returncode=0)
    with pytest.raises(ValueError):pub.publish(s['spec_path'],root,artifact_root=tmp_path/'artifacts',
        import_check=lambda *a:dict(status='VERIFIED'),git_check=lambda *a:dict(runtime_commit=COMMIT),call=call)
    folder=tmp_path/'artifacts'/'exclusive-release'
    evidence=pub.read(folder/'publication.json')
    assert evidence['status']=='UNKNOWN' and evidence['automatic_retry'] is False
    assert len([v for v in calls if v[0]=='scp'])==1
    assert len([v for v in calls if v[0]=='ssh'])==2
    assert (folder/'landing.stdout').read_bytes()==b'partial landing'
    with pytest.raises(FileExistsError):pub.publish(s['spec_path'],root,artifact_root=tmp_path/'artifacts',
        import_check=lambda *a:dict(status='VERIFIED'),git_check=lambda *a:dict(runtime_commit=COMMIT),call=call)
