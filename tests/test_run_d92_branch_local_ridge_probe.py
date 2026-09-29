"""The local radial ridge diagnostic only schedules bound caches once, on CPU."""
from pathlib import Path
import sys
import subprocess
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from test_run_d92_branch_interaction_probe import spec
from test_run_d92_branch_support_probe import fake_launcher as original_launcher
import run_d92_branch_local_ridge_probe as runner


def fake_launcher(s, failing=None):
    original,calls=original_launcher(s,failing)
    def launch(argv,log,cwd,stage):
        original(argv,log,cwd,stage)
        marker=log.parent/'probe/probe_complete.json'
        if marker.exists():
            m=runner.read(marker);n=m['episodes']
            m.update(k1_episodes=n//4,oof_episodes=3*n//4,proxy_anchor_count=35*n//4,
                factorization_count=33*n,optimizer_steps=0)
            runner.write(marker,m)
    return launch,calls


def test_all_eight_cached_rows_complete_without_export(tmp_path):
    s=spec(tmp_path);fn,calls=fake_launcher(s)
    runner.run(s,'commit',launch_fn=fn)
    complete=runner.read(Path(s['execution']['remote_run_root'])/'complete.json')
    assert complete['status']=='SUPPORT_PROBE_COMPLETE' and complete['episodes']==4800
    assert complete['completed_rows']==8 and len(calls)==8 and all(v[1]=='probe' for v in calls)
    assert (complete['k1_episodes'],complete['oof_episodes'],complete['proxy_anchor_count'])==(1200,3600,42000)
    assert complete['factorization_count']==158400 and complete['optimizer_steps']==0
    assert complete['checkpoint_loaded'] is complete['gpu_use'] is False
    with pytest.raises(FileExistsError):runner.run(s,'commit',launch_fn=fn)


def test_failed_lane_preserved_and_no_retry(tmp_path):
    s=spec(tmp_path);rid=s['rows'][0]['row_id'];fn,calls=fake_launcher(s,rid)
    with pytest.raises(RuntimeError,match='lane failed'):runner.run(s,'commit',launch_fn=fn)
    complete=runner.read(Path(s['execution']['remote_run_root'])/'complete.json')
    assert complete['status']=='FAILED' and complete['completed_rows']==7 and calls.count((rid,'probe'))==1


@pytest.mark.parametrize('change',[
    lambda s:s['probe'].update(query_access=True), lambda s:s['probe'].update(reuse_support_cache=False),
    lambda s:s['execution'].update(export_device='cuda:0'),lambda s:s['rows'].pop(),
    lambda s:s['rows'][0].update(support_features=s['rows'][0]['output_root']),
    lambda s:s['rows'][0].update(output_root='/outside')])
def test_bad_scope_before_mutation(tmp_path,change):
    s=spec(tmp_path);change(s)
    with pytest.raises(ValueError):runner.run(s,'x',launch_fn=lambda *a:None)
    assert not Path(s['execution']['remote_run_root']).exists()


def test_cpu_command_and_readback(tmp_path):
    import read_d92_run
    s=spec(tmp_path);argv=runner.command(s,s['rows'][0])
    assert argv[argv.index('--support-features')+1]==s['rows'][0]['support_features']
    assert 'evaluate_d92_branch_local_ridge_probe.py' in ' '.join(argv)
    assert not any(v in ' '.join(argv) for v in ('--truth','--source','--device','--ground','export_d92'))
    text=read_d92_run.readback_script(s);compile(text,'readback','exec')
    assert "candidate_folder='probe'" in text


def test_private_cpu_publisher_compiles_without_mutating_base():
    p=subprocess.run([sys.executable,'-c',
        "import sys;sys.path.insert(0,'tools');import publish_d92_branch_support_probe as b;"
        "old=b.REMOTE;oldpaths=list(b.PATHS);import publish_d92_branch_local_ridge_probe as p;"
        "assert b.REMOTE==old and b.PATHS==oldpaths;"
        "s=p.transport.REMOTE.replace('CONFIG','{}');compile(s,'remote','exec');"
        "assert 'nvidia-smi' not in s;assert 'run_d92_branch_local_ridge_probe.py' in s"],
        cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert p.returncode==0,p.stderr


def test_incomplete_proxy_marker_is_not_completed(tmp_path):
    s=spec(tmp_path);row=s['rows'][0];co=s['probe']['cohorts'][row['cohort']];n=co['expected_split_count']
    marker=tmp_path/'marker.json'
    runner.write(marker,dict(status='SUPPORT_PROBE_COMPLETE',capsule_id=co['capsule_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'],episodes=n,k1_episodes=n//4,
        oof_episodes=3*n//4,proxy_anchor_count=35*n//4-1,query_rows_used=0,source_rows_used=0))
    with pytest.raises(ValueError,match='Incomplete'):runner.verify_marker(marker,s,row)


def test_prepared_real_spec_keeps_exact_raw_cache_bindings():
    import json
    import prepare_d92_branch_local_ridge_probe as prepare
    root=Path(__file__).resolve().parents[1]
    documents=prepare.documents();s=documents[prepare.SPEC]
    parent=json.loads((root/'configs/d92_branch_interaction_support_20260929.json').read_text(encoding='utf-8'))
    runner.validate_spec(s)
    assert s['probe']['max_factorizations']==158400 and 'expected_factorizations' not in s['probe']
    assert s['probe']['expected_proxy_anchors']==42000
    for row,old in zip(s['rows'],parent['rows']):
        assert row['support_features']==old['support_features']
        assert row['seeds']==old['seeds'] and row['data_overrides']==old['data_overrides']
        assert row['expected_checkpoint_sha256']==old['expected_checkpoint_sha256']
    frozen=json.loads((root/'configs/d92_branch_local_ridge_frozen_20260929.json').read_text(encoding='utf-8'))['algorithm']
    for cohort in ('rx3','rx1'):
        assert documents[f'configs/d92_branch_local_ridge_support_{cohort}_20260929.json']==dict(
            algorithm=frozen,matrix=parent['probe']['cohorts'][cohort]['matrix'])


@pytest.mark.parametrize('name',['publish_d92_branch_local_ridge_probe.py','analyze_d92_branch_local_ridge_probe.py'])
def test_specialized_transports_require_explicit_run(name):
    p=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'tools'/name)],
        capture_output=True,text=True)
    assert p.returncode==2 and '--spec' in p.stderr


def test_analysis_archive_dependency_closure_in_isolated_directory(tmp_path):
    import shutil
    import analyze_d92_branch_local_ridge_probe as analyzer
    root=Path(__file__).resolve().parents[1]
    for rel in analyzer.transport.PATHS:
        src=root/rel;dst=tmp_path/rel
        if src.is_dir():shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        else:
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,dst)
    p=subprocess.run([sys.executable,'-I',str(tmp_path/'tools/summarize_d92_branch_local_ridge_probe.py'),'--help'],
        cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
