"""Orbit export must precede each lane; completion counts actual CE work."""
from pathlib import Path
import sys
import subprocess
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from test_run_d92_branch_support_probe import spec as base_spec, fake_launcher as base_launcher
import run_d92_branch_orbit_ce_probe as runner


def spec(tmp_path):
    result=base_spec(tmp_path)
    result['probe'].update(view_count=4,reuse_support_cache=False)
    return result


def fake_launcher(value,fail_row=None,bad_export=False):
    original,calls=base_launcher(value,fail_row,bad_export)
    def launch(argv,log,cwd,stage):
        original(argv,log,cwd,stage)
        out=Path(argv[argv.index('--output')+1])
        row=next(r for r in value['rows'] if r['row_id']==out.parent.name)
        p=out/('features_complete.json' if stage=='export' else 'probe_complete.json')
        m=runner.read(p);m['model_seed']=row['seeds']['model']
        if stage=='export':m['status']='BRANCH_ORBIT_SUPPORT_FEATURES_COMPLETE'
        else:
            n=m['episodes'];m.update(k1_episodes=n//4,oof_episodes=3*n//4,proxy_anchor_count=35*n//4,
                factorization_count=88*n//4,optimizer_steps=99*n)
        runner.write(p,m)
    return launch,calls


def test_full_matrix_and_actual_steps_after_export_barrier(tmp_path):
    s=spec(tmp_path);fn,calls=fake_launcher(s)
    runner.run(s,'commit',launch_fn=fn)
    complete=runner.read(Path(s['execution']['remote_run_root'])/'complete.json')
    assert complete['status']=='SUPPORT_PROBE_COMPLETE'
    assert [complete[k] for k in ('episodes','k1_episodes','oof_episodes','proxy_anchor_count')]==[4800,1200,3600,42000]
    assert complete['factorization_count']==105600 and complete['optimizer_steps']==99*4800
    assert len(calls)==16
    for row in s['rows']:
        assert calls.index((row['row_id'],'export'))<calls.index((row['row_id'],'probe'))
    with pytest.raises(FileExistsError):runner.run(s,'commit',launch_fn=fn)


@pytest.mark.parametrize('bad_marker',[False,True])
def test_failed_export_is_preserved_and_never_probed_or_retried(tmp_path,bad_marker):
    s=spec(tmp_path);rid=s['rows'][0]['row_id'];fn,calls=fake_launcher(s,rid,bad_marker)
    with pytest.raises(RuntimeError,match='One or more'):runner.run(s,'commit',launch_fn=fn)
    done=runner.read(Path(s['execution']['remote_run_root'])/'complete.json')
    assert done['status']=='FAILED' and done['completed_rows']==7
    assert calls.count((rid,'export'))==1 and (rid,'probe') not in calls


@pytest.mark.parametrize('change',[
    lambda s:s['probe'].update(query_access=True),lambda s:s['probe'].update(view_count=1),
    lambda s:s['probe'].update(reuse_support_cache=True),lambda s:s['rows'].pop(),
    lambda s:s['rows'][0].update(output_root='/outside'),lambda s:s['execution'].update(cpu_lanes=8)])
def test_invalid_contract_precedes_output_creation(tmp_path,change):
    s=spec(tmp_path);change(s)
    with pytest.raises(ValueError):runner.run(s,'x',launch_fn=lambda *a:None)
    assert not Path(s['execution']['remote_run_root']).exists()


def test_commands_only_use_support_export_and_new_probe(tmp_path):
    s=spec(tmp_path);export,probe=runner.commands(s,s['rows'][0])
    assert 'export_d92_branch_orbit_support_features.py' in ' '.join(export)
    assert 'evaluate_d92_branch_orbit_ce_probe.py' in ' '.join(probe)
    assert not any(word in ' '.join(export+probe) for word in ('--truth','--ground','--query','score_d92'))


def test_wrong_model_or_incomplete_proxy_marker_is_rejected(tmp_path):
    s=spec(tmp_path);row=s['rows'][0];co=s['probe']['cohorts'][row['cohort']];n=co['expected_split_count']
    p=tmp_path/'marker.json';m=dict(status='SUPPORT_PROBE_COMPLETE',model_seed=row['seeds']['model'],
        capsule_id=co['capsule_id'],checkpoint_sha256=row['expected_checkpoint_sha256'],episodes=n,
        k1_episodes=n//4,oof_episodes=3*n//4,proxy_anchor_count=35*n//4,
        query_rows_used=0,source_rows_used=0)
    for key,value in [('model_seed',42),('proxy_anchor_count',0)]:
        runner.write(p,dict(m,**{key:value}))
        with pytest.raises(ValueError):runner.verify_marker(p,s,row,'probe')


def test_private_transport_preserves_gpu_export_and_old_module():
    p=subprocess.run([sys.executable,'-c',
        "import sys;sys.path.insert(0,'tools');import publish_d92_branch_support_probe as b;"
        "old=b.REMOTE;oldpaths=list(b.PATHS);import publish_d92_branch_orbit_ce_probe as p;"
        "assert b.REMOTE==old and b.PATHS==oldpaths;"
        "s=p.transport.REMOTE.replace('CONFIG','{}');compile(s,'remote','exec');"
        "assert 'nvidia-smi' in s;assert 'run_d92_branch_orbit_ce_probe.py' in s"],
        cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert p.returncode==0,p.stderr


@pytest.mark.parametrize('name',['publish_d92_branch_orbit_ce_probe.py','analyze_d92_branch_orbit_ce_probe.py'])
def test_transports_require_explicit_spec(name):
    p=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'tools'/name)],capture_output=True,text=True)
    assert p.returncode==2 and '--spec' in p.stderr


def test_prepared_spec_retains_source_and_capsule_contracts():
    import json
    import prepare_d92_branch_orbit_ce_probe as prepare
    root=Path(__file__).resolve().parents[1]
    documents=prepare.documents();s=documents[prepare.SPEC]
    original=json.loads((root/'configs/d92_branch_support_probe_20260929.json').read_text(encoding='utf-8'))
    runner.validate_spec(s)
    assert s['checkpoint']['runtime_checkpoint_reload'] is True
    assert s['probe']['max_factorizations']==s['probe']['expected_ce_fits']==105600
    for row,old in zip(s['rows'],original['rows']):
        assert row['source_root']==old['source_root'] and row['seeds']==old['seeds']
        assert row['data_overrides']==old['data_overrides']
        assert row['expected_checkpoint_sha256']==old['expected_checkpoint_sha256']
        assert 'support_features' not in row
    frozen=json.loads((root/'configs/d92_branch_orbit_ce_frozen_20260929.json').read_text(encoding='utf-8'))['algorithm']
    for cohort in ('rx3','rx1'):
        assert documents[f'configs/d92_branch_orbit_ce_support_{cohort}_20260929.json']==dict(
            algorithm=frozen,matrix=original['probe']['cohorts'][cohort]['matrix'])


def test_analysis_archive_dependency_closure(tmp_path):
    import shutil
    import analyze_d92_branch_orbit_ce_probe as analyzer
    root=Path(__file__).resolve().parents[1]
    for rel in analyzer.transport.PATHS:
        src=root/rel;dst=tmp_path/rel
        if src.is_dir():shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        else:
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    p=subprocess.run([sys.executable,'-I',str(tmp_path/'tools/summarize_d92_branch_orbit_ce_probe.py'),'--help'],
        cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
