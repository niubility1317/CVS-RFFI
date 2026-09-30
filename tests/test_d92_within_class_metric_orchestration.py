"""Synthetic data contracts, actual fit accounting and isolated release closure."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tests')]
from test_d92_registration_orchestration import source
import prepare_d92_registration_diagnostic as parent_prep
import prepare_d92_within_class_metric_probe as prep
import run_d92_within_class_metric_probe as run
import publish_d92_within_class_metric_probe as publish
import analyze_d92_within_class_metric_probe as analyze


def documents(source):
    parent=parent_prep.documents(*source,commit='a'*40)[parent_prep.SPEC]
    before=deepcopy(parent);docs=prep.documents(parent,commit='b'*40)
    assert parent==before
    return docs,parent


def local_spec(source,tmp_path):
    docs,_=documents(source);s=docs[prep.SPEC]
    s['code']['cwd']=(tmp_path/'release').as_posix()
    s['execution']['remote_run_root']=(tmp_path/'output').as_posix()
    for name,co in s['probe']['cohorts'].items():
        co['evaluation_config']=(tmp_path/'release'/run.CONFIG_NAMES[name]).as_posix()
        p=Path(co['evaluation_config']);p.parent.mkdir(parents=True,exist_ok=True);run.write(p,docs[run.CONFIG_NAMES[name]])
    for row in s['rows']:row['output_root']=(tmp_path/'output'/row['row_id']).as_posix()
    return s


def marker(s,row):
    co=s['probe']['cohorts'][row['cohort']]
    return dict(status=run.STATUS,capsule_id=co['capsule_id'],checkpoint_sha256=row['expected_checkpoint_sha256'],
        model_seed=row['seeds']['model'],algorithm=run.PROBE_CONFIG,selection=co['selection'],producer_matrix=co['matrix'],
        episodes=40,k1_episodes=10,oof_episodes=30,proxy_anchor_count=350,sequence_paths=440,
        baseline_head_fit_count=792,metric_fit_count=440,metric_nonidentity_count=90,metric_head_fit_count=162,
        head_fit_count=954,factorization_count=954,metric_factorization_count=90,
        diagnostic_fit_count=540,diagnostic_factorization_count=540,optimizer_steps=0,
        query_rows_used=0,source_rows_used=0,truth_read=False)


def test_prepare_preserves_exact_support_identity_and_provenance(source):
    docs,parent=documents(source);s=docs[prep.SPEC];run.validate_spec(s)
    assert s['run_id']!=parent['run_id'] and s['code']['cwd']!=parent['code']['cwd']
    for name,co in s['probe']['cohorts'].items():
        assert co['selection']==parent['probe']['cohorts'][name]['selection']
    for a,b in zip(s['rows'],parent['rows']):
        for key in ('support_features','expected_checkpoint_sha256','data_overrides','seeds'):
            assert a[key]==b[key]
        assert a['output_root']!=b['output_root']
    assert s['probe']['expected_diagnostic_fit_count']==2160
    assert s['probe']['expected_optimizer_steps']==0


@pytest.mark.parametrize('mutation',[
    lambda s:s['probe'].__setitem__('expected_optimizer_steps',1),
    lambda s:s['probe'].__setitem__('expected_diagnostic_fit_count',0),
    lambda s:s['probe'].__setitem__('candidate','R0'),
    lambda s:s['probe'].__setitem__('interpretation','promoted'),
    lambda s:s['permissions'].__setitem__('source_samples',True),
    lambda s:s['permissions'].__setitem__('query_use','adapt'),
    lambda s:s['probe']['cohorts']['rx3'].__setitem__('evaluation_config','/other.json'),
    lambda s:s['probe']['algorithm'].__setitem__('method','different'),
])
def test_reject_contract_drift(source,mutation):
    s=documents(source)[0][prep.SPEC];mutation(s)
    with pytest.raises(ValueError):run.validate_spec(s)


@pytest.mark.parametrize('key,value',[
    ('optimizer_steps',1),('metric_fit_count',439),('diagnostic_fit_count',539),
    ('factorization_count',955),('metric_factorization_count',91),('diagnostic_factorization_count',541),
    ('metric_nonidentity_count',91),('metric_head_fit_count',163),
    ('head_fit_count',792),('query_rows_used',1),('source_rows_used',1),('truth_read',True),
    ('baseline_head_fit_count',True)])
def test_marker_rejects_incomplete_or_forbidden_work(source,tmp_path,key,value):
    s=local_spec(source,tmp_path);row=s['rows'][0];m=marker(s,row);m[key]=value
    path=tmp_path/'marker.json';run.write(path,m)
    with pytest.raises(ValueError):run.verify_marker(path,s,row)


def test_identity_reuse_is_measured_not_forced_extra_fits(source,tmp_path):
    s=local_spec(source,tmp_path);row=s['rows'][0];m=marker(s,row)
    m.update(metric_nonidentity_count=0,metric_head_fit_count=0,head_fit_count=792,
        factorization_count=792,metric_factorization_count=0)
    p=tmp_path/'marker.json';run.write(p,m)
    assert run.verify_marker(p,s,row)['head_fit_count']==792


def test_runner_actual_counters_and_existing_output_protection(source,tmp_path):
    s=local_spec(source,tmp_path)
    def launch(argv,log,cwd,kind):
        output=Path(argv[argv.index('--output')+1]);output.mkdir()
        row=next(r for r in s['rows'] if Path(r['output_root'])==output.parent)
        assert argv[2].endswith('evaluate_d92_within_class_metric_probe.py')
        run.write(output/'probe_complete.json',marker(s,row))
    run.run(s,'c'*40,launch_fn=launch)
    done=run.read(Path(s['execution']['remote_run_root'])/'complete.json')
    assert done['status']==run.STATUS and done['optimizer_steps']==0
    assert done['metric_fit_count']==1760 and done['head_fit_count']==3816
    assert done['diagnostic_fit_count']==2160
    with pytest.raises(FileExistsError):run.run(s,'c'*40,launch_fn=launch)


def test_failed_lane_keeps_other_lanes_without_retry(source,tmp_path):
    s=local_spec(source,tmp_path);calls=[]
    def launch(argv,log,cwd,kind):
        output=Path(argv[argv.index('--output')+1]);calls.append(output.parent.name)
        if output.parent.name==s['rows'][0]['row_id']:raise RuntimeError('synthetic failure')
        output.mkdir();row=next(r for r in s['rows'] if Path(r['output_root'])==output.parent)
        run.write(output/'probe_complete.json',marker(s,row))
    with pytest.raises(RuntimeError):run.run(s,'c'*40,launch_fn=launch)
    done=run.read(Path(s['execution']['remote_run_root'])/'complete.json')
    assert done['status']=='FAILED' and done['completed_rows']==3
    assert done['head_fit_count']==3*954 and len(calls)==len(set(calls))==4


def test_analysis_rejects_partial_completion():
    with pytest.raises(ValueError):analyze.completion_check(dict(status=run.STATUS,model_rows=4,completed_rows=3,episodes=120))


def test_isolated_release_import_closure(source,tmp_path):
    docs,_=documents(source)
    assert not any(p=='code' or 'support_residual_local_ridge' in p for p in publish.PATHS)
    for name in publish.PATHS:
        dest=tmp_path/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if name in docs:dest.write_text(json.dumps(docs[name]),encoding='utf-8')
        else:shutil.copyfile(ROOT/name,dest)
    for name in ('run_d92_within_class_metric_probe.py','evaluate_d92_within_class_metric_probe.py','summarize_d92_within_class_metric_probe.py'):
        p=subprocess.run([sys.executable,'-s',str(tmp_path/'tools'/name),'--help'],cwd=tmp_path,capture_output=True,text=True)
        assert p.returncode==0,p.stderr
