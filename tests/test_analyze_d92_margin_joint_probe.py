"""No remote execution: dynamic analysis boundaries and source bundle contracts."""
from pathlib import Path
from copy import deepcopy
import io
import json
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'code')]
import analyze_d92_margin_joint_probe as analyzer
from run_d92_margin_joint_probe import budget_for_spec, KS, NEW_COUNTS
from test_prepare_d92_margin_joint_probe import spec as prepared_spec


def _spec(rows=3):
    return prepared_spec(rows)


def summary(spec):
    return dict(status=analyzer.SUMMARY_STATUS,summary_schema=analyzer.SUMMARY_SCHEMA,schema=analyzer.SCHEMA,
        method=analyzer.METHOD,run_id=spec['run_id'],model_rows=len(spec['rows']),coverage=budget_for_spec(spec)['total']['exact'],
        algorithm=spec['probe']['algorithm'],qp_resources=spec['probe']['qp_resources'],release_commit='b'*40,
        query_rows_used=0,source_rows_used=0,actual_A=None)


@pytest.mark.parametrize('rows', [1, 2, 3, 6])
def test_dynamic_completion_and_summary_verification(rows):
    spec = _spec(rows); exact = budget_for_spec(spec)['total']['exact']
    done = dict(status=analyzer.STATUS, run_id=spec['run_id'], model_rows=rows, completed_rows=rows,
        qp_resources=spec['probe']['qp_resources'],workload_complete=True,**exact)
    analyzer.completion_check(done, spec)
    result = summary(spec)
    analyzer.summary_check(result, spec)
    done['episodes'] -= 1
    with pytest.raises(ValueError): analyzer.completion_check(done, spec)
    result['coverage'] = dict(exact, episodes=exact['episodes'] - 1)
    with pytest.raises(ValueError): analyzer.summary_check(result, spec)


@pytest.mark.parametrize('name', ['', '../escape', 'slash/path', 'two names', 'Name'])
def test_unsafe_analysis_release_names_rejected(name):
    with pytest.raises(ValueError): analyzer.remote_config(_spec(), name, 'a' * 40)


def test_remote_script_has_dynamic_coverage_exclusive_evidence_and_no_training_launch():
    cfg = analyzer.remote_config(_spec(), 'synthetic-analysis', 'a' * 40)
    compile(analyzer.REMOTE.replace('CONFIG', repr(cfg)), 'synthetic-remote-analysis', 'exec')
    assert len(cfg['rows']) == 3 and cfg['exact']['episodes'] == 60
    assert cfg['output'] == _spec()['execution']['remote_run_root']+'/results/support_summary'
    assert 'summarize_d92_margin_joint_probe.py' in analyzer.REMOTE
    assert 'analysis_process.json' in analyzer.REMOTE and 'analysis_execution.json' in analyzer.REMOTE
    assert "release.mkdir(exist_ok=False)" in analyzer.REMOTE and "open('x'" in analyzer.REMOTE
    assert 'evaluate_d92_margin_joint_probe.py' not in analyzer.REMOTE
    assert 'run_d92_margin_joint_probe.py' not in analyzer.REMOTE


def test_analysis_whitelist_includes_actual_independent_math_dependencies():
    required = {'code/cvsrffi/d92_margin_joint_local_ridge.py', 'code/cvsrffi/d92_margin_qp_head.py',
        'tools/d92_affine_analysis_math.py',
        'tools/summarize_d92_affine_joint_probe.py', 'tools/evaluate_d92_affine_joint_probe.py',
        'tools/summarize_d92_margin_joint_probe.py', 'tools/analyze_d92_margin_joint_probe.py'}
    assert required <= set(analyzer.PATHS)
    assert len(analyzer.PATHS) == len(set(analyzer.PATHS))


def test_post_state_rejects_qp_config_commit_and_identity_substitutions():
    spec=_spec();cfg=analyzer.remote_config(spec,'synthetic-analysis','a'*40)
    execution=dict(status='VERIFIED',commit='a'*40,runtime_commit='b'*40,run_id=spec['run_id'],
        release=cfg['release'],output=cfg['output'],qp_resources=spec['probe']['qp_resources'])
    evidence=dict(summary=summary(spec),execution=execution,resolved_config=deepcopy(spec))
    analyzer.execution_check(evidence,spec,cfg)
    for category,key,value in [('summary','qp_resources',{}),('summary','algorithm',{}),
        ('summary','query_rows_used',1),('summary','release_commit',None),
        ('execution','runtime_commit','c'*40),('execution','status','UNKNOWN'),
        ('execution','run_id','other'),('execution','output','/other')]:
        bad=deepcopy(evidence);bad[category][key]=value
        with pytest.raises(ValueError):analyzer.execution_check(bad,spec,cfg)
    bad=deepcopy(evidence);bad['resolved_config']['probe']['qp_resources']['max_transitions']+=1
    with pytest.raises(ValueError):analyzer.execution_check(bad,spec,cfg)


@pytest.mark.parametrize('failed',[False,True])
def test_real_remote_control_script_preserves_resolved_config_and_technical_failure(tmp_path,monkeypatch,failed):
    spec=_spec(1);cfg=analyzer.remote_config(spec,'synthetic-analysis','a'*40)
    run=tmp_path/'run';run.mkdir();release=tmp_path/'synthetic-analysis';output=run/'results'/'support_summary'
    archive=tmp_path/'analysis.tar'
    cfg.update(run=str(run),release=str(release),archive=str(archive),output=str(output))
    done=dict(status=analyzer.STATUS,run_id=spec['run_id'],model_rows=1,completed_rows=1,commit='b'*40,
        qp_resources=spec['probe']['qp_resources'],workload_complete=True,**cfg['exact'])
    (run/'complete.json').write_text(json.dumps(done),encoding='utf-8')
    (run/'state.json').write_text(json.dumps({spec['rows'][0]['row_id']:dict(status=analyzer.STATUS)}),encoding='utf-8')
    (run/'startup.json').write_text(json.dumps(dict(spec=spec,commit='b'*40)),encoding='utf-8')
    with tarfile.open(archive,'w') as tar:
        content=b'# synthetic summary source\n';item=tarfile.TarInfo('synthetic-analysis/tools/summarize_d92_margin_joint_probe.py')
        item.size=len(content);tar.addfile(item,io.BytesIO(content))
    calls=[]
    class FakeProcess:
        pid=12345
        def __init__(self,argv,**kwargs):calls.append(argv)
        def wait(self):
            if failed:return 2
            output.mkdir(parents=True);(output/'summary.json').write_text(json.dumps(summary(spec)),encoding='utf-8');return 0
    monkeypatch.setattr(subprocess,'Popen',FakeProcess)
    script=analyzer.REMOTE.replace('CONFIG',repr(cfg))
    if failed:
        with pytest.raises(RuntimeError,match='preserve'):exec(compile(script,'synthetic-analysis','exec'),{})
        assert not (output/'analysis_execution.json').exists()
    else:
        exec(compile(script,'synthetic-analysis','exec'),{})
        evidence=dict(summary=json.loads((output/'summary.json').read_text()),
            execution=json.loads((output/'analysis_execution.json').read_text()),
            resolved_config=json.loads((release/'analysis_resolved_config.json').read_text()))
        analyzer.execution_check(evidence,spec,cfg)
    assert len(calls)==1 and str(release/'analysis_resolved_config.json') in calls[0]
    assert (release/'analysis_process.json').exists() and (release/'analysis.log').exists()
    assert json.loads((release/'analysis_resolved_config.json').read_text())==spec
    with pytest.raises(FileExistsError):exec(compile(script,'synthetic-no-retry','exec'),{})
    assert len(calls)==1
