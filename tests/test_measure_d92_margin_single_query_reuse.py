"""Bounded synthetic-only tests; no external data or model loading."""
if __name__=='__main__':
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    for path in (Path(__file__),root/'tools/measure_d92_margin_single_query_reuse.py'):
        text=path.read_text(encoding='utf-8');assert '\ufffd' not in text
        ast.parse(text,filename=str(path))
    print('AST_UTF8_ONLY_OK')
    raise SystemExit(0)

import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pytest
from cvsrffi import d92_margin_joint_local_ridge as mj

spec=importlib.util.spec_from_file_location('measure_reuse',Path(__file__).resolve().parents[1]/'tools/measure_d92_margin_single_query_reuse.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def one_query():return [m.make_features(np,mj,1,m.PLAN['evaluation_seed'])]


def test_fixed_plan_and_shared_physical_old_support():
    assert m.PLAN['ks']==[1,5,10,20]
    assert (m.PLAN['query_count'],m.PLAN['warmup_repetitions'],m.PLAN['timed_repetitions'])==(8,1,3)
    assert len({m.PLAN[key] for key in ('model_seed','data_seed','evaluation_seed')})==3
    b,c=m.build_states(np,mj,1)
    assert c.prior is b and len(b.ids)==6 and len(c.ids)==26 and c.ids[:6]==b.ids
    assert len(c.classes)==26 and b.U.shape==c.U.shape==(736,8)
    assert np.any(b.U) and np.any(c.U)
    for key in mj._NAMES:np.testing.assert_array_equal(c.raw[key][:6],b.raw[key])
    with pytest.raises(ValueError,match='fixed synthetic plan'):m.build_states(np,mj,2)


@pytest.mark.parametrize('case,original_heads,original_pairs',[('PAIR',3,38),('C_ONLY',2,32)])
def test_actual_calls_counts_full_score_bits_and_no_double_count(case,original_heads,original_pairs,monkeypatch):
    b,c=m.build_states(np,mj,1);queries=one_query();calls=[];actual=mj._score_residual
    def trace(problem,*args,**kwargs):
        calls.append(problem);return actual(problem,*args,**kwargs)
    monkeypatch.setattr(mj,'_score_residual',trace)
    original,oa=m.measure_arm(b,c,queries,case=case,optional=False)
    assert len(calls)==original_heads
    assert oa['work']['residual_score_call_count']==len(calls)
    calls.clear()
    reused,ra=m.measure_arm(b,c,queries,case=case,optional=True)
    assert len(calls)==2 and calls[0] is b.problem and calls[1] is c.problem
    m.assert_bitwise_equal(np,original,reused)
    assert oa['work']['kernel_pair_count']==original_pairs
    assert ra['work']['kernel_pair_count']==32
    assert oa['work']['raw_distance_pair_count']==2*original_pairs
    assert ra['work']['raw_distance_pair_count']==64
    assert ra['work']['reference_distance_pair_count']==24
    assert ra['work']['intercept_addition_count']==32
    assert ra['work']['residual_score_call_count']==2 and ra['work']['reused_prior_count']==1
    assert ra['work']['prior_residual_score_call_count']==0
    assert ra['work']['public_score_call_count']==2
    for key in m.EXTRA_SECONDS:
        assert ra['overhead_seconds'][key]['measured_call_count']==1
        assert oa['overhead_seconds'][key]['sum'] is None
    json.dumps(ra,allow_nan=False)


def test_equality_requires_bits_not_argmax_or_tolerance():
    a=np.array([[1.,0.]])
    with pytest.raises(RuntimeError,match='full float64'):
        m.assert_bitwise_equal(np,[a],[np.array([[np.nextafter(1.,2.),0.]])])
    with pytest.raises(RuntimeError,match='full float64'):
        m.assert_bitwise_equal(np,[np.array([[0.]])],[np.array([[-0.]])])


def test_thread_contract_rejects_before_runtime_import(monkeypatch):
    for key in m.THREAD_ENV:monkeypatch.setenv(key,'2')
    assert set(m.check_thread_environment().values())=={'2'}
    monkeypatch.setenv('MKL_NUM_THREADS','1')
    with pytest.raises(ValueError,match='before NumPy import'):m.load_runtime()
    monkeypatch.setenv('MKL_NUM_THREADS','2')
    with pytest.raises(RuntimeError,match='fresh process'):m.load_runtime()


def small_io_plan(monkeypatch):
    monkeypatch.setattr(m,'PLAN',dict(m.PLAN,ks=[1],query_count=1,warmup_repetitions=0,timed_repetitions=1))
    monkeypatch.setattr(m,'load_runtime',lambda:(np,mj,[]))
    monkeypatch.setattr(m,'hardware_metadata',lambda pools:dict(test_fixture=True))


def test_exclusive_output_artifacts_and_paired_scalar_rows(tmp_path,monkeypatch):
    small_io_plan(monkeypatch);out=tmp_path/'measurement'
    assert m.run(out,run_id=m.RUN_ID,code_commit='a'*40)==0
    report=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    assert report['status']=='COMPLETE_SYNTHETIC_SOFTWARE_MEASUREMENT'
    assert report['additional_external_payload_bytes']==0
    assert len(report['cases'])==2
    assert (out/'complete.json').is_file() and not (out/'failed.json').exists()
    startup=json.loads((out/'startup.json').read_text(encoding='utf-8'))
    complete=json.loads((out/'complete.json').read_text(encoding='utf-8'))
    assert startup['pid']==os.getpid() and startup['python']==sys.executable
    assert startup['cwd']==os.getcwd() and startup['argv']==sys.argv
    assert Path(startup['source_file']).resolve()==Path(m.__file__).resolve()
    assert complete['code_commit']==startup['code_commit']=='a'*40
    start=datetime.fromisoformat(startup['started_at_utc']);end=datetime.fromisoformat(complete['ended_at_utc'])
    assert start.utcoffset().total_seconds()==end.utcoffset().total_seconds()==0 and end>=start
    for row_id in ('k1-pair','k1-c_only'):
        row=json.loads((out/row_id/'row.json').read_text(encoding='utf-8'))
        assert row['status']=='COMPLETE' and len(row['samples'])==1
        assert row['samples'][0]['bitwise_equal'] is True
    compact=[json.loads(line) for line in (out/'measurements.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(compact)==2 and all(not isinstance(value,(dict,list)) for row in compact for value in row.values())
    assert len((out/'measurements.csv').read_text(encoding='utf-8').splitlines())==3
    assert 'SYNTHETIC_ONLY' in (out/'measurement.log').read_text(encoding='utf-8')
    previous=(out/'summary.json').read_bytes()
    with pytest.raises(FileExistsError):m.run(out,run_id=m.RUN_ID,code_commit='a'*40)
    assert (out/'summary.json').read_bytes()==previous


def test_late_failure_preserves_completed_row_and_failed_marker(tmp_path,monkeypatch):
    small_io_plan(monkeypatch);out=tmp_path/'failed-measurement';actual=m.measure_case
    def fail_second(*args,**kwargs):
        if args[4]=='C_ONLY':raise RuntimeError('synthetic late failure')
        return actual(*args,**kwargs)
    monkeypatch.setattr(m,'measure_case',fail_second)
    assert m.run(out,run_id=m.RUN_ID,code_commit='b'*40)==1
    failed=json.loads((out/'failed.json').read_text(encoding='utf-8'))
    assert failed['completed_rows']==['k1-pair']
    assert not (out/'complete.json').exists()
    assert json.loads((out/'k1-pair/row.json').read_text(encoding='utf-8'))['status']=='COMPLETE'
    assert json.loads((out/'k1-c_only/row.json').read_text(encoding='utf-8'))['status']=='TECHNICAL_FAILURE'


def test_cli_rejects_missing_contract_and_relative_output(tmp_path):
    with pytest.raises(SystemExit):m.main(['--output-root',str(tmp_path)])
    with pytest.raises(ValueError,match='absolute'):m.run('relative',run_id=m.RUN_ID,code_commit='a'*40)
    with pytest.raises(ValueError,match='run-id'):m.run(tmp_path/'x',run_id='other',code_commit='a'*40)
    with pytest.raises(ValueError,match='40-hex'):m.run(tmp_path/'x',run_id=m.RUN_ID,code_commit='unknown')
    assert not (tmp_path/'x').exists()
