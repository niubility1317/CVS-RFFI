"""Synthetic complete support matrices, optimizer logs and input boundaries."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code')]
import evaluate_d92_branch_orbit_ce_probe as entry
import export_d92_branch_orbit_support_features as cache
import export_d92_branch_support_features as single
from test_evaluate_d92_branch_support_probe import fixture as old_fixture, mutate
from test_export_d92_branch_orbit_support_features import fixture as export_fixture, base_fixture, binding


def fixture(root,ks=(1,2,5)):
    args=old_fixture(root,ks=ks); folder=args['support_features']
    with np.load(folder/single.CACHE_NAME,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    n=len(data['ids']); rng=np.random.default_rng(75)
    for key in cache.BRANCHES:
        original=data[key]; data[key]=rng.normal(size=(n,4,160)).astype(np.float32);data[key][:,0]=original
    data['feature_contract_json']=np.asarray(json.dumps(cache.FEATURE_CONTRACT))
    np.savez(folder/cache.CACHE_NAME,**data)
    for name in ('features_complete.json','startup.json','support_splits.json'):
        path=folder/name;record=cache.read(path);record['schema']=cache.CACHE_SCHEMA
        if name!='support_splits.json':record.update(feature_contract=cache.FEATURE_CONTRACT,native_batch_size=1)
        if name=='features_complete.json':
            arrays={k:data[k] for k in (*cache.BRANCHES,'fft')}
            record.update(status='BRANCH_ORBIT_SUPPORT_FEATURES_COMPLETE',view_count_per_observation=4,
                feature_array_bytes=sum(v.nbytes for v in arrays.values()),
                feature_file_bytes=(folder/cache.CACHE_NAME).stat().st_size,shapes={k:list(v.shape) for k,v in arrays.items()},
                native_physical_forward_count=n*4,native_view_forward_count=n*4,native_batch_calls=n*4,
                identity_reference_checks=n*4,support_physical_observation_count=n,
                smoke_forward_count=4,smoke_batch_calls=4,native_total_physical_forward_count=n*4+4,
                native_total_batch_calls=n*4+4,synthetic_smoke=dict(status='PASS',query_rows_read=0,frozen_state_unchanged=True))
        path.write_text(json.dumps(record),encoding='utf-8')
    args['config']['algorithm']=entry.FROZEN_CONFIG
    return args


def lines(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]


def test_real_core_k1_k2_k5_full_logs_and_current_support_only(tmp_path,monkeypatch):
    args=fixture(tmp_path);real=entry.probe_branch_orbit_ce;calls=[];original=Path.open
    def guard(path,*a,**kw):
        mode=a[0] if a else kw.get('mode','r')
        if 'r' in mode:assert path.parent==args['support_features'] or path==args['capsule']/'manifest.json'
        return original(path,*a,**kw)
    def probe(**kw):
        assert set(kw)==set(cache.BRANCHES)|{'fft','support_ids','support_labels','classes','old_classes'}
        calls.append(len(kw['support_ids']))
        assert kw['z_id'].shape==(calls[-1],4,160)
        return real(**kw)
    monkeypatch.setattr(entry,'probe_branch_orbit_ce',probe)
    captured=io.StringIO()
    with monkeypatch.context() as m,contextlib.redirect_stdout(captured):
        m.setattr(Path,'open',guard); marker=entry.evaluate(**args)
    assert calls==[2,4,10]
    assert (marker['episodes'],marker['k1_episodes'],marker['oof_episodes'],marker['proxy_anchor_count'])==(3,1,2,7)
    assert marker['factorization_count']==24 and marker['standard_factorization_count']==10
    assert marker['optimizer_steps']>0
    out=args['output'];trace=lines(out/'fit_trace.jsonl');compact=lines(out/'compact.jsonl')
    stage=lines(out/'fit_stages.jsonl');steps=lines(out/'training_steps.jsonl')
    assert len(stage)==48
    assert sum(s['optimizer_steps'] for s in stage)==marker['optimizer_steps']
    assert len(steps)==marker['optimizer_steps']+24  # iteration zero per CE fit
    assert trace[0]['oof'] is trace[0]['oneshot_proxy'] is None
    assert trace[0]['folds']==[] and trace[0]['optimizer_steps']==0
    for row in trace[1:]:
        assert set(row['oof'])==set(entry.ARMS)
        assert len(row['oneshot_proxy']['trials'])==row['k']
        for trial in row['oneshot_proxy']['trials']:
            assert not set(trial['training_ids'])&set(trial['held_ids'])
            assert trial['train_k']==1
            assert all('rows' not in v and v['record_count']==2*(row['k']-1) for v in trial['oof'].values())
    assert 'trials' not in compact[1]['oneshot_proxy']
    assert 'rows' not in compact[1]['oof']['orbit_ce']
    assert all('steps' not in s for s in stage)
    for row in steps:
        assert row['arm'].endswith('_ce') and row['learning_rate']>0 and row['gradient_norm']>=0
        assert np.isclose(row['loss_data']+row['loss_ridge'],row['loss_total'])
    with (out/'training_steps.csv').open(encoding='utf-8',newline='') as f:assert len(list(csv.DictReader(f)))==len(steps)
    events=[json.loads(s) for s in captured.getvalue().splitlines()]
    startup=next(e for e in events if e['event']=='STARTUP')
    assert startup['text_step_interval']==25 and 'structured records retain every iteration' in startup['stdout_step_policy']
    key=lambda r:(r['split_id'],r['scope'],r['fold'],r['trial'],r['arm'])
    final={key(s):s['optimizer_steps'] for s in stage}
    expected=[s for s in steps if s['iteration']==0 or s['iteration']%25==0 or s['iteration']==final[key(s)]]
    shown=[{k:v for k,v in e.items() if k!='event'} for e in events if e['event']=='SUPPORT_OPTIMIZER_STEP']
    assert shown==expected
    assert len(shown)<len(steps)
    assert len([e for e in events if e['event']=='SUPPORT_FIT_STAGE'])==len(stage)
    for group in {key(s) for s in steps}:
        selected=[s['iteration'] for s in shown if key(s)==group]
        assert selected[0]==0 and selected[-1]==final[group]
        assert len(selected)==len(set(selected))
    assert not (out/'predictions.jsonl').exists() and not (out/'scores.json').exists()
    with pytest.raises(FileExistsError):entry.evaluate(**args)


def test_actual_export_to_probe_contract(export_fixture):
    args,*_=export_fixture
    with contextlib.redirect_stdout(io.StringIO()):cache.export(**args)
    kw=binding(args);kw['config']['algorithm']=entry.FROZEN_CONFIG;kw['output']=args['output'].parent/'probe'
    with contextlib.redirect_stdout(io.StringIO()):marker=entry.evaluate(**kw)
    assert marker['episodes']==2 and marker['proxy_anchor_count']==2
    assert marker['payload_audit']['native_view_forward_count']==24
    assert marker['payload_audit']['model_incremental_transfer_bytes'] is None


@pytest.mark.parametrize('fault',['algorithm','partial','query','source','step_count','held_overlap'])
def test_binding_or_incomplete_audit_cannot_complete(tmp_path,monkeypatch,fault):
    args=fixture(tmp_path,ks=(2,));real=entry.probe_branch_orbit_ce
    if fault=='algorithm':args['config']['algorithm']={}
    elif fault=='partial':mutate(args['support_features']/'support_splits.json',lambda d:d['splits'].pop())
    elif fault=='query':mutate(args['support_features']/'support_splits.json',lambda d:d['splits'][0].update(query_indices=[1]))
    elif fault=='source':mutate(args['support_features']/'startup.json',lambda d:d.update(source_data_access=True))
    else:
        def broken(**kw):
            audit=real(**kw)
            if fault=='step_count':audit['optimizer_steps']+=1
            else:audit['folds'][0]['held_ids'][0]=audit['folds'][0]['training_ids'][0]
            return audit
        monkeypatch.setattr(entry,'probe_branch_orbit_ce',broken)
    with contextlib.redirect_stdout(io.StringIO()),pytest.raises(ValueError):entry.evaluate(**args)
    assert not (args['output']/'probe_complete.json').exists()


def test_convergence_failure_preserves_optimizer_evidence(tmp_path,monkeypatch):
    from cvsrffi.d92_branch_orbit_ce import OrbitCEConvergenceError
    args=fixture(tmp_path,ks=(2,));evidence=dict(optimizer_steps=1,converged=False,steps=[dict(iteration=0),dict(iteration=1)])
    def fail(**kw):raise OrbitCEConvergenceError('synthetic fixed cap',evidence)
    monkeypatch.setattr(entry,'probe_branch_orbit_ce',fail)
    with contextlib.redirect_stdout(io.StringIO()),pytest.raises(OrbitCEConvergenceError):entry.evaluate(**args)
    failure=entry.read(args['output']/'technical_failure.json')
    assert failure['core_audit']==evidence and failure['completed_episodes']==0
    assert not (args['output']/'probe_complete.json').exists()


def test_true_k1_never_fits_or_fabricates_proxy(tmp_path,monkeypatch):
    import cvsrffi.d92_branch_orbit_ce as core
    args=fixture(tmp_path,ks=(1,))
    monkeypatch.setattr(core,'_fit',lambda *a,**kw:pytest.fail('No fit for genuine K1 diagnostic'))
    with contextlib.redirect_stdout(io.StringIO()):marker=entry.evaluate(**args)
    assert marker['optimizer_steps']==marker['factorization_count']==marker['proxy_anchor_count']==0
