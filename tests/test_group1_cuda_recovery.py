"""Local synthetic recovery checks; no GPU, remote launch, or target truth."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.pamf_phase2_practical import group1_cuda_recovery as recovery
from experiments.pamf_phase2_practical.io import write, read


def fixtures(tmp_path):
    contract = tmp_path / 'contract.json'
    write(contract, {'source_rxs': [1], 'classes': list(range(6))})
    parent = dict(run_id='parent', run_root=str(tmp_path / 'parent'),
        method_profiles=['mopc_hr'], model_seeds=[2026092701,2026092702,2026092703,2026092704],
        capsules=[f'capsules/P01/MIX01/{i}' for i in range(483)], data_root='unused_target',
        source_contract=str(contract), source_cache='synthetic_L_only', source_cpu_threads=1)
    parent['runtime_file'] = str(tmp_path / 'parent_runtime.json')
    write(parent['runtime_file'], parent)
    runtime = dict(parent, run_id='recovery', run_root=str(tmp_path / 'recovery'),
        runtime_file=str(tmp_path / 'runtime.json'), recovery_parent_runtime=parent['runtime_file'],
        allowed_gpus=[1,5], release_root=str(tmp_path), python='never_execute', poll_seconds=0)
    old_jobs = recovery.jobs(parent)
    for job in old_jobs:
        if job['kind'] != 'source':
            continue
        output = Path(job['output'])
        write(job['completion'], dict(status='GROUP1_SOURCE_READY', seed=job['seed'], profile='mopc_hr'))
        (output / 'source.pt').write_bytes(b'checkpoint-for-link-test')
        write(output / 'recipe.json', {'identity': 'parent recipe'})
    return runtime, parent, old_jobs


def complete_prediction(job):
    write(Path(job['output']) / 'predictions.json', dict(model_seed=job['seed'], capsule=job['capsule'],
        capsule_id='fixed_capsule', rows=[dict(prediction='row.npz')]))
    (Path(job['output']) / 'row.npz').write_bytes(b'fixed-prediction')
    write(job['completion'], dict(status='PREDICTIONS_COMPLETE', rows=1,
        capsule_id='fixed_capsule', truth_read=False, query_fit=False))


def test_prepare_reuses_only_complete_predictions_and_scores_and_keeps_source_markers_private(tmp_path):
    runtime, parent, old = fixtures(tmp_path)
    prediction = next(j for j in old if j['kind']=='predict')
    complete_prediction(prediction)
    score = next(j for j in old if j['id']=='score-'+prediction['id'])
    write(score['completion'], dict(status='SCORED_COMPLETE', rows=1, truth_last=True, target_feedback=False))
    write(Path(score['output']) / 'metrics.json', {'not_read_by_recovery': 'metrics'})
    partial = next(j for j in old if j['kind']=='predict' and j['id']!=prediction['id'])
    write(Path(partial['output']) / 'partial.json', {'status': 'FAILED'})
    current, reused = recovery.prepare(runtime, parent)
    assert len(current)==3870 and set(reused)=={prediction['id'],score['id']}
    by = {j['id']:j for j in current}
    assert not Path(by[partial['id']]['output']).exists()
    sources = [j for j in current if j['kind']=='source']
    assert all(set(j['id'] for j in sources) <= set(j['dependencies']) for j in current if j['kind']!='source')
    for job in sources:
        assert not Path(job['completion']).exists()
        old_marker = Path(job['recovery_parent_output']) / 'group1_complete.json'
        before = old_marker.read_bytes()
        write(job['completion'], {'status':'new-private-marker'})
        assert old_marker.read_bytes()==before
        assert Path(job['output'],'source.pt').stat().st_ino == Path(job['recovery_parent_output'],'source.pt').stat().st_ino


def test_complete_prediction_without_complete_score_is_not_recomputed(tmp_path):
    runtime, parent, old = fixtures(tmp_path)
    prediction = next(j for j in old if j['kind']=='predict')
    complete_prediction(prediction)
    _, reused = recovery.prepare(runtime, parent)
    assert prediction['id'] in reused and 'score-'+prediction['id'] not in reused


def test_worker_routes_pending_prediction_to_existing_worker_and_never_source_trainer(monkeypatch):
    from experiments.pamf_phase2_practical import group1_dispatch, group1_source
    observed = []
    monkeypatch.setattr(group1_dispatch, 'worker', lambda rt,j: observed.append(j))
    monkeypatch.setattr(group1_source, 'train', lambda *a,**k: pytest.fail('training is forbidden'))
    job = {'kind':'predict','id':'pending','profile':'mopc_hr'}
    recovery.worker({}, job)
    assert observed==[job]
    monkeypatch.setattr(recovery, 'reuse_source', lambda rt,j: observed.append('source smoke only'))
    recovery.worker({}, {'kind':'source'})
    assert observed[-1]=='source smoke only'


@pytest.mark.parametrize('bad', ['seed','contract'])
def test_real_load_source_rejects_wrong_seed_or_contract_before_smoke(tmp_path, monkeypatch, bad):
    from experiments.pamf_phase2_practical import group1_source
    from experiments.pamf_phase2_practical import io
    runtime, parent, old = fixtures(tmp_path)
    job = next(j for j in recovery.jobs(runtime) if j['kind']=='source')
    job['recovery_parent_output'] = next(j['output'] for j in old if j['id']==job['id'])
    payload = dict(schema='group1_A_v1',recipe=group1_source.recipe('mopc_hr'),seed=job['seed'],
        source_contract=read(runtime['source_contract']),target_contact=False,ancestors=[])
    if bad=='seed':payload['seed']+=1
    else:payload['source_contract']={'wrong':'contract'}
    Path(job['output']).mkdir(parents=True)
    torch.save(payload, Path(job['output'])/'source.pt')
    monkeypatch.setattr(io, 'activate', lambda c: None)
    monkeypatch.setattr(group1_source, 'source_data', lambda *args: pytest.fail('invalid source must not reach smoke'))
    with pytest.raises(ValueError,match='CHECKPOINT_DATA_CONTRACT_MISMATCH_OR_PROVENANCE'):
        recovery.reuse_source(runtime,job,device='cpu')
    assert not Path(job['completion']).exists()


def test_source_smoke_has_zero_training_and_does_not_modify_hardlinked_json(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import group1_source, io
    runtime, parent, _ = fixtures(tmp_path)
    current, _ = recovery.prepare(runtime,parent)
    job = next(j for j in current if j['kind']=='source')
    original = Path(job['recovery_parent_output'])/'recipe.json'
    before = original.read_bytes()
    seen = []
    monkeypatch.setattr(io,'activate',lambda c:None)
    monkeypatch.setattr(group1_source,'train',lambda *a,**k:pytest.fail('no source training'))
    monkeypatch.setattr(group1_source,'load_source',lambda c,p,s,d:dict(model=object(),score_fn=lambda m,x:torch.ones(len(x),6)))
    monkeypatch.setattr(group1_source,'source_data',lambda c,role:(seen.append(role) or SimpleNamespace(x=torch.zeros(3,2,256))))
    recovery.reuse_source(runtime,job,'cpu')
    assert seen==['L_s'] and original.read_bytes()==before
    done=read(job['completion'])
    assert done['status']=='GROUP1_SOURCE_READY' and done['source_new_optimizer_updates']==0
    assert done['reused_from']==job['recovery_parent_output']
    assert read(Path(job['output'])/'recovery_source_smoke.json')['query_rows']==0


def test_resource_limit_and_failure_stop_without_retry(tmp_path):
    assert recovery.eligible_gpu({'free':6000},0)
    assert not recovery.eligible_gpu({'free':5999},0)
    assert not recovery.eligible_gpu({'free':24000},1)
    job=dict(id='failed',completion=str(tmp_path/'absent.json'),status='PREDICTIONS_COMPLETE')
    active={'failed':(SimpleNamespace(poll=lambda:1),dict(pid=123,gpu=1,kind='predict'))}
    (tmp_path/'job_results').mkdir()
    done,failed=set(),set()
    recovery.observe_children(active,{'failed':job},done,failed,tmp_path)
    assert not active and failed=={'failed'} and not done
    recovery.observe_children(active,{'failed':job},done,failed,tmp_path)
    assert failed=={'failed'} and read(tmp_path/'job_results/failed.json')['exit_code']==1


def test_dispatch_failure_drains_owned_workers_and_stops_admission(tmp_path, monkeypatch):
    runtime, parent, _ = fixtures(tmp_path)
    write(runtime['runtime_file'], runtime)
    monkeypatch.setattr(recovery,'prepare',lambda rt,old:([
        dict(id='source',kind='source',profile='mopc_hr',seed=2026092701,dependencies=[],priority=0,gpu=True),
        dict(id='pending',kind='predict',profile='mopc_hr',seed=2026092701,dependencies=['source'],priority=1,gpu=True)],{}))
    root=Path(runtime['run_root']);(root/'logs').mkdir(parents=True);(root/'job_results').mkdir()
    from experiments.pamf_phase2_practical import runner
    monkeypatch.setattr(runner,'occupancy',lambda:{1:dict(free=10000),5:dict(free=10000)})
    launches=[]
    def launch(*args,**kwargs):
        launches.append(args[0])
        return SimpleNamespace(pid=123,poll=lambda:1)
    monkeypatch.setattr(recovery.subprocess,'Popen',launch)
    state=recovery.dispatch(runtime)
    assert len(launches)==1 and state['status']=='PARTIAL_TECHNICAL_FAILURE'
    assert state['failed_jobs']==['source'] and state['counts']['QUEUED']==1


def test_dispatch_complete_reuse_skips_worker_and_gpu_probe(tmp_path, monkeypatch):
    runtime, parent, _ = fixtures(tmp_path)
    root=Path(runtime['run_root']);(root/'logs').mkdir(parents=True);(root/'job_results').mkdir()
    current=[dict(id='predict',kind='predict'),dict(id='score',kind='score')]
    reused={j['id']:dict(kind=j['kind'],rows=22,reused_from='parent') for j in current}
    monkeypatch.setattr(recovery,'prepare',lambda rt,old:(current,reused))
    monkeypatch.setattr(recovery.subprocess,'Popen',lambda *a,**k:pytest.fail('reused jobs must not launch'))
    from experiments.pamf_phase2_practical import runner
    monkeypatch.setattr(runner,'occupancy',lambda:pytest.fail('completed reuse needs no GPU worker'))
    state=recovery.dispatch(runtime)
    assert state['status']=='COMPLETE' and state['counts']['COMPLETE']==2
    assert state['completed_prediction_rows']==22 and state['completed_score_rows']==22


def test_bad_completion_readback_is_failure_without_losing_other_active_workers(tmp_path):
    (tmp_path/'job_results').mkdir()
    marker=tmp_path/'complete.json';marker.write_text('{partial',encoding='utf-8')
    job=dict(kind='predict',completion=str(marker),status='PREDICTIONS_COMPLETE')
    active={'bad':(SimpleNamespace(poll=lambda:0),dict(pid=123)),
            'healthy':(SimpleNamespace(poll=lambda:None),dict(pid=124))}
    done,failed=set(),set()
    recovery.observe_children(active,{'bad':job},done,failed,tmp_path)
    assert set(active)=={'healthy'} and failed=={'bad'} and not done
    assert read(tmp_path/'job_results/bad.json')['failed_layer']=='completion_marker_readback'
