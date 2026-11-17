import json
import pytest


def test_completed_task_is_scored_only_after_prediction_preflight(tmp_path, monkeypatch):
    from experiments.pamf_phase2_practical import clean_incremental_score as incremental
    order=[]
    marker=tmp_path/'prediction/complete.json';marker.parent.mkdir();marker.write_text('{}')
    entry=dict(id='a',group='native_da',job='job',runtime='rt',completion=str(marker))
    item=dict(entry=entry,job={},rt={},folder=marker.parent,payload=dict(rows=[dict(prediction='x.npz')]))
    monkeypatch.setattr(incremental,'verify_all',lambda *args:order.append('verify') or [item])
    monkeypatch.setattr(incremental,'validate_prediction_rows',lambda *_:order.append('preflight'))
    def score(value,output):
        assert order==['verify','preflight']
        order.append('score');output.mkdir(parents=True);(output/'complete.json').write_text('{}')
    monkeypatch.setattr(incremental,'score_job',score)
    output=tmp_path/'scores'
    result=incremental.score_entry((entry,str(output)))
    assert result['status']=='SCORED_COMPLETE' and order==['verify','preflight','score']


def test_partial_or_existing_outputs_never_get_overwritten(tmp_path,monkeypatch):
    from experiments.pamf_phase2_practical import clean_incremental_score as incremental
    entry=dict(id='a',group='native_da',completion=str(tmp_path/'missing'))
    output=tmp_path/'scores';output.mkdir();(output/'prior').write_text('keep')
    monkeypatch.setattr(incremental,'verify_all',lambda *args:pytest.fail('unexpected verification'))
    result=incremental.score_entry((entry,str(output)))
    assert result['status']=='PRESERVED_INCOMPLETE_OUTPUT'
    assert (output/'prior').read_text()=='keep'


def test_identity_keeps_original_and_stabilized_methods_separate(tmp_path,monkeypatch):
    from experiments.pamf_phase2_practical import clean_incremental_score as incremental
    plan=tmp_path/'plans';a=plan/'old';b=plan/'stable';a.mkdir(parents=True);b.mkdir()
    entries=[dict(id='same',job='oldjob',runtime='rt',completion='done',group='hybrid')]
    for folder in (a,b):
        (folder/'scheduler.json').write_text(json.dumps(dict(jobs=entries)))
    result=incremental.unique_entries([(a/'scheduler.json',entries[0]),(b/'scheduler.json',entries[0])])
    assert len(result)==2 and result[0]['score_key']!=result[1]['score_key']


def test_real_fixed_task_produces_per_k_metrics_without_waiting_for_matrix(tmp_path):
    from test_phase2_clean_score import make_case
    from experiments.pamf_phase2_practical.clean_incremental_score import score_entry
    config,_=make_case(tmp_path)
    result=score_entry((config['jobs'][0],str(tmp_path/'single-score')))
    assert result['status']=='SCORED_COMPLETE' and result['rows']==8
    payload=json.loads((tmp_path/'single-score/metrics.json').read_text())
    assert payload['truth_last'] is True and payload['target_feedback'] is False
    assert {r['k'] for r in payload['rows']}=={1,5,10,20}
    assert all(r['common_K_status']=='AVAILABLE' and r['metrics']['accuracy']==1 for r in payload['rows'])


def test_discovery_keeps_legacy_proto_and_isolates_bad_jobs(tmp_path):
    from experiments.pamf_phase2_practical import clean_incremental_score as incremental
    from experiments.pamf_phase2_practical.clean_auto_score import expected_units
    task=dict(capsule='capsules/P01/MIX01/1-1',seed=2026092701,points=[[5,20]],
              profile='protonet_cda',methods=['mopc_hr'])
    plan=tmp_path/'20261114-phase2-clean-legacy';(plan/'jobs').mkdir(parents=True)
    entries=[]
    for name,job in [('good',task),('bad',dict(task,points=[[3,20]]))]:
        file=plan/'jobs'/f'{name}.json';file.write_text(json.dumps(job))
        marker=tmp_path/f'{name}-complete.json';marker.write_text('{"status":"PREDICTIONS_COMPLETE"}')
        entries.append(dict(id=name,group='hybrid',job=str(file),completion=str(marker),runtime='rt'))
    (plan/'scheduler.json').write_text(json.dumps(dict(input_condition='clean_ground_pre_satellite_normalized',jobs=entries)))
    expected={'none':expected_units([task],'none')}
    found,errors=incremental.discover(tmp_path,expected)
    assert [x['entry']['id'] for x in found]==['good']
    assert len(errors)==1 and errors[0]['id']=='bad'
    (tmp_path/'good-complete.json').write_text('{')
    found,errors=incremental.discover(tmp_path,expected)
    assert not found and any(e['id']=='good' for e in errors)
    (tmp_path/'good-complete.json').write_text('{"status":"PREDICTIONS_COMPLETE"}')
    found,_=incremental.discover(tmp_path,expected)
    assert [x['entry']['id'] for x in found]==['good']


def test_worker_pool_failure_retries_safely_with_one_worker(tmp_path,monkeypatch):
    from concurrent.futures.process import BrokenProcessPool
    from experiments.pamf_phase2_practical import clean_incremental_score as incremental
    job=tmp_path/'job.json';job.write_text(json.dumps(dict(profile='dadda_sda')))
    item=dict(score_key='plan/task',entry=dict(id='task',group='native_da',job=str(job)))
    monkeypatch.setattr(incremental,'tasks',lambda:[])
    monkeypatch.setattr(incremental,'expected_units',lambda *a:{})
    monkeypatch.setattr(incremental,'discover',lambda *a:([item],[]))
    monkeypatch.setattr(incremental,'wait',lambda active,**kw:(set(active),set()))
    sizes=[];root=tmp_path/'output'
    class Future:
        def __init__(self,broken):self.broken=broken
        def result(self):
            if self.broken:raise BrokenProcessPool('worker died')
            (root/'USER_STOP.json').write_text('{}')
            return dict(status='SCORED_COMPLETE',rows=1)
    class Pool:
        def __init__(self,max_workers):self.size=max_workers;sizes.append(max_workers)
        def submit(self,*a):return Future(self.size>1)
        def shutdown(self,**kw):pass
    monkeypatch.setattr(incremental,'ProcessPoolExecutor',Pool)
    incremental.run(tmp_path,root,4,60)
    state=json.loads((root/'state.json').read_text())
    assert sizes==[4,1] and state['status']=='STOPPED' and state['pool_restarts']==1
    assert state['counts']['SCORED_COMPLETE']==1 and not list((root/'failures').iterdir())


def test_new_completions_and_existing_backlog_are_both_served():
    from collections import deque
    from experiments.pamf_phase2_practical.clean_incremental_score import next_pending
    backlog=deque(['old1','old2']);recent=deque(['new1','new2']);prefer=True
    selected=[]
    for _ in range(4):
        value,prefer=next_pending(backlog,recent,prefer);selected.append(value)
    assert selected==['new1','old1','new2','old2']
