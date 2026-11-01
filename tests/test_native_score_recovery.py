from pathlib import Path
import pytest
from experiments.pamf_phase2_practical import native_score_recovery as recovery,episodic_score
from experiments.pamf_phase2_practical.io import read,write


def case(tmp_path,monkeypatch,scored=False):
    monkeypatch.setattr(recovery,'PROFILES',('protonet_cda',));monkeypatch.setattr(recovery,'EXPECTED_JOBS',1)
    parent=tmp_path/'oldowner/jobs.json';profile='protonet_cda';pred=tmp_path/'prediction-owner/predictions'/profile/'s1/c0000'
    old=tmp_path/'score-owner/scores'/profile/'s1/c0000';cap='capsules/P01/MIX01/rx'
    job=dict(id='legacy-score-native',kind='score',profile=profile,seed=1,prediction=str(pred),output=str(old),
             completion=str(old/'complete.json'),original_id='score-native',runtime_file='/old/release/runtime.json')
    write(parent,[job]);rt=dict(run_root=str(tmp_path/'own'),native_parent_jobs=str(parent),data_root=str(tmp_path/'data'),poll_seconds=0)
    write(Path(rt['data_root'])/cap/'manifest.json',dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id='cap'))
    row=dict(prediction='frozen.npz',split_id='split',stage='B',arm=profile,input_path='P0',k=5,new_count=0,registered_classes=list('abcdef'))
    write(pred/'predictions.json',dict(capsule=cap,capsule_id='cap',model_seed=1,profile=profile,rows=[row]))
    write(pred/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id='cap',rows=1,truth_read=False,query_fit=False))
    (pred/'frozen.npz').write_bytes(b'frozen prediction, never read by controller')
    payload=dict(capsule_id='cap',model_seed=1,truth_last=True,target_feedback=False,rows=[dict(row,model_seed=1,metrics={'accuracy':.8})])
    if scored:
        write(old/'metrics.json',payload);write(old/'complete.json',dict(status='SCORED_COMPLETE',rows=1,truth_last=True,target_feedback=False))
    own=recovery.manifest(rt)[0]
    return rt,own,pred,old,payload


def test_completed_score_reuses_without_calling_truth_scorer(tmp_path,monkeypatch):
    rt,job,pred,old,payload=case(tmp_path,monkeypatch,True);original=(old/'metrics.json').read_bytes()
    monkeypatch.setattr(episodic_score,'run',lambda *a:pytest.fail('Completed score must not be recomputed'))
    monkeypatch.setattr(recovery,'active_scorers',lambda *a:pytest.fail('Completed score needs no process wait'))
    assert recovery.worker(rt,job)=='reused'
    assert (Path(job['output'])/'metrics.json').read_bytes()==original==(old/'metrics.json').read_bytes()
    assert read(Path(job['output'])/'reuse.json')['parent_owner_unchanged']
    assert not (old/'reuse.json').exists()
    overrides=recovery.bind_external_jobs(rt)
    assert overrides[0]['parent_prediction']==str(pred) and overrides[0]['new_complete']==job['completion']


@pytest.mark.parametrize('fault',['unfixed','wrong_seed','wrong_capsule'])
def test_prediction_or_old_metric_binding_error_blocks_scoring(tmp_path,monkeypatch,fault):
    rt,job,pred,old,payload=case(tmp_path,monkeypatch,True)
    if fault=='unfixed':
        done=read(pred/'complete.json');done['truth_read']=True;write(pred/'complete.json',done,exclusive=False)
    else:
        payload['model_seed']=2 if fault=='wrong_seed' else payload['model_seed']
        payload['capsule_id']='other' if fault=='wrong_capsule' else payload['capsule_id'];write(old/'metrics.json',payload,exclusive=False)
    monkeypatch.setattr(episodic_score,'run',lambda *a:pytest.fail('Illegal input must not reach scorer'))
    with pytest.raises(ValueError):recovery.worker(rt,job)
    assert not Path(job['output']).exists()


def test_missing_score_calls_original_run_only_for_fixed_prediction(tmp_path,monkeypatch):
    rt,job,pred,old,payload=case(tmp_path,monkeypatch);calls=[]
    monkeypatch.setattr(recovery,'active_scorers',lambda job:[])
    def run(local,item):
        calls.append(item);write(Path(item['output'])/'metrics.json',payload)
        write(Path(item['output'])/'complete.json',dict(status='SCORED_COMPLETE',rows=1,truth_last=True,target_feedback=False))
    monkeypatch.setattr(episodic_score,'run',run)
    assert recovery.worker(rt,job)=='new'
    assert calls==[dict(prediction=str(pred),output=job['output'])] and not old.exists()
    assert read(Path(job['output'])/'reuse.json')['reused'] is False


def test_live_original_scorer_is_waited_and_its_completed_score_reused(tmp_path,monkeypatch):
    rt,job,pred,old,payload=case(tmp_path,monkeypatch);waits=[]
    process=dict(pid=123,cwd='/original/release',argv=['python','-m','group1_dispatch','--runtime','/old/release/runtime.json','--job','score-native'])
    assert recovery.matches_process(job,process)
    assert not recovery.matches_process(job,dict(process,argv=process['argv'][:-1]+['different-job']))
    monkeypatch.setattr(recovery,'active_scorers',lambda job:[process])
    def finish(_):
        waits.append(True);write(old/'metrics.json',payload);write(old/'complete.json',dict(status='SCORED_COMPLETE',rows=1,truth_last=True,target_feedback=False))
    monkeypatch.setattr(recovery.time,'sleep',finish)
    monkeypatch.setattr(episodic_score,'run',lambda *a:pytest.fail('Live original scorer must not be duplicated'))
    assert recovery.worker(rt,job)=='reused' and waits==[True]
    assert read(Path(rt['run_root'])/'waits'/f'{job["key"]}.json')['processes'][0]['pid']==123


def test_output_collision_and_incomplete_new_score_are_not_overwritten(tmp_path,monkeypatch):
    rt,job,pred,old,payload=case(tmp_path,monkeypatch)
    output=Path(job['output']);output.mkdir(parents=True);(output/'keep').write_text('preserve')
    with pytest.raises(ValueError,match='collision'):recovery.worker(rt,job)
    assert (output/'keep').read_text()=='preserve'
    # An independent incomplete new result is a technical failure, not a retry.
    other=dict(job,output=str(tmp_path/'another-output'))
    monkeypatch.setattr(recovery,'active_scorers',lambda job:[])
    def incomplete(local,item):
        write(Path(item['output'])/'metrics.json',dict(payload,rows=[]))
        write(Path(item['output'])/'complete.json',dict(status='SCORED_COMPLETE',rows=0,truth_last=True,target_feedback=False))
    monkeypatch.setattr(episodic_score,'run',incomplete)
    with pytest.raises(ValueError,match='row count'):recovery.worker(rt,other)
    assert (Path(other['output'])/'metrics.json').exists() and not (Path(other['output'])/'reuse.json').exists()
