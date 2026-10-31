from pathlib import Path
import numpy as np
import pytest
from experiments.pamf_phase2_practical import milkr_r42_suite as suite


def config(tmp_path):
    rx=['1-1','14-7','2-1','20-1','7-14','7-7','8-8']
    episodes=[f'MIX{i:02d}' for i in range(1,9)]+[f'{p}{i:02d}' for p in ('LS','MU','LU') for i in range(1,6)]
    return dict(kernel_method='adaptive_residual_ensemble',arm='R42',K=[1,5,10,20],N=[2,5,10,20],
        model_seeds=[2026092701,2026092702,2026092703,2026092704],target_receivers=rx,
        capsules=[f'capsules/{p}/{e}/{r}' for p in ('P01','P02','P03') for e in episodes for r in rx],
        run_root=str(tmp_path/'run'),data_root=str(tmp_path/'data'))


def test_complete_matrix_and_no_wrong_candidate(tmp_path):
    rt=config(tmp_path);tasks=suite.tasks(rt)
    assert len(tasks)==1932 and len({t['key'] for t in tasks})==1932
    assert sum('/MIX' in t['capsule'] for t in tasks)==672
    assert sum(len(suite.desired_rows(t['points'])) for t in tasks)==167496
    assert len({suite.task_runtime(rt,t)['run_root'] for t in tasks})==1932
    for wrong in (dict(rt,arm='R43'),dict(rt,K=[5]),dict(rt,capsules=rt['capsules'][:-1])):
        with pytest.raises(ValueError):suite.tasks(wrong)


def test_score_cannot_open_truth_without_global_closure(tmp_path):
    rt=config(tmp_path)
    with pytest.raises(FileNotFoundError):suite.score(rt,suite.tasks(rt)[0])
    assert not (tmp_path/'run/tasks').exists()


def test_dispatch_does_not_score_partial_matrix(tmp_path,monkeypatch):
    rt=config(tmp_path);actions=[]
    monkeypatch.setattr(suite,'queue',lambda rt,items,action,per_gpu:actions.append(action))
    monkeypatch.setattr(suite,'prediction_closed',lambda *a:(_ for _ in ()).throw(ValueError('missing')))
    monkeypatch.setattr(suite.subprocess,'run',lambda *a,**k:actions.append('score'))
    with pytest.raises(ValueError,match='missing'):suite.dispatch(rt)
    assert actions==['source','worker']
    assert not (Path(rt['run_root'])/'predictions_complete.json').exists()


def test_common_query_is_actual_intersection_not_assumed_K20(tmp_path):
    rt=config(tmp_path);rt.update(K=[1,5],N=[2]);task=dict(capsule='capsules/P01/MIX01/1-1')
    cap=Path(rt['data_root'])/task['capsule'];(cap/'stage_rows').mkdir(parents=True);(cap/'splits').mkdir()
    suite.write(cap/'manifest.json',dict(physical_pool='P01',episode_id='MIX01',receiver='1-1'))
    for n in (0,2):
        np.save(cap/'stage_rows'/f'N{n:02d}.npy',np.arange(8))
        for k,sup in ((1,[0]),(5,[1,2])):
            suite.write(cap/'splits'/f'{k}-{n}.json',dict(k=k,new_count=n,support_indices=sup))
    truth=Path(rt['data_root'])/'score_only/P01/MIX01';truth.mkdir(parents=True)
    np.savez(truth/'1-1.npz',labels=np.arange(8))
    prediction=tmp_path/'pred';prediction.mkdir();output=tmp_path/'out';output.mkdir()
    q=np.arange(1,8);np.savez(prediction/'p.npz',query_indices=q,predicted_indices=q)
    row=dict(k=1,new_count=2,prediction='p.npz',registered_classes=list(range(8)))
    suite.common_query_score(rt,task,prediction,dict(rows=[row]),output)
    out=suite.read(output/'common_query_metrics.json')
    assert out['common_query_counts']=={'0':5,'2':5}
    assert out['rows'][0]['metrics']['count']==5 and out['rows'][0]['metrics']['accuracy']==1.


def test_original_pamf_arm_maps_to_localridge():
    row=dict(arm='pamf_tri_sig_time_ssl_milkr',stage='A0',k=5,new_count=0,input_path='Q0')
    assert suite.normal_key(row)==('LocalRidge','A0',5,0,'Q0')
