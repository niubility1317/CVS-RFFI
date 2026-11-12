import json
import numpy as np
import pytest

from experiments.pamf_phase2_practical import clean_score


def put(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj));return path


def make_case(root):
    data=root/'data';cap=data/'capsules/P01/MIX01/1-1';(cap/'splits').mkdir(parents=True);(cap/'stage_rows').mkdir()
    put(cap/'manifest.json',dict(capsule_id='cap',physical_pool='P01',episode_id='MIX01',receiver='1-1'))
    labels=np.repeat(np.arange(8),23);truth=data/'score_only/P01/MIX01';truth.mkdir(parents=True)
    np.savez(truth/'1-1.npz',labels=labels,days=np.zeros(len(labels),np.int8),scene_indices=np.ones(len(labels),np.int8))
    run=root/'run';pred=run/'pred';pred.mkdir(parents=True);rows=[]
    for n in (0,2):
        count=6+n;registered=np.arange(count*23,dtype=np.int32);np.save(cap/'stage_rows'/f'N{n:02d}.npy',registered)
        for k in (1,5,10,20):
            support=np.concatenate([np.arange(c*23,c*23+k) for c in range(count)])
            q=registered[~np.isin(registered,support)];sid=f's{k}n{n}'
            spec=dict(split_id=sid,k=k,new_count=n,registered_classes=list(range(count)),
                      support_indices=support.tolist(),query_count=len(q))
            put(cap/'splits'/(sid+'.json'),spec)
            name=f'{sid}.npz';np.savez(pred/name,query_indices=q,predicted_indices=labels[q],
                                     old_restricted_indices=np.minimum(labels[q],5))
            rows.append(dict(split_id=sid,k=k,new_count=n,registered_classes=list(range(count)),
                arm='lora_triplet_knn',stage='A0' if n==0 else 'C',input_path='P0',prediction=name,
                input_condition='clean_ground_pre_satellite_normalized'))
    put(pred/'predictions.json',dict(capsule='capsules/P01/MIX01/1-1',capsule_id='cap',model_seed=2026092701,rows=rows))
    put(pred/'complete.json',dict(status='PREDICTIONS_COMPLETE',capsule_id='cap',model_seed=2026092701,
                                 rows=len(rows),truth_read=False,query_fit=False))
    job=put(root/'job.json',dict(capsule='capsules/P01/MIX01/1-1',seed=2026092701,profile='lora_triplet_knn',
                                 points=[[k,2] for k in (1,5,10,20)]))
    rt=put(root/'rt.json',dict(data_root=str(data)))
    entry=dict(id='one',group='native_cil',runtime=str(rt),job=str(job),completion=str(pred/'complete.json'))
    return dict(run_root=str(run),jobs=[entry]),dict(jobs=1,blocked=[])


def test_clean_score_reports_natural_and_common_k_without_satellite_scene(tmp_path):
    config,prep=make_case(tmp_path);output=tmp_path/'run/scores'
    clean_score.run(config,prep,output)
    result=json.loads((output/'one/metrics.json').read_text())
    assert result['satellite_scene_labels_used'] is False
    assert len(result['rows'])==8
    for row in result['rows']:
        assert row['metrics']['accuracy']==1.
        assert row['common_K_status']=='AVAILABLE'
        assert row['common_K_metrics']['count']==3*(6+row['new_count'])
        assert 'by_scene' not in row


def test_blocked_or_missing_predictions_never_open_truth(tmp_path,monkeypatch):
    config,prep=make_case(tmp_path)
    monkeypatch.setattr(clean_score.np,'load',lambda *args,**kwargs:pytest.fail('Truth/predictions opened before barrier'))
    with pytest.raises(ValueError,match='blocked'):
        clean_score.run(config,dict(prep,blocked=[{'missing':'MoPC'}]),tmp_path/'run/scores')
    (tmp_path/'run/pred/complete.json').unlink()
    with pytest.raises(FileNotFoundError):
        clean_score.run(config,prep,tmp_path/'run/scores')
    assert not (tmp_path/'run/scores').exists()


def test_parallel_scoring_after_all_predictions_are_fixed(tmp_path):
    config,prep=make_case(tmp_path)
    config['jobs'].append(dict(config['jobs'][0],id='two'))
    prep['jobs']=2
    output=tmp_path/'run/scores'
    clean_score.run(config,prep,output,workers=2)
    assert json.loads((output/'complete.json').read_text())['jobs']==2
    assert (output/'one/metrics.json').exists() and (output/'two/metrics.json').exists()
    with pytest.raises(ValueError,match='between 1 and 4'):
        clean_score.run(config,prep,tmp_path/'run/other',workers=5)
