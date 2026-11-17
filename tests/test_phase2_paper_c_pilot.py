import json
import pytest

from experiments.pamf_phase2_practical import paper_c_pilot


@pytest.mark.parametrize('profile',['lora_paper_v2','orthogonal_paper_v2'])
def test_pilot_runs_source_prediction_then_independent_score(tmp_path,monkeypatch,profile):
    root=tmp_path/profile
    rt=dict(run_root=str(root),release_root=str(tmp_path),pilot_profile=profile,
            pilot_seed=2026092701,pilot_capsule='capsules/P01/MIX01/1-1')
    calls=[]
    def fake_run(command,cwd,env,check):
        stage=command[-1]
        calls.append((stage,env.get('CUDA_VISIBLE_DEVICES')))
        paths={'source':(root/'source'/f'{profile}-s2026092701'/'group1_complete.json','GROUP1_SOURCE_READY'),
               'predict':(root/'predictions'/'complete.json','PREDICTIONS_COMPLETE'),
               'score':(root/'scores'/'complete.json','SCORED_COMPLETE')}
        p,status=paths[stage]
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({'status':status}),encoding='utf8')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    monkeypatch.setattr(paper_c_pilot.subprocess,'run',fake_run)
    paper_c_pilot.dispatch(rt,tmp_path/'runtime.json')
    assert calls==[('source','1'),('predict','1'),('score','')]
    assert json.loads((root/'complete.json').read_text())['status']=='SCORED_COMPLETE'
    with pytest.raises(FileExistsError):paper_c_pilot.dispatch(rt,tmp_path/'runtime.json')


def test_pilot_never_scores_incomplete_predictions(tmp_path,monkeypatch):
    root=tmp_path/'run'
    rt=dict(run_root=str(root),release_root=str(tmp_path),pilot_profile='lora_paper_v2',
            pilot_seed=2026092701,pilot_capsule='capsules/P01/MIX01/1-1')
    calls=[]
    def fake_run(command,**kwargs):
        stage=command[-1];calls.append(stage)
        if stage=='source':
            p=root/'source'/'lora_paper_v2-s2026092701'/'group1_complete.json'
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps({'status':'GROUP1_SOURCE_READY'}),encoding='utf8')
    monkeypatch.setattr(paper_c_pilot.subprocess,'run',fake_run)
    with pytest.raises(ValueError,match='predict did not produce'):
        paper_c_pilot.dispatch(rt,tmp_path/'runtime.json')
    assert calls==['source','predict']
