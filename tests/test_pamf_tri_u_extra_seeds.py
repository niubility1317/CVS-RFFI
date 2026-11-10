import pytest
from experiments.pamf_tri_u_extra import plan, source
from experiments.pamf_tri_u_extra import prepare as build


def test_new_seeds_are_exactly_the_disjoint_paired_extension():
    assert [r['model_seed'] for r in plan.candidate_rows()]==[2026092704,2026092705,2026092706]
    assert all(len(r['views'])==7 and all('weak' not in v for v in r['views']) for r in plan.candidate_rows())
    with pytest.raises(ValueError):plan.configuration({}, {}, '/project', 'commit')


def test_rejects_unregistered_source_before_checkpoint_access():
    with pytest.raises(ValueError):
        source.parent(dict(model_seed=2026092704,run_id='other',row_id='bad',output_root='/bad'))


def test_all_predictions_precede_scores_and_training_never_receives_truth(monkeypatch,tmp_path):
    tasks=build.TASKS
    h={'source_contract':'/source-contract','classes':['a','b'],
       'tests':{t:{'views_root':'/views/'+t,'truth':'/truth/'+t} for t in tasks}}
    context={'handles':{'F0':h},'caches':{'F0':{'L':{'path':'/cache-L'},'LU':{'path':'/cache-LU'}}}}
    writes={}
    def fake_read(path):
        p=str(path)
        if p=='/ctx':return context
        if p=='/source-contract':return {'source':'same'}
        seed=next(s for s in plan.SEEDS if str(s) in p)
        return dict(status='SOURCE_FROZEN',epoch=50,updates=11100,target_access=False,U_truth_read=False,
                    config=dict(model_seed=seed,source_contract='/source-contract'))
    def config(ctx,row,project,commit):
        return dict(model_seed=row['model_seed'],output_root=str(tmp_path/'source'/row['row_id']))
    monkeypatch.setattr(build,'read',fake_read)
    monkeypatch.setattr(build,'write',lambda path,value:writes.__setitem__(str(path),value))
    monkeypatch.setattr(build,'configuration',config)
    monkeypatch.setattr(build.tri,'validate_config',lambda c:None)
    rt=dict(run_id=plan.RUN,run_root=str(tmp_path/'run'),project=str(tmp_path),context='/ctx',
            code_commit='commit',frozen_sources={})
    jobs=build.prepare(rt)
    pred=[j for j in jobs if j['kind']=='predict'];score=[j for j in jobs if j['kind']=='score']
    assert len(pred)==len(score)==6
    assert all(set(j['dependencies'])=={p['completion'] for p in pred} for j in score)
    assert all('truth' not in str(writes[j['config']]) for j in pred)
    assert all('source_output' not in writes[j['config']] and 'restored_caches' not in writes[j['config']] for j in score)
