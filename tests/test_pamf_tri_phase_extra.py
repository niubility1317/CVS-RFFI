import pytest
from experiments.pamf_tri_phase_extra import plan,source
from experiments.pamf_tri_phase_extra import prepare as build


def test_extra_seeds_and_branch_recipe_are_fixed():
    assert [r['model_seed'] for r in plan.candidate_rows()]==[2026092704,2026092705,2026092706]
    assert plan.SUBSET==('time','path','reference')
    with pytest.raises(ValueError):plan.configuration({}, {}, '/project','commit')
    with pytest.raises(ValueError):source.parent(dict(model_seed=2026092704,run_id='other',row_id='bad',output_root='/bad'))


def test_matched_baselines_and_truth_last_barrier(monkeypatch,tmp_path):
    tasks=build.TASKS;project=tmp_path/'project'
    h={'source_contract':'/contract','classes':['a','b'],
       'tests':{t:{'views_root':'/views/'+t,'truth':'/truth/'+t} for t in tasks}}
    caches={'L':{'path':'/cache-L'},'LU':{'path':'/cache-LU'}}
    context={'handles':{'F0':h},'caches':{'F0':caches}}
    old={f'tri-baseline-s{s}':dict(source_output=str(project/'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/F'/
        f'main_split-repl-shuffled_time-3view-s{s}'/'source'),predictions=f'/pred/{s}',scores=f'/score/{s}') for s in plan.SEEDS}
    writes={}
    def fake_read(path):
        p=str(path)
        if p=='/ctx':return context
        if p.endswith('sources.json'):return old
        seed=next(s for s in plan.SEEDS if str(s) in p)
        if 'predict__' in p:
            return dict(model_seed=seed,restored_caches={k:v['path'] for k,v in caches.items()},classes=h['classes'],
                        tests={t:{'views_root':h['tests'][t]['views_root']} for t in tasks})
        if '/pred/' in p.replace('\\','/'):
            return dict(status='ALL_PREDICTIONS_COMPLETE',truth_read=False,query_fit=False,run_id=build.BASELINE_RUN,
                        row_id=f'tri-baseline-s{seed}',views=plan.candidate_rows()[0]['views'],readouts=list(build.READOUTS))
        return dict(status='SCORED_COMPLETE')
    monkeypatch.setattr(build,'read',fake_read)
    monkeypatch.setattr(build,'write',lambda path,value:writes.__setitem__(str(path),value))
    monkeypatch.setattr(build,'configuration',lambda ctx,row,project,commit:dict(output_root=str(tmp_path/'source'/row['row_id'])))
    rt=dict(run_id=plan.RUN,run_root=str(tmp_path/'run'),project=str(project),context='/ctx',code_commit='commit',frozen_sources={})
    jobs=build.prepare(rt);pred=[j for j in jobs if j['kind']=='predict'];scores=[j for j in jobs if j['kind']=='score']
    assert len(pred)==len(scores)==3
    assert all(set(j['dependencies'])=={p['completion'] for p in pred} for j in scores)
    assert all('truth' not in str(writes[j['config']]) for j in pred)
    assert all('source_output' not in writes[j['config']] for j in scores)
