import json

from experiments.pamf_phase2_practical import clean_issl_short_plan as plan


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')
    return str(path)


def test_short_plan_pairs_every_budget_with_original_capsules(tmp_path,monkeypatch):
    monkeypatch.setattr(plan,'completion_matches',lambda entry:True)
    monkeypatch.setattr(plan,'validate_plan',lambda config:None)
    jobs=[]
    for i in range(5):
        job=dict(profile='issl_fixed_v1',seed=2026092701,points=[[5,20]],capsule=f'capsule-{i}')
        jobs.append(dict(id=f'cil-issl_fixed_v1-{i:02d}',group='native_cil',
                         job=put(tmp_path/f'old/job{i}.json',job),
                         runtime=put(tmp_path/f'old/rt{i}.json',{}),completion='unused'))
    original=put(tmp_path/'old/scheduler.json',dict(jobs=jobs))
    result=plan.prepare(original,tmp_path/'plan',tmp_path/'run',tmp_path/'release',[6,7])
    assert len(result['jobs'])==20
    values=[json.loads(open(item['job']).read()) for item in result['jobs']]
    assert {tuple(v['issl_budget']) for v in values}==set(plan.BUDGETS)
    assert all(sum(v['capsule']==f'capsule-{i}' for v in values)==4 for i in range(5))
    assert all(v['points']==[[5,20]] and v['seed']==2026092701 for v in values)
