import json

from experiments.pamf_phase2_practical import clean_issl_lr_plan as plan


def test_paired_arms_and_single_idle_gpu(tmp_path,monkeypatch):
    monkeypatch.setattr(plan,'completion_matches',lambda entry:True)
    monkeypatch.setattr(plan,'validate_plan',lambda config:None)
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value),encoding='utf-8')
        return str(path)
    jobs=[]
    for i in range(5):
        job=dict(profile='issl_fixed_v1',seed=2026092701,points=[[5,20]],capsule=f'capsule-{i}')
        jobs.append(dict(id=f'cil-issl-{i:02d}',group='native_cil',job=put(tmp_path/f'old/job{i}.json',job),
                         runtime=put(tmp_path/f'old/rt{i}.json',{}),completion='unused'))
    original=put(tmp_path/'old/scheduler.json',dict(jobs=jobs))
    result=plan.prepare(original,tmp_path/'plan',tmp_path/'run',tmp_path/'release',[5])
    values=[json.loads(open(entry['job']).read()) for entry in result['jobs']]
    assert len(values)==20 and result['allowed_gpus']==[5] and result['max_per_gpu']==2
    assert {(v['issl_budget'][0],v['issl_budget'][1],v['issl_ssl_lr']) for v in values}==set(plan.ARMS)
    assert all(sum(v['capsule']==f'capsule-{i}' for v in values)==4 for i in range(5))
    assert result['auto_score']['workers']==1
