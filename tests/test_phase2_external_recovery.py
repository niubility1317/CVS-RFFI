import json
from pathlib import Path
import pytest
from experiments.pamf_phase2_practical.external_dispatch import make_jobs,is_complete,external_ready


def put(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x),encoding='utf-8')


def fixture(tmp_path):
    root=tmp_path/'old';old=dict(run_root=str(root),runtime_file='/r01/runtime.json',release_root='/r01')
    rows=[dict(id='source-mopc_hr-s1',kind='source',profile='mopc_paper',output=str(root/'source_checks/mopc'),dependencies=[],gpu=True,priority=0),
        dict(id='predict-mopc-s1-P0',kind='predict',profile='mopc_paper',seed=1,capsule='capsules/P01/MIX01/r',input_path='P0',output=str(root/'predictions/mopc/s1/c0000/P0'),dependencies=['source-mopc_hr-s1'],gpu=True,priority=2),
        dict(id='score-mopc-s1',kind='score',profile='mopc_paper',output=str(root/'scores/mopc/s1/c0000'),prediction=str(root/'predictions/mopc/s1/c0000'),dependencies=['predict-mopc-s1-P0'],gpu=False,priority=1),
        dict(id='predict-radio',kind='predict',profile='radionet_triplet',output=str(root/'predictions/radio'),dependencies=[],gpu=True,priority=3),
        dict(id='core__predict-c0000-s1',kind='core',original_kind='predict',profile='PAMF_core',output='/core/predictions',dependencies=[],gpu=True,priority=4)]
    put(root/'jobs.json',rows);put(tmp_path/'old.json',old)
    rt=dict(continuation_runtime=str(tmp_path/'old.json'),run_root=str(tmp_path/'new'),repair_profiles=['mopc_paper','mopc_public_code'],model_seeds=[1],capsules=['capsules/P01/MIX01/r'])
    return rt,old,rows


def test_recovery_moves_only_mopc_predictions_and_scores_to_new_root(tmp_path):
    rt,old,original=fixture(tmp_path);rows=make_jobs(rt);by={j['id']:j for j in rows}
    for key in ('predict-mopc-s1-P0','score-mopc-s1'):
        j=by[key]
        assert Path(j['output']).is_relative_to(rt['run_root']) and 'repair-source-smoke' in j['dependencies']
        assert 'worker_runtime' not in j
    assert Path(by['score-mopc-s1']['prediction']).is_relative_to(rt['run_root'])
    # Reuse completed source qualification even when its representative
    # profile is one of the repaired methods; no source retraining.
    for key in ('source-mopc_hr-s1','predict-radio','core__predict-c0000-s1'):
        j=by[key];before=next(x for x in original if x['id']==key)
        assert j['output']==before['output'] and j['worker_runtime']==old['runtime_file']
        assert j['worker_release']==old['release_root']
    assert by['repair-source-smoke']['kind']=='repair_smoke'
    assert json.loads((Path(old['run_root'])/'jobs.json').read_text())==original


def test_smoke_completion_is_required_before_repaired_work(tmp_path):
    rt,old,original=fixture(tmp_path);smoke=next(j for j in make_jobs(rt) if j['kind']=='repair_smoke')
    assert not is_complete(smoke)
    put(Path(smoke['output'])/'complete.json',dict(status='PASS',target_rows=0,query_rows=0))
    assert is_complete(smoke)
    rt['repair_profiles'].append('radionet_triplet')
    with pytest.raises(Exception,match='limited'):make_jobs(rt)
