import json
import pytest

from experiments.pamf_phase2_practical.satellite_native_da import make_plan, validate_target, verify_prediction


def test_satellite_da_requires_complete_missing_a0_and_b_rows():
    job = dict(profile='poster_ft', capsule='capsules/P01/MIX02/1-1', seed=2026092701,
               satellite_ks=[1, 10, 20])
    missing = {(job['capsule'], job['seed'], stage, k, 0)
               for stage in ('A0', 'B') for k in job['satellite_ks']}
    assert validate_target(job, missing) == [1, 10, 20]
    missing.remove((job['capsule'], job['seed'], 'B', 10, 0))
    with pytest.raises(ValueError):
        validate_target(job, missing)


@pytest.mark.parametrize('change', [
    {'satellite_ks': [5]}, {'satellite_ks': [1, 1]},
    {'capsule': 'capsules/P01/MIX01/1-1'}, {'capsule': 'capsules/P01/LU01/1-1'},
    {'profile': 'orthogonal'}, {'seed': 99},
])
def test_satellite_da_rejects_previously_covered_or_wrong_scope(change):
    job = dict(profile='poster_ft', capsule='capsules/P01/MIX02/1-1', seed=2026092701,
               satellite_ks=[1, 10, 20])
    job.update(change)
    with pytest.raises(ValueError):
        validate_target(job, set())


def test_score_preflight_rejects_wrong_input_or_missing_k(tmp_path):
    job=dict(profile='poster_ft',capsule='capsules/P01/MIX02/1-1',seed=2026092701,
             satellite_ks=[1],output=str(tmp_path))
    rows=[dict(stage=stage,k=1,new_count=0,input_path='P0',arm='poster_ft',
               input_condition='satellite_received',prediction=stage+'.npz') for stage in ('A0','B')]
    for row in rows:(tmp_path/row['prediction']).touch()
    (tmp_path/'complete.json').write_text(json.dumps(dict(status='PREDICTIONS_COMPLETE',truth_read=False,
        query_fit=False,profile='poster_ft',model_seed=job['seed'],capsule_id='id',rows=2)))
    index=dict(capsule=job['capsule'],capsule_id='id',model_seed=job['seed'],rows=rows)
    (tmp_path/'predictions.json').write_text(json.dumps(index))
    verify_prediction(job)
    rows[0]['input_condition']='clean_ground_pre_satellite_normalized'
    (tmp_path/'predictions.json').write_text(json.dumps(index))
    with pytest.raises(ValueError):verify_prediction(job)
    rows[0]['input_condition']='satellite_received';rows[1]['k']=10
    (tmp_path/'predictions.json').write_text(json.dumps(index))
    with pytest.raises(ValueError):verify_prediction(job)


def test_planner_uses_only_missing_rows_with_matching_runtime(tmp_path):
    profiles=('mrior_sda','dadda_sda','poster_ft');seeds=(2026092701,2026092702,2026092703,2026092704)
    receivers=('1-1','14-7','2-1','20-1','7-14','7-7','8-8')
    source=tmp_path/'source';data=tmp_path/'data';cache=tmp_path/'cache';release=tmp_path/'release'
    for path in (source,data,cache,release):path.mkdir()
    contract=tmp_path/'contract.json';contract.write_text('{}')
    runtime_file=tmp_path/'runtime.json';runtime_file.write_text(json.dumps(dict(
        data_root=str(data),source_cache=str(cache),source_contract=str(contract),phase1_release=str(release),
        run_root=str(source),source_run_root=str(source))))
    inventory=dict(status='INCOMPLETE',issues=[['unrelated-profile','missing']],counts={},missing={})
    for profile in profiles:
        for seed in seeds:
            path=source/'source'/f'{profile}-s{seed}';path.mkdir(parents=True);(path/'source.pt').touch()
        inventory['counts'][profile]=dict(expected_rows=7896,verified_existing=4368,missing=3528,score_jobs=1933)
        inventory['missing'][profile]=[dict(capsule=f'capsules/{pool}/MIX{episode:02}/{rx}',seed=seed,
            stage=stage,k=k,n=0) for pool in ('P01','P02','P03') for episode in range(2,9)
            for rx in receivers for seed in seeds for stage in ('A0','B') for k in (1,10,20)]
    result=make_plan(inventory,runtime_file,tmp_path/'new-run',release)
    assert len(result['jobs'])==1764 and result['runtime_file']==str(runtime_file)
    inventory['issues'].append(['mrior_sda-old-row','missing'])
    with pytest.raises(ValueError):make_plan(inventory,runtime_file,tmp_path/'new-run',release)
