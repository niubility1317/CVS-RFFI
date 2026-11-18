from pathlib import Path

import pytest

from experiments.pamf_phase2_practical import mopc_clean_author


def test_clean_scope_requires_author_version_and_exact_four_points():
    job = dict(profile='mopc_hr', capsule='capsules/P01/MIX08/1-1', seed=2026092701,
               points=[[5, 5], [5, 10], [10, 5], [10, 10]],
               mopc_sampling_version='author_drop_last_v1', mopc_stabilization='none')
    mopc_clean_author.validate_scope(job)
    with pytest.raises(ValueError):
        mopc_clean_author.validate_scope(dict(job, points=[[5, 5]]))
    with pytest.raises(ValueError):
        mopc_clean_author.validate_scope(dict(job, mopc_sampling_version='old'))
    with pytest.raises(ValueError):
        mopc_clean_author.validate_scope(dict(job, capsule='capsules/P01/LS01/1-1'))


def test_clean_prediction_rejects_missing_sampling_identity(tmp_path: Path):
    job = dict(profile='mopc_hr', capsule='capsules/P01/MIX08/1-1', seed=2026092701,
               points=[[5, 5], [5, 10], [10, 5], [10, 10]],
               mopc_sampling_version='author_drop_last_v1', mopc_stabilization='none',
               output=str(tmp_path))
    from experiments.pamf_phase2_practical.io import write
    write(tmp_path/'complete.json', dict(status='PREDICTIONS_COMPLETE', truth_read=False,
          query_fit=False, profile='mopc_hr', model_seed=job['seed'], capsule_id='x', rows=4))
    rows=[]
    for k,n in job['points']:
        name=f'{k}-{n}.npz';(tmp_path/name).write_bytes(b'placeholder')
        rows.append(dict(stage='C', k=k, new_count=n, arm='mopc_hr', input_path='P0',
                         input_condition='clean_ground_pre_satellite_normalized', prediction=name,
                         resources={'native_resources':{'mopc_semantics':'public_code'}}))
    write(tmp_path/'predictions.json', dict(capsule=job['capsule'], capsule_id='x',
          model_seed=job['seed'], rows=rows))
    with pytest.raises(ValueError, match='sampling'):
        mopc_clean_author.verify_prediction(job)


def test_rebuild_clean_job_excludes_old_registration_and_sets_fresh_output(tmp_path):
    old=dict(profile='protonet_cda', capsule='capsules/P01/MIX01/1-1', seed=2026092701,
             points=[[1,2],[5,5]], methods=['mopc_hr'], key='old', output='old',
             save_B_for_mopc=True)
    job=mopc_clean_author.new_job(old, '/new-run')
    assert job['points']==mopc_clean_author.POINTS
    assert job['mopc_bridge']=='proto_affine_v1'
    assert job['mopc_sampling_version']=='author_drop_last_v1'
    assert job['output'].startswith('/new-run/predictions/')
    assert 'reuse' not in job


def test_pilot_partition_contains_no_duplicate_full_jobs():
    jobs=[]
    for profile in mopc_clean_author.PROFILES:
        for capsule in ('capsules/P01/MIX01/1-1','capsules/P01/MIX02/1-1'):
            jobs.append(dict(profile=profile,capsule=capsule,seed=2026092701))
    pilot=mopc_clean_author.partition(jobs, 'pilot')
    rest=mopc_clean_author.partition(jobs, 'remaining')
    assert len(pilot)==6 and len(rest)==6
    assert not set((j['profile'],j['capsule']) for j in pilot) & set((j['profile'],j['capsule']) for j in rest)
