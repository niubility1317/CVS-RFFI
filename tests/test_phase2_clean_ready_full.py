import json

import pytest

from experiments.pamf_phase2_practical import clean_ready_full as full
from experiments.pamf_phase2_practical.clean_score import expected_rows


def test_requires_all_pilot_methods_before_excluding_any_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(full,'completion_matches',lambda entry:True)
    jobs=[]
    groups=(('pamf',('R42',)),('native_da',full.DA),('native_cil',full.CIL),('hybrid',full.DA))
    for i in range(5):
        capsule=f'capsules/P01/MIX01/rx{i}'
        for group,profiles in groups:
            for profile in profiles:
                name=f'{i}-{group}-{profile}';folder=tmp_path/name;folder.mkdir()
                job=dict(capsule=capsule,seed=2026092701,points=[[5,20]],profile=profile)
                if group=='hybrid':job['methods']=['orthogonal','lora']
                (folder/'job.json').write_text(json.dumps(job))
                rows=[]
                for arm,stage,k,n,path in expected_rows(group,job):
                    file=f'{len(rows)}.npz';(folder/file).write_bytes(b'x')
                    rows.append(dict(arm=arm,stage=stage,k=k,new_count=n,input_path=path,
                                     prediction=file,input_condition='clean_ground_pre_satellite_normalized'))
                (folder/'predictions.json').write_text(json.dumps(dict(capsule=capsule,model_seed=job['seed'],rows=rows)))
                jobs.append(dict(group=group,job=str(folder/'job.json'),completion=str(folder/'complete.json')))
    config=tmp_path/'config.json';config.write_text(json.dumps(dict(jobs=jobs)))
    assert len(full.prior_pilot_identities([config]))==5
    config.write_text(json.dumps(dict(jobs=jobs[:-1])))
    with pytest.raises(ValueError,match='incomplete'):
        full.prior_pilot_identities([config])
