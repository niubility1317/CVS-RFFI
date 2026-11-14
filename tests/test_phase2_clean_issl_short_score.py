import json
import pytest

from experiments.pamf_phase2_practical import clean_issl_short_score as scorer


def test_paired_score_rejects_missing_budget_before_scoring(tmp_path,monkeypatch):
    reference={'jobs':[{'group':'native_cil','job':f'old-{i}'} for i in range(5)]}
    short={'jobs':[{'group':'native_cil','job':f'new-{i}'} for i in range(20)]}
    def read(path):
        name=str(path)
        if name=='reference':return reference
        if name=='short':return short
        if name.startswith('old-'):return {'profile':'issl_fixed_v1'}
        if name.startswith('completion-'):return {'issl_budget':[50,25]}
        raise AssertionError(name)
    monkeypatch.setattr(scorer,'read',read)
    def verify(config,prep):
        if len(config['jobs'])==5:
            return [dict(job=dict(capsule=f'cap-{i}',seed=1,points=[[5,20]])) for i in range(5)]
        return [dict(job=dict(capsule=f'cap-{i%5}',seed=1,points=[[5,20]],issl_budget=[50,25]),
                     payload={'issl_budget':[50,25]},entry={'completion':'completion-x'}) for i in range(20)]
    monkeypatch.setattr(scorer,'verify_all',verify)
    monkeypatch.setattr(scorer,'score_job',lambda *args:pytest.fail('Scoring began before pairing was checked'))
    with pytest.raises(ValueError,match='Duplicate short-budget'):
        scorer.score('reference','short',tmp_path/'output')
    assert not (tmp_path/'output').exists()
