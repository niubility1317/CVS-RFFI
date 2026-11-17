import pytest
from experiments.pamf_repl_priority import plan as original
from experiments.pamf_repl_priority_extension import plan
from experiments.pamf_repl_component_guard import plan as base


def test_extension_is_disjoint_and_scope_uses_exact_new_run(tmp_path):
    old=original.candidate_rows()
    rows=plan.candidate_rows()
    assert len(rows)==4 and {r['row_id'] for r in rows}.isdisjoint(r['row_id'] for r in old)
    assert {plan.RECIPES[r['recipe']]['weight'] for r in rows}=={.75}
    assert {plan.RECIPES[r['recipe']]['kl_factor'] for r in rows}=={1.,1.5}
    assert {plan.RECIPES[r['recipe']]['mode'] for r in rows}=={'native','kl'}
    with plan.scoped():
        assert base.RUN==plan.RUN and base.candidate_rows()==rows
        with pytest.raises(FileNotFoundError):base.configuration({},rows[0],tmp_path,'commit')
    assert original.candidate_rows()==old
