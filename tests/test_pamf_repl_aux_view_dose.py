import json

from experiments.pamf_repl_aux_view_dose import plan
from experiments.pamf_repl_view_dose import plan as prior


def test_both_matrices_cross_doses_and_reuse_the_exact_central_recipe():
    assert len({row['row_id'] for row in plan.candidate_rows()}) == 16
    for setting, axes in plan.FAMILIES.items():
        recipes = [recipe for recipe in plan.RECIPES.values() if recipe['setting'] == setting]
        aux_key = axes['auxiliary']
        central = prior.RECIPES['ten_kl_ordinary' if setting == 'ten_class' else 'six_kl_raw']
        assert (central[aux_key], central['kl_factor']) == axes['anchor']
        for recipe in recipes:
            assert json.loads(json.dumps(recipe)) == recipe
            assert {k: v for k, v in recipe.items() if k not in (aux_key, 'kl_factor')} == {
                k: v for k, v in central.items() if k not in (aux_key, 'kl_factor')}
        assert {(r[aux_key], r['kl_factor']) for r in recipes} == {
            (a, k) for a in axes['aux_values'] for k in axes['kl_values']} - {axes['anchor']}
