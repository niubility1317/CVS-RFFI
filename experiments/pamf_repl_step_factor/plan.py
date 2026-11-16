"""Isolated factor matrix using the original full-L AdamW guard."""
from contextlib import contextmanager
from experiments.pamf_repl_component_guard import plan as component
from experiments.pamf_repl_step_guard import plan as original

RUN = '20261116-phase1-pamf-repl-step-factor-clean-m4-r01'
RECIPES = {
    'step_weight3': {'weight': 3.0},
    'step_guard05': {'guard_cap': 0.5},
}


@contextmanager
def scoped():
    """Temporarily register this matrix in a single-purpose worker process."""
    old = original.RUN, original.RECIPES, component.RUN, component.RECIPES, component.apply_recipe
    apply = component.apply_recipe
    def apply_factor(configuration, name):
        result = apply(configuration, name)
        if 'guard_cap' in RECIPES[name]:
            result['hard_pl_gradient_guard']['U_to_L_norm_cap'] = RECIPES[name]['guard_cap']
        return result
    original.RUN, original.RECIPES = RUN, RECIPES
    component.RUN, component.RECIPES, component.apply_recipe = RUN, RECIPES, apply_factor
    try:
        yield
    finally:
        original.RUN, original.RECIPES, component.RUN, component.RECIPES, component.apply_recipe = old


def candidate_rows():
    with scoped():
        return original.candidate_rows()
