"""Fixed original-loss, complete-L-anchored AdamW step protection."""
from contextlib import contextmanager
from experiments.pamf_repl_component_guard import plan as base

RUN='20261116-phase1-pamf-repl-adamw-guard-clean-m2-r01'
SEED=base.SEED
SETTINGS=base.SETTINGS
RECIPES={'step_guard':{'anchor':'complete_L','original_U_weight':6.0}}


@contextmanager
def _scope():
    old=base.RUN,base.RECIPES
    base.RUN,base.RECIPES=RUN,RECIPES
    try:yield
    finally:base.RUN,base.RECIPES=old


def candidate_rows():
    with _scope():return base.candidate_rows()


def configuration(ctx,row,project,commit):
    with _scope():c=base.configuration(ctx,row,project,commit,run_id=RUN)
    c['algorithm_id']='pamf_repl_adamw_full_L_guard_v1'
    return c


def validate(c):
    if c.get('algorithm_id')!='pamf_repl_adamw_full_L_guard_v1':
        raise ValueError('Registered optimizer-step method required')
    with _scope():return base.validate(c)


def parent(c):
    with _scope():return base.parent(c)


def check_graph(model,c):return base.check_graph(model,c)
