"""Keep old method implementations scoped to this new no-reference matrix."""
from contextlib import contextmanager
from copy import deepcopy

from .plan import RUN, CE_VARIANT, rows


@contextmanager
def paper_scope():
    from experiments.pamf_paper_completion import plan
    old=(plan.RUN,plan.make_plan,plan.CE_VARIANT)
    plan.RUN,plan.make_plan,plan.CE_VARIANT=RUN,lambda:dict(rows=rows()),CE_VARIANT
    try: yield
    finally: plan.RUN,plan.make_plan,plan.CE_VARIANT=old


@contextmanager
def structure_scope():
    from experiments.cric_dura_comparison import supervised_physics as physics
    original=physics.install
    kinds={'NR_no_time_coordinate':'NO_TIME','NR_moment':'COV_X3','NR_shuffled_time':'SHUFFLE_TIME'}
    def install(model, arm, seed):
        if arm in kinds:
            model=original(model,'NO_REFERENCE',seed)
            return original(model,kinds[arm],seed)
        return original(model,arm,seed)
    physics.install=install
    try: yield
    finally: physics.install=original


@contextmanager
def seed_scope(seed):
    """Extend the original three-seed argument template without changing RNG."""
    if seed in (2026092701,2026092702,2026092703):
        yield
        return
    if seed not in range(2026092704,2026092709):
        raise ValueError('Unregistered extended model seed')
    from experiments.cric_dura_comparison import f_train
    from experiments.cvs_phase1_stack import runtime
    original_config,original_arguments,original_installed=f_train.original_config,f_train.arguments,runtime.installed
    def config(value):
        if value!=seed:raise ValueError('Unexpected model seed')
        return original_config(2026092701)
    def arguments(c,device):
        args=original_arguments(c,device)
        args.seed=seed
        return args
    def installed(c):
        # The native adapter initializes the actual network from this seed in
        # a forked RNG context, independently from the argument template.
        return original_installed(dict(c,model_seed=seed))
    f_train.original_config,f_train.arguments,runtime.installed=config,arguments,installed
    try: yield
    finally: f_train.original_config,f_train.arguments,runtime.installed=original_config,original_arguments,original_installed


def ce_build(variant, seed):
    if variant!=CE_VARIANT: raise ValueError('Only no-reference E200 CE graph')
    from experiments.standard_ce_baselines.sigtime import SIGTimeCVS
    from experiments.cvs_rdrop_sinc_units.x3_features import CompositePool
    from experiments.cric_dura_comparison.structural import remove_branch
    net=SIGTimeCVS(seed)
    net.sequence_pool=CompositePool(net.sequence_pool.channels,'SIG_TIME_X3',seed)
    remove_branch(net,'F10')
    net._architecture.update(name='PAMF-CE Three-view without reference',signature_views=3,
                             temporal_permutation=False,internal_reference_response=False)
    if sum(p.numel() for p in net.parameters())!=220193:
        raise ValueError('No-reference E200 graph parameter count differs')
    return net


@contextmanager
def ce_scope():
    from experiments.pamf_paper_completion import model
    old=model.ce_build
    model.ce_build=ce_build
    try:
        with paper_scope(),model.ce_scope(): yield
    finally: model.ce_build=old


def train(c):
    from .plan import validate
    validate(c)
    if c['paper_row']['family']=='structure_ce':
        from experiments.pamf_repl_completion import source
        old=source.validate_config
        source.validate_config=validate
        try:
            with structure_scope(): return source.train(c)
        finally: source.validate_config=old
    from experiments.pamf_paper_completion import source
    if c['paper_row']['family']=='architecture':
        with paper_scope(),ce_scope(): return source.train(c)
    from .fast_practical import installed as fast_as_scope
    with seed_scope(c['model_seed']),paper_scope(),fast_as_scope(c): return source.train(c)


def load_reader(folder, contract, row, device):
    if row['family']=='structure_ce':
        from experiments.pamf_repl_completion.source import load_reader as load
        with structure_scope(): return load(folder,contract,device,require_no_unlabeled=True)
    from experiments.pamf_paper_completion.model import load_reader as load
    with seed_scope(row['model_seed']),paper_scope(): return load(folder,contract,row,device)
