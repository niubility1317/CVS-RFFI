"""Install the same graph in training, checkpoint reconstruction and CE comparison."""
from contextlib import contextmanager
from types import SimpleNamespace

from .plan import RUN, CE_VARIANT, BRANCHES, rows, validate
from .graph import install as install_graph


@contextmanager
def paper_scope():
    from experiments.pamf_paper_completion import plan
    old = plan.RUN, plan.make_plan, plan.CE_VARIANT
    plan.RUN, plan.make_plan, plan.CE_VARIANT = RUN, lambda: dict(rows=rows()), CE_VARIANT
    try:
        yield
    finally:
        plan.RUN, plan.make_plan, plan.CE_VARIANT = old


@contextmanager
def main_graph_scope():
    from experiments.pamf_paper_completion import model as old
    from experiments.cric_dura_comparison import supervised_physics
    original = old.structure
    def structure(net, row, seed):
        # The old builder must first remove the reference and verify its own
        # original graph; the frequency removal then happens before training.
        base = dict(row, structure='no_reference')
        net = original(net, base, seed)
        if row['structure'] == 'no_time_coordinate':
            net = supervised_physics.install(net, 'NO_TIME', seed)
        elif row['structure'] == 'moment':
            net = supervised_physics.install(net, 'COV_X3', seed)
        install_graph(net, 1, 1, 1, 1)
        return net
    old.structure = structure
    try:
        yield
    finally:
        old.structure = original


@contextmanager
def structural_graph_scope():
    from experiments.cric_dura_comparison import supervised_physics as physics
    original = physics.install
    def structure(net, arm, seed):
        if not arm.startswith('SS_'):
            return original(net, arm, seed)
        name = arm[3:]
        net = original(net, 'NO_REFERENCE', seed)
        if name in BRANCHES:
            t, v, p, s, _ = BRANCHES[name]
        else:
            t = v = p = s = 1
            if name == 'single_view':
                from experiments.cvs_rdrop_sinc_units.x3_features import CompositePool
                ref = net.id_backbone.encoder
                ref.sequence_pool = CompositePool(ref.sequence_pool.channels, 'SIG_TIME', seed)
            elif name == 'no_time_coordinate':
                net = original(net, 'NO_TIME', seed)
            elif name == 'moment':
                net = original(net, 'COV_X3', seed)
            elif name == 'shuffled_time':
                net = original(net, 'SHUFFLE_TIME', seed)
            else:
                raise ValueError('Unknown structural graph')
        return install_graph(net, t, v, p, s)
    physics.install = structure
    try:
        yield
    finally:
        physics.install = original


def ce_build(variant, seed):
    if variant != CE_VARIANT:
        raise ValueError('Registered CE graph required')
    from experiments.standard_ce_baselines.sigtime import SIGTimeCVS
    from experiments.cvs_rdrop_sinc_units.x3_features import CompositePool
    from experiments.cric_dura_comparison.structural import remove_branch
    net = SIGTimeCVS(seed)
    net.sequence_pool = CompositePool(net.sequence_pool.channels, 'SIG_TIME_X3', seed)
    remove_branch(net, 'F10')
    install_graph(SimpleNamespace(id_backbone=SimpleNamespace(encoder=net)), 1, 1, 1, 1)
    net._architecture.update(name='PAMF split spectrum CE', signature_views=3,
                             temporal_permutation=False, internal_reference_response=False)
    return net


@contextmanager
def ce_scope():
    from experiments.pamf_paper_completion import model as old
    original = old.ce_build
    old.ce_build = ce_build
    try:
        with paper_scope(), old.ce_scope():
            yield
    finally:
        old.ce_build = original


def train(c):
    validate(c)
    row = c['paper_row']
    from experiments.pamf_no_reference_completion.model import seed_scope
    if row['family'] in ('branch', 'structure_ce'):
        from experiments.pamf_repl_completion import source
        from experiments.pamf_branch_factorial.method import optimizer_scope
        original = source.validate_config
        source.validate_config = validate
        try:
            no_time = row['family'] == 'branch' and not BRANCHES[row['structure']][0]
            with structural_graph_scope(), optimizer_scope(no_time):
                return source.train(c)
        finally:
            source.validate_config = original
    from experiments.pamf_paper_completion import source
    if row['family'] == 'architecture':
        with paper_scope(), ce_scope():
            return source.train(c)
    with seed_scope(c['model_seed']), paper_scope(), main_graph_scope():
        return source.train(c)


def load_reader(folder, contract, row, device):
    from experiments.pamf_no_reference_completion.model import seed_scope
    if row['family'] in ('branch', 'structure_ce'):
        from experiments.pamf_repl_completion.source import load_reader as native
        with structural_graph_scope():
            return native(folder, contract, device, require_no_unlabeled=True)
    from experiments.pamf_paper_completion.model import load_reader as native
    with seed_scope(row['model_seed']), paper_scope(), main_graph_scope():
        return native(folder, contract, row, device)
