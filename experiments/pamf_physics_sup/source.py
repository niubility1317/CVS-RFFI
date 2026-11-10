"""Keep the original L-only F20 and add the true-label physical branch relation."""
from contextlib import contextmanager
from pathlib import Path

from experiments.cric_dura_comparison.io import read
from experiments.pamf_branch_training.source import checked_e20_parent
from experiments.pamf_branch_training.source_ids import indexed_source_scope
from experiments.pamf_paper_completion import model as paper_model, plan as paper
from experiments.pamf_physics_branch.objective import install_labeled
from experiments.pamf_physics_tri.source import train_trajectory
from .plan import PREFIX_RUN, RUN, SEEDS


def parent(c):
    seed = c['model_seed']
    root = Path(c['output_root'])
    baseline = f'main_split-factorial-sup-full-ordered-s{seed}'
    if (seed not in SEEDS or c['run_id'] != RUN or c['row_id'] != f'sup-label-tangent-s{seed}'
            or root.parts[-5:] != ('runs', RUN, 'F', c['row_id'], 'source')
            or c['paper_row']['row_id'] != baseline or c['algorithm_id'] != 'pamf_sup_label_tangent_v1'
            or c['physical_relation'] != dict(beta=.5, l_weight=.05, u_weight=0.)):
        raise ValueError('Unregistered supervised physical continuation')
    expected = root.parents[3] / PREFIX_RUN / f'F20-s{seed}' / 'source' / 'segment20'
    if Path(c['shared_prefix']) != expected:
        raise ValueError('Only the original same-seed F20 prefix is allowed')
    proxy = dict(c, run_id=paper.RUN, row_id=baseline)
    proxy.pop('shared_prefix')
    paper.validate_config(proxy)
    evidence = checked_e20_parent(c)
    progress = read(expected / 'progress.json')
    if progress['U_actual'] or progress['PL_seen'] or evidence['config']['ablation'] != 'F20':
        raise ValueError('The inherited supervised trajectory must have zero U exposure')
    return evidence


@contextmanager
def installed(c, holder):
    from experiments.cric_dura_comparison import f_train, t1_model

    old_install, old_args = t1_model.install, f_train.arguments

    def build(model, cfg):
        model = paper_model.structure(old_install(model, cfg), cfg['paper_row'], cfg['model_seed'])
        model = install_labeled(model, cfg, protect_identity=True)
        holder['model'] = model
        return model

    t1_model.install = build
    f_train.arguments = lambda cfg, device: paper_model.source_arguments(old_args, cfg, device)
    try:
        with indexed_source_scope():
            yield
    finally:
        t1_model.install, f_train.arguments = old_install, old_args


def train(c):
    return train_trajectory(c, parent(c), installed, ssl=False)
