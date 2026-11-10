"""Install source-only PTBR without altering the released PAMF graph."""
from contextlib import contextmanager
from experiments.pamf_branch_training.source_ids import indexed_source_scope
from .objective import PhysicalPL, install_labeled


@contextmanager
def installed(c, holder):
    from experiments.cric_dura_comparison import f_train, t1_model
    from experiments.pamf_paper_completion import model as paper
    from experiments.pamf_repl_completion.objectives import loader_scope
    from experiments.pamf_ssl_research.hard_pl_train import installed as hard_scope
    old_install, old_arguments = t1_model.install, f_train.arguments

    def build(model, cfg):
        model = paper.structure(old_install(model, cfg), cfg['paper_row'], cfg['model_seed'])
        return install_labeled(model, cfg)

    def arguments(cfg, device):
        return paper.source_arguments(old_arguments, cfg, device)

    t1_model.install, f_train.arguments = build, arguments
    try:
        with indexed_source_scope(), loader_scope(), hard_scope(c, holder, objective_cls=PhysicalPL):
            yield
    finally:
        t1_model.install, f_train.arguments = old_install, old_arguments
