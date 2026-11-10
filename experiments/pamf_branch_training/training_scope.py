"""Install branch supervision around the original matched PAMF PL trainer."""
from contextlib import contextmanager

from .objective import GuardedBranchPL, install_labeled
from .source_ids import indexed_source_scope


@contextmanager
def installed(config, holder, *, branch_u_enabled):
    from experiments.cric_dura_comparison import f_train, t1_model
    from experiments.pamf_paper_completion import model as paper_model
    from experiments.pamf_repl_completion.objectives import loader_scope
    from experiments.pamf_ssl_research.hard_pl_train import installed as hard_scope

    original_install, original_arguments = t1_model.install, f_train.arguments

    def build(model, cfg):
        model = original_install(model, cfg)
        model = paper_model.structure(model, cfg['paper_row'], cfg['model_seed'])
        return install_labeled(model, cfg)

    def arguments(cfg, device):
        return paper_model.source_arguments(original_arguments, cfg, device)

    t1_model.install, f_train.arguments = build, arguments
    try:
        with indexed_source_scope(), loader_scope():
            if branch_u_enabled:
                with hard_scope(config, holder, objective_cls=GuardedBranchPL):
                    yield
            else:
                yield
    finally:
        t1_model.install, f_train.arguments = original_install, original_arguments
