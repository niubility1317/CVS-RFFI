"""Retain the Tri SSL objective and remove identity-direction projection only."""
from contextlib import contextmanager

from experiments.pamf_branch_training.source_ids import indexed_source_scope
from experiments.pamf_repl_view_shuffle.source import installed as original_tri
from experiments.pamf_physics_branch.objective import install_labeled


@contextmanager
def installed(c, holder):
    from experiments.cric_dura_comparison import t1_model

    with indexed_source_scope(), original_tri(c, holder, enabled=True):
        original_install = t1_model.install

        def build(model, cfg):
            return install_labeled(original_install(model, cfg), cfg, protect_identity=False)

        t1_model.install = build
        try:
            yield
        finally:
            t1_model.install = original_install
