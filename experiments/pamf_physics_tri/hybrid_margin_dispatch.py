"""Attach the fixed margin queue to the existing task-owned GPU dispatcher."""
from . import hybrid_fullbranch_dispatch as base
from .hybrid_margin_plan import RUN

base.OLD_RELEASE = base.scheduler.PROJECT/'releases/20261113-phase1-pamf-hybrid-fullbranch-match-m4-r01-release'
base.OLD_MODULE = 'experiments.pamf_physics_tri.hybrid_fullbranch_dispatch'
base.NEW_RUN = RUN
base.NEW_MODULE = 'experiments.pamf_physics_tri.hybrid_margin_runner'
base.EXISTING = (
    ('20261113-phase1-pamf-hybrid-tracked-failing-m26-r01', 'experiments.pamf_physics_tri.hybrid_tracked_runner', None),
    ('20261113-phase1-pamf-hybrid-structure-abcd-m16-r01', 'experiments.pamf_physics_tri.hybrid_structure_runner', None),
    ('20261113-phase1-pamf-hybrid-fullbranch-match-m4-r01', 'experiments.pamf_physics_tri.hybrid_fullbranch_runner', None),
)

if __name__ == '__main__':
    base.main()
