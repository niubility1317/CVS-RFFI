"""Staged reduced-weight Hybrid continuations and automatic seven-view scoring."""
import argparse
import math
from contextlib import contextmanager
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write
from . import hybrid_tracked_plan as plan, hybrid_confidence_runner as common
from .hybrid_tracked import HybridTracked
from .practical_gate import installed as practical_scope


@contextmanager
def installed(c, holder):
    from experiments.pamf_paper_completion.model import installed as paper_scope

    @contextmanager
    def adapted(cfg, state, enabled, objective_factory):
        with paper_scope(cfg, state, ssl_enabled=enabled, objective_factory=objective_factory,
                         validate=plan.validate):
            yield

    with practical_scope(c, holder, base_scope=adapted, objective_factory=HybridTracked):
        yield


def stage(rt):
    old = common.plan
    common.plan = plan
    try:
        return common.stage(rt)
    finally:
        common.plan = old


def check_tracked(state):
    details = common.check_hybrid(state)
    hybrid = state['U_hybrid_contrast']
    gradients = hybrid['smoke_gradients']
    if state['U_fused_physics']['extra_student_forwards'] != 2 or hybrid['anchors'] != 2 * state['accepted']:
        raise ValueError('Every accepted sample must reach the isolated auxiliary forwards')
    if any(not math.isfinite(v) or v <= 0 for component in gradients.values() for v in component.values()):
        raise ValueError('Both auxiliary components must deliver finite branch gradients')
    return dict(details, U_hybrid_contrast=hybrid, U_tracking=state['U_tracking'])


def smoke(rt):
    from .runner import smoke as smoke_rows
    checks = []
    for row in (r for r in plan.candidate_rows() if r['model_seed'] == plan.SEEDS[0]):
        checks.extend(smoke_rows(rt, rows=lambda: [row], configure=plan.configuration, check_parent=plan.parent,
                                scope=installed, check_u=check_tracked, graph_check=plan.check_graph,
                                segment_output=True, write_result=False))
    write(Path(rt['run_root']) / 'smoke.json', dict(status='PASS', checks=checks, real_checkpoint=True,
          optimizer_updates=0, query_reads=0, U_truth_reads=0))


def worker(c, kind):
    if kind == 'score':
        from experiments.pamf_paper_completion.evaluation import score
        return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind == 'smoke':
        return smoke(c)
    if kind == 'train':
        from .source import train_trajectory
        return train_trajectory(c, plan.parent(c), installed, ssl=True, physical_L=False)
    if kind == 'predict':
        from .development_evaluation import predict
        return predict(c)
    raise ValueError('Only source training, frozen prediction and independent scoring are allowed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['worker', 'stage'])
    parser.add_argument('--kind')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = read(args.config)
    stage(config) if args.action == 'stage' else worker(config, args.kind)
