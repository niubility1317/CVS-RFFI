"""Fixed three-seed F0 L-only physical continuation."""
from pathlib import Path

from experiments.pamf_paper_completion import plan as paper
from experiments.pamf_physics_tri.plan import SEEDS
from experiments.pamf_branch_training.plan import VIEWS


RUN = '20261110-phase1-pamf-sup-label-tangent-f0-m3-r01'
PREFIX_RUN = '20261022-phase1-T1-DURA-F-ablation-binding-m63-r01'


def candidate_rows():
    return [dict(row_id=f'sup-label-tangent-s{seed}', model_seed=seed, views=list(VIEWS)) for seed in SEEDS]


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only the three matched PAMF-Sup continuations are allowed')
    seed = row['model_seed']
    paper_row = next(r for r in paper.make_plan()['rows']
                     if r['row_id'] == f'main_split-factorial-sup-full-ordered-s{seed}')
    c = paper.configuration(context, paper_row, project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_sup_label_tangent_v1',
             shared_prefix=context['sources'][str(seed)]['F20'] + '/segment20',
             physical_relation=dict(beta=.5, l_weight=.05, u_weight=0.),
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    return c
