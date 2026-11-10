"""Three fixed continuations of the original same-seed Tri-SIG-TIME prefixes."""
from pathlib import Path

from experiments.cric_dura_comparison.io import read
from experiments.pamf_paper_completion import plan as paper
from experiments.pamf_repl_view_shuffle import plan as tri


RUN = '20261110-phase1-pamf-tri-label-tangent-f0-m3-r01'
SEEDS = (2026092701, 2026092702, 2026092703)


def candidate_rows():
    return [dict(row_id=f'tri-label-tangent-s{seed}', model_seed=seed, views=list(tri.VIEWS)) for seed in SEEDS]


def configuration(context, row, project, commit, *, run_id=RUN):
    if row not in candidate_rows() or run_id != RUN:
        raise ValueError('Only the registered three Tri-SIG-TIME continuations are allowed')
    seed = row['model_seed']
    original = f'main_split-repl-shuffled_time-3view-s{seed}'
    prefix = Path(project) / 'runs' / tri.RUN / 'F' / original / 'source' / 'segment20'
    c = dict(read(prefix / 'complete.json')['config'])
    tri.validate_config(c)
    for key in ('parent_source', 'segment_epochs', 'epoch_offset', 'step_offset'):
        c.pop(key, None)
    paper_row = next(r for r in paper.make_plan()['rows']
                     if r['row_id'] == f'main_split-factorial-ssl-full-permuted-s{seed}')
    c.update(run_id=RUN, row_id=row['row_id'], paper_row=paper_row, code_commit=commit,
             algorithm_id='pamf_tri_label_tangent_v1', shared_prefix=str(prefix),
             physical_relation=dict(beta=.5, l_weight=.05, u_weight=0.),
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    if read(c['source_contract']) != read(context['handles']['F0']['source_contract']):
        raise ValueError('Original Tri-SIG-TIME prefix does not match the registered F0 handle')
    return c
