"""Fixed classification-margin replacement on the matched exposed seeds."""
from pathlib import Path
from . import full_matrix_plan as original, hybrid_structure_plan as structure

RUN = '20261113-phase1-pamf-hybrid-margin-m4-r01'
SEEDS = structure.SEEDS
MATRIX = structure.MATRIX


def candidate_rows():
    return [dict(r, row_id=f"{r['setting']}-hybrid-margin-s{r['model_seed']}")
            for r in structure.candidate_rows() if r['structure_arm'] == 'C']


baseline_row = structure.baseline_row
proxy = structure.proxy
check_graph = structure.check_graph


def validate(c):
    row = next((r for r in candidate_rows() if r['row_id'] == c.get('row_id')), None)
    if row is None or c.get('hybrid_weight_row') != row or c.get('run_id') != RUN or c.get('source_development'):
        raise ValueError('Only the fixed full-source margin matrix is allowed')
    root = Path(c['output_root'])
    if root.name in ('segment30', 'segment50'):
        root = root.parent
    if root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source'):
        raise ValueError('Independent output root required')
    if (c.get('practical_gate') != dict(mode='all', strength=.25, schedule='taper', all_confident=True)
            or c.get('hybrid_auxiliary_weight') != .1 or c.get('structure_arm') != 'C'
            or c.get('margin_cap_rule') != 'correct_source_L_median'
            or c.get('source_prototype') is not None):
        raise ValueError('Fixed margin route and source-only cap required')
    original.validate(proxy(c, row))
    return row


def configuration(context, row, project, commit, *, run_id=RUN):
    if run_id != RUN or row not in candidate_rows():
        raise ValueError('Unregistered margin row')
    c = original.configuration(context, baseline_row(row), project, commit)
    c.update(run_id=RUN, row_id=row['row_id'], hybrid_weight_row=dict(row),
             algorithm_id='pamf_tri_hybrid_margin_v1', hybrid_auxiliary_weight=.1,
             structure_arm='C', tracked_aux_mode='contrast', margin_cap_rule='correct_source_L_median',
             practical_gate=dict(mode='all', strength=.25, schedule='taper', all_confident=True),
             output_root=str(Path(project)/'runs'/RUN/'F'/row['row_id']/'source'))
    validate(c)
    return c


def parent(c):
    return original.parent(proxy(c, validate(c)))
