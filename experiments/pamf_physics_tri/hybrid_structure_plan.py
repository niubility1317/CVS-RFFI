"""A/B/C/D input-routing and fusion-only ablation on fixed exposed seeds."""
from pathlib import Path
from experiments.pamf_repl_view_shuffle.plan import VIEWS
from . import full_matrix_plan as original

RUN = '20261113-phase1-pamf-hybrid-structure-abcd-m16-r01'
SEEDS = (2026092701,2026092702,2026092703)
MATRIX = (('main_split','F0',(2026092701,)),('ten_class','ManyRx',SEEDS))


def candidate_rows():
    return [dict(row_id=f'{setting}-hybrid-structure-{arm}-s{seed}', setting=setting, protocol_id=pid,
                 model_seed=seed, auxiliary_weight=.1, structure_arm=arm, views=list(VIEWS))
            for setting,pid,seeds in MATRIX for arm in ('A','B','C','D') for seed in seeds]


def baseline_row(row):
    return next(r for r in original.candidate_rows() if r['setting']==row['setting']
                and r['model_seed']==row['model_seed'] and r['arm']=='hybrid')


def proxy(c,row):
    old=baseline_row(row);root=Path(c['output_root'])
    segment=root.name if root.name in ('segment30','segment50') else None
    source=root.parent if segment else root
    path=source.parents[3]/original.RUN/'F'/old['row_id']/'source'
    if segment:path/=segment
    return dict(c,run_id=original.RUN,row_id=old['row_id'],algorithm_id='pamf_tri_two_method_matrix_v1',
                output_root=str(path),practical_gate=dict(mode='branch',strength=.25,schedule='taper'))


def validate(c):
    row=next((r for r in candidate_rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c.get('hybrid_weight_row')!=row or c.get('run_id')!=RUN or c.get('source_development'):
        raise ValueError('Only the fixed full-source structural ablation is allowed')
    root=Path(c['output_root'])
    if root.name in ('segment30','segment50'):root=root.parent
    if root.parts[-5:]!=('runs',RUN,'F',row['row_id'],'source'):
        raise ValueError('Independent output root required')
    if (c.get('practical_gate')!=dict(mode='all',strength=.25,schedule='taper',all_confident=True)
        or c.get('hybrid_auxiliary_weight')!=.1 or c.get('structure_arm')!=row['structure_arm']
        or c.get('source_prototype') is not None):
        raise ValueError('Fixed route, strength schedule and fusion-only ablation required')
    original.validate(proxy(c,row));return row


def configuration(context,row,project,commit,*,run_id=RUN):
    if run_id!=RUN or row not in candidate_rows():raise ValueError('Unregistered structural ablation row')
    c=original.configuration(context,baseline_row(row),project,commit)
    c.update(run_id=RUN,row_id=row['row_id'],hybrid_weight_row=dict(row),algorithm_id='pamf_tri_hybrid_structure_abcd_v1',
        hybrid_auxiliary_weight=.1,structure_arm=row['structure_arm'],tracked_aux_mode='contrast',
        practical_gate=dict(mode='all',strength=.25,schedule='taper',all_confident=True),
        output_root=str(Path(project)/'runs'/RUN/'F'/row['row_id']/'source'))
    validate(c);return c


def parent(c):return original.parent(proxy(c,validate(c)))
check_graph=original.check_graph
