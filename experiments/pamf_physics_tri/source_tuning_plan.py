"""One-seed, source-development pilot for branch-prototype temperature."""
from pathlib import Path

from experiments.pamf_paper_completion import plan as paper

RUN = '20261112-phase1-pamf-branch-prototype-source-tuning-s2026092701-r01'
SPLIT_SEED = 20261112


def candidate_rows():
    return [dict(row_id=f'ten-class-prototype-temp-{name}-s2026092701',
                 setting='ten_class', protocol_id='ManyRx', model_seed=2026092701,
                 temperature=value)
            for name, value in (('010', 0.1), ('020', 0.2), ('040', 0.4))]


def validate_candidate(row):
    if row not in candidate_rows():
        raise ValueError('Candidate must match the registered source-only pilot')
    return row


def configuration(context, row, project, commit, partition):
    validate_candidate(row)
    original_id = f"{row['setting']}-factorial-ssl-full-permuted-s{row['model_seed']}"
    paper_row = next(r for r in paper.make_plan()['rows'] if r['row_id'] == original_id)
    c = paper.configuration(context, paper_row, project, commit)
    gate = dict(mode='branch', strength=.25, schedule='taper')
    if 'required_votes' in row:
        gate['required_votes'] = row['required_votes']
    c.update(run_id=RUN, row_id=row['row_id'], algorithm_id='pamf_tri_source_tuning_v1',
             tuning_row=dict(row), source_development=dict(partition),
             practical_gate=gate,
             day_relation_weight=.025, fused_physics=dict(weight='original_joint', raw_detached=True),
             hybrid_contrast=dict(mode='fusion_branch', temperature=.2),
             source_prototype=dict(mode='branch', temperature=row['temperature'], coefficient='original_joint'),
             output_root=str(Path(project) / 'runs' / RUN / 'F' / row['row_id'] / 'source'))
    validate(c)
    return c


def validate(c):
    row = validate_candidate(c.get('tuning_row'))
    part = c.get('source_development')
    gate = dict(mode='branch', strength=.25, schedule='taper')
    if 'required_votes' in row:
        gate['required_votes'] = row['required_votes']
    root = Path(c['output_root'])
    if root.name in ('segment20', 'segment30', 'segment50'):
        root = root.parent
    if (c.get('run_id') != RUN or c.get('row_id') != row['row_id']
            or c.get('model_seed') != row['model_seed'] or c.get('algorithm_id') != 'pamf_tri_source_tuning_v1'
            or c.get('protocol_id') != row['protocol_id']
            or c.get('paper_row', {}).get('row_id') != f"{row['setting']}-factorial-ssl-full-permuted-s{row['model_seed']}"
            or root.parts[-5:] != ('runs', RUN, 'F', row['row_id'], 'source')
            or not isinstance(part, dict) or set(part) != {'train_ids', 'dev_ids', 'seed'}
            or part['seed'] != SPLIT_SEED or not part['train_ids'] or not part['dev_ids']
            or c.get('practical_gate') != gate
            or c.get('day_relation_weight') != .025
            or c.get('fused_physics') != dict(weight='original_joint', raw_detached=True)
            or c.get('hybrid_contrast') != dict(mode='fusion_branch', temperature=.2)
            or c.get('source_prototype') != dict(mode='branch', temperature=row['temperature'], coefficient='original_joint')):
        raise ValueError('Source development candidate differs from the registered recipe')
    proxy = dict(c, run_id=paper.RUN, row_id=c['paper_row']['row_id'])
    proxy.pop('parent_source', None)
    proxy.pop('shared_prefix', None)
    paper.validate_config(proxy)
    return row
