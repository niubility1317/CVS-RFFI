from copy import deepcopy
from pathlib import Path

from experiments.pamf_paper_completion import plan as paper
from experiments.pamf_tri_u_phase.plan import RECIPE

_validate_paper = paper.validate_config

RUN = '20261111-phase1-pamf-tri-phase-complement-splits-m12-r02'
SETTINGS = ('receiver_split_1', 'receiver_split_2', 'labeled_1pct', 'labeled_5pct')
SEEDS = (2026092701, 2026092702, 2026092703)
SUBSET = ('time', 'path', 'reference')
ARM = 'tri-phase-complement-splits'


def candidate_rows():
    return [r for r in paper.make_plan()['rows'] if r['data_setting'] in SETTINGS
            and r['family'] == 'factorial' and r['method'] == 'ssl'
            and r['model_seed'] in SEEDS]


def candidate_id(row):
    return f"{ARM}-{row['data_setting']}-s{row['model_seed']}"


def configuration(context, row, project, commit):
    if row not in candidate_rows():
        raise ValueError('Unregistered paper setting or seed')
    c = paper.configuration(context, row, str(project), commit)
    rid = candidate_id(row)
    c.update(run_id=RUN, row_id=rid, algorithm_id=ARM, phase_relation=deepcopy(RECIPE),
             phase_subset=list(SUBSET),
             shared_prefix=str(Path(project) / 'runs' / paper.RUN / 'F' / row['row_id'] / 'source' / 'segment20'),
             output_root=str(Path(project) / 'runs' / RUN / 'F' / rid / 'source'))
    return c


def validate_candidate(c):
    row = c.get('paper_row')
    if (row not in candidate_rows() or c.get('run_id') != RUN or c.get('row_id') != candidate_id(row)
        or c.get('algorithm_id') != ARM or c.get('phase_relation') != RECIPE
        or c.get('phase_subset') != list(SUBSET) or c.get('physical_relation')):
        raise ValueError('Unregistered phase split candidate')
    root = Path(c['output_root'])
    # Segment training has its own output folder but retains the registered root.
    if root.name in ('segment30', 'segment50'):
        root = root.parent
    if root.parts[-5:] != ('runs', RUN, 'F', c['row_id'], 'source'):
        raise ValueError('Wrong candidate output root')
    project = root.parents[3].parent
    expected = project / 'runs' / paper.RUN / 'F' / row['row_id'] / 'source' / 'segment20'
    if Path(c['shared_prefix']) != expected:
        raise ValueError('Original same-setting and same-seed E20 required')
    proxy = dict(c, run_id=paper.RUN, row_id=row['row_id'])
    proxy.pop('shared_prefix')
    return _validate_paper(proxy)
