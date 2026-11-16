"""Fixed scratch-only supervised controls for the two requested data settings."""
from contextlib import contextmanager
from copy import deepcopy

from experiments.pamf_paper_completion import plan as paper

RUN = '20261116-phase1-pamf-tri-label-only-clean-m2-r01'
SEED = 2026092701
SETTINGS = {'ten_class': 'ManyRx', 'source_rx_6': 'm6-natural'}


def candidate_rows():
    template = next(r for r in paper.make_plan()['rows'] if r['family']=='U_permission'
                    and r['model_seed']==SEED)
    rows=[]
    for setting,pid in SETTINGS.items():
        row=deepcopy(template)
        row.update(row_id=f'{setting}-tri-label-only-permuted-s{SEED}', data_setting=setting,
                   setting=setting, protocol_id=pid, split_name=paper.PROTOCOLS[setting][1], views=['clean'],
                   readouts=['raw'], reuse_key=None, new_training=True)
        rows.append(row)
    return rows


@contextmanager
def installed():
    rows=candidate_rows()
    old_run,old_make=paper.RUN,paper.make_plan
    paper.RUN=RUN
    paper.make_plan=lambda:dict(run_id=RUN,rows=rows)
    try:
        yield
    finally:
        paper.RUN,paper.make_plan=old_run,old_make


def configuration(ctx,row,project,commit):
    if row not in candidate_rows():
        raise ValueError('Only fixed labeled-only rows are registered')
    with installed():
        c=paper.configuration(ctx,row,project,commit)
    if (c['U_use'] or c['regime']!='matched_sat_supervised' or c['ablation']!='F20'
            or not row['permuted'] or row['learning_regime']!='supervised'):
        raise ValueError('Complete labeled-only supervised Tri graph required')
    return c


def validate(c):
    with installed():
        paper.validate_config(c)
    if c['U_use'] or c['regime']!='matched_sat_supervised' or c['paper_row']['views']!=['clean']:
        raise ValueError('No U input or satellite evaluation permitted')
    return c
