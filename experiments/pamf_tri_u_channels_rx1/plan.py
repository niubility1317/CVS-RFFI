from copy import deepcopy
from pathlib import Path
from experiments.pamf_paper_completion import plan as paper
from experiments.pamf_tri_u_phase.plan import RECIPE
from experiments.pamf_tri_u_channels.plan import ARMS, SEEDS

RUN='20261111-phase1-pamf-tri-u-channels-rx1-m12-r01'
SETTING='receiver_split_1'
_validate_paper=paper.validate_config


def candidate_rows():
    return [dict(row_id=f'{arm}-rx1-s{r["model_seed"]}', arm=arm, model_seed=r['model_seed'],
                 paper_row=r, views=list(r['views']))
            for r in paper.make_plan()['rows'] if r['data_setting']==SETTING and r['family']=='factorial'
            and r['method']=='ssl' and r['model_seed'] in SEEDS for arm in ARMS]


def configuration(context,row,project,commit,*,run_id=RUN):
    if row not in candidate_rows() or run_id!=RUN:
        raise ValueError('Only four RX1 channel arms and three seeds registered')
    p=row['paper_row'];arm=row['arm'];c=paper.configuration(context,p,str(project),commit)
    c.update(run_id=RUN,row_id=row['row_id'],algorithm_id=arm,phase_relation=deepcopy(RECIPE),
        channel_perturbation=deepcopy(ARMS[arm]),
        shared_prefix=str(Path(project)/'runs'/paper.RUN/'F'/p['row_id']/'source'/'segment20'),
        output_root=str(Path(project)/'runs'/RUN/'F'/row['row_id']/'source'))
    return c


def validate_candidate(c):
    row=next((r for r in candidate_rows() if r['row_id']==c.get('row_id')),None)
    if (row is None or c.get('run_id')!=RUN or c.get('paper_row')!=row['paper_row']
        or c.get('model_seed')!=row['model_seed'] or c.get('algorithm_id')!=row['arm']
        or c.get('phase_relation')!=RECIPE or c.get('channel_perturbation')!=ARMS[row['arm']]):
        raise ValueError('Unregistered RX1 channel candidate')
    root=Path(c['output_root'])
    if root.name in ('segment30','segment50'):root=root.parent
    if root.parts[-5:]!=('runs',RUN,'F',row['row_id'],'source'):
        raise ValueError('Candidate output identity differs')
    project=root.parents[3].parent
    expected=project/'runs'/paper.RUN/'F'/row['paper_row']['row_id']/'source'/'segment20'
    if Path(c['shared_prefix'])!=expected:
        raise ValueError('Same RX1 setting and seed E20 required')
    proxy=dict(c,run_id=paper.RUN,row_id=row['paper_row']['row_id']);proxy.pop('shared_prefix')
    return _validate_paper(proxy)
