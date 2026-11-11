from copy import deepcopy
from pathlib import Path
from experiments.pamf_paper_completion import plan as paper

RUN='20261111-phase1-pamf-tri-u-coherence-cal-m1-m15-r01'
SETTING='source_rx_1'
SEEDS=(2026092701,2026092702,2026092703)
ARMS={'identity-ce025':dict(mode='identity',frequency=False,weight=.25),
      'identity-ce1':dict(mode='identity',frequency=False,weight=1.),
      'calibrated-ref025':dict(mode='expanded',frequency=False,weight=.25),
      'calibrated-ref1':dict(mode='expanded',frequency=False,weight=1.),
      'matched-ref1':dict(mode='expanded_uniform',frequency=False,weight=1.)}
RECIPE=dict(weight=.25,selected=128,min_scale=.25)
_validate_paper=paper.validate_config


def candidate_rows():
    return [dict(row_id=f'{arm}-m1-s{r["model_seed"]}',arm=arm,model_seed=r['model_seed'],
                 paper_row=r,views=list(r['views']))
            for r in paper.make_plan()['rows'] if r['data_setting']==SETTING
            and r['family']=='factorial' and r['method']=='ssl' and r['model_seed'] in SEEDS for arm in ARMS]


def configuration(context,row,project,commit,*,run_id=RUN):
    if row not in candidate_rows() or run_id!=RUN:raise ValueError('Only registered m1 calibration arms')
    p=row['paper_row'];c=paper.configuration(context,p,str(project),commit)
    c.update(run_id=RUN,row_id=row['row_id'],algorithm_id=row['arm'],coherence_branch=deepcopy(ARMS[row['arm']]),
        coherence_recipe=dict(RECIPE,weight=ARMS[row['arm']]['weight']),
        shared_prefix=str(Path(project)/'runs'/paper.RUN/'F'/p['row_id']/'source'/'segment20'),
        output_root=str(Path(project)/'runs'/RUN/'F'/row['row_id']/'source'))
    return c


def validate_candidate(c):
    row=next((r for r in candidate_rows() if r['row_id']==c.get('row_id')),None)
    if (row is None or c.get('run_id')!=RUN or c.get('paper_row')!=row['paper_row']
        or c.get('model_seed')!=row['model_seed'] or c.get('algorithm_id')!=row['arm']
        or c.get('coherence_recipe')!=dict(RECIPE,weight=ARMS[row['arm']]['weight'])
        or c.get('coherence_branch')!=ARMS[row['arm']]):raise ValueError('Unregistered calibration candidate')
    root=Path(c['output_root'])
    if root.name in ('segment30','segment50'):root=root.parent
    if root.parts[-5:]!=('runs',RUN,'F',row['row_id'],'source'):raise ValueError('Wrong output root')
    expected=root.parents[3].parent/'runs'/paper.RUN/'F'/row['paper_row']['row_id']/'source'/'segment20'
    if Path(c['shared_prefix'])!=expected:raise ValueError('Same single-RX seed E20 required')
    proxy=dict(c,run_id=paper.RUN,row_id=row['paper_row']['row_id']);proxy.pop('shared_prefix')
    return _validate_paper(proxy)
