from copy import deepcopy
from pathlib import Path
from experiments.pamf_paper_completion import plan as paper

RUN='20261112-phase1-pamf-tri-u-balancedcap-m1-m27-r01'
SETTING='source_rx_1';SEEDS=(2026092701,2026092702,2026092703)
ARMS={f'cap{label}-{rank}':dict(rank=rank,mode='expanded',frequency=False,cap=cap)
      for label,cap in (('1125',1.125),('125',1.25),('150',1.5))
      for rank in ('random','confidence','physical')}
_validate_paper=paper.validate_config


def candidate_rows():
    return [dict(row_id=f'{arm}-m1-s{r["model_seed"]}',arm=arm,model_seed=r['model_seed'],paper_row=r,views=list(r['views']))
            for r in paper.make_plan()['rows'] if r['data_setting']==SETTING and r['family']=='factorial'
            and r['method']=='ssl' and r['model_seed'] in SEEDS for arm in ARMS]


def configuration(context,row,project,commit,*,run_id=RUN):
    if row not in candidate_rows() or run_id!=RUN:raise ValueError('Only registered balanced U arms')
    p=row['paper_row'];c=paper.configuration(context,p,str(project),commit)
    c.update(run_id=RUN,row_id=row['row_id'],algorithm_id=row['arm'],balanced_recipe=deepcopy(ARMS[row['arm']]),
        shared_prefix=str(Path(project)/'runs'/paper.RUN/'F'/p['row_id']/'source'/'segment20'),
        output_root=str(Path(project)/'runs'/RUN/'F'/row['row_id']/'source'))
    return c


def validate_candidate(c):
    row=next((r for r in candidate_rows() if r['row_id']==c.get('row_id')),None)
    if (row is None or c.get('run_id')!=RUN or c.get('paper_row')!=row['paper_row']
        or c.get('model_seed')!=row['model_seed'] or c.get('algorithm_id')!=row['arm']
        or c.get('balanced_recipe')!=ARMS[row['arm']]):raise ValueError('Unregistered balanced candidate')
    root=Path(c['output_root'])
    if root.name in ('segment30','segment50'):root=root.parent
    if root.parts[-5:]!=('runs',RUN,'F',row['row_id'],'source'):raise ValueError('Wrong output root')
    expected=root.parents[3].parent/'runs'/paper.RUN/'F'/row['paper_row']['row_id']/'source'/'segment20'
    if Path(c['shared_prefix'])!=expected:raise ValueError('Same single-RX seed E20 required')
    proxy=dict(c,run_id=paper.RUN,row_id=row['paper_row']['row_id']);proxy.pop('shared_prefix')
    return _validate_paper(proxy)
