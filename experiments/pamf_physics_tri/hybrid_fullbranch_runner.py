"""Matched fusion-plus-five-branch control for the fusion-only C condition."""
import argparse
from contextlib import contextmanager
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from . import hybrid_fullbranch_plan as plan,hybrid_confidence_runner as common
from .hybrid_structure import HybridStructure
from .practical_gate import installed as practical_scope


@contextmanager
def installed(c,holder):
    from experiments.pamf_paper_completion.model import installed as paper_scope
    from .hybrid_update_probe import installed as probe_scope
    @contextmanager
    def adapted(cfg,state,enabled,objective_factory):
        with paper_scope(cfg,state,ssl_enabled=enabled,objective_factory=objective_factory,validate=plan.validate):yield
    with probe_scope() as probe:
        with practical_scope(c,holder,base_scope=adapted,objective_factory=lambda cfg:HybridStructure(cfg,probe)):yield


def stage(rt):
    old=common.plan;common.plan=plan
    try:common.stage(rt)
    finally:common.plan=old
    from experiments.practical_full_2x2.cache import atomic
    root=Path(rt['run_root']);project=Path(rt['project']);ctx=read(rt['context']);jobs=read(root/'jobs.json');state=read(root/'pipeline_state.json')
    for job in jobs:
        if job['kind'] not in ('predict','score'):continue
        c=read(job['config']);c['readouts']=['raw','DURA-LU']
        if job['kind']=='predict':c['restored_caches']={'LU':ctx['caches'][c['protocol_id']]['LU']['path']}
        write(job['config'],c,exclusive=False)
    prior=project/'runs'/'20261113-phase1-pamf-hybrid-structure-abcd-m16-r01'
    for row in plan.candidate_rows():
        rid=row['row_id'];c_row=f"{row['setting']}-hybrid-structure-C-s{row['model_seed']}"
        paths={'C':str(prior/'readouts'/c_row/'predictions'),'E':str(root/'readouts'/rid/'predictions')}
        output=root/'paired'/rid;cfg=root/'configs'/('score__paired-'+rid+'.json')
        config=dict(run_id=plan.RUN,row_id='paired-'+rid,predictions=paths,base_arm='C',compare_arms=['E'],
                    tests=ctx['handles'][row['protocol_id']]['tests'],readouts=['raw','DURA-LU'],
                    views=row['views'],output_root=str(output))
        write(cfg,config)
        job=dict(id='score__paired-'+rid,kind='score',config=str(cfg),completion=str(output/'complete.json'),
                 successful_statuses=['PAIRED_SCORED_COMPLETE'],dependencies=[str(Path(v)/'complete.json') for v in paths.values()]
                    + [str(prior/'readouts'/c_row/'scores'/'complete.json')])
        jobs.append(job);state['jobs'][job['id']]=dict(status='QUEUED')
    write(root/'jobs.json',jobs,exclusive=False);atomic(root/'pipeline_state.json',state)


def smoke(rt):
    from .runner import smoke as smoke_rows
    from .hybrid_structure_runner import check_structure
    checks=[]
    for row in (r for r in plan.candidate_rows() if r['model_seed']==plan.SEEDS[0]):
        checks.extend(smoke_rows(rt,rows=lambda:[row],configure=plan.configuration,check_parent=plan.parent,
                                scope=installed,check_u=check_structure,graph_check=plan.check_graph,
                                segment_output=True,write_result=False))
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',checks=checks,real_checkpoint=True,
          optimizer_updates=0,query_reads=0,U_truth_reads=0))


def worker(c,kind):
    if kind=='score':
        if 'predictions' in c:
            from .hybrid_paired_score import score
        else:
            from experiments.pamf_paper_completion.evaluation import score
        return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind=='smoke':return smoke(c)
    if kind=='train':
        from .source import train_trajectory
        return train_trajectory(c,plan.parent(c),installed,ssl=True,physical_L=False)
    if kind=='predict':
        from .hybrid_structure_runner import predict
        return predict(c)
    raise ValueError('Only source training, frozen prediction and independent scoring are allowed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['worker','stage']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);stage(c) if a.action=='stage' else worker(c,a.kind)
