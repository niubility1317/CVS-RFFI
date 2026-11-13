"""A/B/C/D source training followed by raw and frozen DURA-LU scoring."""
import argparse
import math
from contextlib import contextmanager
from pathlib import Path

from experiments.cric_dura_comparison.io import read,write
from . import hybrid_structure_plan as plan, hybrid_confidence_runner as common
from .hybrid_structure import HybridStructure
from .practical_gate import installed as practical_scope


@contextmanager
def installed(c,holder):
    from experiments.pamf_paper_completion.model import installed as paper_scope
    from .hybrid_update_probe import installed as probe_scope
    @contextmanager
    def adapted(cfg,state,enabled,objective_factory):
        with paper_scope(cfg,state,ssl_enabled=enabled,objective_factory=objective_factory,validate=plan.validate):
            yield
    with probe_scope() as probe:
        with practical_scope(c,holder,base_scope=adapted,objective_factory=lambda cfg:HybridStructure(cfg,probe)):
            yield


def stage(rt):
    old=common.plan;common.plan=plan
    try:common.stage(rt)
    finally:common.plan=old
    root=Path(rt['run_root']);context=read(rt['context']);jobs=read(root/'jobs.json')
    for job in jobs:
        if job['kind'] not in ('predict','score'):continue
        c=read(job['config']);c['readouts']=['raw','DURA-LU']
        if job['kind']=='predict':c['restored_caches']={'LU':context['caches'][c['protocol_id']]['LU']['path']}
        write(job['config'],c,exclusive=False)
    from experiments.practical_full_2x2.cache import atomic
    state=read(root/'pipeline_state.json')
    for setting,pid,seeds in plan.MATRIX:
        for seed in seeds:
            candidates={r['structure_arm']:r for r in plan.candidate_rows() if r['setting']==setting and r['model_seed']==seed}
            rid=f'{setting}-paired-s{seed}';folder=root/'paired'/rid
            predictions={arm:str(root/'readouts'/row['row_id']/'predictions') for arm,row in candidates.items()}
            config=dict(run_id=plan.RUN,row_id=rid,predictions=predictions,tests=context['handles'][pid]['tests'],
                        readouts=['raw','DURA-LU'],views=candidates['A']['views'],output_root=str(folder))
            cfg=root/'configs'/('score__'+rid+'.json');write(cfg,config)
            job=dict(id='score__'+rid,kind='score',config=str(cfg),completion=str(folder/'complete.json'),
                     successful_statuses=['PAIRED_SCORED_COMPLETE'],
                     dependencies=[str(Path(v)/'complete.json') for v in predictions.values()])
            jobs.append(job);state['jobs'][job['id']]=dict(status='QUEUED')
    write(root/'jobs.json',jobs,exclusive=False);atomic(root/'pipeline_state.json',state)


def check_structure(state):
    from .pair_runner import check_single_rx
    details=check_single_rx(state);aux=state['U_fused_physics'];enabled=aux['auxiliary_enabled']
    if enabled:
        gradients=state['U_hybrid_contrast']['smoke_gradients']
        if aux['extra_student_forwards']!=2 or any(not math.isfinite(v) or v<=0 for d in gradients.values() for v in d.values()):
            raise ValueError('Fusion-only auxiliary forward and gradients required')
    elif aux['extra_student_forwards'] or aux['selected']:
        raise ValueError('Zero-auxiliary controls may not add an auxiliary student forward')
    return dict(details,U_fused_physics=aux,U_tracking=state['U_tracking'])


def smoke(rt):
    from .runner import smoke as smoke_rows
    checks=[]
    for row in (r for r in plan.candidate_rows() if r['model_seed']==plan.SEEDS[0]):
        checks.extend(smoke_rows(rt,rows=lambda:[row],configure=plan.configuration,check_parent=plan.parent,
                                scope=installed,check_u=check_structure,graph_check=plan.check_graph,
                                segment_output=True,write_result=False))
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',checks=checks,real_checkpoint=True,
          optimizer_updates=0,query_reads=0,U_truth_reads=0))


def predict(c):
    from experiments.pamf_single_view_paper import evaluation as native
    from experiments.pamf_paper_completion import evaluation as paper
    old=native.READOUTS;native.READOUTS=('raw','DURA-LU')
    try:return paper.predict(c)
    finally:native.READOUTS=old


def worker(c,kind):
    if kind=='score':
        if 'predictions' in c:
            from .hybrid_paired_score import score
            return score(c)
        from experiments.pamf_paper_completion.evaluation import score
        return score(c)
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    if kind=='smoke':return smoke(c)
    if kind=='train':
        from .source import train_trajectory
        return train_trajectory(c,plan.parent(c),installed,ssl=True,physical_L=False)
    if kind=='predict':return predict(c)
    raise ValueError('Only source training, frozen prediction and independent scoring are allowed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['worker','stage']);p.add_argument('--kind');p.add_argument('--config',required=True)
    a=p.parse_args();c=read(a.config);stage(c) if a.action=='stage' else worker(c,a.kind)
