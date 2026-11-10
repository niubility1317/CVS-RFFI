import argparse
from functools import partial
from pathlib import Path
from experiments.cric_dura_comparison.io import read,write
from experiments.pamf_physics_tri_u import runner as physical
from experiments.pamf_physics_branch import runner as evaluation
from experiments.pamf_physics_route import runner as dispatching
from experiments.pamf_repl_view_shuffle.source import seed_scope
from .plan import candidate_rows,configuration,paper_row
from .source import parent,train
from .prepare import prepare


def smoke(rt):
    old=physical.candidate_rows,physical.configuration,physical.parent
    physical.candidate_rows,physical.configuration,physical.parent=candidate_rows,configuration,parent
    try:
        physical.smoke(rt)
        import numpy as np
        import torch
        from experiments.cric_dura_comparison.io import tensor
        from experiments.pamf_paper_completion.model import load_reader
        from experiments.pamf_repl_view_shuffle.plan import RUN as ORIGINAL_RUN
        ctx=read(rt['context']);checks=[]
        for row in candidate_rows():
            c=configuration(ctx,row,rt['project'],rt['code_commit']);seed=row['model_seed']
            folder=Path(rt['project'])/'runs'/ORIGINAL_RUN/'F'/f'main_split-repl-shuffled_time-3view-s{seed}'/'source'
            with seed_scope():
                reader,_=load_reader(folder,read(c['source_contract']),paper_row(seed),'cuda:0')
            x=np.load(Path(c['source_cache'])/'L_s_x.npy',mmap_mode='r',allow_pickle=False)[:2]
            with torch.inference_mode():logits=reader.classify_features(reader.features(tensor(x,'cuda:0')))
            if logits.shape!=(2,6) or not bool(torch.isfinite(logits).all()):
                raise ValueError('Original same-seed E50 prediction reader smoke failed')
            checks.append(dict(seed=seed,real_original_E50=True,finite_logits=True,source_only=True))
            del reader,logits;torch.cuda.empty_cache()
        path=Path(rt['run_root'])/'smoke.json';result=read(path);result['original_baseline_checks']=checks
        write(path,result,exclusive=False)
    finally:physical.candidate_rows,physical.configuration,physical.parent=old


def worker(c,kind):
    if kind=='score':return evaluation.worker(c,kind)
    if kind=='predict':
        with seed_scope():return evaluation.worker(c,kind)
    if kind not in ('train','smoke'):raise ValueError('Only source and practical evaluation stages allowed')
    from experiments.pamf_repl_completion.runner import activate
    activate(c)
    return train(c) if kind=='train' else smoke(c)


def dispatch(rt):
    launch=partial(dispatching.launch,module='experiments.pamf_tri_u_extra.runner')
    dispatching.dispatch(rt,prepare_run=prepare,launch_worker=launch)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch','worker'])
    p.add_argument('--kind');p.add_argument('--config',required=True);a=p.parse_args()
    dispatch(read(a.config)) if a.action=='dispatch' else worker(read(a.config),a.kind)
