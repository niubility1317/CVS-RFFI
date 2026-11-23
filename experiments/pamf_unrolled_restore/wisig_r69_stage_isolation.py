"""Score original R69 intermediate readouts without changing its weights or solve."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_action_restore.models import SCALE
from experiments.pamf_lora_main.runner import activate

from . import wisig_joint_preamble_source as tri
from . import wisig_overlap_isolation as existing
from . import wisig_uncertain_source as core
from .wisig_anchored_window_source import classify
from .wisig_tri_learned_replay import BASELINE_COMMIT, parent
from .wisig_uncertain_prior import as_complex


RUN = '20261124-diagnostic-wisig-r69-stages-source-m2-r01'
ROUTES, VIEWS, BATCH = existing.ROUTES, existing.VIEWS, existing.BATCH
MODES = ('direct','r69','coarse','prior_1','map_1','prior_2')
COMPARISONS = (('coarse','direct'),('prior_1','coarse'),('map_1','prior_1'),
               ('prior_2','map_1'),('r69','prior_2'),('r69','map_1'),
               *((mode,'r69') for mode in MODES[2:]))


class StageReadout:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, received):
        records = []

        def observe(_module, args, residual):
            _,base,previous = args[0][:,:6].split(2,dim=1)
            records.append((base,previous,base+SCALE*residual))

        handles = [layer.register_forward_hook(observe) for layer in self.pipeline.model.prior]
        try:
            final,valid = self.pipeline(received)
        finally:
            for handle in handles:
                handle.remove()
        (base,initial,prior_1),(_,map_1,prior_2) = records
        if not bool(valid) or not torch.equal(base, initial):
            raise ValueError('The first R69 update must start from its coarse waveform')
        return final,base,prior_1,map_1,prior_2


def decide(reader, output):
    return [classify(reader,as_complex(x)).argmax(-1).tolist() for x in output]


@torch.no_grad()
def smoke(rt):
    source,_,indices = core.source_iq(Path(rt['source_root']))
    received = tensor(source[indices[:4]],'cuda:0')
    reader = tri.reader_factory(rt).eval().requires_grad_(False)
    engine = core.baseline.original_r69(Path(rt['parent_root']))
    stages = StageReadout(engine.pipeline)(received)
    graph, = engine(received)
    torch.testing.assert_close(stages[0],graph,rtol=5e-5,atol=5e-6)
    if decide(reader,(graph,))[0]!=decide(reader,(received,*stages))[1]:
        raise ValueError('Hooked R69 final decisions differ from the original Graph-RR engine')
    write(Path(rt['run_root'])/'smoke.json',dict(status='PASS',count=len(received),source_V_reads=0,
          target_reads=0,parameters_added=0))


@torch.no_grad()
def evaluate(rt, route):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache,_,_,_,ids,_,_ = core.baseline.handles(rt)
    prior = parent(rt,tri.RUN,BASELINE_COMMIT)/'predictions'
    binding = read(prior/'complete.json')
    if (binding['status'],binding['modes'],binding['count'],binding['source_truth_reads'],binding['target_reads']) != (
            'FROZEN',list(tri.MODES),len(ids),0,0):
        raise ValueError('Expected existing same-reader blind baseline predictions')
    reader = tri.reader_factory(rt).eval().requires_grad_(False)
    engine = core.baseline.original_r69(Path(rt['parent_root']))
    inspect = StageReadout(engine.pipeline)
    output = Path(rt['run_root'])/'predictions'
    summary = {}
    for view in VIEWS:
        started = perf_counter()
        key = f'{route}_{view}'
        wave = np.load(cache/ROUTES[route]/f'{view}.npy',mmap_mode='r',allow_pickle=False)
        with np.load(prior/f'{key}.npz',allow_pickle=False) as existing_decisions:
            if not np.array_equal(existing_decisions['ids'],ids) or wave.shape!=(len(ids),2,256) or wave.dtype!=np.float32:
                raise ValueError('R69 stage audit source rows differ')
            expected = existing_decisions['decision'][[tri.MODES.index(name) for name in MODES[:2]]]
        decisions, edits, max_error = [], [], 0.
        for start in range(0,len(wave),BATCH):
            received = tensor(wave[start:start+BATCH],'cuda:0')
            final, coarse, p1, x1, p2 = inspect(received)
            graph, = engine(received)
            torch.testing.assert_close(final,graph,rtol=5e-5,atol=5e-6)
            max_error = max(max_error,float((final-graph).abs().max()))
            decisions.append(decide(reader,(received,final,coarse,p1,x1,p2)))
            sequence = (received,coarse,p1,x1,p2,final)
            edits.append(torch.stack([(b-a).square().mean((1,2)).sqrt()
                                     for a,b in zip(sequence[:-1],sequence[1:],strict=True)]))
        matrix = np.concatenate(decisions,axis=1).astype(np.uint8)
        if matrix.shape!=(len(MODES),len(ids)) or not np.array_equal(matrix[:2],expected):
            raise ValueError('The hooked path must reproduce original frozen direct and R69 decisions')
        with (output/f'{key}.npz').open('xb') as stream:
            np.savez(stream,ids=ids,decision=matrix)
        summary[key] = dict(count=len(ids),max_graph_iq_difference=max_error,
                            median_edit_rms=torch.cat(edits,dim=1).median(1).values.tolist(),
                            seconds=perf_counter()-started)
        print('WISIG_OVERLAP_FROZEN',key,len(ids),max_error,flush=True)
    write(output/f'{route}-complete.json',summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action',required=True)
    pub = actions.add_parser('publish')
    for key in ('repo','output','context','ssh-config'):
        pub.add_argument('--'+key,type=Path,required=True)
    for action in ('dispatch','smoke','evaluate','score'):
        worker = actions.add_parser(action)
        worker.add_argument('runtime',type=Path)
        if action!='dispatch':
            worker.add_argument('name',choices=ROUTES if action=='evaluate' else ('source',))
    args = vars(parser.parse_args())
    action = args.pop('action')
    if action=='publish':
        existing.publish(**args,run=RUN,runner=__spec__.name)
        return
    rt = read(args['runtime'])
    if action=='dispatch':
        existing.dispatch(rt,modes=MODES,paired_source_iq_isolated=False)
    elif action=='score':
        core.baseline.score(rt,MODES,comparisons=COMPARISONS)
    else:
        activate(rt)
        if action=='smoke':
            smoke(rt)
        else:
            evaluate(rt,args['name'])


if __name__=='__main__':
    main()
