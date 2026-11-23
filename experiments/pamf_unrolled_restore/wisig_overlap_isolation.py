"""Isolate WiSig edge and overlap effects on a frozen Main split source reader."""

import argparse
from pathlib import Path, PurePosixPath
from time import perf_counter

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.publish import release, release_commit
from experiments.pamf_lora_main.runner import activate

from . import wisig_joint_preamble_source as tri
from . import wisig_uncertain_source as core
from .wisig_anchored_window_source import classify
from .wisig_timing_diagnostic import oracle_readouts, paired_delay
from .wisig_tri_learned_replay import BASELINE_COMMIT, parent
from .wisig_uncertain_prior import as_complex


RUN = '20261124-diagnostic-wisig-main-overlap-source-m2-r01'
ROUTES, VIEWS, BATCH = core.baseline.ROUTES, core.baseline.VIEWS, core.baseline.BATCH
MODES = ('direct', 'r69', 'paired_align', 'r69_edges', 'r69_overlap', 'source_edges',
         'source_edges_scalar', 'r69_overlap_phase', 'r69_overlap_amplitude')
COMPARISONS = (('paired_align','direct'), ('r69_edges','paired_align'), ('r69_overlap','paired_align'),
               ('r69','r69_edges'), ('r69','r69_overlap'), ('source_edges','r69_edges'),
               ('source_edges_scalar','source_edges'), ('r69_overlap_phase','r69_edges'),
               ('r69_overlap_amplitude','r69_edges'), ('r69','r69_overlap_phase'),
               ('r69','r69_overlap_amplitude'), *((mode,'r69') for mode in MODES[2:5]))


def complex_array(iq):
    return iq[:, 0] + 1j * iq[:, 1]


def as_array(wave):
    return np.stack((wave.real, wave.imag), axis=1).astype(np.float32, copy=False)


def extend_hybrids(aligned, restored, valid):
    raw_magnitude, restored_magnitude = np.abs(aligned), np.abs(restored)
    raw_phase = np.divide(aligned, raw_magnitude, out=np.ones_like(aligned), where=raw_magnitude>0.)
    restored_phase = np.divide(restored, restored_magnitude, out=raw_phase.copy(), where=restored_magnitude>0.)
    r69_edges = np.where(valid, aligned, restored)
    r69_overlap = np.where(valid, restored, aligned)
    phase = np.where(valid, raw_magnitude * restored_phase, restored)
    amplitude = np.where(valid, restored_magnitude * raw_phase, restored)
    return r69_edges, r69_overlap, phase, amplitude


def isolated_views(raw, clean, restored):
    (aligned, source_edges, scalar), info = oracle_readouts(raw, clean)
    index = np.arange(raw.shape[-1]) + info['paired_delta'][:, None]
    valid = (index >= 0) & (index < raw.shape[-1])
    r69_edges, r69_overlap, phase, amplitude = extend_hybrids(aligned, restored, valid)
    r69_delay, r69_shifted, r69_zero = paired_delay(restored, clean)
    info.update(r69_delay=r69_delay, r69_shifted=r69_shifted, r69_zero=r69_zero,
                missing=(~valid).sum(-1))
    return (aligned, r69_edges, r69_overlap, source_edges, scalar, phase, amplitude), info


def decision(reader, iq):
    y = as_complex(tensor(iq, 'cuda:0'))
    return classify(reader, y).argmax(-1).tolist()


@torch.no_grad()
def smoke(rt):
    source, _, indices = core.source_iq(Path(rt['source_root']))
    raw = source[indices[:4]]
    f = tri.reader_factory(rt).eval().requires_grad_(False)
    engine = core.baseline.original_r69(Path(rt['parent_root']))
    original, = engine(tensor(raw, 'cuda:0'))
    result = complex_array(np.asarray(original.tolist(), dtype=np.float32))
    z = complex_array(raw)
    modes, _ = isolated_views(z, z, result)
    if not np.array_equal(modes[0], z) or not np.array_equal(modes[2], result):
        raise ValueError('Zero-shift overlap decomposition failed')
    for wave in (z, result, *modes):
        decision(f, as_array(wave))
    write(Path(rt['run_root'])/'smoke.json',
          dict(status='PASS', count=len(raw), source_V_reads=0, target_reads=0, parameters_added=0))


@torch.no_grad()
def evaluate(rt, route):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache, _, _, _, ids, _, _ = core.baseline.handles(rt)
    old = parent(rt, tri.RUN, BASELINE_COMMIT)/'predictions'
    proof = read(old/'complete.json')
    if (proof['status'],proof['modes'],proof['count'],proof['source_truth_reads'],proof['target_reads']) != (
            'FROZEN',list(tri.MODES),len(ids),0,0):
        raise ValueError('Expected same-reader, truth-blind direct and DURA')
    root = Path(rt['run_root'])/'predictions'
    folder = cache/ROUTES[route]
    clean = np.load(folder/'clean.npy', mmap_mode='r', allow_pickle=False)
    f = tri.reader_factory(rt).eval().requires_grad_(False)
    engine = core.baseline.original_r69(Path(rt['parent_root']))
    summaries = {}
    for view in VIEWS:
        key = f'{route}_{view}'
        t0 = perf_counter()
        raw = np.load(folder/f'{view}.npy', mmap_mode='r', allow_pickle=False)
        wanted = (len(ids),2,256)
        if any(x.shape!=wanted or x.dtype!=np.float32 for x in (raw,clean)):
            raise ValueError('Unexpected paired source IQ')
        with np.load(old/f'{key}.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['ids'],ids) or saved['decision'].shape != (len(tri.MODES), len(ids)):
                raise ValueError('Frozen comparator rows differ')
            ref = saved['decision'][[tri.MODES.index('direct'),tri.MODES.index('r69')]]
        batches, details = [], []
        for start in range(0,len(ids),BATCH):
            sl = slice(start,start+BATCH)
            raw_iq = tensor(raw[sl], 'cuda:0')
            restored, = engine(raw_iq)
            z = complex_array(np.asarray(restored.tolist(), dtype=np.float32))
            options, information = isolated_views(complex_array(raw[sl]),complex_array(clean[sl]),z)
            details.append(information)
            low = (classify(f,as_complex(raw_iq)).argmax(-1).tolist(),
                   classify(f,as_complex(restored)).argmax(-1).tolist())
            predictions = [*low, *(decision(f,as_array(y)) for y in options)]
            batches.append(np.asarray(predictions, dtype=np.uint8))
        matrix = np.concatenate(batches,axis=1)
        with (root/f'{key}.npz').open('xb') as stream:
            np.savez(stream, ids=ids, decision=matrix)
        merged = {name:np.concatenate([batch[name] for batch in details]) for name in details[0]}
        summaries[key] = dict(count=len(ids), seconds=perf_counter()-t0,
              comparator_disagreements=(matrix[:2]!=ref).sum(-1).tolist(),
              abs_source_delay_median=float(np.median(np.abs(merged['paired_delta']))),
              source_aligned_cos2_median=float(np.median(merged['aligned_cos2'])),
              r69_abs_delay_median=float(np.median(np.abs(merged['r69_delay']))),
              r69_colocated_within_two=int((np.abs(merged['r69_delay'])<=2).sum()),
              r69_original_cos2_median=float(np.median(merged['r69_zero'])),
              mean_missing=float(merged['missing'].mean()))
        print('WISIG_OVERLAP_FROZEN',key,len(ids),*summaries[key]['comparator_disagreements'],flush=True)
    write(root/f'{route}-complete.json',summaries)


def dispatch(rt):
    root = Path(rt['run_root'])
    root.mkdir(exist_ok=False)
    def state(name, **fields):
        write(root/'state.json',dict(status=name, **fields),exclusive=False)

    def start(action, name, gpu=None):
        return core.baseline.start(rt,action,name,gpu,module=rt['runner'],extra_keys=('recognizer_run',))

    try:
        state('SMOKE')
        if start('smoke','source',core.baseline.select_gpu()).wait():
            raise RuntimeError('Real-checkpoint WiSig isolation smoke failed')
        pred = root/'predictions'
        pred.mkdir()
        first = core.baseline.select_gpu()
        gpus = (first,core.baseline.select_gpu(exclude=(first,)))
        state('EVALUATE',gpus=dict(zip(ROUTES,gpus,strict=True)))
        jobs = [start('evaluate',route,gpu) for route,gpu in zip(ROUTES,gpus,strict=True)]
        if any([process.wait() for process in jobs]):
            raise RuntimeError('WiSig isolated prediction failed')
        reports = {key:value for route in ROUTES for key,value in read(pred/f'{route}-complete.json').items()}
        if set(reports) != {f'{route}_{view}' for route in ROUTES for view in VIEWS} or any(
                value['count']!=1800 for value in reports.values()):
            raise ValueError('Both isolation routes must freeze all rows before truth access')
        write(pred/'complete.json',dict(status='FROZEN',count=1800,modes=list(MODES),views=list(VIEWS),
              source_truth_reads=0,target_reads=0,summaries=reports,paired_source_iq_isolated=True))
        state('SCORE')
        if start('score','source').wait():
            raise RuntimeError('Independent WiSig isolation scoring failed')
        state('SOURCE_SCORED')
    except Exception as error:
        state('FAILED',error=str(error))
        raise


def publish(repo, output, context, ssh_config):
    cfg = read(context)
    rt = {key:cfg[key] for key in ('project','python','frozen_sources')}
    p = PurePosixPath(rt['project'])
    runs = p/'runs'
    rt.update(run_id=RUN, runner=__spec__.name, code_commit=release_commit(repo),
              run_root=str(runs/RUN), release_root=str(p/'releases'/f'{RUN}-release'),
              source_root=str(runs/core.baseline.CACHE_RUN/'cache'), parent_root=str(runs/core.baseline.R17_RUN),
              recognizer_run=str(runs/tri.tri_plan.RUN))
    release(repo,output,rt,ssh_config,archive_extras=('tools/r17_activation_factorization.py',))


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
        publish(**args)
        return
    rt = read(args['runtime'])
    if action=='dispatch':
        dispatch(rt)
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
