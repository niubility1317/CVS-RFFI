"""Train and independently score a fixed-dose LoRa direction update on source IQ."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from json import JSONDecodeError
import os
from pathlib import Path, PurePosixPath
import subprocess
from time import perf_counter, sleep

import numpy as np
import torch

if __package__ is None:
    from experiments import pamf_lora_main

    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.full_packet_views import native, source_contract
from experiments.pamf_lora_main.runner import activate, available
from experiments.pamf_lora_main.w24_aperture_noise_audit import center_rms, model_iq, native_aperture
from experiments.pamf_lora_main.w24_edit_budget_audit import WAVEFORM_NAMESPACE
from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN, load_classifier, panel, score_source
from experiments.pamf_lora_main.w24_fixed_dose_direction_model import (
    ApertureDirection, DirectionHead, direction_objective, matched_aperture_edit,
)
from experiments.pamf_lora_main.w24_native_full import (
    DATA_RUN, SOURCE, VIEWS, W24_COMMIT, W24_RUN, classify_aperture, native_full_view,
)
from experiments.pamf_lora_main.w24_practical_supervision import training_sets
from experiments.pamf_lora_main.w24_third_prior import PRACTICAL_ARMS, PRACTICAL_CACHE_RUN, PRACTICAL_RUN, load_candidate
from experiments.pamf_lora_main.w24_wide_transfer import TRAIN_RUN, load_restorer, received_aperture, training_runtime


RUN = '20261127-phase1-lora-w24-fixed-dose-direction-s2026112991-r01'
EVAL_RUN = '20261127-phase1-lora-w24-fixed-dose-direction-s2026112991-r02'
COMPARATOR = '20261127-diagnostic-lora-w24-practical-direction-factor-s2026112559-r01'
ARM = PRACTICAL_ARMS[1]
SEED, STEPS, BATCH, FIT_PER_RX = 2026112991, 24000, 16, 4000
MODES = ('direct', 'frozen_half', 'fixed_dose_direction')
CONTRASTS = (('fixed_dose_direction', 'frozen_half'),)
FIT_FIELDS = ('project', 'run_root', 'source_root', 'practical_cache', 'training_release_root')
ISOLATION = dict(F_reads=0, TX_reads=0, V_reads=0, target_reads=0)


def source_parent(rt, device):
    import experiments

    runs = Path(rt['project'])/'runs'
    root = runs/PRACTICAL_RUN
    if read(root/'state.json') != dict(status='SOURCE_FROZEN', arms=list(PRACTICAL_ARMS), steps=STEPS):
        raise ValueError('Expected both terminal Practical r02 source checkpoints')
    contract = source_contract(Path(rt['source_root']))
    # The archived loader temporarily prepends its module directory.
    if not isinstance(experiments.__path__, list):
        experiments.__path__ = list(experiments.__path__)
    load_restorer(*training_runtime(rt), contract, device)
    model, info = load_candidate(root, contract, device, ARM)
    if info['steps'] != STEPS or any(info[k] != 0 for k in ISOLATION):
        raise ValueError('Practical exact-additive source provenance differs')
    return model


def dtype_policy():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def fit_root(rt):
    if set(rt) != set(FIT_FIELDS):
        raise ValueError('Fit accepts the registered U-only paths and no evaluation handles')
    root, project = Path(rt['run_root']), Path(rt['project'])
    if (root != project/'runs'/RUN or Path(rt['practical_cache']) !=
            project/'runs'/PRACTICAL_CACHE_RUN/'source_practical'):
        raise ValueError('Expected the immutable Practical source bank and owned run')
    return root


def smoke_source(rt):
    root = fit_root(rt)
    torch.set_num_threads(2)
    parent = source_parent(rt,'cpu')
    torch.manual_seed(SEED)
    head, aperture = DirectionHead().eval(), ApertureDirection().eval()
    (smoke, _, _), _ = training_sets(rt,'cpu',smoke=True,fit_per_rx=FIT_PER_RX,bank_per_rx=0)
    _, y, s = smoke[:BATCH].unbind(1)
    with torch.no_grad():
        base = torch.lerp(y,parent.restore_native(y),.5)-y
        loss, _ = direction_objective(aperture,y,base,head(y,base),s)
    if not torch.isfinite(loss):
        raise ValueError('Real frozen Practical checkpoint source smoke failed')
    write(root/'fit-smoke.json',dict(status='PASS',parent=PRACTICAL_RUN,source_only=True,device='cpu',**ISOLATION))


def fit_source(rt):
    root = fit_root(rt)
    device = 'cuda:0'
    dtype_policy()
    parent = source_parent(rt, device)
    torch.manual_seed(SEED)
    head = DirectionHead().to(device)
    aperture = ApertureDirection().to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-4)
    lr = torch.optim.lr_scheduler.CosineAnnealingLR(optim, STEPS, eta_min=2e-5)

    def evaluate(batch, with_identity=False):
        clean, y, s = batch.unbind(1)
        mask = None
        if with_identity:
            y, s = (torch.cat((w[:-2], clean[-2:])) for w in (y, s))
            mask = torch.arange(len(y), device=device) >= len(y)-2
        with torch.no_grad():
            base = torch.lerp(y, parent.restore_native(y), .5) - y
        edit = head(y, base)
        return direction_objective(aperture, y, base, edit, s, clean_mask=mask)

    (train, _, held), _ = training_sets(rt, device, smoke=False, fit_per_rx=FIT_PER_RX, bank_per_rx=0)
    if (len(train), len(held)) != (20000,500):
        raise ValueError('Unexpected source fit and held physical counts')
    rng = np.random.default_rng(SEED+1)
    trajectory = {}
    start = perf_counter()
    for step in range(1,STEPS+1):
        rows = rng.choice(len(train), BATCH, replace=False)
        index = tensor(rows, device)
        optim.zero_grad(set_to_none=True)
        loss, diagnostics = evaluate(train[index], with_identity=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optim.step()
        lr.step()
        if step == 1 or step % 200 == 0:
            print('W24_FIXED_DOSE_DIRECTION_FIT',step,float(loss.detach()),
                  {k:float(v) for k,v in diagnostics.items()},flush=True)
        if step in (2000,6000,12000,18000,STEPS):
            with torch.no_grad():
                checks = [(len(group),evaluate(group)[1]) for group in held.split(BATCH)]
            trajectory[str(step)] = {k:sum(n*float(row[k]) for n,row in checks)/len(held) for k in checks[0][1]}
            print('W24_FIXED_DOSE_DIRECTION_HELD',step,trajectory[str(step)],flush=True)
    info = dict(status='SOURCE_FROZEN', parent=PRACTICAL_RUN, parent_arm=ARM, seed=SEED, batch=BATCH,
                steps=STEPS, fit=len(train), held=len(held), initialization='random_direction_head',
                parameters=sum(p.numel() for p in head.parameters()), seconds=perf_counter()-start,
                peak_cuda=int(torch.cuda.max_memory_allocated()), held_trajectory=trajectory, **ISOLATION)
    with (root/'direction.pt').open('xb') as f:
        torch.save(dict(weights=head.state_dict(), metadata=info),f)
    write(root/'fit-complete.json',info)


def load_head(root):
    info = read(root/'fit-complete.json')
    checkpoint = torch.load(root/'direction.pt', map_location='cpu', weights_only=True)
    if (checkpoint['metadata'] != info or info['status'] != 'SOURCE_FROZEN' or info['steps'] != STEPS
            or info['parent'] != PRACTICAL_RUN or info['parent_arm'] != ARM
            or any(info[k] != 0 for k in ISOLATION)):
        raise ValueError('Expected the fixed-dose direction terminal source checkpoint')
    model = DirectionHead().eval().requires_grad_(False)
    model.load_state_dict(checkpoint['weights'],strict=True)
    return model, info


@torch.no_grad()
def aperture_components(parent, model, y):
    wave = tensor(np.ascontiguousarray(y).view(np.float32),'cpu').view(torch.complex64)
    frozen = torch.lerp(wave,parent.restore_native(wave),.5)
    proposal = wave + model(wave, frozen-wave)
    a, b, d = (native_aperture(np.from_dlpack(z.contiguous())) for z in (wave, frozen, proposal))
    base, proposed = (tensor(np.ascontiguousarray(z,dtype=np.complex64).view(np.float32),'cpu').view(torch.complex64)
                      for z in (b-a,d-a))
    return wave, a, b, matched_aperture_edit(base,proposed)


@torch.no_grad()
def readouts(parent, model, y):
    _, a, b, edit = aperture_components(parent, model, y)
    candidate = a + np.from_dlpack(edit.contiguous())
    result = tuple(tensor(model_iq(v,center_rms(v)),'cpu') for v in (a,b,candidate))
    return result


def source_observation(scene, item):
    row, receiver, packet = item
    return native_full_view(packet, scene, receiver, row, WAVEFORM_NAMESPACE)


def checkpoint_root(rt):
    root = Path(rt['run_root'])
    runs = Path(rt['project'])/'runs'
    if rt['run_id'] == RUN and root == runs/RUN:
        return root
    if rt['run_id'] == EVAL_RUN and root == runs/EVAL_RUN:
        checkpoint = Path(rt['fixed_checkpoint_root'])
        if checkpoint == runs/RUN:
            return checkpoint
    raise ValueError('Unexpected fixed-dose source checkpoint binding')


def wait_for_checkpoint(root):
    while True:
        if (root/'direction.pt').is_file():
            try:
                proof = read(root/'fit-complete.json')
            except (FileNotFoundError, JSONDecodeError):
                pass
            else:
                if proof.get('status') != 'SOURCE_FROZEN' or proof.get('steps') != STEPS:
                    raise ValueError('Expected a complete terminal source checkpoint')
                return
        try:
            failed = read(root/'state.json')['status'] == 'FAILED'
        except (FileNotFoundError, JSONDecodeError):
            failed = False
        if failed:
            raise RuntimeError('Original source fit exited before checkpoint freeze')
        sleep(20)


def predict_source(rt):
    activate(rt)
    torch.set_num_threads(2)
    root, source = (Path(rt[k]) for k in ('run_root','source_root'))
    parent = source_parent(rt,'cpu')
    fit = checkpoint_root(rt)
    model, info = load_head(fit)
    contract = source_contract(source)
    f = load_classifier(rt,contract,'cpu')
    legal_u = native(source,'U_s',contract['counts']['U_s'])[:1]
    raw = source_observation(VIEWS[-1],(0,SOURCE[0],legal_u[0]))[None]
    direct, baseline, candidate = readouts(parent,model,raw)
    torch.testing.assert_close(direct, received_aperture(raw,'cpu'),rtol=0,atol=0)
    classify_aperture(f,candidate)
    wave = tensor(np.ascontiguousarray(raw).view(np.float32),'cpu').view(torch.complex64)
    with torch.inference_mode():
        frozen = torch.lerp(wave,parent.restore_native(wave),.5)
    torch.testing.assert_close(baseline, received_aperture(np.from_dlpack(frozen),'cpu'),rtol=0,atol=0)
    write(root/'smoke.json',dict(status='PASS',classifier_epoch=104,fit_steps=info['steps'],source_u=1,
                                target_reads=0,source_truth_reads=0))
    rows, rx, ids = panel(source,contract)
    ref = root.parent/COMPARATOR
    marker = read(ref/'source-predictions.json')
    if (marker['status'],marker['physical_ids'],marker['views'],marker['target_reads']) != (
        'PREDICTIONS_FROZEN',ids,list(VIEWS),0):
        raise ValueError('Source exact-dose matrix does not share the previous fixed panel')
    packets = native(source,'V',contract['counts']['V'])[rows]
    jobs = [(int(row),SOURCE[int(r)],x) for row,r,x in zip(rows,rx,packets,strict=True)]
    output={}
    with (ProcessPoolExecutor(max_workers=2) as pool,
          np.load(ref/'predictions.npz',allow_pickle=False) as previous):
        for scene in VIEWS:
            stream = pool.map(partial(source_observation,scene),jobs,chunksize=6)
            batches = []
            for i in range(0,len(jobs),BATCH):
                y=np.stack([next(stream) for _ in jobs[i:i+BATCH]])
                batches.append(np.stack([classify_aperture(f,z) for z in readouts(parent,model,y)]))
            output[scene]=np.concatenate(batches,axis=1)
            old=[marker['modes'].index(m) for m in MODES[:2]]
            if not np.array_equal(output[scene][:2],previous[scene][old]):
                raise ValueError(f'Source matched controls changed on {scene}')
            print('W24_FIXED_DOSE_DIRECTION_PREDICTED',scene,len(jobs),flush=True)
    with (root/'predictions.npz').open('xb') as f:
        np.savez(f,**output)
    write(root/'source-predictions.json',dict(status='PREDICTIONS_FROZEN', views=list(VIEWS), modes=list(MODES),
          physical_ids=ids, target_reads=0, source_truth_reads=0, waveform=dict(parent=PRACTICAL_RUN,
               fit_run=fit.name, dose_rule='fixed_center_edge_norm_before_w24_rms', device='cpu', received_only=True)))


def eligible_from_source(scores):
    ref, candidate = (scores['scores'][m] for m in (MODES[1],MODES[2]))
    direct = scores['scores'][MODES[0]]
    contrasted = scores['paired_contrasts'][f'{MODES[2]}_vs_{MODES[1]}']
    return (candidate['six_correct'] >= ref['six_correct']+18
            and candidate['low_urban_correct'] >= ref['low_urban_correct']+6
            and candidate['by_scene']['clean'] >= direct['by_scene']['clean']
            and contrasted['CI95_net_six'][0] > 0)


def dispatch(runtime):
    rt = read(runtime)
    root, reference = (Path(rt[k]) for k in ('run_root','w24_release_root'))
    root.mkdir()
    state = root/'state.json'
    if read(reference/'submit.json')['code_commit'] != W24_COMMIT:
        raise ValueError('Expected pinned W24 evaluator archive')
    script = str(Path(__file__).resolve())
    own = Path(script).parents[2]

    def stage(phase,config,cwd,gpu=None):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu), PYTHONPATH=str(cwd),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (root/f'{phase}.log').open('x',encoding='utf-8') as log:
            job = subprocess.Popen([rt['python'],'-u',script,phase,str(config)],cwd=cwd,env=env,
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write(state,dict(status='RUNNING',phase=phase,pid=job.pid,gpu=gpu),exclusive=False)
        print('W24_FIXED_DOSE_DIRECTION_STARTED',phase,job.pid,gpu,flush=True)
        if job.wait():
            raise RuntimeError(f'{phase} failed')

    try:
        fit = checkpoint_root(rt)
        if fit == root:
            body = root/'fit-inputs.json'
            write(body,{k:rt[k] for k in FIT_FIELDS})
            stage('smoke-source',body,own)
            write(state,dict(status='WAIT_GPU_TRAINING'),exclusive=False)
            while not (cards := available(rt['allowed_gpus'],2,training_cap=True)):
                sleep(20)
            gpu = next(g for g in rt['allowed_gpus'] if g in cards)
            stage('fit-source',body,own,gpu)
        else:
            write(state,dict(status='WAIT_SOURCE_FREEZE',fit_run=fit.name))
            wait_for_checkpoint(fit)
        stage('predict-source',runtime,reference)
        stage('score-source',runtime,reference)
        promote = eligible_from_source(read(root/'source-scores.json'))
        write(state,dict(status='SOURCE_SCORED',promote=promote,target_reads=0),exclusive=False)
    except Exception as exc:
        write(state,dict(status='FAILED',error=str(exc)),exclusive=False)
        raise


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    for phase in ('dispatch','smoke-source','fit-source','predict-source','score-source'):
        sub.add_parser(phase).add_argument('runtime',type=Path)
    pub = sub.add_parser('publish')
    pub.add_argument('--frozen-source-eval',action='store_true')
    for name in ('repo','context','output','ssh-config'):
        pub.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.action == 'publish':
        ctx = read(args.context)
        rt = {k:ctx[k] for k in ('project','python','frozen_sources')}
        project = PurePosixPath(rt['project'])
        releases,runs = project/'releases',project/'runs'
        run = EVAL_RUN if args.frozen_source_eval else RUN
        rt.update(run_id=run,runner=__spec__.name,code_commit=release_commit(args.repo),
                  release_root=str(releases/f'{run}-release'),run_root=str(runs/run),
                  source_root=str(releases/f'{DATA_RUN}-release'/'data/source'),classifier_root=str(runs/AUG_RUN),
                  w24_release_root=str(releases/f'{W24_RUN}-release'),
                  training_release_root=str(releases/f'{TRAIN_RUN}-release'),
                  practical_cache=str(runs/PRACTICAL_CACHE_RUN/'source_practical'),
                  allowed_gpus=[0,7,2,3,6,5,1,4])
        rt['external_releases']=[rt[k] for k in ('source_root','classifier_root','w24_release_root',
             'training_release_root','practical_cache')]+[str(runs/r) for r in (PRACTICAL_RUN,COMPARATOR)]
        if args.frozen_source_eval:
            rt['fixed_checkpoint_root'] = str(runs/RUN)
            rt['external_releases'].append(rt['fixed_checkpoint_root'])
        release(args.repo,args.output,rt,args.ssh_config)
    elif args.action == 'dispatch':
        dispatch(args.runtime)
    elif args.action == 'smoke-source':
        smoke_source(read(args.runtime))
    elif args.action == 'fit-source':
        fit_source(read(args.runtime))
    elif args.action == 'predict-source':
        predict_source(read(args.runtime))
    else:
        score_source(args.runtime,modes=MODES,contrasts=CONTRASTS)


if __name__ == '__main__':
    main()
