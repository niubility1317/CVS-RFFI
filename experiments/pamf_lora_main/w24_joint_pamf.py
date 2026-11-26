"""Matched source study of differentiable LoRa recovery and PAMF W24."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial
import importlib
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from time import monotonic, sleep

import numpy as np
import torch
from torch.nn import functional as F

# Spawned CPU replay workers must import from the same archived W24 CWD
# before the parent's activated historical paths.
if __name__ == '__mp_main__':
    sys.path.insert(0, str(Path.cwd()))

if not __package__:
    from experiments import pamf_lora_main
    pamf_lora_main.__path__.append(str(Path(__file__).resolve().parent))

from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.dura_repair import real_iq
from experiments.pamf_lora_main.full_packet_views import native, source_contract, source_role, target_contract
from experiments.pamf_lora_main.packet_runner import schedule
from experiments.pamf_lora_main.runner import activate, available
from experiments.pamf_lora_main.w24_edit_cross import check_controls
from experiments.pamf_lora_main.w24_factor_audit import AUG_RUN, load_classifier, panel, score_source
from experiments.pamf_lora_main.w24_formal import FULL, RUN_OUTER, append_blind_prediction, receipt, wide_delta
from experiments.pamf_lora_main.w24_native_full import (
    DATA_RUN, LIMIT, SOURCE, VIEWS, W24_COMMIT, W24_RUN, channel_job, classify_aperture,
)
from experiments.pamf_lora_main.w24_observed_dose import FULL_CASES, prior_target, source_receive
from experiments.pamf_lora_main.w24_third_prior import UNET_FIDELITY_ARMS, UNET_FIDELITY_RUN, load_candidate
from experiments.pamf_lora_main.w24_wide_transfer import TRAIN_RUN, load_restorer, received_aperture, training_runtime
from experiments.pamf_lora_main.w24_outer_objective_source import RUN as DURA_RUN


RUN = '20261126-phase1-lora-w24-joint-pamf-s2026112991-r01'
RUN_PL = '20261126-phase1-lora-w24-joint-pamf-clean-pl-s2026112991-r01'
ARMS = ('pamf_only', 'frozen_pamf_ce', 'joint')
PL = 'joint_clean_pl'
PL_THRESHOLD = .95
SEED, STEPS, BATCH, PRED_BATCH = 2026112991, 6000, 16, 32
CENTER, CANDIDATE = 384, 'frozen_pamf_ce'
MODES = ('direct', 'outer_half', 'unet_parent', *ARMS, 'joint_g_fixed_e104', 'joint_f_parent_g')
PL_MODES = (*MODES, PL)
CONTRASTS = (('joint','pamf_only'), ('joint','frozen_pamf_ce'), ('joint','unet_parent'),
             ('joint','joint_g_fixed_e104'), ('joint','joint_f_parent_g'),
             ('frozen_pamf_ce','unet_parent'), ('frozen_pamf_ce','outer_half'))
FIT_FIELDS = ('project','frozen_sources','run_root','source_root','classifier_root','training_release_root','unet_parent')


def allowed_fit_runtime(rt, root):
    return {**{k: rt[k] for k in FIT_FIELDS}, 'run_root': str(root)}


def training_labels(source, contract):
    # Only L_s may supply truth during fit.
    with np.load(source/'L_s_metadata.npz', allow_pickle=False) as meta:
        labels = meta['y']
        if meta['ids'].tolist() != contract['role_ids']['L_s'] or labels.shape != (contract['counts']['L_s'],):
            raise ValueError('Invalid registered source L_s target mapping')
    if labels.dtype.kind not in 'iu' or (labels.min(), labels.max()) != (0, 9):
        raise ValueError('Expected the ten source classifier labels')
    return labels.astype(np.int64, copy=False)


def normalize_aperture(iq):
    scale = iq[..., :CENTER].abs().square().mean(dim=(1,2), keepdim=True).clamp_min(1e-12).sqrt()
    return real_iq(iq/scale), scale


def recovered(g, tracker, observation):
    update = .5 * wide_delta(g, tracker, observation)
    return torch.cat((observation[..., :CENTER]+update, observation[..., CENTER:]), dim=-1)


def boot(rt, device):
    activate(rt)
    project = Path(rt['project'])
    if (Path(rt['unet_parent']) != project/'runs'/UNET_FIDELITY_RUN
            or Path(rt['classifier_root']) != project/'runs'/AUG_RUN):
        raise ValueError('Expected exact lawful source parent and registered E104')
    contract = source_contract(Path(rt['source_root']))
    _, tracker, _, _ = load_restorer(*training_runtime(rt), contract, device)
    classifier = load_classifier(rt, contract, device)
    g, _ = load_candidate(Path(rt['unet_parent']), contract, device, UNET_FIDELITY_ARMS[0])
    return contract, classifier, g, tracker


def clean_pseudo_loss(f, anchor, clean_u, restored_u):
    with torch.no_grad():
        confidence, pseudo = anchor(clean_u)[0].softmax(-1).max(-1)
        active = confidence >= PL_THRESHOLD
    accepted = active.sum()
    if not active.any():
        return confidence.new_zeros(()), accepted
    loss = F.cross_entropy(f(restored_u[active])[0],pseudo[active],reduction='sum') / len(clean_u)
    return loss, accepted


def loss_parts(g, f, anchor, tracker, generic, raw, label, rng, *, pseudo=False):
    labeled = len(label)
    if labeled == 0 or len(raw) != 2 * labeled:
        raise ValueError('Expected equal labeled and unlabeled IQ groups, with labels only for the first group')
    raw = raw / raw.abs().square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt()
    clean = tracker.aperture(raw)
    disturbed = tracker.aperture(generic(raw.unflatten(-1,(8,1024)), rng).flatten(-2))
    base, scale = normalize_aperture(disturbed)
    restore = recovered(g, tracker, base)
    clean_input, _ = normalize_aperture(clean[:labeled])
    clean_out = recovered(g, tracker, clean_input)
    f_restore, f_clean, f_clean_out = f(torch.cat((restore[:labeled], clean_input, clean_out)))[0].split(labeled)
    with torch.no_grad():
        teacher = anchor(clean_input)[0].softmax(-1)
    restore_ce = F.cross_entropy(f_restore, label)
    clean_ce = .5 * (F.cross_entropy(f_clean, label) + F.cross_entropy(f_clean_out, label))
    anchor_kl = .5 * (F.kl_div(f_clean.log_softmax(-1),teacher,reduction='batchmean')
                      + F.kl_div(f_clean_out.log_softmax(-1),teacher,reduction='batchmean'))
    target = real_iq(clean/scale)[..., :CENTER]
    change = (restore[..., :CENTER]-target).permute(0,1,3,2).contiguous()
    ref = target.permute(0,1,3,2).contiguous()
    error = torch.fft.fft(torch.view_as_complex(change),norm='ortho')[...,1:].abs().square()
    power = torch.fft.fft(torch.view_as_complex(ref),norm='ortho')[...,1:].abs().square()
    spectral = (error/(.001+power+.03*power.mean(-1,keepdim=True))).log1p().mean()
    amplitude = change.square().mean()
    identity = (clean_out-clean_input).square().mean()
    parts = (restore_ce,clean_ce,anchor_kl,spectral,amplitude,identity)
    if pseudo:
        clean_u, _ = normalize_aperture(clean[labeled:])
        parts += clean_pseudo_loss(f,anchor,clean_u,restore[labeled:])
    return torch.stack(parts)


def norm_gradient(values):
    return sum(float(x.detach().float().square().sum()) for x in values if x is not None)**.5


def fit(config, arm, *, smoke, device='cuda:0'):
    rt = read(config)
    if set(rt) != set(FIT_FIELDS):
        raise ValueError('Unexpected non-source input in joint training')
    out = Path(rt['run_root'])
    source = Path(rt['source_root'])
    contract, f, g, tracker = boot(rt, device)
    generic = importlib.import_module('lora_wide_dura_archive.dura_w24_model').condition_waveforms
    labeled, _, _, _ = source_role(source, contract, 'L_s')
    unlabeled, _, _, _ = source_role(source, contract, 'U_s')
    labels = training_labels(source,contract)
    anchor = deepcopy(f)
    learn_g, learn_f = arm != 'pamf_only', arm != 'frozen_pamf_ce'
    g.requires_grad_(learn_g).train(learn_g)
    # Train the selected W24 weights against its deployed normalization and dropout behavior.
    f.requires_grad_(learn_f).eval()
    groups = []
    if learn_g:
        groups.append(dict(params=g.parameters(), lr=8e-5))
    if learn_f:
        groups.append(dict(params=f.parameters(), lr=3e-5))
    optimizer = torch.optim.AdamW(groups,weight_decay=1e-4)
    steps = 1 if smoke else STEPS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=2e-6)
    order = np.random.default_rng(SEED)
    rng = torch.Generator(device=device).manual_seed(SEED)
    pseudo = arm == PL
    weights = torch.tensor((1.,.65,.3,.30,.25,1.5) + ((1.,0.) if pseudo else ()),device=device)
    start = monotonic()
    ce_norm = None
    accepted = torch.zeros((),device=device)
    for step in range(1,steps+1):
        l = order.choice(len(labeled), BATCH if not smoke else 2, replace=False)
        u = order.choice(len(unlabeled),len(l),replace=False)
        raw = np.concatenate((labeled[l],unlabeled[u]))
        raw = tensor(raw.view('float32'),device).view(torch.complex64)
        target = tensor(labels[l],device).long()
        optimizer.zero_grad(set_to_none=True)
        parts = loss_parts(g,f,anchor,tracker,generic,raw,target,rng,pseudo=pseudo)
        if pseudo:
            accepted += parts[-1].detach()
        if not torch.isfinite(parts).all():
            raise FloatingPointError('Nonfinite source objective')
        if step == 1 and learn_g:
            ce_norm = norm_gradient(torch.autograd.grad(parts[0],tuple(g.parameters()),allow_unused=True,retain_graph=True))
            if not (0 < ce_norm < float('inf')):
                raise ValueError('No supervised CE gradient reaches the restorer')
        (parts @ weights).backward()
        gradients = {name: norm_gradient(p.grad for p in model.parameters())
                     for name,model in (('G',g),('F',f))} if step in (1,steps) else None
        if gradients is not None and any(not ((0 < gradients[k] < float('inf')) if active else gradients[k] == 0)
                                         for k,active in (('G',learn_g),('F',learn_f))):
            raise ValueError('Unexpected source model gradient activity')
        torch.nn.utils.clip_grad_norm_([p for model in (g,f) for p in model.parameters() if p.requires_grad],
                                       1.,error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0:
            print('W24_JOINT_FIT', arm, step, *parts.detach().tolist(),flush=True)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    proof = dict(arm=arm, steps=steps, seed=SEED, g_parent=UNET_FIDELITY_RUN, f_parent=AUG_RUN,
                 L_s=len(labeled), U_s=len(unlabeled), labeled_exposures=steps*(2 if smoke else BATCH),
                 unlabeled_exposures=steps*(2 if smoke else BATCH), U_TX_reads=0, V_reads=0, target_reads=0,
                 practical_reads=0, trace_reads=0, G_trainable=learn_g, F_trainable=learn_f)
    if pseudo:
        proof.update(U_PL_teacher=AUG_RUN,U_PL_threshold=PL_THRESHOLD,U_PL_accepted=int(accepted))
    summary = dict(**proof, status='PASS' if smoke else 'SOURCE_FROZEN', CE_to_G_norm=ce_norm,
                   gradients=gradients, worker_seconds=monotonic()-start,
                   peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
                   G_parameters=sum(x.numel() for x in g.parameters()),F_parameters=sum(x.numel() for x in f.parameters()))
    if not smoke:
        with (out/'model.pt').open('xb') as stream:
            torch.save(dict(contract=contract, proof=proof,G=g.state_dict(),F=f.state_dict()),stream)
    write(out/('smoke.json' if smoke else 'complete.json'),summary)


def load_row(root, arm, contract, f0, g0, device):
    folder = root / arm
    summary = read(folder/'complete.json')
    state = torch.load(folder/'model.pt',map_location=device,weights_only=True)
    proof = state['proof']
    if (summary['status'] != 'SOURCE_FROZEN' or state['contract'] != contract
            or (proof['arm'],proof['steps'],proof['seed'],proof['g_parent'],proof['f_parent']) != (
                arm,STEPS,SEED,UNET_FIDELITY_RUN,AUG_RUN)
            or any(proof[k] for k in ('U_TX_reads','V_reads','target_reads','practical_reads','trace_reads'))
            or (proof['G_trainable'],proof['F_trainable']) != (arm!='pamf_only',arm!='frozen_pamf_ce')
            or any(summary[k] != v for k,v in proof.items()) or read(folder/'smoke.json')['status'] != 'PASS'):
        raise ValueError(f'Invalid source freeze for {arm}')
    if arm == PL and (proof.get('U_PL_teacher') != AUG_RUN or proof.get('U_PL_threshold') != PL_THRESHOLD
                      or not 0 <= proof['U_PL_accepted'] <= proof['unlabeled_exposures']):
        raise ValueError('Unexpected clean U_s pseudo-label policy')
    g, f = deepcopy(g0), deepcopy(f0)
    for model, key in ((g,'G'),(f,'F')):
        model.load_state_dict(state[key],strict=True)
        prior = (g0 if key=='G' else f0).state_dict()
        changed = any(not torch.equal(v,prior[k]) for k,v in model.state_dict().items())
        if changed != proof[f'{key}_trainable']:
            raise ValueError(f'Unexpected checkpoint update of {key} in {arm}')
    return g.eval().requires_grad_(False), f.eval().requires_grad_(False)


def source_predictions(rt, source, contract, render):
    indices, rx, ids = panel(source,contract)
    raw = native(source,'V',contract['counts']['V'])[indices]
    jobs = [(int(i),SOURCE[int(r)],x) for i,r,x in zip(indices,rx,raw,strict=True)]
    matrix = {}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for view in VIEWS:
            stream = pool.map(partial(source_receive,view),jobs,chunksize=8)
            groups = []
            for offset in range(0,len(jobs),PRED_BATCH):
                raw = np.stack([next(stream) for _ in range(min(PRED_BATCH,len(jobs)-offset))])
                groups.append(render(raw))
            matrix[view] = np.concatenate(groups,axis=1)
            print('W24_JOINT_SOURCE',view,len(jobs),flush=True)
    return ids, matrix


def write_source_predictions(root, ids, matrix, modes, **waveform):
    out = root/FULL
    out.mkdir()
    np.savez(out/'predictions.npz',**matrix)
    write(out/'source-predictions.json',dict(status='PREDICTIONS_FROZEN',modes=list(modes),views=list(VIEWS),
          physical_ids=ids,target_reads=0,source_truth_reads=0,waveform=dict(route=FULL,warm_steps=STEPS,**waveform)))


@torch.no_grad()
def predict_source(rt):
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    contract, f0, g0, tracker = boot(rt,device)
    rows = [load_row(root,arm,contract,f0,g0,device) for arm in ARMS]
    (g_pamf,f_pamf),(g_fixed,_),(g_joint,f_joint) = rows

    def render(raw):
        iq = received_aperture(raw,device)
        base = recovered(g0,tracker,iq)
        updated = recovered(g_joint,tracker,iq)
        pairs = ((f0,iq),(f0,base),(f_pamf,recovered(g_pamf,tracker,iq)),
                 (f0,recovered(g_fixed,tracker,iq)),(f_joint,updated),(f0,updated),(f_joint,base))
        return np.stack([classify_aperture(f,x) for f,x in pairs])

    u = native(source,'U_s',contract['counts']['U_s'])[0]
    sample = np.stack([source_receive(v,(0,SOURCE[0],u)) for v in ('clean', VIEWS[-1])])
    cpu_frontend = received_aperture(np.stack((u,u)),device)
    source_frontend, _ = normalize_aperture(tracker.aperture(
        tensor(np.stack((u,u)).view('float32'),device).view(torch.complex64)))
    torch.testing.assert_close(source_frontend,cpu_frontend,rtol=1e-5,atol=2e-6)
    if render(sample).shape != (len(MODES)-1,2):
        raise ValueError('Expected a complete real-checkpoint source smoke')

    ids, matrix = source_predictions(rt,source,contract,render)
    check_controls(Path(rt['unet_parent'])/FULL,('direct','unet_standard_long_half'),[0,1],ids,matrix)
    outer = Path(rt['project'])/'runs'/DURA_RUN/FULL
    mark = read(outer/'source-predictions.json')
    if (mark['status'], mark['physical_ids'], mark['views'], mark['target_reads'], mark['source_truth_reads']) != (
            'PREDICTIONS_FROZEN', ids, list(VIEWS), 0, 0):
        raise ValueError('Archived DURA source predictions differ')
    col = mark['modes'].index('outer_objective_half')
    with np.load(outer/'predictions.npz',allow_pickle=False) as old:
        matrix = {v: np.concatenate((data[:1],old[v][[col]],data[1:])) for v,data in matrix.items()}
    write_source_predictions(root,ids,matrix,MODES,frozen_E104_formal_candidate=CANDIDATE,joint_source_only=True)


@torch.no_grad()
def predict_pseudo(rt):
    root, source, device = Path(rt['run_root']), Path(rt['source_root']), 'cuda:0'
    contract, f0, g0, tracker = boot(rt,device)
    g, f = load_row(root,PL,contract,f0,g0,device)

    def render(raw):
        observed = received_aperture(raw,device)
        return np.stack((classify_aperture(f0,observed),classify_aperture(f,recovered(g,tracker,observed))))

    u = native(source,'U_s',contract['counts']['U_s'])[0]
    smoke = np.stack([source_receive(v,(0,SOURCE[0],u)) for v in ('clean',VIEWS[-1])])
    if render(smoke).shape != (2,2):
        raise ValueError('Expected real frozen clean-PL predictions')
    ids, matrix = source_predictions(rt,source,contract,render)
    comparison = root.parent/RUN/FULL
    reference = read(comparison/'source-predictions.json')
    if reference['modes'] != list(MODES):
        raise ValueError('Expected the preregistered true-label comparison')
    check_controls(comparison,('direct',),[0],ids,matrix)
    with np.load(comparison/'predictions.npz',allow_pickle=False) as archived:
        predictions = {view: np.concatenate((archived[view],matrix[view][1:])) for view in VIEWS}
    write_source_predictions(root,ids,predictions,PL_MODES,
                             joint_source_only=True,matched_run=RUN,U_PL_teacher=AUG_RUN,U_PL_threshold=PL_THRESHOLD)


def qualifies(scores):
    test = scores[CANDIDATE]
    controls = [scores[key] for key in ('direct','outer_half','unet_parent')]
    return (test['six_correct'] > max(x['six_correct'] for x in controls)
            and test['low_urban_correct'] >= max(x['low_urban_correct'] for x in controls)
            and test['by_scene']['clean'] >= scores['direct']['by_scene']['clean'])


def formal_receipt(case):
    if case not in FULL_CASES:
        raise ValueError('Expected Full-only shard')
    old = receipt(case,True,True)
    return dict(old,modes=[*old['modes'],CANDIDATE],candidate_run=RUN,reference_run=RUN_OUTER)


@torch.no_grad()
def predict_formal(rt,case):
    root, device = Path(rt['run_root']), 'cuda:0'
    proof = read(root/FULL/'source-scores.json')
    if (proof['status'] != 'ANALYZED' or not proof['source_truth_last'] or proof['target_reads']
            or proof['n_packets'] != 300 or proof['n_practical'] != 1800 or not qualifies(proof['scores'])):
        raise ValueError('Frozen E104 source qualification required')
    expected = formal_receipt(case)
    baseline = prior_target(rt,case)
    activity = read(baseline/'activity.json')
    contract,f0,g0,tracker = boot(rt,device)
    model,_ = load_row(root,CANDIDATE,contract,f0,g0,device)
    target = Path(rt['native_target_root'])
    target_contract(target)
    out = root/'predictions'/case
    out.mkdir(parents=True)
    counts = {}
    with ProcessPoolExecutor(max_workers=rt['cpu_workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for rx in expected['receivers']:
            raw = native(target,rx,LIMIT)
            decisions = {}
            with np.load(baseline/f'{rx}.npz',allow_pickle=False) as old:
                for view in VIEWS:
                    stream = iter(raw) if view=='clean' else pool.map(partial(channel_job,view,rx),enumerate(raw),chunksize=16)
                    pieces = []
                    for i in range(0,LIMIT,PRED_BATCH):
                        observed = received_aperture(np.stack([next(stream) for _ in range(min(PRED_BATCH,LIMIT-i))]),device)
                        rows = (classify_aperture(f0,observed),classify_aperture(f0,recovered(model,tracker,observed)))
                        pieces.append(np.stack(rows))
                    key = f'{rx}/{view}'
                    decisions[view],counts[key] = append_blind_prediction(
                        old[view],np.concatenate(pieces,axis=1)[None],activity[key],LIMIT)
                    print('W24_JOINT_FIXED_E104_FORMAL',rx,view,flush=True)
            np.savez(out/f'{rx}.npz',**decisions)
    write(out/'activity.json',counts)
    write(out/'complete.json',expected)


def dispatch(runtime):
    rt = read(runtime)
    root, reference = Path(rt['run_root']), Path(rt['w24_release_root'])
    root.mkdir()
    state = root/'state.json'
    with_pl = rt['run_id'] == RUN_PL
    arms = (PL,) if with_pl else ARMS

    def status(**values):
        write(state,values,exclusive=False)

    def launch(action, label, gpu, config, *details):
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                   PYTHONPATH=str(reference),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        cmd=[rt['python'],'-u',str(Path(__file__).resolve()),action,str(config),*details]
        with (root/f'{label}.log').open('x',encoding='utf-8') as log:
            job=subprocess.Popen(cmd,cwd=reference,env=env,stdout=log,stderr=subprocess.STDOUT)
        print('W24_JOINT_LAUNCHED',label,job.pid,gpu,flush=True)
        return job

    try:
        if read(reference/'submit.json')['code_commit'] != W24_COMMIT:
            raise ValueError('Expected archived W24 implementation')
        for arm in arms:
            out = root/arm
            out.mkdir()
            write(out/'fit-inputs.json',allowed_fit_runtime(rt,out))
        cards = available(rt['allowed_gpus'],rt['max_compute_processes'])
        if not cards:
            raise RuntimeError('No sufficient N607 training memory')
        status(status='SOURCE_TRAINING',steps=STEPS,arms=list(arms))
        schedule(rt,[('fit-row',arm) for arm in arms],lambda _,action,arm,gpu:
                 launch(action,f'{arm}-fit',gpu,root/arm/'fit-inputs.json',arm))
        if with_pl:
            comparison = root.parent/RUN
            deadline = monotonic()+12*60*60
            status(status='WAITING_FOR_MATCHED_SOURCE_PREDICTIONS',comparison=RUN)
            while not (comparison/FULL/'source-predictions.json').is_file():
                if read(comparison/'state.json')['status']=='FAILED' or monotonic()>=deadline:
                    raise RuntimeError('Matched true-label source predictions are unavailable')
                sleep(30)
        deadline=monotonic()+600
        while not (cards := available(rt['allowed_gpus'],rt['max_compute_processes'])):
            if monotonic()>=deadline:
                raise RuntimeError('No GPU free for frozen source predictions')
            sleep(15)
        for action,gpu in (('predict-source',cards[0]),('score-source',None)):
            job=launch(action,action,gpu,runtime)
            status(status='RUNNING',phase=action,pid=job.pid,gpu=gpu)
            if job.wait():
                raise RuntimeError(f'{action} failed')
        promote = not with_pl and qualifies(read(root/FULL/'source-scores.json')['scores'])
        status(status='SOURCE_SCORED',promote=promote)
        if promote:
            status(status='FORMAL_PREDICTING',route=FULL)
            schedule(rt,[('predict-formal',case) for case in FULL_CASES],lambda _,action,case,gpu:
                     launch(action,case,gpu,runtime,case))
            if any(read(root/'predictions'/c/'complete.json') != formal_receipt(c) for c in FULL_CASES):
                raise RuntimeError('Incomplete formal Full predictions')
            status(status='PREDICTIONS_FROZEN',cases=list(FULL_CASES),target_truth_read=False)
    except Exception as error:
        status(status='FAILED',error=str(error))
        raise


def collect(args):
    from experiments.pamf_lora_main.w24_formal_score import full_target_acceptance, load_frozen, score_matrix

    root = PurePosixPath(read(args.submit)['run_root'])
    if root.name != RUN or not re.fullmatch(r'/[A-Za-z0-9_./-]+',str(root)):
        raise ValueError('Expected the immutable joint run root')
    conn=['-F',str(args.ssh_config),'-o','BatchMode=yes','-o','ConnectTimeout=10']
    deadline,last=monotonic()+12*60*60,None
    while monotonic()<deadline:
        state=None
        try:
            remote=subprocess.run(['ssh',*conn,'-T','N607','cat '+str(root/'state.json')],timeout=25,capture_output=True)
            if remote.returncode==0:
                state=json.loads(remote.stdout)
        except (subprocess.TimeoutExpired,json.JSONDecodeError):
            pass
        if state:
            if state['status']!=last:
                print(state,flush=True)
                last=state['status']
            if last=='FAILED':
                raise RuntimeError(state['error'])
            if last=='SOURCE_SCORED' and not state['promote']:
                return
            if last=='PREDICTIONS_FROZEN':
                break
        sleep(40)
    else:
        raise TimeoutError('Formal E104 collector exceeded its observation window')
    args.output.mkdir(parents=True)
    subprocess.run(['scp',*conn,'-r','N607:'+str(root/'predictions'),str(args.output)],check=True,timeout=300)
    plans={case:formal_receipt(case) for case in FULL_CASES}
    matrices,activity=load_frozen(args.output/'predictions',ungated=True,observed_plans=plans)
    spec=plans[FULL_CASES[0]]
    score_matrix(matrices,activity,{FULL:spec['modes']},args.dataset,args.output/'scores',tuple(spec['seeds']),
                 lambda result:full_target_acceptance(result[FULL][CANDIDATE]['pooled']))


def main():
    from experiments.pamf_lora_main.publish import release, release_commit

    root = argparse.ArgumentParser(description=__doc__)
    actions = root.add_subparsers(dest='action',required=True)
    for name in ('dispatch','fit-row','fit-smoke','fit','predict-source','score-source','predict-formal'):
        cmd=actions.add_parser(name)
        cmd.add_argument('runtime',type=Path)
        if name.startswith('fit'):
            cmd.add_argument('arm',choices=(*ARMS,PL))
        if name=='predict-formal':
            cmd.add_argument('case',choices=FULL_CASES)
    pub=actions.add_parser('publish')
    pub.add_argument('--clean-pl',action='store_true')
    col=actions.add_parser('collect')
    for cmd,fields in ((pub,('repo','context','ssh-config','output')),(col,('submit','dataset','ssh-config','output'))):
        for field in fields:
            cmd.add_argument('--'+field,type=Path,required=True)
    args=root.parse_args()
    if args.action=='dispatch':
        dispatch(args.runtime)
    elif args.action=='fit-row':
        # Separate processes exercise the true path before starting the fixed fit.
        entry=str(Path(__file__).resolve())
        for name in ('fit-smoke','fit'):
            subprocess.run([sys.executable,'-u',entry,name,str(args.runtime),args.arm],check=True)
    elif args.action in ('fit-smoke','fit'):
        fit(args.runtime,args.arm,smoke=args.action=='fit-smoke')
    elif args.action=='predict-source':
        runtime=read(args.runtime)
        (predict_pseudo if runtime['run_id']==RUN_PL else predict_source)(runtime)
    elif args.action=='score-source':
        runtime=read(args.runtime)
        modes, contrasts = ((PL_MODES,((PL,'joint'),(PL,'outer_half'),(PL,'direct')))
                            if runtime['run_id']==RUN_PL else (MODES,CONTRASTS))
        score_source(args.runtime,modes=modes,output=Path(runtime['run_root'])/FULL,contrasts=contrasts)
    elif args.action=='predict-formal':
        predict_formal(read(args.runtime),args.case)
    elif args.action=='collect':
        collect(args)
    else:
        ctx=read(args.context)
        rt={k:ctx[k] for k in ('project','python','frozen_sources')}
        project=PurePosixPath(rt['project'])
        releases,runs=project/'releases',project/'runs'
        run = RUN_PL if args.clean_pl else RUN
        rt.update(run_id=run,runner=__spec__.name,code_commit=release_commit(args.repo),
                  release_root=str(releases/f'{run}-release'),run_root=str(runs/run),
                  source_root=str(releases/f'{DATA_RUN}-release'/'data/source'),
                  w24_release_root=str(releases/f'{W24_RUN}-release'),training_release_root=str(releases/f'{TRAIN_RUN}-release'),
                  classifier_root=str(runs/AUG_RUN),unet_parent=str(runs/UNET_FIDELITY_RUN),
                  allowed_gpus=[2,6,5] if args.clean_pl else [3,0,7,2,6,5],
                  max_workers=1 if args.clean_pl else 3,max_compute_processes=8,cpu_workers=6)
        rt['external_releases']=[rt[k] for k in ('source_root','w24_release_root',
                                                'training_release_root','classifier_root','unet_parent')]
        rt['external_releases'] += [str(runs/r) for r in (DURA_RUN,TRAIN_RUN)]
        if args.clean_pl:
            rt['external_releases'].append(str(runs/RUN))
        else:
            rt['native_target_root']=str(releases/f'{DATA_RUN}-release'/'data/target')
            rt['external_releases'] += [rt['native_target_root'],str(runs/RUN_OUTER)]
        release(args.repo,args.output,rt,args.ssh_config)


if __name__=='__main__':
    main()
