"""Single-receiver, two-view source validation of the LoRa and WiSig fusion."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic, perf_counter, sleep

import numpy as np
import torch
from torch import nn

from experiments.cric_dura_comparison.bootstrap import numerical_policy
from experiments.cric_dura_comparison.io import read, tensor, write
from experiments.pamf_lora_main.gain_dura_fusion import companion_only, frequency_real, fuse
from experiments.pamf_lora_main.w24_native_dura import capture_scale, iq_error
from experiments.pamf_lora_main.w24_native_spectral_innovation import SpectralInnovation
from experiments.pamf_lora_main.w24_packet_unet import PacketUNet
from experiments.pamf_restoration.models import complex_iq, pair


RUN = '20261128-phase1-lora-wisig-gain-dura-screen-s2026113047-r01'
SCORE_RUN = RUN.removesuffix('r01') + 'r02'
FACTOR_RUN = '20261128-phase1-lora-wisig-gain-dura-factor-s2026113051-r01'
FREQUENCY_RUN = '20261128-phase1-lora-wisig-dura-real-update-s2026113061-r01'
FREQUENCY_RETRY = FREQUENCY_RUN.removesuffix('r01') + 'r02'
STAGE_RUN = '20261128-phase1-lora-wisig-gain-dura-stages-s2026113067-r01'
PAIRED_RUN = '20261128-phase1-lora-wisig-gain-dura-paired-detail-s2026113071-r01'
COMPLEX_RUN = '20261128-phase1-lora-wisig-dura-complex-residual-s2026113081-r01'
VARIANCE_RUN = '20261128-phase1-lora-wisig-dura-variance-cap-s2026113088-r01'
TRAIN_BOUND_RUN = '20261128-phase1-lora-wisig-dura-train-bound-s2026113097-r01'
SPECTRAL_RUN = '20261128-phase1-lora-wisig-dura-spectral-guard-s2026113103-r01'
DRIFT_RUN = '20261128-phase1-lora-wisig-dura-source-drift-s2026113113-r01'
SEED, COUNT, STEPS, BATCH = 2026113047, 192, 8000, 24
FACTOR_SEED, FACTOR_COUNT = 2026113051, 160
FREQUENCY_SEED = 2026113061
STAGE_SEED = 2026113067
PAIRED_SEED = 2026113071
COMPLEX_SEED = 2026113081
VARIANCE_SEED = 2026113088
TRAIN_BOUND_SEED = 2026113097
SPECTRAL_SEED = 2026113103
DRIFT_SEED = 2026113113
DEV = 'cuda:0'
LO_RA = '20261128-phase1-lora-raw8192-frequency-quadrature-s2026113037-r01'
LO_RA_COMMIT = '38c7b18d779e4ee5815a2ef3b49ef2c8d23297bf'
ARCHIVE = '20261127-phase1-lora-wisig-packet-dura-s2026113021-r03'
WISIG_REF = '20261125-phase1-wisig-native-fir-source-m4-r01'
MODES = ('direct', 'dura', 'real_gain', 'fusion')
FACTOR_MODES = (*MODES,'data_only','companion_only')
FREQUENCY_MODES = (*MODES[:3], 'frequency_real', 'data_only')
STAGE_MODES = (*MODES[:3], 'stage_frequency', 'stage_packet')
PAIRED_MODES = (*MODES[:3], 'paired_detail', 'matched_waveform')
COMPLEX_MODES = (*MODES[:3], 'complex_residual', 'aligned_residual')
VARIANCE_MODES = (*MODES[:3], 'variance_capped', 'same_unbounded')
TRAIN_BOUND_MODES = (*MODES[:3], 'train_bounded', 'post_bounded')
SPECTRAL_MODES = (*MODES[:3], 'spectral_guard', 'packet_guard')
DRIFT_MODES = (*MODES[:3], 'structured_nuisance', 'generic_nuisance')
SCREEN_VIEWS = ('clean', 'practical_low_urban')
RX = {'lora': 0, 'wisig': 1}


def screen_rows(ids, receivers, receiver, *, excluded=(), limit=COUNT, seed=SEED):
    if len(ids) != len(receivers) or len(set(ids.tolist())) != len(ids):
        raise ValueError('Expected one blind receiver record per physical ID')
    rows = np.setdiff1d(np.flatnonzero(receivers == receiver), excluded)
    if len(rows) < limit:
        raise ValueError('Insufficient fixed receiver coverage')
    return np.sort(np.random.default_rng(seed).choice(rows,limit,replace=False))


def paired_panel(change, draws):
    return {view: dict(repairs=int((diff>0).sum()),harms=int((diff<0).sum()),
               net=int(diff.sum()),ci95=np.quantile(diff[draws].sum(1),(.025,.975)).tolist())
            for view,diff in zip(SCREEN_VIEWS,change,strict=True)}


def tx_counts(predictions, labels, modes):
    hit = predictions == labels
    groups = {}
    for tx in np.unique(labels):
        mask = labels == tx
        counts = hit[..., mask].sum(-1).tolist()
        groups[int(tx)] = dict(n=int(mask.sum()), correct={
            view: dict(zip(modes, row, strict=True))
            for view, row in zip(SCREEN_VIEWS, counts, strict=True)})
    return groups


def score_panel(predictions, labels, *, count=COUNT, seed=SEED, modes=MODES):
    if predictions.shape != (len(SCREEN_VIEWS), len(modes), count) or labels.shape != (count,):
        raise ValueError('Incomplete same-packet screen')
    hit = predictions == labels
    correct = hit.sum(-1)
    idx = np.random.default_rng(seed+2).integers(0,count,size=(4000,count))
    candidate = hit[:,-1].astype(np.int8)
    paired = {name:paired_panel(candidate-hit[:,i],idx) for i,name in enumerate(modes[:-1])}
    clean_pass = bool(correct[0,-1] >= correct[0,:-1].max())
    difficult_pass = all(out[SCREEN_VIEWS[-1]]['net'] >= 8 and out[SCREEN_VIEWS[-1]]['ci95'][0] > 0
                         for out in paired.values())
    return dict(correct={v:dict(zip(modes,c.tolist(),strict=True))
                         for v,c in zip(SCREEN_VIEWS,correct,strict=True)},
                paired=paired,clean_pass=clean_pass,low_urban_pass=difficult_pass,
                promote=clean_pass and difficult_pass)


def score_factor_panel(predictions, labels):
    if predictions.shape != (2,len(FACTOR_MODES),FACTOR_COUNT) or labels.shape!=(FACTOR_COUNT,):
        raise ValueError('Incomplete independent factor screening matrix')
    candidates = {mode:score_panel(predictions[:,[0,1,2,i]],labels,count=FACTOR_COUNT,seed=FACTOR_SEED)
                  for i,mode in enumerate(FACTOR_MODES) if i>=3}
    hit = (predictions==labels).astype(np.int8)
    correct = {v:dict(zip(FACTOR_MODES,c.tolist(),strict=True))
               for v,c in zip(SCREEN_VIEWS,hit.sum(-1),strict=True)}
    draws = np.random.default_rng(FACTOR_SEED+2).integers(0,FACTOR_COUNT,size=(4000,FACTOR_COUNT))
    pairs = {(f'{FACTOR_MODES[a]}_vs_{FACTOR_MODES[b]}'):paired_panel(hit[:,a]-hit[:,b],draws)
             for a,b in ((3,4),(3,5),(4,5))}
    return dict(correct=correct,candidates=candidates,factor_pairs=pairs)


def score_frequency_panel(predictions, labels):
    if predictions.shape != (2, len(FREQUENCY_MODES), FACTOR_COUNT):
        raise ValueError('Incomplete real-gain constraint comparison')
    result = score_panel(predictions[:, :4], labels, count=FACTOR_COUNT, seed=FREQUENCY_SEED, modes=FREQUENCY_MODES[:4])
    hit = (predictions == labels).astype(np.int8)
    draws = np.random.default_rng(FREQUENCY_SEED+2).integers(0, FACTOR_COUNT, size=(4000, FACTOR_COUNT))
    result['paired']['data_only'] = paired_panel(hit[:, 3] - hit[:, 4], draws)
    for i, view in enumerate(SCREEN_VIEWS):
        result['correct'][view]['data_only'] = int(hit[i, 4].sum())
    result['per_tx'] = tx_counts(predictions, labels, FREQUENCY_MODES)
    return result


def score_stage_panel(predictions, labels, *, modes=STAGE_MODES, seed=STAGE_SEED):
    if predictions.shape != (2, len(modes), FACTOR_COUNT):
        raise ValueError('Incomplete matched in-stage screen')
    results = {}
    for i, candidate in enumerate(modes[3:], 3):
        results[candidate] = score_panel(predictions[:, [0, 1, 2, i]], labels, count=FACTOR_COUNT,
                                         seed=seed, modes=(*modes[:3], candidate))
    hit = (predictions == labels).astype(np.int8)
    wrong = ~(hit[:, 1].astype(bool) | hit[:, 2].astype(bool))
    gains = {name: {view: int((hit[i, j] & wrong[i]).sum()) for i, view in enumerate(SCREEN_VIEWS)}
             for j, name in enumerate(modes[3:], 3)}
    draws = np.random.default_rng(seed+2).integers(0, FACTOR_COUNT, size=(4000, FACTOR_COUNT))
    return dict(candidates=results, per_tx=tx_counts(predictions, labels, modes),
                matched_ablation=paired_panel(hit[:, 3]-hit[:, 4], draws),
                both_experts_wrong=dict(zip(SCREEN_VIEWS, wrong.sum(-1).tolist(), strict=True)),
                corrected_when_both_wrong=gains)


def panel_dimensions(rt):
    if rt.get('drift_from'):
        return FACTOR_COUNT, DRIFT_MODES
    if rt.get('spectral_from'):
        return FACTOR_COUNT, SPECTRAL_MODES
    if rt.get('train_bound_from'):
        return FACTOR_COUNT, TRAIN_BOUND_MODES
    if rt.get('variance_from'):
        return FACTOR_COUNT, VARIANCE_MODES
    if rt.get('complex_from'):
        return FACTOR_COUNT, COMPLEX_MODES
    if rt.get('paired_from'):
        return FACTOR_COUNT, PAIRED_MODES
    if rt.get('stage_from'):
        return FACTOR_COUNT, STAGE_MODES
    if rt.get('frequency_from'):
        return FACTOR_COUNT, FREQUENCY_MODES
    return (FACTOR_COUNT,FACTOR_MODES) if rt.get('factor_from') else (COUNT,MODES)


def receiver_index(dataset, modes):
    if dataset == 'wisig' and modes in (SPECTRAL_MODES, DRIFT_MODES):
        return 8
    if dataset == 'wisig' and modes in (VARIANCE_MODES, TRAIN_BOUND_MODES):
        return 6
    if dataset == 'wisig' and modes in (PAIRED_MODES, COMPLEX_MODES):
        return 4
    return 3 if dataset == 'wisig' and modes in (FREQUENCY_MODES, STAGE_MODES) else RX[dataset]


def panel_rows(rt, dataset, ids, rx, *, excluded=()):
    ids, rx = np.asarray(ids), np.asarray(rx)
    count, modes = panel_dimensions(rt)
    rule = {
        MODES: SEED, FACTOR_MODES: FACTOR_SEED, FREQUENCY_MODES: FREQUENCY_SEED,
        STAGE_MODES: STAGE_SEED, PAIRED_MODES: PAIRED_SEED, COMPLEX_MODES: COMPLEX_SEED,
        VARIANCE_MODES: VARIANCE_SEED, TRAIN_BOUND_MODES: TRAIN_BOUND_SEED, SPECTRAL_MODES: SPECTRAL_SEED,
        DRIFT_MODES: DRIFT_SEED,
    }
    history = (
        ('factor_from', RUN, COUNT, MODES),
        ('frequency_from', FACTOR_RUN, FACTOR_COUNT, FACTOR_MODES),
        ('stage_from', FREQUENCY_RETRY, FACTOR_COUNT, FREQUENCY_MODES),
        ('paired_from', STAGE_RUN, FACTOR_COUNT, STAGE_MODES),
        ('complex_from', PAIRED_RUN, FACTOR_COUNT, PAIRED_MODES),
        ('variance_from', COMPLEX_RUN, FACTOR_COUNT, COMPLEX_MODES),
        ('train_bound_from', VARIANCE_RUN, FACTOR_COUNT, VARIANCE_MODES),
        ('spectral_from', TRAIN_BOUND_RUN, FACTOR_COUNT, TRAIN_BOUND_MODES),
        ('drift_from', SPECTRAL_RUN, FACTOR_COUNT, SPECTRAL_MODES),
    )
    blocked = excluded
    for key, run_id, old_count, old_modes in history:
        if not rt.get(key):
            continue
        prior = Path(rt[key])
        folder = prior / dataset
        if (prior != Path(rt['run_root']).parent / run_id
                or read(folder/'complete.json') != panel_receipt(dataset, old_count, old_modes)):
            raise ValueError(f'Expected frozen panel from {run_id}')
        with np.load(folder/'predictions.npz', allow_pickle=False) as old:
            past = np.flatnonzero(np.isin(ids, old['ids']))
        if len(past) != old_count or not (rx[past] == receiver_index(dataset, old_modes)).all():
            raise ValueError(f'Physical ID mismatch for prior screen {run_id}')
        blocked = np.union1d(blocked, past)
    return screen_rows(ids, rx, receiver_index(dataset, modes), excluded=blocked, limit=count, seed=rule[modes])


def fusion_outputs(rt, observed, anchor, companion, action, variance):
    if rt.get('frequency_from'):
        ranked = (frequency_real(observed, anchor, action, variance),
                  fuse(observed, anchor, anchor, action, variance))
    else:
        ranked = (fuse(observed,anchor,companion,action,variance),)
        if rt.get('factor_from'):
            ranked += (fuse(observed,anchor,anchor,action,variance),companion_only(observed,anchor,companion))
    if not all(bool(torch.isfinite(v).all()) for v in ranked):
        raise ValueError('Nonfinite fusion')
    return ranked


def time_readouts(path, operations, scope):
    costs = {}
    for method, operation in operations:
        for _ in range(8):
            operation()
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            started = perf_counter()
            operation()
            torch.cuda.synchronize()
            times.append(1e3 * (perf_counter() - started))
        costs[method] = dict(median_ms=float(np.median(times)), p95_ms=float(np.quantile(times, .95)))
    write(path, dict(scope=scope,
                     gpu=torch.cuda.get_device_name(), precision='FP32', timings=costs))


def time_frequency_update(path, observed, anchor, action, variance):
    time_readouts(path, (
        ('frequency_real', lambda: frequency_real(observed, anchor, action, variance)),
        ('data_only', lambda: fuse(observed, anchor, anchor, action, variance)),
    ), 'B1 standalone observation update only, fixed action and anchor')


def stage_outputs(gates, received, real_gain, context, priors, timing, modes):
    operations = [(name, partial(gate, received, real_gain, context, priors))
                  for name, gate in zip(modes[3:], gates, strict=True)]
    if timing is not None:
        time_readouts(timing, operations, 'B1 two stages including priors and CG, frozen inputs/context')
    outputs = tuple(operation()[-1] for _, operation in operations)
    if not all(bool(torch.isfinite(x).all()) for x in outputs):
        raise ValueError('Nonfinite in-stage restoration')
    return outputs


def variance_outputs(gate, received, real_gain, context, priors, raw, scale, curve, timing):
    from experiments.pamf_lora_main.gain_dura_variance import project_output

    def unbounded():
        return gate(received, real_gain, context, priors)[-1] * scale

    def bounded(wave):
        return project_output(raw, wave, complex_iq(received), context[2], curve)

    if timing is not None:
        time_readouts(timing, [('variance_capped', lambda: bounded(unbounded())), ('same_unbounded', unbounded)],
                      'B1 two-stage recovery with or without final variance projection')
    recovered = unbounded()
    outputs = (bounded(recovered), recovered)
    if not all(bool(torch.isfinite(x).all()) for x in outputs):
        raise ValueError('Expected finite independent and variance-projected outputs')
    return outputs


def stage_readout_config(rt, dataset):
    from experiments.pamf_lora_main.gain_dura_stage_fit import load_gates

    if rt.get('train_bound_from'):
        from experiments.pamf_lora_main.gain_dura_variance import load_curve

        gate_root = rt.get('spectral_from', rt['run_root']) if not rt.get('drift_from') else rt['run_root']
        gates = load_gates(dict(rt, run_root=gate_root), dataset, DEV)
        return gates, load_curve(dict(rt, run_root=rt['train_bound_from']), dataset)
    if rt.get('variance_from'):
        from experiments.pamf_lora_main.gain_dura_variance import load_curve

        gates = load_gates(dict(rt, run_root=rt['variance_from']), dataset, DEV)
        return gates, load_curve(rt, dataset)
    return load_gates(rt, dataset, DEV), None


def bounded_fit_outputs(gates, received, real_gain, context, priors, raw, scale, curve, reference, timing,
                        *, spectral=False, readout_modes=TRAIN_BOUND_MODES[3:]):
    from experiments.pamf_lora_main.gain_dura_variance import project_reference_increment, protect_weak_spectrum

    observed = complex_iq(received)

    def predict(gate):
        return gate(received, real_gain, context, priors, reference=reference)[-1]

    def finish(proposal, local_guard=False):
        state = protect_weak_spectrum(reference, proposal) if local_guard else proposal
        bounded = project_reference_increment(observed, reference, state, context[2], curve)
        return raw + (bounded - observed) * scale

    if spectral:
        if timing is not None:
            time_readouts(timing, [(SPECTRAL_MODES[3], lambda: finish(predict(gates[0]), True)),
                                   (SPECTRAL_MODES[4], lambda: finish(predict(gates[0])))],
                          'B1 frozen reference recovery with weak-bin and packet-only envelopes')
        proposal = predict(gates[0])
        outputs = (finish(proposal, True), finish(proposal))
    else:
        def apply(gate):
            return finish(predict(gate))

        operations = [(mode, partial(apply, gate)) for mode, gate in zip(readout_modes, gates, strict=True)]
        if timing is not None:
            time_readouts(timing, operations, 'B1 reference anchored recovery with final uncertainty envelope')
        outputs = tuple(operation() for _, operation in operations)
    if not all(bool(torch.isfinite(z).all()) for z in outputs):
        raise ValueError('Expected finite bounded reference outputs')
    return outputs


class WiSigRealGain(nn.Module):
    """WiSig-specific repeated-preamble coordinates and shared phase-aware real gain."""

    def __init__(self):
        super().__init__()
        self.coarse = PacketUNet(across_chirps=True,center=32)
        self.gain = SpectralInnovation(phase_aware=True,samples=32)
        self.register_buffer('time',torch.arange(256,dtype=torch.float32),persistent=False)

    def forward(self, raw):
        adjacent = (raw[:,100:200] * raw[:,80:180].conj()).sum(-1,keepdim=True)
        rotation = torch.polar(torch.ones_like(raw.real), adjacent.angle()*self.time/20)
        framed = (raw*rotation.conj()).unflatten(-1,(8,32))
        scale = capture_scale(framed)
        observed = framed/scale
        coherence = (observed[...,1:]*observed[...,:-1].conj()).mean(-1,keepdim=True).abs().clamp(max=1.)
        edges = torch.cat((framed[...,:24],framed[...,-24:]),dim=-1)
        coarse = self.coarse(observed,coherence,scale,edges)[-1]
        state = observed + .5*(coarse-observed)
        first = raw + ((state-observed)*scale).flatten(1)*rotation
        final = first + (self.gain(observed,state)*scale).flatten(1)*rotation
        return first,final


def archive_replay(rt):
    package = importlib.import_module('experiments.pamf_lora_main')
    archived = Path(rt['archive_root'])/'experiments'/'pamf_lora_main'
    package.__path__.append(str(archived))
    return importlib.import_module('experiments.pamf_lora_main.packet_dura_crossdata')


def wifi_context(rt):
    replay = archive_replay(rt)
    config = replay.reference(rt,'wisig')
    return replay.wisig_deps(config),config


def fit_wifi(folder, original, source):
    data, ids, smoke_rows = original.core.source_iq(source)
    if len(ids) != 52604 or len(np.intersect1d(ids,smoke_rows)):
        raise ValueError('Expected legal WiSig gain fit mask and disjoint source smoke')
    torch.manual_seed(SEED)
    model = WiSigRealGain().to(DEV).train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=2e-5)
    draws = np.random.default_rng(SEED)
    rng = torch.Generator(device=DEV).manual_seed(SEED+1)

    def objective(clean):
        clean = clean/capture_scale(clean)
        options = dict(device=DEV,generator=rng)
        n = torch.randn(clean.shape,dtype=clean.dtype,**options)
        echo = torch.randn((len(clean),1),dtype=clean.dtype,**options)
        shifted = torch.nn.functional.pad(clean[:,:-3],(3,0))
        snr = -8. + 32.*torch.rand((len(clean),1),**options)
        y = clean + .08*echo*shifted + 10.**(-snr/20)*n/capture_scale(n)
        p,x = model(y)
        _,identity = model(clean[:6])
        e = torch.fft.fft(x-clean,norm='ortho').abs().square()
        e = (e/(.03+torch.fft.fft(clean,norm='ortho').abs().square())).log1p().mean()
        return iq_error(x,clean) + .35*iq_error(p,clean) + .3*e + 1.5*iq_error(identity,clean[:6])

    smoke = objective(complex_iq(tensor(data[smoke_rows[:8]],DEV)))
    grad, = torch.autograd.grad(smoke,model.gain.gain[-1].weight)
    if not bool(torch.isfinite(grad).all() and grad.norm()>0):
        raise ValueError('Real source smoke has no gain gradient')
    write(folder/'fit-smoke.json',dict(status='PASS',gain_gradient=float(grad.norm()),target_reads=0,F_reads=0))
    rng.manual_seed(SEED+1)
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    for step in range(1,STEPS+1):
        clean = complex_iq(tensor(data[draws.choice(ids,BATCH,replace=False)],DEV))
        loss = objective(clean)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        schedule.step()
        if step==1 or step%100==0:
            print('WISIG_REAL_GAIN',step,float(loss.detach()),flush=True)
    torch.cuda.synchronize()
    with (folder/'real_gain.pt').open('xb') as stream:
        torch.save(model.state_dict(),stream)
    write(folder/'fit.json',dict(status='SOURCE_FROZEN',seed=SEED,steps=STEPS,batch=BATCH,
          training_ids=len(ids),F_reads=0,TX_reads=0,V_reads=0,target_reads=0,
          parameters=sum(p.numel() for p in model.parameters()),seconds=perf_counter()-started,
          peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    return model.eval().requires_grad_(False)


def lora_panel(rt, joint, source, contract):
    old,_,_ = joint.panel(source,contract)
    with np.load(source/'V_metadata.npz',allow_pickle=False) as metadata:
        ids,rx = metadata['ids'].copy(),metadata['rx_i'].copy()
    rows = panel_rows(rt,'lora',ids,rx,excluded=old)
    return rows,ids[rows]


def panel_receipt(dataset, count, modes):
    return dict(dataset=dataset,receiver=receiver_index(dataset,modes),views=list(SCREEN_VIEWS),modes=list(modes),n=count,
                source_truth_reads=0,target_reads=0,status='PREDICTIONS_FROZEN')


def freeze(folder, dataset, ids, decisions, *, count=COUNT, modes=MODES):
    if len(ids)!=count or set(decisions)!=set(SCREEN_VIEWS):
        raise ValueError('Incomplete preregistered screen')
    for d in decisions.values():
        if d.shape!=(len(modes),count):
            raise ValueError('Wrong primary and comparison coverage')
    with (folder/'predictions.npz').open('xb') as stream:
        np.savez(stream,ids=ids,**decisions)
    write(folder/'complete.json',panel_receipt(dataset,count,modes))


def classify_lora(joint, reader, wave):
    packets = np.asarray(wave.contiguous().view(torch.float32).cpu().tolist(),dtype=np.float32).view(np.complex64)
    return joint.classify_aperture(reader,joint.received_aperture(packets,DEV))


def lora_archives(rt):
    config = read(Path(rt['lora_ref'])/'runtime.json')
    if (config['run_id'],config['code_commit']) != (LO_RA,LO_RA_COMMIT):
        raise ValueError('Wrong frozen LoRa 8192-point real gain version')
    w24 = Path(config['w24_release_root'])
    sys.path.insert(0,str(w24))
    archive_replay(rt).archive_imports(config,extra=(w24,))
    from experiments.pamf_lora_main import w24_joint_pamf as joint, w24_native_training as native
    return config, joint, native


def load_lora_gain(config, native, contract, device):
    stage = native.PHASE_ARMS[0]
    gain = native.task_model(None,device)
    gain.core.innovation = native.spectral_innovation(stage,device)
    native.load_stage(Path(config['run_root']),stage,contract,gain.core,device)
    return gain.eval().requires_grad_(False)


@torch.no_grad()
def predict_lora(rt, folder):
    count,modes = panel_dimensions(rt)
    unified = rt.get('frequency_from')
    if unified:
        # Bind the native-length solver before the historical W24 import path.
        from experiments.pamf_lora_main.lora_dura_stage2_reference import load_frozen_stage2

    config, joint, native = lora_archives(rt)
    source = Path(config['source_root'])
    contract,reader,_,_ = joint.boot(config,DEV)
    gain = load_lora_gain(config, native, contract, DEV)
    if unified:
        dura = load_frozen_stage2(Path(rt['lora_stage2']), DEV)
    else:
        dura = native.dura_model(Path(rt['project'])/'runs'/native.DURA_RUN,contract,DEV).eval().requires_grad_(False)
    gates, curve = None, None
    if rt.get('stage_from'):
        gates, curve = stage_readout_config(rt, 'lora')

    def apply(packets, timing=None):
        y = tensor(np.ascontiguousarray(packets).view(np.float32),DEV).view(torch.complex64)
        scale = capture_scale(y)
        received = pair(y/scale)
        real_gain = gain(y)
        if unified:
            context = dura.estimate_context(received)
            _, action, variance = context
            anchor = complex_iq(dura.stage_outputs(received, context)[0][-1])
            recovered, companion = anchor*scale, real_gain/scale
        else:
            coarse,action,variance = dura.estimate(received)
            recovered = complex_iq(dura.unroll(received,coarse,action,variance))*scale
            anchor, companion = real_gain/scale, recovered/scale
        if rt.get('train_bound_from'):
            extras = bounded_fit_outputs(gates, received, real_gain/scale, context, dura.prior,
                                         y, scale, curve, real_gain/scale, timing,
                                         spectral=modes == SPECTRAL_MODES, readout_modes=modes[3:])
        elif curve is not None:
            extras = variance_outputs(gates[0], received, real_gain/scale, context, dura.prior, y, scale, curve, timing)
        elif gates is None:
            if timing is not None:
                time_frequency_update(timing, y/scale, anchor, action, variance)
            extras = (z*scale for z in fusion_outputs(rt,y/scale,anchor,companion,action,variance))
        else:
            extras = (z*scale for z in stage_outputs(gates, received, real_gain/scale, context, dura.prior, timing, modes))
        values = (y,recovered,real_gain,*extras)
        return np.stack([classify_lora(joint,reader,z) for z in values])

    sample = joint.source_role(source,contract,'U_s')[0][:2]
    if apply(sample).shape != (len(modes),2):
        raise ValueError('LoRa no-query checkpoint smoke failed')
    write(folder/'smoke.json',dict(status='PASS',source_count=2,source_V_reads=0,target_reads=0))
    if unified:
        apply(sample[:1], timing=folder/'timing.json')
    rows,ids = lora_panel(rt,joint,source,contract)
    x = joint.native(source,'V',contract['counts']['V'])[rows]
    jobs = list(zip(rows.tolist(),[joint.SOURCE[receiver_index('lora',modes)]]*count,x,strict=True))
    decisions = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for view in SCREEN_VIEWS:
            packets = np.stack(list(pool.map(partial(joint.source_receive,view),jobs)))
            decisions[view] = np.concatenate([apply(packets[i:i+8]) for i in range(0,count,8)],axis=1)
    freeze(folder,'lora',ids,decisions,count=count,modes=modes)


def wifi_predictor(rt, original, config, gain):
    reader = original.tri.reader_factory(config).eval().requires_grad_(False)
    engine = original.tri.core.original_r69(Path(config['parent_root']))
    base = engine.pipeline.model
    _, modes = panel_dimensions(rt)
    gates, curve = None, None
    if rt.get('stage_from'):
        gates, curve = stage_readout_config(rt, 'wisig')

    @torch.no_grad()
    def apply(block, timing=None):
        iq = tensor(block,DEV)
        y = complex_iq(iq)
        context = base.estimate_context(iq)
        _,action,variance = context
        (output,) = engine(iq)
        anchor = complex_iq(output)
        companion = gain(y)[-1]
        if rt.get('train_bound_from'):
            reference = complex_iq(base.stage_outputs(iq, context)[0][-1])
            solved = bounded_fit_outputs(gates, iq, companion, context, base.prior, y, 1., curve, reference, timing,
                                          spectral=modes == SPECTRAL_MODES, readout_modes=modes[3:])
        elif curve is not None:
            solved = variance_outputs(gates[0], iq, companion, context, base.prior, y, 1., curve, timing)
        elif gates is None:
            if timing is not None:
                time_frequency_update(timing, y, anchor, action, variance)
            solved = fusion_outputs(rt,y,anchor,companion,action,variance)
        else:
            solved = stage_outputs(gates, iq, companion, context, base.prior, timing, modes)
        return np.asarray([original.classify(reader,z).argmax(1).tolist()
                           for z in (y,anchor,companion,*solved)],dtype=np.int16)

    return apply


def smoke_wifi(rt, folder):
    original,config = wifi_context(rt)
    source = Path(config['source_root'])
    gain = WiSigRealGain().to(DEV).eval().requires_grad_(False)
    data,_,rows = original.core.source_iq(source)
    if wifi_predictor(rt,original,config,gain)(data[rows[:4]]).shape != (len(MODES),4):
        raise ValueError('Original R69 and Tri-SIG default head smoke failed')
    write(folder/'smoke.json',dict(status='PASS',source_count=4,source_V_reads=0,target_reads=0))


def load_wifi_gain(rt, device):
    prior = Path(rt['factor_from'])/'wisig'
    proof = read(prior/'fit.json')
    if (proof['status'],proof['steps'],proof['seed'],proof['training_ids'])!=('SOURCE_FROZEN',STEPS,SEED,52604) \
            or any(proof[k] for k in ('TX_reads','V_reads','F_reads','target_reads')):
        raise ValueError('Missing legal frozen WiSig real-gain ancestor')
    gain = WiSigRealGain().to(device).eval().requires_grad_(False)
    gain.load_state_dict(torch.load(prior/'real_gain.pt',map_location=device,weights_only=True),strict=True)
    return gain


def predict_wifi(rt, folder, runtime):
    count,modes = panel_dimensions(rt)
    if not rt.get('factor_from'):
        subprocess.run([sys.executable,'-u','-m',__spec__.name,'wifi-smoke',str(runtime)],check=True)
    original,config = wifi_context(rt)
    source = Path(config['source_root'])
    if rt.get('factor_from'):
        gain = load_wifi_gain(rt, DEV)
    else:
        gain = fit_wifi(folder,original,source)
        gain.load_state_dict(torch.load(folder/'real_gain.pt',map_location=DEV,weights_only=True),strict=True)
    apply = wifi_predictor(rt,original,config,gain)
    data,_,smoke_rows = original.core.source_iq(source)
    if apply(data[smoke_rows[:4]]).shape != (len(modes),4):
        raise ValueError('WiSig no-query checkpoint smoke failed')
    write(folder/'frozen-smoke.json',dict(status='PASS',source_count=4,source_V_reads=0,target_reads=0))
    if rt.get('frequency_from'):
        apply(data[smoke_rows[:1]], timing=folder/'timing.json')
    cache,_,_,_,ids,rx,_ = original.tri.core.handles(config)
    selected = panel_rows(rt,'wisig',ids,rx)
    decisions={}
    for view in SCREEN_VIEWS:
        array=np.load(cache/'validation_full'/f'{view}.npy',mmap_mode='r',allow_pickle=False)
        if array.shape!=(len(ids),2,256) or array.dtype!=np.float32:
            raise ValueError('Mismatched WiSig source V capture')
        decisions[view]=np.concatenate([apply(array[selected[i:i+32]]) for i in range(0,count,32)],axis=1)
    freeze(folder,'wisig',np.asarray(ids)[selected],decisions,count=count,modes=modes)


def independent_score(rt):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Expected separate CPU truth connection')
    root = Path(rt['run_root'])
    count,modes = panel_dimensions(rt)
    predictions = root
    if rt.get('score_from'):
        predictions = Path(rt['score_from'])
        if predictions != root.parent/RUN or read(predictions/'state.json')['status'] != 'FAILED':
            raise ValueError('Expected the immutable, previously failed score stage')
    rows={}
    for dataset in RX:
        folder = predictions/dataset
        proof = read(folder/'complete.json')
        if proof!=panel_receipt(dataset,count,modes):
            raise ValueError('Both screening matrices must freeze before source truth')
        with np.load(folder/'predictions.npz',allow_pickle=False) as data:
            ids = data['ids'].copy()
            matrix = np.stack([data[v] for v in SCREEN_VIEWS])
        classes = 10 if dataset=='lora' else 6
        if (matrix.shape!=(2,len(modes),count) or matrix.dtype.kind not in 'iu'
                or not ((0<=matrix)&(matrix<classes)).all() or len(set(ids.tolist()))!=count):
            raise ValueError('Invalid frozen predictions')
        rows[dataset] = ids,matrix

    # Activate the WiSig archive before importing any LoRa package that can load
    # another pamf_unrolled_restore namespace.
    original,wifi=wifi_context(rt)
    _,_,wifi_source,_,ids,rx,order=original.tri.core.handles(wifi)
    selected=panel_rows(rt,'wisig',ids,rx)
    if not np.array_equal(rows['wisig'][0],np.asarray(ids)[selected]):
        raise ValueError('WiSig screen order drift')

    lora = read(Path(rt['lora_ref'])/'runtime.json')
    source = Path(lora['source_root'])
    lora_old = read(Path(lora['run_root'])/'full_native'/'source-predictions.json')
    if lora_old['status']!='PREDICTIONS_FROZEN' or lora_old['target_reads'] or lora_old['source_truth_reads']:
        raise ValueError('Expected the fixed earlier source panel exclusion')
    with np.load(source/'V_metadata.npz',allow_pickle=False) as data:
        names,rx = data['ids'].copy(),data['rx_i'].copy()
        role = read(source/'contract.json')['role_ids']['V']
        if names.tolist()!=role:
            raise ValueError('LoRa source validation roles differ')
        exclude = np.flatnonzero(np.isin(names,lora_old['physical_ids']))
        if len(exclude)!=300:
            raise ValueError('The original 300 source validation packets must be excluded')
        index = panel_rows(rt,'lora',names,rx,excluded=exclude)
        if not np.array_equal(rows['lora'][0],names[index]):
            raise ValueError('LoRa screen order drift')
        lora_truth=data['y'][index]
    wifi_truth=np.load(wifi_source/'V/labels.npy',allow_pickle=False)[np.asarray(order)[selected]]
    if rt.get('stage_from'):
        seed = {STAGE_MODES: STAGE_SEED, PAIRED_MODES: PAIRED_SEED,
                COMPLEX_MODES: COMPLEX_SEED, VARIANCE_MODES: VARIANCE_SEED, TRAIN_BOUND_MODES: TRAIN_BOUND_SEED,
                SPECTRAL_MODES: SPECTRAL_SEED, DRIFT_MODES: DRIFT_SEED}[modes]
        calculate = partial(score_stage_panel, modes=modes, seed=seed)
    else:
        calculate = score_frequency_panel if rt.get('frequency_from') else score_factor_panel if rt.get('factor_from') else score_panel
    outputs={k:calculate(rows[k][1],y) for k,y in (('lora',lora_truth),('wisig',wifi_truth))}
    write(root/'scores.json',dict(status='ANALYZED',truth_last=True,target_reads=0,n_per_view=count,
                                  modes=list(modes),views=list(SCREEN_VIEWS),results=outputs))


def wait_for_lora_fit(rt, wisig):
    root = Path(rt['lora_stage2'])
    deadline = monotonic() + 90 * 60
    while not (root / 'fit.json').is_file():
        if read(root / 'state.json')['status'] == 'FAILED':
            raise RuntimeError('LoRa structural baseline failed before producing frozen weights')
        if wisig.poll() not in (None, 0):
            raise RuntimeError('WiSig screen worker failed')
        if monotonic() >= deadline:
            raise RuntimeError('LoRa structural baseline exceeded the registered dependency wait')
        sleep(15)
    # The fitting worker writes this once, after closing the strict checkpoint.
    while not (root / 'fit.json').read_bytes().endswith(b'\n'):
        if monotonic() >= deadline:
            raise RuntimeError('Incomplete frozen LoRa fitting record')
        sleep(1)


def dispatch(runtime, rt):
    from experiments.pamf_lora_main.runner import available

    root=Path(rt['run_root'])
    root.mkdir()
    state=root/'state.json'

    def launch(action, gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                 OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        with (root/f'{action}.log').open('x',encoding='utf-8') as log:
            return subprocess.Popen([sys.executable,'-u','-m',__spec__.name,action,str(runtime)],
                                     stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)

    try:
        if rt.get('score_from'):
            job = launch('score',None)
            write(state,dict(status='SCORING_FROZEN',pid=job.pid),exclusive=False)
            if job.wait():
                raise RuntimeError('Independent scoring of the previously frozen screen failed')
            write(state,dict(status='ANALYZED',target_reads=0,reused_predictions=rt['score_from']),exclusive=False)
            return
        deadline=monotonic()+600
        while not (cards:=available(rt['allowed_gpus'],max_compute_processes=8)):
            if monotonic()>=deadline:
                raise RuntimeError('No registered GPU slot')
            sleep(10)
        if rt.get('reuse_wisig'):
            prior = Path(rt['reuse_wisig'])
            if (prior != root.parent / FREQUENCY_RUN or read(prior/'state.json')['status'] != 'FAILED'
                    or read(prior/'wisig'/'complete.json') != panel_receipt('wisig', FACTOR_COUNT, FREQUENCY_MODES)):
                raise ValueError('Expected complete WiSig predictions from the failed paired run')
            shutil.copytree(prior/'wisig', root/'wisig')
            card = cards[0]
            job = launch('lora', card)
            write(state, dict(status='RUNNING', workers={'lora':job.pid}, gpu=card,
                              reused_wisig=str(prior)), exclusive=False)
            if job.wait() or launch('score', None).wait():
                raise RuntimeError('Frozen LoRa/WiSig evaluation recovery failed')
            write(state, dict(status='ANALYZED', target_reads=0, reused_wisig=str(prior)), exclusive=False)
            return
        jobs={}
        for i,dataset in enumerate(('wisig','lora')):
            card=cards[min(i,len(cards)-1)]
            if dataset == 'lora' and rt.get('frequency_from') and not rt.get('stage_from'):
                wifi = jobs['wisig']
                write(state, dict(status='RUNNING', workers={'wisig':wifi.pid},
                                  pending='lora_stage2_source_freeze'), exclusive=False)
                wait_for_lora_fit(rt, wifi)
                deadline = monotonic() + 600
                while True:
                    wifi_exit = wifi.poll()
                    if wifi_exit not in (None, 0):
                        raise RuntimeError('WiSig screen worker failed')
                    fresh = available(rt['allowed_gpus'], max_compute_processes=8)
                    fresh = [gpu for gpu in fresh if wifi_exit == 0 or gpu != cards[0]]
                    if fresh:
                        card = fresh[0]
                        break
                    if monotonic() >= deadline:
                        raise RuntimeError('No registered GPU slot for the ready LoRa screen')
                    sleep(10)
            elif i and len(cards)==1 and jobs['wisig'].wait():
                raise RuntimeError('WiSig fit or prediction failed')
            jobs[dataset]=launch(dataset,card)
            write(state,dict(status='RUNNING',workers={k:p.pid for k,p in jobs.items()}),exclusive=False)
        exits={k:p.wait() for k,p in jobs.items()}
        if any(exits.values()) or launch('score',None).wait():
            raise RuntimeError(f'Screen pipeline failed after worker exits {exits}')
        write(state,dict(status='ANALYZED',target_reads=0),exclusive=False)
    except Exception as error:
        write(state,dict(status='FAILED',error=str(error)),exclusive=False)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('publish','dispatch','lora','wisig','wifi-smoke','score','stage-fit-lora','stage-fit-wisig'))
    parser.add_argument('runtime',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--ssh-config',type=Path)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--retry-score',action='store_true')
    modes.add_argument('--factor-screen',action='store_true')
    modes.add_argument('--frequency-screen',action='store_true')
    modes.add_argument('--retry-frequency',action='store_true')
    modes.add_argument('--stage-screen',action='store_true')
    modes.add_argument('--paired-detail-screen',action='store_true')
    modes.add_argument('--complex-residual-screen',action='store_true')
    modes.add_argument('--variance-cap-screen',action='store_true')
    modes.add_argument('--train-bound-screen',action='store_true')
    modes.add_argument('--spectral-guard-screen',action='store_true')
    modes.add_argument('--source-drift-screen',action='store_true')
    args=parser.parse_args()
    rt=read(args.runtime)
    if args.action=='publish':
        from experiments.pamf_lora_main.publish import release,release_commit
        project=rt['project']
        releases=f'{project}/releases'
        if args.retry_frequency:
            run = FREQUENCY_RETRY
        elif args.source_drift_screen:
            run = DRIFT_RUN
        elif args.spectral_guard_screen:
            run = SPECTRAL_RUN
        elif args.train_bound_screen:
            run = TRAIN_BOUND_RUN
        elif args.variance_cap_screen:
            run = VARIANCE_RUN
        elif args.complex_residual_screen:
            run = COMPLEX_RUN
        elif args.paired_detail_screen:
            run = PAIRED_RUN
        elif args.stage_screen:
            run = STAGE_RUN
        else:
            run = SCORE_RUN if args.retry_score else FREQUENCY_RUN if args.frequency_screen else FACTOR_RUN if args.factor_screen else RUN
        rt={k:rt[k] for k in ('project','python','frozen_sources')}
        rt.update(run_id=run,code_commit=release_commit(Path.cwd()),runner=__spec__.name,
                  release_root=f'{releases}/{run}-release',run_root=f'{project}/runs/{run}',
                  lora_ref=f'{releases}/{LO_RA}-release',wisig_ref=f'{releases}/{WISIG_REF}-release',
                  archive_root=f'{releases}/{ARCHIVE}-release',allowed_gpus=[5,2,6,3,0,7])
        rt['external_releases']=[rt[key] for key in ('lora_ref','wisig_ref','archive_root')]
        if args.retry_score:
            rt['score_from']=f'{project}/runs/{RUN}'
            rt['external_releases'].append(rt['score_from'])
        staged = (STAGE_RUN, PAIRED_RUN, COMPLEX_RUN, VARIANCE_RUN, TRAIN_BOUND_RUN, SPECTRAL_RUN, DRIFT_RUN)
        if args.factor_screen or run in (FREQUENCY_RUN, FREQUENCY_RETRY, *staged):
            rt['factor_from']=f'{project}/runs/{RUN}'
            rt['external_releases'].append(rt['factor_from'])
        if run in (FREQUENCY_RUN, FREQUENCY_RETRY, *staged):
            from experiments.pamf_lora_main.lora_dura_stage2_reference import RUN as LORA_STAGE2
            rt['frequency_from']=f'{project}/runs/{FACTOR_RUN}'
            rt['lora_stage2']=f'{project}/runs/{LORA_STAGE2}'
            rt['external_releases'].extend((rt['frequency_from'], rt['lora_stage2']))
            rt['allowed_gpus'] = [2, 3, 5, 7]
        if args.retry_frequency:
            rt['reuse_wisig'] = f'{project}/runs/{FREQUENCY_RUN}'
            rt['external_releases'].append(rt['reuse_wisig'])
        if run in staged:
            rt['stage_from'] = f'{project}/runs/{FREQUENCY_RETRY}'
            rt['external_releases'].append(rt['stage_from'])
            rt['allowed_gpus'] = [0, 2, 3, 5, 7]
        if run in (PAIRED_RUN, COMPLEX_RUN, VARIANCE_RUN, TRAIN_BOUND_RUN, SPECTRAL_RUN, DRIFT_RUN):
            rt['paired_from'] = f'{project}/runs/{STAGE_RUN}'
            rt['external_releases'].append(rt['paired_from'])
        if run in (COMPLEX_RUN, VARIANCE_RUN, TRAIN_BOUND_RUN, SPECTRAL_RUN, DRIFT_RUN):
            rt['complex_from'] = f'{project}/runs/{PAIRED_RUN}'
            rt['external_releases'].append(rt['complex_from'])
        if run in (VARIANCE_RUN, TRAIN_BOUND_RUN, SPECTRAL_RUN, DRIFT_RUN):
            rt['variance_from'] = f'{project}/runs/{COMPLEX_RUN}'
            rt['external_releases'].append(rt['variance_from'])
        if run in (TRAIN_BOUND_RUN, SPECTRAL_RUN, DRIFT_RUN):
            rt['train_bound_from'] = f'{project}/runs/{VARIANCE_RUN}'
            rt['external_releases'].append(rt['train_bound_from'])
        if run in (SPECTRAL_RUN, DRIFT_RUN):
            rt['spectral_from'] = f'{project}/runs/{TRAIN_BOUND_RUN}'
            rt['external_releases'].append(rt['spectral_from'])
        if run == DRIFT_RUN:
            rt['drift_from'] = f'{project}/runs/{SPECTRAL_RUN}'
            rt['external_releases'].append(rt['drift_from'])
        release(Path.cwd(),args.output,rt,args.ssh_config)
    elif args.action=='dispatch':
        dispatch(args.runtime,rt)
    elif args.action=='score':
        independent_score(rt)
    elif args.action=='wifi-smoke':
        numerical_policy()
        smoke_wifi(rt,Path(rt['run_root'])/'wisig')
    elif args.action.startswith('stage-fit-'):
        if rt.get('variance_from') and not rt.get('train_bound_from'):
            from experiments.pamf_lora_main.gain_dura_variance import calibrate_source as fit
        else:
            from experiments.pamf_lora_main.gain_dura_stage_fit import fit
        numerical_policy()
        fit(rt, args.action.removeprefix('stage-fit-'), DEV)
    else:
        numerical_policy()
        if rt.get('stage_from') and (not rt.get('spectral_from') or rt.get('drift_from')):
            subprocess.run([sys.executable,'-u','-m',__spec__.name,'stage-fit-'+args.action,
                            str(args.runtime)], check=True)
        folder=Path(rt['run_root'])/args.action
        folder.mkdir()
        if args.action=='lora':
            predict_lora(rt,folder)
        else:
            predict_wifi(rt,folder,args.runtime)


if __name__=='__main__':
    main()
