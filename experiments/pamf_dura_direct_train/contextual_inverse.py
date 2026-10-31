"""Scratch contextual spectral/time restoration, source-only and truth-last.

The inherited seven-view evaluator uses the historical F50.  It is a
restoration regression test, not the required Phase3 F0/open-world test.
"""
import gc
import json
import math
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from . import runner as base
from experiments.pamf_dura_prior_refine.model import spectral_error
from .contextual_model import ContextualSpectralRestorer

RUN = '20261101-phase1-contextual-inverse-E30-s2026092701-r01'
MODULE = 'experiments.pamf_dura_direct_train.contextual_inverse'
ARM = 'contextual_inverse'
EPOCHS, STEPS = 30, 7410


def make_model(seed=2026092701):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ContextualSpectralRestorer()


def configuration():
    value = base_configuration()
    value.update(run=RUN, arms=[ARM], recipe='scratch_conditional_spectral_and_time',
        parameters=sum(v.numel() for v in make_model().parameters()),
        epochs=EPOCHS, updates=STEPS, L_exposures=6300*EPOCHS, U_exposures=56700*EPOCHS,
        initialization='scratch_no_checkpoint', training_stages=1, inference_stages=1,
        objective='paired_IQ_NMSE + 0.05 * multiscale_STFT_logpower_error',
        source_supervision='L_s/U_s source simulated received and paired clean IQ; no TX/F/OE/V/target',
        peak_lr=1e-3, final_lr=2e-5, warmup=247, corruption_epochs=20,
        corruption_schedule='Original fixed 20 source views repeated in order for 30 epochs',
        source_freeze='fixed final E30/7410, no early stop or target selection',
        prior_calls=0, CG_updates=0, training_F_access=False, training_TX_labels=0,
        V_training=False, target_training=False, OE_training=False,
        goal_phase1='six practical scenes each within 3pp of paired raw clean; clean damage <=1pp',
        phase3='Historical F50 seven-view regression does not establish Phase3 F0 or open-world performance',
        max_total_visible_GPU_processes=3, allowed_GPUs=[4, 6, 7], max_own_workers=1,
        score_owner='one fixed scorer per row; all seven predictions before truth',
        target_feedback=False)
    return value


def learning_rate(update):
    if not 0 <= update < STEPS:
        raise ValueError('Fixed E30 update index required')
    if update < 247:
        return .001 * (update + 1) / 247
    fraction = (update - 247) / (STEPS - 247 - 1)
    return .00002 + (.001 - .00002) * .5 * (1 + math.cos(math.pi * fraction))


def objective(model, received, clean):
    restored = model(received)
    energy = clean.detach().square().mean((1, 2)).clamp_min(1e-4)
    iq = (restored - clean.detach()).square().mean((1, 2)) / energy
    spectrum = spectral_error(restored, clean)
    return iq + .05 * spectrum, {'iq_nmse': iq, 'spectral_error': spectrum}


def save_model(path, model, arm, epoch, updates, commit):
    payload = dict(schema='contextual_inverse_v1', arm=arm, epoch=epoch, updates=updates,
        code_commit=commit, config=configuration(), source_contract='original7/63/30',
        initialization='scratch_no_checkpoint', training_TX_labels_read=0,
        training_F_access=False, V_training=False, target_training=False,
        OE_training=False, model=model.state_dict())
    with path.open('xb') as stream:
        torch.save(payload, stream)


def load_model(path, device, *, arm, commit, epoch=None):
    if epoch is None:
        epoch = EPOCHS
    payload = torch.load(path, map_location='cpu', weights_only=False)
    expected = dict(schema='contextual_inverse_v1', arm=arm, epoch=epoch, updates=epoch*247,
        code_commit=commit, config=configuration(), source_contract='original7/63/30',
        initialization='scratch_no_checkpoint', training_TX_labels_read=0,
        training_F_access=False, V_training=False, target_training=False, OE_training=False)
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError('Exact source-only contextual checkpoint required')
    model = make_model()
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).eval().requires_grad_(False)


def smoke(args, p):
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    clean, roles = base.source_training(p)
    indices = np.flatnonzero(roles == 'L_s')[:2]
    if len(indices) != 2:
        raise ValueError('Two source L_s records required')
    source = np.load(p['cache']/'epochs/01/received.npy', mmap_mode='r', allow_pickle=False)
    x, y = base.tensor(clean[indices], 'cpu'), base.tensor(source[indices], 'cpu')
    model = make_model()
    initial = model(y)
    torch.testing.assert_close(initial, y, rtol=0, atol=0)
    loss, _ = objective(model, y, x)
    loss.mean().backward()
    gradients = [v.grad for v in model.parameters() if v.grad is not None]
    if not torch.isfinite(loss).all() or not gradients or not all(torch.isfinite(v).all() for v in gradients) or not any(v.abs().sum() > 0 for v in gradients):
        raise ValueError('Finite nonzero actual source gradient required')
    path = p['output']/'smoke-contextual.pt'
    save_model(path, model, ARM, 0, 0, args.commit)
    loaded = load_model(path, 'cpu', arm=ARM, commit=args.commit, epoch=0)
    reader = load_reader(p['reader'], p['source_cache'], 'cpu').eval().requires_grad_(False)
    with torch.no_grad():
        restored = loaded(y)
        torch.testing.assert_close(restored, initial, rtol=0, atol=0)
        logits = reader(restored)
    if logits.shape != (2, 6) or not torch.isfinite(logits).all():
        raise ValueError('Actual historical frozen F50/defaulthead smoke failed')
    base.write_json(p['output']/'smoke.json', dict(status='PASS', real_Ls=2,
        checkpoint_roundtrip=True, frozen_historical_F50=True, G_initialization_checkpoint_loads=0,
        TX_labels_read=0, F_training_inputs=0, V_reads=0, OE_reads=0, target_reads=0))


def train(args, p):
    clean, roles = base.source_training(p)
    model = make_model().to('cuda:0').train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    row = p['output']/args.arm
    row.mkdir()
    base.write_json(row/'launch.json', dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv,
        physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'], commit=args.commit))
    base.write_json(row/'resolved_config.json', configuration())
    rng = np.random.default_rng(2026103004)
    updates = seen_l = seen_u = 0
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (row/'training.jsonl').open('x', encoding='utf-8') as stream:
        for epoch in range(1, EPOCHS+1):
            view = (epoch-1) % 20 + 1
            received = np.load(p['cache']/'epochs'/f'{view:02d}'/'received.npy', mmap_mode='r', allow_pickle=False)
            if received.shape != (63000, 2, 256) or received.dtype != np.float32:
                raise ValueError('Full matching source received cache required')
            totals = dict(loss=0., iq_nmse=0., spectral_error=0.)
            clipped = 0
            for offset in range(0, 63000, 256):
                if offset == 0:
                    order = rng.permutation(63000)
                take = order[offset:offset+256]
                x, y = base.tensor(clean[take], 'cuda:0'), base.tensor(received[take], 'cuda:0')
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate(updates)
                loss, terms = objective(model, y, x)
                if not torch.isfinite(loss).all():
                    raise FloatingPointError('Nonfinite source objective')
                loss.mean().backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                clipped += int(norm > 1)
                updates += 1
                seen_l += int((roles[take] == 'L_s').sum())
                seen_u += int((roles[take] == 'U_s').sum())
                for key, value in dict(loss=loss, **terms).items():
                    totals[key] += float(value.detach().sum())
            record = dict(epoch=epoch, source_cache_view=view, updates=updates,
                losses={key: value/63000 for key, value in totals.items()},
                L_exposures=seen_l, U_exposures=seen_u, clip_fraction=clipped/247,
                seconds=time.perf_counter()-start, TX_labels_read=0, F_inputs=0, OE_inputs=0)
            stream.write(json.dumps(record, allow_nan=False)+'\n')
            stream.flush()
            print(json.dumps(dict(event='SOURCE_EPOCH_COMPLETE', **record)), flush=True)
    if (updates, seen_l, seen_u) != (STEPS, 6300*EPOCHS, 56700*EPOCHS):
        raise ValueError('Incomplete fixed source budget')
    save_model(row/'final.pt', model, args.arm, EPOCHS, updates, args.commit)
    base.write_json(row/'training_complete.json', dict(status='SOURCE_TRAINED', arm=args.arm,
        epoch=EPOCHS, updates=updates, L_exposures=seen_l, U_exposures=seen_u,
        training_TX_labels_read=0, training_F_access=False, V_reads=0, OE_reads=0, target_reads=0,
        initialization_checkpoint_loads=0, parameters=sum(v.numel() for v in model.parameters()),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), seconds=time.perf_counter()-start))


@torch.no_grad()
def predict(args, p, role):
    from . import evaluation
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    complete = base.read_json(p['output']/args.arm/'training_complete.json')
    if (complete['status'], complete['epoch'], complete['updates'], complete['target_reads']) != ('SOURCE_TRAINED', EPOCHS, STEPS, 0):
        raise ValueError('Fixed source training must finish before prediction')
    folder, ids, _, _ = evaluation.reference(p, role)
    model = load_model(p['output']/args.arm/'final.pt', 'cuda:0', arm=args.arm,
                       commit=getattr(args, 'checkpoint_commit', args.commit))
    reader = load_reader(p['reader'], p['source_cache'], 'cuda:0').eval().requires_grad_(False)
    output = p['output']/args.arm/(role+'_prediction')
    output.mkdir()
    start = time.perf_counter()
    timings = {}
    for view in base.VIEWS:
        received = np.load(folder/(view+'.npy'), mmap_mode='r', allow_pickle=False)
        if received.shape != (len(ids), 2, 256) or received.dtype != np.float32:
            raise ValueError('Original fixed received view required')
        iq = np.lib.format.open_memmap(output/(view+'.iq.npy'), mode='w+', dtype=np.float32, shape=received.shape)
        logits = np.lib.format.open_memmap(output/(view+'.logits.npy'), mode='w+', dtype=np.float32, shape=(len(ids), 6))
        raw = np.lib.format.open_memmap(output/(view+'.raw_logits.npy'), mode='w+', dtype=np.float32, shape=(len(ids), 6)) if role == 'source' else None
        view_start = time.perf_counter()
        for offset in range(0, len(ids), 128):
            end = min(offset+128, len(ids))
            x = base.tensor(received[offset:end], 'cuda:0')
            restored = model(x)
            scores = reader(restored)
            if restored.shape != x.shape or scores.shape != (len(x), 6) or not torch.isfinite(restored).all() or not torch.isfinite(scores).all():
                raise FloatingPointError('Invalid frozen G/F output')
            iq[offset:end] = base.numpy_values(restored)
            logits[offset:end] = base.numpy_values(scores)
            if raw is not None:
                raw[offset:end] = base.numpy_values(reader(x))
        iq.flush(); logits.flush()
        if raw is not None:
            raw.flush()
        timings[view] = time.perf_counter()-view_start
        print(json.dumps(dict(event='FROZEN_VIEW_COMPLETE', role=role, view=view, count=len(ids))), flush=True)
    np.save(output/'physical_ids.npy', ids, allow_pickle=False)
    base.write_json(output/'complete.json', dict(status='PREDICTIONS_COMPLETE', count=len(ids),
        views=list(base.VIEWS), role=role, arm=args.arm, F_run=base.F_RUN, F_seed=2026092701,
        original_default_head=True, TX_truth_reads=0, state_updates=0, head_fit_updates=0,
        code_commit=args.commit, seconds=time.perf_counter()-start, per_view_seconds=timings,
        inference='direct eager FP32; no DURA solver or fabricated variance'))


def profile(args, p):
    from experiments.pamf_fcrossrx_readout.frozen import load_reader
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    from .evaluation import conv_linear_mac
    student = load_model(p['output']/args.arm/'final.pt', 'cuda:0', arm=args.arm,
                         commit=getattr(args, 'checkpoint_commit', args.commit))
    baseline, _ = base.parent_model(p, 'cuda:0')
    baseline.eval().requires_grad_(False)
    reader = load_reader(p['reader'], p['source_cache'], 'cuda:0').eval().requires_grad_(False)
    x = base.tensor(np.load(p['cache']/'validation/practical_low_urban.npy', mmap_mode='r', allow_pickle=False)[:128], 'cuda:0')
    mac = conv_linear_mac(student, x[:1])
    results = {}
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for scope in ('G', 'G_plus_F'):
            old = InferenceEngine(baseline, reader if scope == 'G_plus_F' else None, reuse=True, graph=True, native=False)
            def new(z):
                restored = student(z)
                return reader(restored) if scope == 'G_plus_F' else restored
            for batch in (1, 128):
                times = {'DURA_graph': [], 'context_eager': []}
                for repeat in range(3):
                    order = [('DURA_graph', old), ('context_eager', new)]
                    if repeat % 2:
                        order.reverse()
                    for name, fn in order:
                        for _ in range(3):
                            fn(x[:batch])
                        torch.cuda.synchronize()
                        for _ in range(16):
                            t = time.perf_counter(); fn(x[:batch]); torch.cuda.synchronize()
                            times[name].append((time.perf_counter()-t)*1000)
                results[scope+'_B'+str(batch)] = {name: dict(median_ms=float(np.median(v)),
                    p95_ms=float(np.quantile(v, .95)), throughput_samples_s=float(batch*1000/np.mean(v)),
                    iterations=len(v)) for name, v in times.items()}
    base.write_json(p['output']/args.arm/'matched_profile.json', dict(status='PROFILE_COMPLETE', cases=results,
        parameters=sum(v.numel() for v in student.parameters()), ConvLinear_MAC=mac,
        checkpoint_bytes=(p['output']/args.arm/'final.pt').stat().st_size, spectral_FFT_IFFT=3,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), physical_GPU=os.environ['CUDA_VISIBLE_DEVICES'],
        timing_scope='Same GPU and FP32; DURA graph includes its validity checks; candidate eager; preloaded input; setup excluded',
        memory_scope='Shared resident F and both G models; not standalone VRAM'))


def worker(args, p):
    try:
        train(args, p)
        gc.collect(); torch.cuda.empty_cache()
        profile(args, p)
        for role in ('source', 'target'):
            predict(args, p, role)
            import fcntl
            with (p['output']/'score.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                subprocess.run(base.command(args, 'score', args.arm, role),
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
        base.write_json(p['output']/args.arm/'complete.json', dict(status='ANALYZED',
            all_seven_predictions_before_truth=True, target_feedback=False,
            phase3_test_complete=False))
    except BaseException as error:
        row = p['output']/args.arm
        if row.exists():
            base.write_json(row/'failure.json', dict(status='FAILED_TECHNICAL', error=repr(error),
                traceback=traceback.format_exc(), outputs_preserved=True))
        raise


base_configuration = base.configuration
base_command = base.command


def configure():
    def command(args, mode, arm=None, role=None):
        return [MODULE if x == 'experiments.pamf_dura_direct_train.runner' else x
                for x in base_command(args, mode, arm, role)]
    base.RUN = RUN
    base.ARMS = (ARM,)
    base.configuration = configuration
    base.command = command
    base.smoke = smoke
    base.worker = worker


if __name__ == '__main__':
    configure()
    base.main()
