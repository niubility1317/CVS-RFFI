"""Frozen source-only GPU timing for an expert mixture and its distilled student."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.pamf_phase1_flow.anchor_source_smoke import frozen_dura
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_dura_orthogonal.expert_teacher_contract import EXPERT_SOURCE, EXPERT_COMMIT, TEACHER_SOURCE, TEACHER_COMMIT
from experiments.phase1_dura_orthogonal.multiscale_spectral import DuraMultiscaleSpectral
from experiments.phase1_dura_orthogonal.source_expert_diagnostic import fixed_model
from experiments.phase1_dura_orthogonal.train_source import SCHEMA, WIDE_FIT_SIZE, extract_context

STUDENT_SOURCE = '20261107-phase1-dura-expert-distill-source-s2026092701-r01'
STUDENT_COMMIT = 'f1836093cfff10a735ebe68b42e11dcda9ebd93e'


def load_student(root, device):
    folder = root/'runs'/STUDENT_SOURCE
    summary = core.read(folder/'source_summary.json')
    state = torch.load(folder/'final.pt', map_location='cpu', weights_only=False)
    expected = dict(commit=STUDENT_COMMIT, architecture='dura_multiscale', objective='wide_expert_distill',
        epochs=5, source_fit_size=WIDE_FIT_SIZE, source_expert_teacher=TEACHER_SOURCE,
        source_expert_teacher_commit=TEACHER_COMMIT, source_expert_teacher_weight=.5,
        expert_teacher_inference_reads=0, DURA_training_reads=1, DURA_frozen=True,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0)
    if (summary.get('status') != 'SOURCE_TRAINED' or state.get('schema') != SCHEMA or
            summary.get('fit_ids') != WIDE_FIT_SIZE or any(
                obj.get(k) != v for obj in (summary, state) for k, v in expected.items())):
        raise ValueError('Fixed source-only distilled student required')
    parent = torch.load(root/'runs'/EXPERT_SOURCE/'final.pt', map_location='cpu', weights_only=False)
    fit = np.asarray(state.get('fit_ids', []))
    if len(fit) != WIDE_FIT_SIZE or not np.array_equal(fit, parent.get('fit_ids')):
        raise ValueError('Student and source expert teacher must use matching fit IDs')
    model = DuraMultiscaleSpectral(width=state['width'], taps=9)
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval().requires_grad_(False)


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, (time.perf_counter()-start)*1000


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(batches=len(a), mean_ms=float(a.mean()), median_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a, 95)))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Unique benchmark output required')
    if args.device != 'cuda:0' or not torch.cuda.is_available():
        raise ValueError('Registered CUDA device required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, rx, scene, _, received, _ = source_inputs(args.root)
    selected = []
    for receivers in ((4, 6), (8,)):
        for view in core.VIEWS:
            positions = np.flatnonzero(np.isin(rx, receivers) & (scene == view))
            if len(positions) < 8:
                raise ValueError('Eight source samples per group required')
            selected.extend(positions[:8].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) != 112 or len(np.unique(selected)) != 112:
        raise ValueError('Distinct source samples required')
    expert = fixed_model(args.root, EXPERT_SOURCE, EXPERT_COMMIT, ids, args.device)
    student = load_student(args.root, args.device)
    dura = frozen_dura(args.root, args.device)
    x = core.tensor(received[ids[selected]], args.device)
    # Real-checkpoint no-query smoke, then fixed same-input paired timings.
    ctx = extract_context(dura, x[:8])
    for model in (expert, student):
        output = model(x[:8], *ctx)
        if output.shape != x[:8].shape or not torch.isfinite(output).all():
            raise ValueError('Finite fixed source output required')
    print('EXPERT_STUDENT_LATENCY_SMOKE_PASS', flush=True)
    results = {}
    for batch_size in (1, 8):
        batches = [x[i:i+batch_size] for i in range(0, len(x), batch_size)]
        contexts = [extract_context(dura, batch) for batch in batches]
        for _ in range(5):
            expert(batches[0], *contexts[0]); student(batches[0], *contexts[0])
            expert(batches[0], *extract_context(dura, batches[0]))
            student(batches[0], *extract_context(dura, batches[0]))
        torch.cuda.synchronize()
        times = {k: [] for k in ('expert_correction', 'student_correction', 'dura_expert', 'dura_student')}
        # Alternate pair order per batch; two passes at B8 provide 28 paired samples.
        for repeat in range(2 if batch_size == 8 else 1):
            for i, (batch, context) in enumerate(zip(batches, contexts)):
                names = ('expert', 'student') if (i+repeat) % 2 == 0 else ('student', 'expert')
                for name in names:
                    model = expert if name == 'expert' else student
                    _, elapsed = timed(lambda: model(batch, *context))
                    times[name+'_correction'].append(elapsed)
                for name in reversed(names):
                    model = expert if name == 'expert' else student
                    _, elapsed = timed(lambda: model(batch, *extract_context(dura, batch)))
                    times['dura_'+name].append(elapsed)
        results[str(batch_size)] = {k: stats(v) for k, v in times.items()}
        print('EXPERT_STUDENT_LATENCY_BATCH_COMPLETE', batch_size, flush=True)
    args.out.mkdir(parents=True)
    core.write(args.out/'result.json', dict(status='SOURCE_LATENCY_COMPLETE', commit=args.commit,
        expert_source=EXPERT_SOURCE, expert_commit=EXPERT_COMMIT, student_source=STUDENT_SOURCE,
        student_commit=STUDENT_COMMIT, source_records=len(selected), device=args.device,
        expert_parameters=sum(p.numel() for p in expert.parameters()),
        student_parameters=sum(p.numel() for p in student.parameters()), results=results,
        TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0,
        scope='source-only eager FP32 synchronized wall time; no target accuracy or official graph latency'))
    print('EXPERT_STUDENT_LATENCY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
