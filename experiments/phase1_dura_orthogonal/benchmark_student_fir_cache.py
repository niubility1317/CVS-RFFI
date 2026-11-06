"""Full-budget source-only parity and timing of frozen student precomputation."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import torch

from experiments.phase1_dura_orthogonal.direct_student_fir import DirectStudentFIR, SOURCE_COMMIT
from experiments.phase1_dura_orthogonal.train_source import source_loss, posterior_teacher_loss


def load_inputs(cache_path, parent_path):
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    contract = cache['contract']
    expected = dict(schema='phase1_fixed_dura_source_context_v1', source_epochs=[21, 22, 23],
                    batch_size=32, precision='FP32_TF32off', dura='dura_v1_R65_E20_4940',
                    deterministic_algorithms=True, TX_labels_read=0, F_weights_read=0,
                    F_outputs_read=0, V_reads=0, target_reads=0)
    parent_expected = dict(commit=SOURCE_COMMIT, architecture='direct_spectral_student',
        objective='wide_source_distill', epochs=20, source_fit_size=32768, DURA_training_reads=1,
        source_iq_teacher='dura_v1_R65_E20_4940', source_iq_teacher_weight=.5,
        inference_received_iq_only=True, TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_reads=0, DURA_frozen=True)
    if (any(contract.get(k) != v for k, v in expected.items()) or
            any(parent.get(k) != v for k, v in parent_expected.items()) or
            len(contract['fit_ids']) != 32768 or
            not np.array_equal(np.asarray(contract['fit_ids']), np.asarray(parent['fit_ids']))):
        raise ValueError('Source-only parent and ordered source-cache IDs must match')
    shapes = ((98304, 2, 256), (98304, 2, 256), (98304, 2, 256), (98304,), (98304, 2, 256))
    data = cache['data']
    if (len(data) != 5 or any(v.dtype != torch.float32 or tuple(v.shape) != shape
                              for v, shape in zip(data, shapes)) or sum(cache['counts'].values()) != 98304):
        raise ValueError('Complete source-only cache required')
    return tuple(v.to('cuda:0') for v in data), parent['model']


def train(label, data, initial, cached, epochs):
    model = DirectStudentFIR().to('cuda:0').train()
    model.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.0003, weight_decay=.001)
    generator = torch.Generator().manual_seed(2026110613)
    torch.cuda.synchronize()
    started = time.monotonic()
    if cached:
        with torch.no_grad():
            base = torch.cat([model.student(data[0][i:i+32]) for i in range(0, len(data[0]), 32)])
        torch.cuda.synchronize()
    precompute_seconds = time.monotonic() - started if cached else 0.
    step_losses, epoch_losses = [], []
    for epoch in range(1, epochs+1):
        order = torch.randperm(len(data[0]), generator=generator).to('cuda:0')
        total = torch.zeros((), device='cuda:0')
        for start in range(0, len(order), 32):
            index = order[start:start+32]
            x, anchor, _, _, y = (v[index] for v in data)
            if cached:
                student = base[index]
            else:
                with torch.no_grad():
                    student = model.student(x, validate=False)
            output = model.refine(x, student)
            loss = source_loss(output, student, y, objective='dura_guard') + .5*posterior_teacher_loss(output, anchor, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Finite source loss required')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if (model.tap_head.weight.grad is None or not torch.isfinite(model.tap_head.weight.grad).all()
                    or any(p.requires_grad or p.grad is not None for p in model.student.parameters())):
                raise ValueError('Frozen student and finite FIR gradients required')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach()*len(index)
            step_losses.append(loss.detach())
        epoch_losses.append(float(total.cpu())/len(order))
        print('FIR_CACHE_EPOCH', label, epoch, 'loss', epoch_losses[-1],
              'elapsed', round(time.monotonic()-started, 2), flush=True)
    torch.cuda.synchronize()
    return model, torch.stack(step_losses).cpu(), epoch_losses, time.monotonic()-started, precompute_seconds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    if args.out.exists() or not torch.cuda.is_available():
        raise ValueError('Unique output and available GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    data, parent = load_inputs(args.cache, args.parent)
    torch.manual_seed(2026110612)
    initial = DirectStudentFIR().to('cuda:0')
    initial.student.load_state_dict(parent, strict=True)
    with torch.no_grad():
        base = initial.student(data[0][:32])
        if not torch.equal(initial.refine(data[0][:32], base), base):
            raise ValueError('Real source parent zero-head smoke required')
    initial_state = initial.state_dict()
    print('FIR_CACHE_REAL_SOURCE_SMOKE_PASS', flush=True)
    args.out.mkdir(parents=True)
    small = tuple(torch.cat([v[i:i+64] for i in (0, 32768, 65536)]) for v in data)
    for cached in (False, True):
        warm, _, _, _, _ = train('warm_cached' if cached else 'warm_original', small, initial_state, cached, 1)
        del warm
    old, old_steps, old_epochs, old_seconds, _ = train('original', data, initial_state, False, 5)
    new, new_steps, new_epochs, new_seconds, precompute = train('cached', data, initial_state, True, 5)
    different = [k for k in old.state_dict() if not torch.equal(old.state_dict()[k], new.state_dict()[k])]
    maximum = max((float((old.state_dict()[k]-new.state_dict()[k]).abs().max().cpu()) for k in different), default=0.)
    old.eval(); new.eval()
    equal_outputs = True
    with torch.no_grad():
        for offset in (0, 32768, 65536):
            for start in (offset, offset+32):
                equal_outputs &= torch.equal(old(data[0][start:start+32]), new(data[0][start:start+32]))
    result = dict(status='FIR_CACHE_FULL_SOURCE_PARITY_COMPLETE', commit=args.commit, source_parent_commit=SOURCE_COMMIT,
        source_ids=32768, source_views=98304, epochs=5, batch=32, steps=len(old_steps),
        step_losses_exact=torch.equal(old_steps, new_steps), epoch_losses_exact=old_epochs == new_epochs,
        parameters_exact=not different, different_parameters=different, max_parameter_abs_difference=maximum,
        source_outputs_exact=equal_outputs, source_output_count=192, precompute_seconds=precompute,
        elapsed_seconds={'uncached':old_seconds, 'cached_including_precompute':new_seconds},
        device=torch.cuda.get_device_name(), TX_labels_read=0, F_weights_read=0, F_outputs_read=0,
        V_reads=0, target_reads=0,
        caveat='Same initial state and full source budget on a local GPU; fixed order, no target predictions, not N607 run parity.')
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print('FIR_CACHE_FULL_SOURCE_PARITY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
