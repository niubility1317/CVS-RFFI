"""One clean, truth-blind U_s probability pass per new noCFO CE E20 teacher."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from . import design as d
from experiments.cvs_nocfo_foundation import design as old
from .teacher import load as teacher_load
from experiments.cvs_equivariant_identity.precision import numerical_context, actual_flags
from experiments.cvs_stage2_hybrid.semisupervised import validate_unlabeled

SCHEMA = 'nocfo_ce_peer_probability_cache_v1'
COUNT = 56700
BATCH_SIZE = 256
SEEDS = (2026092701, 2026092702, 2026092703)


def validate_inputs(cache, expected_ids, expected_count=COUNT):
    count = validate_unlabeled(cache, expected_ids)
    if expected_count < 1 or count != expected_count:
        raise ValueError('Nonempty full U_s physical pool required')
    return count


def _rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state().clone(),
                cuda=torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None)


def _same_rng(before, after):
    a, b = before['numpy'], after['numpy']
    numpy_same = a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]
    cuda_same = before['cuda'] is None and after['cuda'] is None
    if before['cuda'] is not None and after['cuda'] is not None:
        cuda_same = torch.equal(before['cuda'], after['cuda'])
    return before['python'] == after['python'] and numpy_same and torch.equal(before['cpu'], after['cpu']) and cuda_same


def _model_state(model):
    # Include nonpersistent registered buffers as well as saved checkpoint state.
    return {kind + ':' + name: value.detach().cpu().clone()
            for kind, values in [('parameter', model.named_parameters()), ('buffer', model.named_buffers())]
            for name, value in values}


def infer_probabilities(model, x, device='cuda:0', batch_size=BATCH_SIZE, progress=None):
    """Exactly one forward per packet, with measured state/RNG invariance."""
    device = torch.device(device)
    if not torch.is_tensor(x) or x.ndim != 3 or x.shape[1:] != (2, 256) or len(x) < 1:
        raise ValueError('Nonempty U IQ[B,2,256] required')
    if x.dtype != torch.float32 or x.device.type != 'cpu' or not torch.isfinite(x).all():
        raise ValueError('Finite CPU float32 clean IQ required')
    if batch_size < 1:
        raise ValueError('Positive batch size required')
    model.eval()
    model.requires_grad_(False)
    before = _model_state(model)
    rng_before = _rng_state(device)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Teacher must be fresh and have no accumulated gradient')
    probabilities = torch.empty((len(x), 6), dtype=torch.float32, device='cpu')
    forwards = 0
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            logits = model(x[start:stop].to(device))
            if not torch.is_tensor(logits) or logits.shape != (stop - start, 6):
                raise ValueError('Teacher must return six native class logits per U packet')
            if logits.dtype != torch.float32 or logits.requires_grad or not torch.isfinite(logits).all():
                raise ValueError('Finite detached float32 teacher logits required')
            probability = logits.softmax(1)
            if not torch.isfinite(probability).all():
                raise FloatingPointError('Nonfinite U teacher probability')
            probabilities[start:stop].copy_(probability.cpu())
            forwards += 1
            if progress is not None and (forwards % 32 == 0 or stop == len(x)):
                progress(dict(packets=stop, total_packets=len(x), forwards=forwards,
                              seconds=time.perf_counter() - started))
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    after = _model_state(model)
    state_same = before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
    rng_same = _same_rng(rng_before, _rng_state(device))
    no_grad = all(parameter.grad is None for parameter in model.parameters())
    eval_mode = all(not module.training for module in model.modules())
    if not state_same or not rng_same or not no_grad or not eval_mode:
        raise ValueError('Teacher inference changed parameters/buffers, RNG, gradients or eval modes')
    if not torch.allclose(probabilities.sum(1), torch.ones(len(x)), atol=2e-6, rtol=0):
        raise ValueError('Native probability rows do not sum to one')
    proof = dict(eval_mode=True, no_grad=True, parameters_and_buffers_unchanged=True,
                 rng_unchanged=True, optimizer_updates=0, EMA_updates=0,
                 auxiliary_forwards=0, validation_packets=0, target_packets=0,
                 physical_packets=len(x), packet_occurrences=len(x), forward_calls=forwards,
                 batch_size=batch_size, dtype='float32', temperature=1.0,
                 inference_seconds=inference_seconds)
    return probabilities, proof


def _memory():
    try:
        import resource
        # N607 is Linux, where ru_maxrss is KiB.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024), None
    except ImportError:
        return None, 'Process peak RSS unavailable on this platform'


def prepare(device='cuda:0'):
    if tuple(d.SEEDS) != SEEDS:
        raise ValueError('Original three E20 teacher seeds differ')
    device = torch.device(device)
    root = d.BASE / 'peer_cache'
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(2)
    configs = [old.config(next(row for row in old.rows() if row['arm'] == 'CE' and row['model_seed'] == seed))
               for seed in SEEDS]
    if any(c['multilevel']['enabled'] or c['unlabeled'] for c in configs):
        raise ValueError('Teacher construction must use native CE only')
    cache_paths = {c['unlabeled_cache'] for c in configs}
    if len(cache_paths) != 1:
        raise ValueError('Teachers must use the same legal physical U cache')
    cache_path = Path(next(iter(cache_paths)))
    contract = d.read(d.SOURCE)
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    validate_inputs(cache, contract['role_ids']['U_s'])
    if set(cache['ids']) & (set(contract['role_ids']['L_s']) | set(contract['role_ids']['V'])):
        raise ValueError('U pool overlaps another source role')
    provenance, resources = {}, {}
    with numerical_context(d.FULL_FP32_POLICY):
        for seed, c in zip(SEEDS, configs):
            seed_started = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            print('PEER_CACHE_START ' + json.dumps(dict(seed=seed, count=COUNT, source_role='U_s',
                checkpoint=(Path(c['output_root'])/'final_ssdg.pth').as_posix(), batch_size=BATCH_SIZE, device=str(device),
                target_access=False, U_truth=False, backend_flags=actual_flags())), flush=True)
            model,verified = teacher_load(seed,device)
            if (verified.get('status') != 'VERIFIED' or verified.get('seed') != seed
                    or verified.get('epoch') != 20 or verified.get('optimizer_steps') != 4440
                    or verified.get('source_roles') != 'EXACT_MATCH'
                    or verified.get('training_target_access') is not False
                    or verified.get('scratch_only') is not True or verified.get('ancestors') != []
                    or Path(verified['checkpoint']).as_posix() != (Path(c['output_root'])/'final_ssdg.pth').as_posix()):
                raise ValueError('New noCFO CE E20 teacher provenance differs')
            if hasattr(model, 'multilevel'):
                raise ValueError('Teacher unexpectedly contains auxiliary heads')
            probabilities, proof = infer_probabilities(model, cache['x'], device,
                progress=lambda value: print('PEER_CACHE_PROGRESS ' + json.dumps(dict(seed=seed, **value)), flush=True))
            destination = root / f'seed-{seed}.pt'
            payload = dict(schema=SCHEMA, seed=seed, source_role='U_s', ids=list(cache['ids']),
                           probabilities=probabilities, provenance=verified)
            with destination.open('xb') as stream:
                torch.save(payload, stream)
            peak_rss, memory_note = _memory()
            measured = dict(proof, total_seconds=time.perf_counter() - seed_started,
                backend_flags=actual_flags(),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
                process_peak_rss_bytes=peak_rss, process_peak_rss_note=memory_note,
                teacher_parameters=sum(parameter.numel() for parameter in model.parameters()),
                trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                probability_tensor_bytes=probabilities.numel() * probabilities.element_size(),
                output_bytes=destination.stat().st_size, output=str(destination))
            provenance[str(seed)] = verified
            resources[str(seed)] = measured
            print('PEER_CACHE_COMPLETE ' + json.dumps(dict(seed=seed, **measured)), flush=True)
            del model, payload, probabilities
            gc.collect()
    complete = dict(schema=SCHEMA, status='COMPLETE', source_role='U_s', count=COUNT,
        seeds=list(SEEDS), target_access=False, U_truth=False, provenance=provenance,
        resources=resources, packet_occurrences=COUNT * len(SEEDS), physical_packets=COUNT,
        total_seconds=time.perf_counter() - started,
        output_bytes=sum(item['output_bytes'] for item in resources.values()),
        output_bytes_scope='three serialized probability cache files; completion JSON excluded',
        source_contract=str(d.SOURCE), unlabeled_cache=str(cache_path),
        preprocessing=cache['preprocessing'], device=str(device),
        hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        deployment_scope='Phase1 ground U supervision only; no teacher cache in prediction model',
        teacher_training_cost='Three existing new noCFO CE E20 models reused; historical training cost is not zero')
    with (root / 'completion.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(complete, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print('PEER_CACHE_ALL_COMPLETE ' + json.dumps(complete, ensure_ascii=False), flush=True)
    return complete


def self_check(device='cpu'):
    """Bounded synthetic protocol negatives; never reads real data or checkpoints."""
    from torch import nn
    count = 7
    cache = dict(x=torch.zeros(count, 2, 256), ids=[f'U-{index}' for index in range(count)],
                 role='U_s', preprocessing='synthetic test')
    cache.update({key: torch.zeros(count, dtype=torch.long) for key in ('rx', 'day', 'domain', 'eq', 'sig', 'base_index')})
    validate_inputs(cache, cache['ids'], count)
    rejected = []
    cases = [('truth', dict(cache, y=torch.zeros(count))),
             ('empty', dict(cache, x=torch.empty(0, 2, 256), ids=[])),
             ('duplicate_id', dict(cache, ids=['duplicate'] * count)),
             ('poison', dict(cache, x=torch.full_like(cache['x'], float('nan')))),
             ('wrong_role', dict(cache, role='V'))]
    for name, bad in cases:
        try:
            validate_inputs(bad, cache['ids'], count)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Invalid U cache accepted: ' + name)
    model = nn.Sequential(nn.Flatten(), nn.Linear(512, 12), nn.BatchNorm1d(12), nn.Dropout(.5), nn.Linear(12, 6)).to(device)
    with numerical_context(d.FULL_FP32_POLICY):
        probabilities, proof = infer_probabilities(model, cache['x'], device, batch_size=3)
        repeated, _ = infer_probabilities(model, cache['x'], device, batch_size=3)
    if not torch.equal(probabilities, repeated) or proof['forward_calls'] != 3 or proof['packet_occurrences'] != count:
        raise AssertionError('Teacher probability extraction is not deterministic/exactly one pass')
    class Mutating(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('counter', torch.tensor(0))
        def forward(self, x):
            self.counter.add_(1)
            return torch.zeros(len(x), 6, device=x.device)
    try:
        infer_probabilities(Mutating().to(device), cache['x'], device)
    except ValueError:
        rejected.append('state_mutation')
    else:
        raise AssertionError('Persistent teacher mutation accepted')
    class RandomTeacher(nn.Module):
        def forward(self, x):
            return torch.rand(len(x), 6, device=x.device)
    try:
        infer_probabilities(RandomTeacher().to(device), cache['x'], device)
    except ValueError:
        rejected.append('random_inference')
    else:
        raise AssertionError('Random teacher inference accepted')
    return dict(status='PASS', device=str(device), synthetic_only=True, rejected=rejected, proof=proof)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(args.device)))
    else:
        prepare(args.device)
