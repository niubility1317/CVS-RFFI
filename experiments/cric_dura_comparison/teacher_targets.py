"""Bounded source-only reuse of deterministic, frozen teacher predictions.

The epoch and source selector belong to the cache, not the student seed. Student
sampling, optimizer updates and loss denominators remain outside this module.
"""
from pathlib import Path
import time
import numpy as np
import torch
from .io import read, write, tensor


def exposure_count(n):
    if n < 1:
        raise ValueError('A nonempty source pool is required')
    cycles, rest = divmod(247, (n + 255) // 256)
    return cycles * n + min(256 * rest, n)


def frozen_predict(module, received, predict=None):
    if module.training or any(p.requires_grad for p in module.parameters()):
        raise ValueError('Only eval-mode frozen teacher/prefix predictions may be cached')
    devices = [received.device.index] if received.is_cuda else []
    # A producer must not consume the student's random stream, even if a future
    # frozen implementation internally makes a deterministic random draw.
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        values = predict(received) if predict is not None else module.stage_outputs(received)[0]
        values = tuple(value.detach() for value in values)
    if not values or any(v.shape[0] != len(received) for v in values):
        raise ValueError('Batched teacher outputs are required')
    if any(v.dtype not in (torch.float32, torch.complex64) for v in values):
        raise ValueError('Cache retains FP32/complex64 numerical policy')
    if not all(bool(torch.isfinite(v).all()) for v in values):
        raise FloatingPointError('Nonfinite frozen teacher target')
    return values


class EpochTargets:
    """Lazy cache: evaluate each selected source index at most once this epoch."""
    def __init__(self, module, count, epoch, identity, predict=None,
                 storage='auto', device_limit_mib=64, max_mib=512):
        if count < 1 or not 1 <= epoch <= 20 or not identity:
            raise ValueError('Explicit teacher/source/epoch binding required')
        if storage not in ('auto', 'cpu', 'device'):
            raise ValueError('Unknown teacher cache storage')
        self.module, self.count, self.epoch, self.identity = module, count, epoch, identity
        self.predict, self.storage = predict, storage
        self.device_limit = int(device_limit_mib * 2**20)
        self.maximum = int(max_mib * 2**20)
        self.present = np.zeros(count, bool)
        self.buffers = None
        self.calls = self.teacher_samples = self.hits = self.requested = self.bytes = 0

    def get(self, indices, received, epoch=None):
        if epoch is not None and epoch != self.epoch:
            raise ValueError('Teacher targets cannot cross epochs')
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(indices) != len(received):
            raise ValueError('Matched source indices and teacher input are required')
        if not len(indices) or indices.min() < 0 or indices.max() >= self.count:
            raise ValueError('Forbidden teacher source index')
        self.requested += len(indices)
        self.hits += int(self.present[indices].sum())
        # Preserve first-occurrence order; a batch may itself contain repeats.
        missing, positions, seen = [], [], set()
        for pos, idx in enumerate(indices.tolist()):
            if not self.present[idx] and idx not in seen:
                seen.add(idx); missing.append(idx); positions.append(pos)
        if missing:
            take = torch.tensor(positions, dtype=torch.long, device=received.device)
            values = frozen_predict(self.module, received.index_select(0, take), self.predict)
            if self.buffers is None:
                self.bytes = sum(self.count * v[0].numel() * v.element_size() for v in values)
                if self.bytes > self.maximum:
                    raise ValueError('Teacher epoch cache exceeds declared memory budget')
                keep_device = self.storage == 'device' or (self.storage == 'auto' and self.bytes <= self.device_limit)
                device = received.device if keep_device else torch.device('cpu')
                self.buffers = tuple(torch.empty((self.count, *v.shape[1:]), dtype=v.dtype, device=device) for v in values)
            if len(values) != len(self.buffers):
                raise ValueError('Teacher trajectory changed inside epoch')
            put = torch.tensor(missing, dtype=torch.long, device=self.buffers[0].device)
            for buffer, value in zip(self.buffers, values):
                if buffer.shape[1:] != value.shape[1:] or buffer.dtype != value.dtype:
                    raise ValueError('Teacher trajectory shape/dtype changed')
                buffer.index_copy_(0, put, value.to(buffer.device))
            self.present[missing] = True
            self.calls += 1; self.teacher_samples += len(missing)
        take = torch.tensor(indices.tolist(), dtype=torch.long, device=self.buffers[0].device)
        return tuple(buffer.index_select(0, take).to(received.device).detach() for buffer in self.buffers)

    def stats(self):
        return dict(requested=self.requested, hit_samples=self.requested-self.teacher_samples,
                    teacher_samples=self.teacher_samples, teacher_calls=self.calls,
                    unique_cached=int(self.present.sum()), cache_bytes=self.bytes,
                    storage=str(self.buffers[0].device) if self.buffers else None)


def binding(cache_config, teacher_checkpoint, teacher_stage):
    """Use existing immutable checkpoint and data identities; no new hash gate."""
    if cache_config['roles'] not in ('L', 'LU'):
        raise ValueError('Source L/LU only')
    return dict(schema='dura_teacher_targets_v1', teacher_checkpoint=str(teacher_checkpoint),
                teacher_stage=teacher_stage, source_contract=read(cache_config['source_contract']),
                roles=cache_config['roles'], observation=dict(old_action=cache_config['old_action'],
                old_coarse=cache_config['old_coarse'], new_cache=cache_config['new_cache']),
                input_view='fixed_source_received_by_epoch', dtype='float32')


def produce_shared_epoch(root, identity, view, teacher, epoch, positions=(-2, -1), batch_size=64):
    """One explicit producer, immutable completion; readers never open partials."""
    folder = Path(root) / ('epoch%02d' % epoch)
    folder.mkdir(parents=True, exist_ok=False)
    if not 1 <= epoch <= 20 or not view.ready():
        raise ValueError('Complete source observations and registered epoch required')
    started = time.perf_counter(); values = None; calls = 0
    for start in range(0, len(view), batch_size):
        selected = np.arange(start, min(start + batch_size, len(view)), dtype=np.int64)
        arrays = view.batch(selected, epoch)
        device = next(teacher.parameters()).device
        result = frozen_predict(teacher, tensor(arrays['received'], device))
        chosen = tuple(result[pos] for pos in positions)
        if any(v.dtype != torch.float32 or v.shape[1:] != (2, 256) for v in chosen):
            raise ValueError('Only full FP32 IQ trajectories may be shared')
        if values is None:
            values = torch.empty((len(view), len(positions), 2, 256), dtype=torch.float32)
        values[start:start+len(selected)] = torch.stack(chosen, 1).cpu()
        calls += 1
    torch.save(values, folder / 'targets.pt')
    write(folder / 'complete.json', dict(status='TEACHER_TARGETS_COMPLETE', binding=identity,
        epoch=epoch, ids=view.ids.tolist(), positions=list(positions), shape=list(values.shape),
        teacher_samples=len(view), teacher_calls=calls, bytes=values.numel()*4,
        seconds=time.perf_counter()-started, TX_labels_read=0, F_reads=0, target_reads=0))


class SharedEpochTargets:
    def __init__(self, root, identity, ids, epoch, positions=(-2, -1)):
        folder = Path(root) / ('epoch%02d' % epoch)
        meta = read(folder / 'complete.json')
        if meta.get('status') != 'TEACHER_TARGETS_COMPLETE' or meta.get('binding') != identity or meta.get('epoch') != epoch:
            raise ValueError('Teacher/source/epoch cache binding differs')
        if meta['ids'] != np.asarray(ids).tolist() or meta['positions'] != list(positions):
            raise ValueError('Teacher physical IDs or trajectory positions differ')
        if any(meta.get(k) != 0 for k in ('TX_labels_read', 'F_reads', 'target_reads')):
            raise ValueError('Forbidden shared teacher provenance')
        self.values = torch.load(folder / 'targets.pt', map_location='cpu', weights_only=True, mmap=True)
        if self.values.dtype != torch.float32 or list(self.values.shape) != meta['shape'] or self.values.shape != (len(ids), len(positions), 2, 256):
            raise ValueError('Shared FP32 teacher tensor shape/dtype differs')
        self.epoch = epoch; self.requested = 0; self.meta = meta

    def get(self, indices, received, epoch=None):
        indices = np.asarray(indices)
        if epoch is not None and epoch != self.epoch:
            raise ValueError('Teacher targets cannot cross epochs')
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(indices) != len(received) or not len(indices) or indices.min() < 0 or indices.max() >= len(self.values):
            raise ValueError('Forbidden shared teacher source index')
        self.requested += len(indices)
        selected = self.values.index_select(0, torch.tensor(indices.tolist(), dtype=torch.long)).to(received.device)
        return tuple(selected[:, i].detach() for i in range(selected.shape[1]))

    def stats(self):
        return dict(requested=self.requested, teacher_samples=0, teacher_calls=0,
                    hit_samples=self.requested, shared_target_bytes=self.meta['bytes'])


def frozen_prefix_predictor(model, stage):
    if stage == 'action':
        return model.coarse, lambda x: (model.coarse(x),)
    if stage == 'r17':
        def predict(x):
            base, action, variance = model.frozen.estimate(x)
            return base, action.kernel, action.main, action.image, variance
        return model.frozen, predict
    raise ValueError('Only actually frozen action/R17 prefixes may be cached')


def r17_outputs_with_prefix(model, received, context):
    """Original R17 student computation; only its frozen context is supplied."""
    from experiments.pamf_action_restore.operators import Action
    from experiments.pamf_action_restore.models import SCALE
    from experiments.pamf_restoration.models import complex_iq, pair
    from experiments.pamf_uncertainty_restore.models import weighted_map
    base, kernel, main, image, native = context
    action = Action(kernel, main, image)
    variance = native[:, None].expand(-1, 256)
    observed = complex_iq(received); current = base; stages = []
    logvar = native.log().clamp(-16, 4)[:, None, None].expand(-1, 1, 256)/8
    for k in range(3):
        innovation = pair(action.adjoint(observed-action(complex_iq(current))))
        innovation = innovation/(innovation.square().mean((1, 2), keepdim=True)+1e-6).sqrt()
        inputs = torch.cat((received, base, current, innovation, logvar), 1)
        prior = base+SCALE*model.prior[0 if len(model.prior) == 1 else k](inputs)
        current = pair(weighted_map(action, observed, complex_iq(prior), variance, steps=2, initial=complex_iq(current)))
        stages.append(current)
    return stages
