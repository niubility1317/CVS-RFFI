"""One nominal tail-As occurrence pool, reused once under two allocations.

Preparation reads only the explicitly supplied legal L trace.  Allocation never
generates, rejects, or replaces an observation and uses private NumPy RNGs.
"""
from contextlib import contextmanager
import gzip
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch

KEYS = ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')
BINS = (3., 8., 15.)
METHODS = ('POOL_RANDOM', 'QUALITY_STRAT')
_ROLE = re.compile(r'^\s*\{\s*"role"\s*:\s*"([^"]+)"')
_CACHE = {}
_SCHEDULE_CACHE = {}


class PoolPreparationError(ValueError):
    def __init__(self, summary):
        self.summary = summary
        super().__init__('Nominal As pool technical stop: ' + str(summary.get('reason', summary)))


def _serial(value):
    if isinstance(value, torch.Tensor): return _serial(value.detach().cpu().tolist())
    if isinstance(value, np.generic): return _serial(value.item())
    if isinstance(value, np.ndarray): return _serial(value.tolist())
    if isinstance(value, dict): return {str(k): _serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_serial(v) for v in value]
    if isinstance(value, complex): return [value.real, value.imag]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(_serial(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    os.replace(temporary, path)


def _read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


@contextmanager
def _preserve_rng():
    py, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(py); np.random.set_state(numpy); torch.set_rng_state(cpu)
        if cuda is not None: torch.cuda.set_rng_state_all(cuda)


def _new_channel():
    from experiments.cvs_clock_factorial30.channel import Practical
    from experiments.cvs_clock_factorial30.model import Shifted
    return Shifted(Practical('train'))


def _l_batches(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig') as stream:
        for line in stream:
            role = _ROLE.match(line)
            if role is None:
                raise ValueError('Trace must expose its role before any payload')
            if role.group(1) != 'L':
                continue  # Do not decode U IDs, labels, or any U payload.
            record = json.loads(line)
            ids = record.get('ids')
            if not isinstance(ids, list) or len(ids) != 128:
                raise ValueError('One original L128 occurrence list required')
            if any(not isinstance(pid, list) or len(pid) != 5 or any(type(x) is not int for x in pid) for pid in ids):
                raise ValueError('Five integer physical ID fields required')
            yield [tuple(pid) for pid in ids]


def _lookup(ctx):
    dataset = ctx['train_loader'].dataset
    if getattr(dataset, 'transform', None) is not None or getattr(dataset.base, 'transform', None) is not None:
        raise ValueError('Pool must read native raw-before-ordinary L IQ')
    selected = list(map(int, dataset.selected))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Unique selected legal L base indices required')
    lookup = {}
    for local, base_index in enumerate(selected):
        it = dataset.base.index[base_index]
        pid = (int(it.eq_i), int(it.rx_i), int(it.day_i), int(it.sig_i), base_index)
        if pid[0] != 1 or pid[1] not in (1, 3, 4, 6, 8) or pid[2] not in (1, 2, 3):
            raise ValueError('Only original equalized source RX/day L allowed')
        lookup[pid] = local
    domain_map = {int(k): int(v) for k, v in ctx['domain_label_map'].items()}
    if not domain_map or any(v not in range(15) for v in domain_map.values()):
        raise ValueError('Original compact source domain map required')
    return dataset, lookup, domain_map


def _batch(dataset, lookup, domain_map, ids, step):
    if any(pid not in lookup for pid in ids):
        raise ValueError('Trace physical ID outside selected legal L')
    slots = [slot for slot in range(128) if (slot + step) % 2 != 0]
    waves, labels, domains, source = [], [], [], []
    for slot in slots:
        pid = ids[slot]
        x, y, raw_domain, meta = dataset[lookup[pid]]
        if isinstance(y, torch.Tensor):
            if y.ndim != 0 or y.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError('A scalar integer true L label is required')
            y = y.detach().item()
        if tuple(int(meta[k]) for k in KEYS) != pid or isinstance(y, (bool, np.bool_)) or not isinstance(y, (int, np.integer)) or not 0 <= int(y) < 6:
            raise ValueError('Actual labeled L sample/physical metadata mismatch')
        if int(raw_domain) not in domain_map:
            raise ValueError('L raw domain absent from native compact map')
        if not isinstance(x, torch.Tensor) or x.shape != (2, 256) or not bool(torch.isfinite(x).all()):
            raise ValueError('Finite original L IQ2x256 required')
        waves.append(x.detach().cpu().float()); labels.append(int(y)); domains.append(domain_map[int(raw_domain)])
        source.append(_serial(dict(meta)))
    metadata = {key: torch.tensor([item[key] for item in source], dtype=torch.long) for key in KEYS}
    return torch.stack(waves), labels, domains, source, slots, metadata


def _request(trace_path, seed, updates, steps):
    if type(seed) is not int or type(updates) is not int or type(steps) is not int or updates <= 0 or steps <= 0 or updates % steps:
        raise ValueError('Explicit integer seed and complete fixed epoch/update budget required')
    return dict(schema='quality_risk70_nominal_pool_v1', trace_path=str(Path(trace_path).resolve()),
                seed=seed, updates=updates, steps=steps, N=updates * 64, batch_size=64, bins=list(BINS))


def _existing(directory, request):
    if (directory / 'failed.json').is_file():
        raise PoolPreparationError(_read(directory / 'failed.json'))
    if (directory / 'complete.json').is_file():
        value = _read(directory / 'complete.json')
        if any(value.get(k) != v for k, v in request.items()) or value.get('status') != 'READY':
            raise ValueError('Existing nominal pool request differs')
        return value
    return None


def prepare(ctx, directory, trace_path, *, seed, updates=2220, steps=222):
    """Generate the exact original As64 for each supplied tail L occurrence."""
    request = _request(trace_path, seed, updates, steps)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    lock = directory / 'prepare.lock'
    while True:
        existing = _existing(directory, request)
        if existing is not None: return existing
        try:
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            time.sleep(.2)
            continue
        else:
            with os.fdopen(fd, 'w', encoding='utf8') as stream:
                stream.write(json.dumps(dict(pid=os.getpid(), request=request)) + '\n')
            break
    arrays = {}; written = 0; counts = np.zeros(4, dtype=np.int64); started = time.perf_counter()
    try:
        existing = _existing(directory, request)
        if existing is not None: return existing
        if (directory / 'iq.npy').exists():
            raise ValueError('Partial nominal pool exists; preserve it instead of retrying')
        with _preserve_rng():
            dataset, lookup, domain_map = _lookup(ctx)
            channel = _new_channel()
            shape = dict(iq=((request['N'], 2, 256), np.float32), labels=((request['N'],), np.int64),
                         domains=((request['N'],), np.int64), physical_ids=((request['N'], 5), np.int64),
                         groups=((request['N'],), np.int64), quality=((request['N'],), np.float64),
                         origin_steps=((request['N'],), np.int64), origin_slots=((request['N'],), np.int64),
                         metadata_offsets=((request['N'],), np.int64))
            arrays = {name: np.lib.format.open_memmap(directory / (name + '.npy'), mode='w+', dtype=dtype, shape=size)
                      for name, (size, dtype) in shape.items()}
            arrays['quality'][:] = np.nan; arrays['groups'][:] = -1
            step = 0
            with (directory / 'metadata.jsonl').open('wb') as metadata_file:
                for ids in _l_batches(trace_path):
                    step += 1
                    if step > updates: raise ValueError('More than the declared original tail L updates')
                    epoch = (step - 1) // steps + 1
                    raw, labels, domains, source, slots, meta = _batch(dataset, lookup, domain_map, ids, step)
                    iq, observed = channel.apply(raw, meta, epoch=epoch, step=step)
                    if not isinstance(iq, torch.Tensor) or iq.shape != (64, 2, 256) or len(observed) != 64 or not bool(torch.isfinite(iq).all()):
                        raise ValueError('Every original As observation must be finite and retained')
                    begin, end = written, written + 64
                    arrays['iq'][begin:end] = np.asarray(iq.detach().cpu().tolist(), dtype=np.float32)
                    arrays['labels'][begin:end] = labels; arrays['domains'][begin:end] = domains
                    arrays['physical_ids'][begin:end] = [ids[slot] for slot in slots]
                    arrays['origin_steps'][begin:end] = step; arrays['origin_slots'][begin:end] = slots
                    bad_quality = False
                    for j, actual in enumerate(observed):
                        pid = ids[slots[j]]
                        if tuple(actual.get('sample_id', ())) != pid:
                            raise ValueError('Channel output lost the original physical source ID')
                        value = actual.get('signal_to_added_noise_db')
                        valid = isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))
                        group = int(np.searchsorted(BINS, float(value), side='right')) if valid else -1
                        row = _serial(dict(actual, **dict(zip(KEYS, pid)), physical_id=pid,
                                           label=labels[j], domain=domains[j], group=group,
                                           origin_step=step, origin_slot=slots[j], source_metadata=source[j]))
                        arrays['metadata_offsets'][begin+j] = metadata_file.tell()
                        metadata_file.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n').encode('utf8'))
                        if valid:
                            arrays['quality'][begin+j] = float(value); arrays['groups'][begin+j] = group; counts[group] += 1
                        else: bad_quality = True
                    written = end
                    if bad_quality:
                        raise ValueError('Missing/nonfinite signal_to_added_noise_db; preserve pool and stop without dropping')
            if step != updates or written != request['N']:
                raise ValueError('Original tail L trace does not match declared update count')
            if np.any((counts > 0) & (counts < updates)):
                raise ValueError('Each nonempty fixed quality group must support every update')
            for array in arrays.values(): array.flush()
            result = dict(request, status='READY', written=written, counts=counts.tolist(), group_counts=counts.tolist(),
                          p_g=(counts / request['N']).tolist(), channel=channel.summary(), actual_channel=channel.audit(),
                          positive_groups=np.flatnonzero(counts > 0).tolist(), empty_groups=np.flatnonzero(counts == 0).tolist(),
                          minimum_positive_group_count=int(counts[counts > 0].min()),
                          iq_dtype='float32', iq_shape=[request['N'], 2, 256], iq_bytes=request['N'] * 2 * 256 * 4,
                          source_IQ_observations=written, real_CVS_NN=0, extra_proposals=0, dropped=0,
                          occurrence_one_use=True, physical_ID_deduplication=False,
                          wall_seconds=time.perf_counter() - started)
            _write(directory / 'complete.json', result)
            return result
    except PoolPreparationError:
        raise
    except Exception as error:
        result = dict(request, status='TECHNICAL_STOP', reason=type(error).__name__ + ': ' + str(error),
                      written=written, counts=counts.tolist(), real_CVS_NN=0, dropped=0,
                      partial_pool_preserved=True, automatic_retry=False, wall_seconds=time.perf_counter() - started)
        _write(directory / 'failed.json', result)
        raise PoolPreparationError(result) from error
    finally:
        for array in arrays.values(): array.flush()
        lock.unlink()


def _schedule(groups, method, *, seed, updates):
    groups = np.asarray(groups)
    if method not in METHODS or groups.shape != (updates * 64,) or not np.issubdtype(groups.dtype, np.integer) or np.any((groups < 0) | (groups > 3)):
        raise ValueError('Fixed method and complete four-group pool required')
    totals = np.bincount(groups, minlength=4)
    if np.any((totals > 0) & (totals < updates)): raise ValueError('Every nonempty quality group must support every update')
    rng = np.random.default_rng(seed)
    if method == 'POOL_RANDOM':
        result = rng.permutation(len(groups)).reshape(updates, 64)
    else:
        counts = np.tile(totals // updates, (updates, 1))
        cursor = 0
        for group, remainder in enumerate(totals % updates):
            rows = (np.arange(int(remainder)) + cursor) % updates
            counts[rows, group] += 1; cursor += int(remainder)
        counts = counts[rng.permutation(updates)]
        shuffled = [rng.permutation(np.flatnonzero(groups == group)) for group in range(4)]
        offsets = [0] * 4; result = np.empty((updates, 64), dtype=np.int64)
        for step in range(updates):
            pieces = []
            for group in range(4):
                count = int(counts[step, group]); offset = offsets[group]
                pieces.append(shuffled[group][offset:offset + count]); offsets[group] += count
            row = np.concatenate(pieces)
            if len(row) != 64: raise ValueError('Balanced allocation row sum differs')
            result[step] = row[rng.permutation(64)]
        if offsets != totals.tolist(): raise ValueError('Balanced allocation column totals differ')
    if not np.array_equal(np.sort(result.ravel()), np.arange(len(groups))):
        raise ValueError('Pool allocation must use each occurrence exactly once')
    result.flags.writeable = False
    return result


def _tensor(array, dtype, device):
    packed = np.ascontiguousarray(array)
    return torch.frombuffer(bytearray(packed.tobytes()), dtype=dtype).reshape(packed.shape).to(device)


class Reader:
    def __init__(self, shared, schedule, method, seed):
        self._shared, self._schedule, self.method, self.seed = shared, schedule, method, seed
        self.calls = self.packets = 0; self.used_counts = np.zeros(4, dtype=np.int64)
    def __deepcopy__(self, memo):
        value = type(self)(self._shared, self._schedule, self.method, self.seed)
        memo[id(self)] = value
        return value
    def consume(self, step, epoch, device):
        spec = self._shared['summary']
        if step != self.calls + 1 or step not in range(1, spec['updates'] + 1) or epoch != (step - 1) // spec['steps'] + 1:
            raise ValueError('One ordered pool64 consumption per declared update')
        used = self._schedule[step - 1]; arrays = self._shared['arrays']; groups = arrays['groups'][used]
        counts = np.bincount(groups, minlength=4)
        metadata = []
        with (self._shared['directory'] / 'metadata.jsonl').open('rb') as stream:
            for index in used:
                stream.seek(int(arrays['metadata_offsets'][index])); metadata.append(json.loads(stream.readline()))
        record = dict(method=self.method, count=64, step=step, epoch=epoch,
                      used_pool_indices=used.tolist(), counts=counts.tolist(), group_counts=counts.tolist(),
                      p_g=list(spec['p_g']), groups=groups.tolist(),
                      origin_steps=arrays['origin_steps'][used].tolist(), origin_slots=arrays['origin_slots'][used].tolist())
        iq = _tensor(arrays['iq'][used], torch.float32, device)
        labels = _tensor(arrays['labels'][used], torch.long, device)
        domains = _tensor(arrays['domains'][used], torch.long, device)
        self.calls += 1; self.packets += 64; self.used_counts += counts
        return iq, labels, domains, metadata, record
    def audit(self):
        result = dict(self._shared['summary']['actual_channel'])
        result.update(enabled=True, fixed_pool=True, method=self.method, seed=self.seed,
                      summary=self._shared['summary'], actual_channel=self._shared['summary']['actual_channel'],
                      new_channel_calls_during_training=0, formal_test_changed=False)
        return result
    def summary(self):
        result = dict(self._shared['summary']['channel'])
        result.update(calls=self.calls, packets=self.packets, method=self.method, fixed_pool=True,
                      pool_N=self._shared['summary']['N'], group_counts=self.used_counts.tolist(),
                      p_g=list(self._shared['summary']['p_g']), new_channel_calls=0,
                      pool_channel_generation=self._shared['summary']['channel'],
                      unique_occurrences_used=self.packets, dropped=0, physical_ID_deduplication=False)
        return result


def load(directory, method, *, seed):
    directory = Path(directory).resolve(); key = str(directory)
    if method not in METHODS: raise ValueError('Unregistered nominal pool allocation')
    if key not in _CACHE:
        if (directory / 'failed.json').exists(): raise PoolPreparationError(_read(directory / 'failed.json'))
        summary = _read(directory / 'complete.json')
        if summary.get('status') != 'READY' or summary.get('seed') != seed:
            raise ValueError('Complete matching nominal pool required')
        names = ('iq', 'labels', 'domains', 'physical_ids', 'groups', 'quality', 'origin_steps', 'origin_slots', 'metadata_offsets')
        arrays = {name: np.load(directory / (name + '.npy'), mmap_mode='r', allow_pickle=False) for name in names}
        if arrays['iq'].shape != (summary['N'], 2, 256) or arrays['iq'].dtype != np.float32 or any(len(a) != summary['N'] for a in arrays.values()):
            raise ValueError('Complete nominal pool array shape/dtype differs')
        _CACHE[key] = dict(directory=directory, summary=summary, arrays=arrays)
    shared = _CACHE[key]
    if shared['summary']['seed'] != seed: raise ValueError('Pool allocation seed differs')
    schedule_key = (key, method, seed)
    if schedule_key not in _SCHEDULE_CACHE:
        _SCHEDULE_CACHE[schedule_key] = _schedule(shared['arrays']['groups'], method, seed=seed, updates=shared['summary']['updates'])
    return Reader(shared, _SCHEDULE_CACHE[schedule_key], method, seed)
