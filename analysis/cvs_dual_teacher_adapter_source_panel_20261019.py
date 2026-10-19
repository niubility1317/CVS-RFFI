"""Fixed L180 metadata recovery and full-physics named-stream replay, without NN.

No Dataset, checkpoint or IQ file is opened here.  The caller supplies the
already role-checked raw L tensor.  Public helpers return (value, audit).
"""
from collections import Counter
from dataclasses import asdict, is_dataclass
import gzip
import json
from pathlib import Path

import numpy as np

STEPS = 2220
PER_EPOCH = 222
PARENT_STEPS = 4440
KEYS = ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')
FIT = (1, 3, 4, 6, 8)
VIEW_ORDER = ('raw', 'observed_last', 'fresh_mid_d0', 'fresh_mid_d1',
              'fresh_low_d0', 'fresh_low_d1')
FRESH_NAMESPACES = ('FdualAdapter/sourceFit/20261019/draw0',
                    'FdualAdapter/sourceFit/20261019/draw1')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pid(value):
    _require(isinstance(value, (list, tuple)) and len(value) == 5
             and all(type(v) is int for v in value), 'Five integer physical coordinates required')
    result = tuple(value)
    _require(result[0] == 1 and result[1] in FIT and result[2] in (1, 2, 3)
             and result[3] >= 0 and result[4] >= 0, 'Only original source eq/RX/day coordinates allowed')
    return result


def _selected(selected):
    _require(len(selected) == 180, 'Reuse exactly the existing fixed L180 panel')
    result = {}
    physical = set()
    cells = Counter()
    for row in selected:
        pid = _pid(row['pid'])
        identity = row['physical_id']
        _require(isinstance(identity, str) and len(identity) == 24
                 and all(c in '0123456789abcdef' for c in identity), 'Existing opaque physical ID required')
        _require(pid not in result and identity not in physical, 'Duplicate selected physical L')
        _require(type(row['TX']) is int and row['TX'] in range(6)
                 and (row['RX'], row['day']) == (pid[1], pid[2]), 'Selected visible L metadata differs')
        result[pid] = row
        physical.add(identity)
        cells[(row['TX'], row['RX'], row['day'])] += 1
    _require(cells == Counter({(t, r, d): 2 for t in range(6) for r in FIT for d in (1, 2, 3)}),
             'Fixed L180 requires two unique samples in each of ninety source cells')
    return result


def _jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            _require(bool(line.strip()), 'Blank training metadata row')
            value = json.loads(line)
            _require(isinstance(value, dict), 'Training metadata row must be a mapping')
            yield value


def _policy(channel_design=None, stable_seed=None):
    if channel_design is None:
        from experiments.cvs_fullsource_concat20 import design as channel_design
    if stable_seed is None:
        from leo_practical.channel import stable_seed
    scenes = tuple(channel_design.TRAIN_SCENES)
    _require(bool(scenes) and len(set(scenes)) == len(scenes), 'Original training scene choices required')
    return channel_design, stable_seed, scenes


def _scene(pid, namespace, channel_design, stable_seed):
    rng = np.random.default_rng(stable_seed(channel_design.CHANNEL_SEEDS['train'],
                                           namespace, 'scene', tuple(pid)))
    return str(rng.choice(channel_design.TRAIN_SCENES))


def _namespace(local_step):
    cumulative_step = PARENT_STEPS + local_step
    epoch = 20 + (local_step - 1) // PER_EPOCH + 1
    return f'FcrossRX/train/e{epoch}/step{cumulative_step}/L_S'


def _recover(sequence, mechanism, selected, channel_design, stable_seed):
    wanted = _selected(selected)
    sequence, mechanism = iter(sequence), iter(mechanism)
    last, batches, scene_counts = {}, {}, {}
    consumption = Counter()
    epochs = {e: Counter() for e in range(1, 11)}
    selected_occurrences = Counter()
    for step in range(1, STEPS + 1):
        epoch = (step - 1) // PER_EPOCH + 1
        pair = []
        for role, size in (('L', 128), ('U', 124 if step % PER_EPOCH == 0 else 256)):
            try:
                record = next(sequence)
            except StopIteration as error:
                raise ValueError('Incomplete ordered L/U training sequence') from error
            _require(record.get('role') == role and record.get('batch') == step,
                     'Exactly ordered L/U roles and cumulative local batch numbers required')
            ids = record.get('ids')
            _require(isinstance(ids, list) and len(ids) == size, 'Actual native L128/U256-or-last124 size differs')
            ids = [_pid(pid) for pid in ids]
            pair.append(ids)
            consumption[role + '_batches'] += 1
            consumption[role + '_packets'] += len(ids)
            epochs[epoch][role + '_batches'] += 1
            epochs[epoch][role + '_packets'] += len(ids)
        try:
            mech = next(mechanism)
        except StopIteration as error:
            raise ValueError('Incomplete mechanism step sequence') from error
        namespace = _namespace(step)
        _require(mech.get('step') == step and mech.get('epoch') == epoch
                 and mech.get('cumulative_step') == PARENT_STEPS + step
                 and mech.get('cumulative_epoch') == 20 + epoch,
                 'Mechanism local/cumulative step and epoch differ')
        expected = dict(L_forwards=2, student_L_packets=256, As_packets=128,
                        student_As_packets=128, channel_generated_packets=128,
                        fresh_As_packets=128, Tc_packets=128, Ts_packets=128)
        _require(all(mech.get(k) == v for k, v in expected.items()), 'Complete S3 L/As/teacher128 exposure differs')
        _require(mech.get('first_input') == 'raw' and mech.get('raw_first_forward_verified') is True
                 and mech.get('Ts_same_fullAs_IQ') is True and mech.get('As_student_same_IQ') is True
                 and mech.get('As_namespaces') == [namespace], 'Actual raw/full-As S3 input or namespace differs')
        counts = mech.get('As_scene_counts')
        _require(isinstance(counts, dict) and set(counts) <= set(channel_design.TRAIN_SCENES)
                 and all(type(n) is int and n >= 0 for n in counts.values())
                 and sum(counts.values()) == 128, 'Full per-step As scene counts required')
        batches[step], scene_counts[step] = pair[0], Counter({k: v for k, v in counts.items() if v})
        for slot, pid in enumerate(pair[0]):
            if pid in wanted:
                selected_occurrences[pid] += 1
                last[pid] = dict(local_step=step, cumulative_step=PARENT_STEPS + step,
                                 local_epoch=epoch, epoch=20 + epoch, slot=slot,
                                 pid=list(pid), physical_id=wanted[pid]['physical_id'], namespace=namespace)
    _require(next(sequence, None) is None and next(mechanism, None) is None,
             'Extra sequence or mechanism rows outside the fixed 2220 updates')
    _require(len(last) == 180 and set(last) == set(wanted), 'A selected L physical ID was never consumed')
    checked = []
    for step in sorted({row['local_step'] for row in last.values()}):
        namespace = _namespace(step)
        actual = Counter(_scene(pid, namespace, channel_design, stable_seed) for pid in batches[step])
        _require(actual == scene_counts[step], 'Reconstructed complete L128 scene choices disagree with mechanism')
        checked.append(dict(local_step=step, cumulative_step=PARENT_STEPS + step,
                            namespace=namespace, packets=128, scene_counts=dict(actual), matched=True))
    origins = []
    for row in selected:
        origin = dict(last[tuple(row['pid'])])
        origin['scene'] = _scene(origin['pid'], origin['namespace'], channel_design, stable_seed)
        origin['step_scene_counts_verified'] = True
        origins.append(origin)
    _require(consumption == Counter(L_batches=2220, U_batches=2220, L_packets=284160, U_packets=567000),
             'Complete source continuation consumption differs')
    _require(all(v == Counter(L_batches=222, U_batches=222, L_packets=28416, U_packets=56700)
                 for v in epochs.values()), 'Native per-epoch consumption differs')
    audit = dict(status='VERIFIED_METADATA_ONLY', selected_physical_IDs=180,
                 selection='Existing DR L180, unchanged order; last S3 L consumption including within-batch slot order',
                 consumption=dict(consumption), epochs=[dict(local_epoch=e, **dict(v)) for e, v in epochs.items()],
                 selected_L_consumptions=sum(selected_occurrences.values()),
                 selected_min_consumptions=min(selected_occurrences.values()),
                 selected_max_consumptions=max(selected_occurrences.values()),
                 training_scenes=list(channel_design.TRAIN_SCENES),
                 checked_origin_steps=checked, complete128_scene_choice_packets=128 * len(checked),
                 all_origin_scene_counts_match=True, U_TX_read=False, IQ_reads=0, NN=0, target_access=False)
    return origins, audit


def recover_origins(sequence_path, mechanism_path, selected, *, channel_design=None, stable_seed=None):
    """Return selected-order last L origins and a full 2220-step metadata audit.

    Optional policy injection exists for synthetic metadata tests.  Production
    defaults import the same original scene/seed policy as ContinuationPractical.
    `epoch` in each origin is cumulative21..30; `local_epoch` is1..10.
    """
    channel_design, stable_seed, _ = _policy(channel_design, stable_seed)
    return _recover(_jsonl(sequence_path), _jsonl(mechanism_path), selected,
                    channel_design, stable_seed)


def _check_origin(origin, selected):
    pid = _pid(origin['pid'])
    step, slot = origin['local_step'], origin['slot']
    _require(type(step) is int and step in range(1, STEPS + 1)
             and type(slot) is int and slot in range(128), 'Origin step or slot outside actual L128')
    _require(pid == tuple(selected['pid']) and origin['physical_id'] == selected['physical_id']
             and origin['cumulative_step'] == PARENT_STEPS + step
             and origin['epoch'] == 20 + (step - 1) // PER_EPOCH + 1
             and origin['namespace'] == _namespace(step)
             and origin.get('step_scene_counts_verified') is True,
             'Origin physical identity, cumulative coordinates or scene audit differs')


def build_views(raw, selected, origins, channel_design, Practical, serial):
    """Return CPU float32[N,6,2,256] and exact900-packet physics metadata.

    Uses the original single native normalize_input and full residual process
    once per generated observation, without output RMS, EQ, filtering or redraw.
    """
    import torch
    _selected(selected)
    _require(len(origins) == 180 and raw.shape == (180, 2, 256)
             and raw.dtype == torch.float32 and bool(torch.isfinite(raw).all()), 'Finite original raw L180 float32 required')
    for origin, record in zip(origins, selected):
        _check_origin(origin, record)
    p = Practical('train')
    _require(set(channel_design.TRAIN_SCENES) <= set(p.configs), 'Original training configurations missing')
    view_specs = [('observed_last', None, None),
                  ('fresh_mid_d0', 'practical_mid_urban', FRESH_NAMESPACES[0]),
                  ('fresh_mid_d1', 'practical_mid_urban', FRESH_NAMESPACES[1]),
                  ('fresh_low_d0', 'practical_low_urban', FRESH_NAMESPACES[0]),
                  ('fresh_low_d1', 'practical_low_urban', FRESH_NAMESPACES[1])]
    array = np.asarray(raw.detach().cpu().tolist(), dtype=np.float64)
    views = torch.empty((180, 6, 2, 256), dtype=torch.float32)
    views[:, 0] = raw.detach().cpu()
    records, coverage, sessions = [], Counter(), {}
    for vi, (view, fixed_scene, fixed_namespace) in enumerate(view_specs, 1):
        for i, (record, origin) in enumerate(zip(selected, origins)):
            pid = tuple(record['pid'])
            namespace = fixed_namespace or origin['namespace']
            scene = fixed_scene or _scene(pid, namespace, channel_design, p.ch.stable_seed)
            if view == 'observed_last':
                _require(scene == origin['scene'], 'Actual channel policy differs from recovered last scene')
            cfg = p.configs[scene]
            _require(cfg.fs_hz == 25000000. and cfg.fc_hz == 2462000000.
                     and cfg.processing_route == 'residual' and cfg.mode == 'post_sync'
                     and cfg.equalization_enabled is False and cfg.reference_snr_db == 30.
                     and cfg.boundary == 'reflect', 'Original full physical configuration differs')
            session = f'rx{pid[1]}:day{pid[2]}:eq{pid[0]}'
            seed = p.ch.stable_seed(channel_design.CHANNEL_SEEDS['train'], namespace, scene, pid)
            hardware = p.ch.receiver_for_session(cfg, channel_design.CHANNEL_SEEDS['receiver'], session)
            hw = asdict(hardware) if is_dataclass(hardware) else serial(hardware)
            _require(sessions.setdefault(session, hw) == hw, 'Physical receiver hardware changed within session')
            normalized, _ = p.ch.normalize_input(array[i, 0] + 1j * array[i, 1])
            signal, info = p.residual.ResidualChannel(cfg, seed, hardware).process(normalized)
            _require(signal.shape == (256,) and np.isfinite(signal).all()
                     and info['channel_equalization_applied'] is False, 'Finite full noEQ observation required')
            a = np.ascontiguousarray(np.stack((signal.real, signal.imag)), dtype=np.float32)
            views[i, vi] = torch.frombuffer(bytearray(a.tobytes()), dtype=torch.float32).reshape(2, 256)
            details = serial(dict(info, view=view, panel_index=i, physical_id=record['physical_id'],
                                  sample_id=pid, TX=record['TX'], RX=pid[1], day=pid[2],
                                  scene=scene, namespace=namespace, session_id=session, seed=int(seed)))
            records.append(details)
            coverage[(view, scene, info['state_end'], info['receiver_lock_status'])] += 1
    _require(len(records) == 900 and bool(torch.isfinite(views).all())
             and torch.equal(views[:, 0], raw.detach().cpu()), 'Exact raw+900 full-channel view budget differs')
    metadata = dict(generated_packets=900, observed_last_packets=180, fresh_packets=720,
                    raw_packets=180, views=list(VIEW_ORDER), shape=list(views.shape),
                    training_scenes=list(channel_design.TRAIN_SCENES),
                    channel_seeds=serial(channel_design.CHANNEL_SEEDS),
                    fresh_namespaces=list(FRESH_NAMESPACES), channel_audit=serial(p.audit()),
                    observed_scene_counts_check=dict(status='VERIFIED',
                        full128_counts_checked_in_recover=True,
                        selected_origin_steps=len({r['local_step'] for r in origins}),
                        selected_scenes_match_actual_channel_policy=True),
                    session_count=len(sessions), sessions=sessions, records=records,
                    coverage=[dict(view=k[0], scene=k[1], state=k[2], lock=k[3], n=v)
                              for k, v in sorted(coverage.items())],
                    native_normalizations=900, extra_RMS=0, output_normalization=False,
                    named_stream_reconstruction=True, historical_IQ_byte_comparison=False,
                    quality_filter=False, dropped_packets=0, optimizer_updates=0, NN=0,
                    U_TX_read=False, target_access=False)
    return views, metadata


def self_test():
    """Synthetic metadata negatives; no actual IQ, checkpoint, NN or channel."""
    import hashlib
    from types import SimpleNamespace
    def stable(seed, *parts):
        text = json.dumps([int(seed), *parts], ensure_ascii=False, separators=(',', ':'))
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')
    policy = SimpleNamespace(TRAIN_SCENES=('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban'),
                             CHANNEL_SEEDS=dict(train=20261016421, receiver=20261016423))
    selected = []
    for tx in range(6):
        for rx in FIT:
            for day in (1, 2, 3):
                for j in range(2):
                    i = len(selected)
                    selected.append(dict(pid=[1, rx, day, i, i], physical_id=format(i, '024x'), TX=tx, RX=rx, day=day))
    def batch(step):
        return [selected[((step - 1) * 128 + j) % 180]['pid'] for j in range(128)]
    def sequence(bad_role=False):
        for step in range(1, STEPS + 1):
            yield dict(role='V' if bad_role and step == 1 else 'L', batch=step, ids=batch(step))
            size = 124 if step % 222 == 0 else 256
            yield dict(role='U', batch=step, ids=[[1, 1, 1, j, 100000 + j] for j in range(size)])
    def mechanism(skip_first=False, wrong_counts=False):
        for step in range(2 if skip_first else 1, STEPS + 1):
            namespace = _namespace(step)
            counts = Counter(_scene(pid, namespace, policy, stable) for pid in batch(step)) if step >= STEPS - 1 else Counter({policy.TRAIN_SCENES[0]: 128})
            if wrong_counts and step == STEPS:
                counts = Counter({policy.TRAIN_SCENES[0]: 128})
            yield dict(step=step, epoch=(step - 1) // 222 + 1, cumulative_step=4440 + step,
                       cumulative_epoch=20 + (step - 1) // 222 + 1, L_forwards=2, student_L_packets=256,
                       As_packets=128, student_As_packets=128, channel_generated_packets=128,
                       fresh_As_packets=128, Tc_packets=128, Ts_packets=128, first_input='raw',
                       raw_first_forward_verified=True, Ts_same_fullAs_IQ=True, As_student_same_IQ=True,
                       As_namespaces=[namespace], As_scene_counts=dict(counts))
    origins, audit = _recover(sequence(), mechanism(), selected, policy, stable)
    _require(len(origins) == 180 and audit['consumption']['U_packets'] == 567000,
             'Synthetic positive recovery failed')
    _require({r['local_step'] for r in origins} == {STEPS - 1, STEPS}, 'Must recover last consumption, not first')
    negatives = []
    def reject(name, call):
        try:
            call()
        except (ValueError, KeyError):
            negatives.append(name)
        else:
            raise AssertionError('Expected rejection: ' + name)
    reject('wrong role', lambda: _recover(sequence(True), mechanism(), selected, policy, stable))
    reject('missing step', lambda: _recover(sequence(), mechanism(True), selected, policy, stable))
    reject('scene count mismatch', lambda: _recover(sequence(), mechanism(wrong_counts=True), selected, policy, stable))
    reject('slot128', lambda: _check_origin(dict(origins[0], slot=128), selected[0]))
    reject('wrong cumulative step', lambda: _check_origin(dict(origins[0], cumulative_step=1), selected[0]))
    reject('non-source RX', lambda: _pid([1, 2, 1, 0, 0]))
    reject('duplicate selection', lambda: _selected(selected[:-1] + [selected[0]]))
    # Exercise the complete public shape/order/budget interface with toy physics.
    # Neither the real Practical nor any physical source IQ is used by this test.
    import torch
    calls = Counter()
    class ToyChannel:
        stable_seed = staticmethod(stable)
        @staticmethod
        def receiver_for_session(cfg, seed, session):
            return dict(receiver_seed=seed, session=session)
        @staticmethod
        def normalize_input(z):
            calls['normalize'] += 1
            return z / np.sqrt(np.mean(abs(z) ** 2)), {}
    class ToyResidual:
        def __init__(self, cfg, seed, hardware):
            self.seed = seed
        def process(self, z):
            calls['process'] += 1
            return (z * (1 + .1 * (self.seed % 5))).astype(np.complex64), dict(
                channel_equalization_applied=False, state_end='LOS', receiver_lock_status='locked')
    class ToyPractical:
        def __init__(self, role):
            _require(role == 'train', 'Synthetic role differs')
            self.ch = ToyChannel
            self.residual = SimpleNamespace(ResidualChannel=ToyResidual)
            self.configs = {s: SimpleNamespace(fs_hz=25000000., fc_hz=2462000000.,
                processing_route='residual', mode='post_sync', equalization_enabled=False,
                reference_snr_db=30., boundary='reflect') for s in policy.TRAIN_SCENES}
        def audit(self):
            return dict(synthetic=True, scene_configs=list(self.configs))
    raw = torch.ones((180, 2, 256), dtype=torch.float32)
    rendered, metadata = build_views(raw, selected, origins, policy, ToyPractical, lambda x: x)
    _require(rendered.shape == (180, 6, 2, 256) and torch.equal(rendered[:, 0], raw)
             and metadata['generated_packets'] == 900 and calls == Counter(normalize=900, process=900)
             and metadata['views'] == list(VIEW_ORDER), 'Synthetic view interface/budget differs')
    return dict(status='PASS', synthetic=True, selected=180, steps=2220, negative_cases=negatives,
                complete_L_packets=284160, complete_U_packets=567000,
                recovered_origin_steps=2, synthetic_channel_packets=900,
                actual_IQ_reads=0, checkpoint_reads=0, NN=0, target_access=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if not args.self_test:
        parser.error('Only --self-test is a command-line action; use helpers from the caller')
    print(json.dumps(self_test(), ensure_ascii=False))
