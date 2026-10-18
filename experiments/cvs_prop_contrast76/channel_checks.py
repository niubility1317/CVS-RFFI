"""Rich fake-native plumbing only: zero native imports, real physics, IQ, CP or NN."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from . import channel


@dataclass(frozen=True)
class Config:
    scenario: str
    fs_hz: float = 25000000.
    fc_hz: float = 2462000000.
    processing_route: str = 'residual'
    mode: str = 'post_sync'
    equalization_enabled: bool = False
    reference_snr_db: float = 30.
    boundary: str = 'reflect'


@dataclass(frozen=True)
class Hardware:
    session_id: str
    oscillator_cfo_hz: float
    image: float = .027


@dataclass(frozen=True)
class Geometry:
    elevation_deg: float
    distance_m: float
    range_rate_mps: float


def stable_seed(seed, *parts):
    value = json.dumps([int(seed), *parts], ensure_ascii=False, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(value.encode('utf-8')).digest()[:8], 'little')


def rng_snapshot(rngs):
    return {k: channel.serial(deepcopy(v.bit_generator.state)) for k, v in rngs.items()}


def fixture():
    # Every sig_i is deliberately outside the six-class label range.
    raw = (torch.arange(64*2*256, dtype=torch.float32).reshape(64, 2, 256)*.071).sin()+.12
    meta = dict(eq_i=torch.ones(64, dtype=torch.long),
        rx_i=torch.tensor([channel.FIT[i % 5] for i in range(64)]),
        day_i=(torch.arange(64)//5) % 3+1, sig_i=torch.arange(64)+1000,
        base_index=torch.arange(64)+5000, fakeTX=torch.arange(64) % 6)
    return raw, meta


class Primary:
    """Artificial algebra, not a propagation implementation or native wrapper."""
    def __init__(self):
        self.sessions = {}; self.hardware = {}; self.made = []; self.hardware_calls = []
        self.calls = self.packets = self.normalize_calls = self.fake_processes = 0
        self.coverage = Counter(); self.fail = False; self.used_on_create = False
        self.missing_stream = False; self.bad_boundary = False; self.bad_metadata = False
        self.configs = {scene: Config(scene) for scene in channel.TRAIN_SCENES}
        self.ch = SimpleNamespace(stable_seed=stable_seed, normalize_input=self.normalize,
                                  receiver_for_session=self.receiver_for_session)
        self.residual = SimpleNamespace(ResidualChannel=self.constructor)

    def receiver_for_session(self, cfg, seed, session):
        assert seed == channel.RECEIVER_SEED
        self.hardware_calls.append((cfg.scenario, seed, session))
        if session not in self.hardware:
            v = np.random.default_rng(stable_seed(seed, 'receiver', session))
            self.hardware[session] = Hardware(session, float(v.normal(0, 3)))
        return self.hardware[session]

    def normalize(self, raw):
        self.normalize_calls += 1
        rms = float(np.sqrt(np.mean(abs(raw)**2)))
        return raw/rms, rms

    def constructor(self, cfg, seed, hardware):
        owner = self

        class Residual:
            def __init__(self):
                self.cfg, self.seed, self.receiver = cfg, int(seed), hardware
                names = ('geometry', 'states', 'shadow', 'scatter', 'phase', 'noise', 'sync', 'tracking', 'equalizer')
                r = {name: np.random.default_rng(stable_seed(seed, name)) for name in names}
                geom = Geometry(float(r['geometry'].uniform(10, 80)),
                                float(r['geometry'].uniform(.3, 1.2)), float(r['geometry'].normal()))
                self.latent = SimpleNamespace(rngs=r, geometry=geom,
                    probs=np.array([.3, .4, .3]), state=int(r['states'].integers(0, 3)),
                    z=float(r['shadow'].normal()), ray_hz=r['scatter'].uniform(-1, 1, (3, 4)),
                    ray_coef=r['scatter'].normal(size=(3, 4))+1j*r['scatter'].normal(size=(3, 4)),
                    f_doppler=-.1*geom.range_rate_mps, phase=float(r['phase'].uniform(-math.pi, math.pi)),
                    tracking_z=float(r['tracking'].normal()), residual_hz=float(r['sync'].normal()),
                    coarse_doppler_error_hz=float(r['sync'].normal()),
                    pdp=np.array([.6, .3, .1]), delays=np.array([[0., 1., 3.]]*3), reference_power=1.3)
                a = self.latent
                a.f_total = a.f_doppler+hardware.oscillator_cfo_hz
                a.f_hat = a.f_total-a.residual_hz
                if owner.missing_stream: del r['noise']
                self.used = owner.used_on_create; self.process_calls = self.agc_calls = 0
                self.initial_private = {name: getattr(a, name) for name in channel.PRIVATE_SCALARS}
                self.initial_rng_objects = dict(r); self.initial_rng_states = rng_snapshot(r)
                self.initial_constants = (a.pdp.copy(), a.delays.copy(), a.reference_power)
                owner.made.append(self)

            def process(self, raw):
                if owner.fail: raise RuntimeError('fake-native-process-failure')
                assert not self.used
                owner.fake_processes += 1; self.process_calls += 1
                a = self.latent
                self.at_process_rng_states = rng_snapshot(a.rngs)
                self.at_process_private = {name: getattr(a, name) for name in channel.PRIVATE_SCALARS}
                # H is just a diagnostic vector assembled from the seven fields.
                # There is no FIR, orbit, channel import or physical simulation.
                h = a.ray_coef.sum(axis=1)*.005+(a.state+1)*.03+a.z*.002+a.geometry.distance_m
                self.h = h.copy(); self.raw = raw.copy()
                phase = a.phase+a.rngs['phase'].normal(0, .01, len(raw))
                tracking = a.tracking_z*.001+a.rngs['tracking'].normal(0, .001, len(raw))
                rotation = np.exp(1j*(phase+tracking))
                signal = h[0]*raw*rotation+hardware.image*raw.conj()
                draws = a.rngs['noise'].normal(size=(len(raw), 2))
                noise = math.sqrt(.001/2)*(draws[:, 0]+1j*draws[:, 1])*rotation
                y = signal+noise
                requested = -20*np.log10(max(float(np.sqrt(np.mean(abs(y)**2))), 1e-300))
                gain = float(np.clip(requested, -30, 30)); y *= 10**(gain/20)
                self.agc_calls += 1; self.used = True; self.noise = noise.copy()
                self.agc_gain = gain
                ps, pn = float(np.mean(abs(signal)**2)), float(np.mean(abs(noise)**2))
                quality = 10*np.log10(float(np.mean(abs(raw)**2))*float(np.sum(abs(h)**2))/.001)
                metadata = dict(seed=self.seed, receiver=asdict(hardware),
                    boundary='wrong' if owner.bad_boundary else 'synthetic_reflect',
                    geometry=asdict(a.geometry), state_start=('LOS', 'PARTIAL', 'BLOCKED')[a.state],
                    state_end=('LOS', 'PARTIAL', 'BLOCKED')[a.state], receiver_lock_status='locked',
                    quality_snr_db=float(quality), signal_to_added_noise_db=float(10*np.log10(ps/pn)),
                    agc_gain_db=gain, agc_limited=bool(abs(requested)>30),
                    channel_equalization_applied=False, residual_kernel_energy=float(np.sum(abs(h)**2)),
                    measured_added_noise_power_pre_agc=pn, output_metrics=dict(rms=float(np.sqrt(np.mean(abs(y)**2)))))
                if owner.bad_metadata: del metadata['quality_snr_db']
                return y.astype(np.complex64), metadata

        return Residual()


def assert_rng_equal(before):
    py, np_before, th = before
    after = np.random.get_state()
    assert random.getstate() == py and torch.equal(torch.get_rng_state(), th)
    assert np_before[0] == after[0] and np.array_equal(np_before[1], after[1]) and np_before[2:] == after[2:]


def invoke(primary, raw, meta, namespace, share):
    raw_before = raw.clone(); meta_before = deepcopy(meta)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    result = channel.grouped_view(primary, raw, meta, namespace, share)
    assert torch.equal(raw, raw_before) and all(torch.equal(meta[k], meta_before[k]) for k in meta)
    assert_rng_equal(before)
    return result


def matrix_checks():
    raw, meta = fixture(); primaries = {arm: Primary() for arm in ('C00', 'C10', 'C01', 'C11')}
    previous = {}; generated = 0
    for call, (epoch, step) in enumerate(((21, 4441), (30, 6660)), 1):
        ns = f'FcrossRX/train/e{epoch}/step{step}/L_S'; rows = {}
        for arm, primary in primaries.items():
            share = arm[1] == '1'; before = len(primary.made)
            out, records = invoke(primary, raw, meta, ns, share); rows[arm] = (out, records)
            assert out.shape == raw.shape and out.dtype == torch.float32 and bool(torch.isfinite(out).all())
            instances = primary.made[before:]; by_seed = {obj.seed: obj for obj in instances}
            groups = {tuple(record['group']) for record in records}; assert len(groups) == 15
            assert len(instances) == 64+len(groups) and primary.fake_processes == call*64
            assert len(primary.hardware_calls) == primary.normalize_calls == call*64
            assert primary.calls == call and primary.packets == call*64 and sum(primary.coverage.values()) == call*64
            assert len(primary.sessions) == 15
            assert primary._prop76_environment_initializations == call*15
            assert primary._prop76_personal_initializations == primary._prop76_native_process_calls == call*64
            expected_coverage = Counter()
            for index, record in enumerate(records):
                pid = tuple(int(meta[k][index]) for k in channel.PHYSICAL_FIELDS)
                group = (pid[1], pid[2], pid[0]); assert record['sample_id'] == list(pid)
                assert record['group'] == list(group) and record['sig_i_is_TX'] is False
                assert record['TX_used_in_group'] is False and record['namespace'] == ns
                scene_seed = stable_seed(channel.TRAIN_SEED, ns, 'scene', group)
                expected_scene = str(np.random.default_rng(scene_seed).choice(channel.TRAIN_SCENES))
                assert record['scene'] == expected_scene and record['scene_seed'] == scene_seed
                personal_seed = stable_seed(channel.TRAIN_SEED, ns, expected_scene, pid)
                env_seed = stable_seed(channel.TRAIN_SEED, ns, expected_scene, 'environment', group)
                assert record['personal_seed'] == record['packet_seed'] == record['seed'] == personal_seed
                assert record['environment_seed'] == env_seed
                a, env = by_seed[personal_seed], by_seed[env_seed]
                assert a.used and a.process_calls == a.agc_calls == 1 and not env.used and env.process_calls == 0
                assert all(a.latent.rngs[k] is v for k, v in a.initial_rng_objects.items())
                assert a.at_process_rng_states == a.initial_rng_states
                assert a.at_process_private == a.initial_private
                assert record['private_streams_preserved'] and record['native_agc_recomputed']
                assert record['agc_gain_db'] == a.agc_gain
                assert np.array_equal(a.latent.pdp, a.initial_constants[0])
                assert np.array_equal(a.latent.delays, a.initial_constants[1])
                assert a.latent.reference_power == a.initial_constants[2]
                assert a.latent.f_total == a.latent.f_doppler+a.receiver.oscillator_cfo_hz == record['f_total']
                assert a.latent.f_hat == a.latent.f_total-a.latent.residual_hz == record['initial_f_hat']
                assert a.receiver is primary.hardware[record['session_id']]
                assert record['receiver'] == primary.sessions[record['session_id']]
                assert record['environment_initializations_this_step'] == 15
                assert record['personal_initializations_this_step'] == record['native_waveforms_this_step'] == 64
                assert record['environment_processes_this_step'] == 0
                for name in channel.PRIVATE_STREAMS:
                    assert record['private_stream_seeds'][name] == stable_seed(personal_seed, name)
                if share:
                    assert record['propagation_matches_environment']
                    for name in channel.PROPAGATION:
                        left, right = getattr(a.latent, name), getattr(env.latent, name)
                        assert np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right
                    for name in ('probs', 'ray_hz', 'ray_coef'):
                        assert not np.shares_memory(getattr(a.latent, name), getattr(env.latent, name))
                else:
                    assert not record['propagation_matches_environment']
                expected_coverage[(group[0], group[1], record['scene'], record['state_end'], record['receiver_lock_status'])] += 1
            for name in channel.PRIVATE_STREAMS:
                assert len({r['private_stream_seeds'][name] for r in records}) == 64
            assert len({r['packet_phase'] for r in records}) == 64
            assert len({r['agc_gain_db'] for r in records}) == 64
            if call == 1: assert primary.coverage == expected_coverage
            else: assert primary.coverage == previous[arm]+expected_coverage
            previous[arm] = primary.coverage.copy()
            if share:
                for group in groups:
                    pack = [r for r in records if tuple(r['group']) == group]
                    assert len({json.dumps(r['geometry'], sort_keys=True) for r in pack}) == 1
                    assert len({r['state_end'] for r in pack}) == 1
                    assert len({r['residual_kernel_energy'] for r in pack}) == 1
                    assert len({r['packet_phase'] for r in pack}) == len(pack)
            generated += primary.fake_processes-(call-1)*64
        for left, right in (('C00', 'C01'), ('C10', 'C11')):
            assert torch.equal(rows[left][0], rows[right][0]) and rows[left][1] == rows[right][1]
        fields = ('sample_id', 'group', 'scene', 'scene_seed', 'environment_seed', 'personal_seed',
                  'private_stream_seeds', 'packet_phase', 'packet_tracking_z', 'packet_residual_hz', 'packet_coarse_error_hz')
        for same_rows in zip(*(rows[arm][1] for arm in primaries)):
            assert all(all(r[k] == same_rows[0][k] for k in fields) for r in same_rows)
    return generated


def identity_order_checks():
    raw, meta = fixture(); ns = 'FcrossRX/train/e21/step4441/L_S'
    for share in (False, True):
        first, records = invoke(Primary(), raw, meta, ns, share)
        changed = dict(meta, fakeTX=(meta['fakeTX']+3) % 6, tx_i=torch.full((64,), 999))
        again, again_records = invoke(Primary(), raw, changed, ns, share)
        assert torch.equal(first, again) and records == again_records
        order = torch.arange(63, -1, -1)
        reordered, details = invoke(Primary(), raw[order], {k:v[order] for k,v in meta.items()}, ns, share)
        assert torch.equal(reordered[order], first)
        for a, b in zip(records, reversed(details)):
            for key in ('sample_id', 'group', 'scene', 'scene_seed', 'personal_seed', 'environment_seed',
                        'private_stream_seeds', 'geometry', 'state_end', 'agc_gain_db'):
                assert a[key] == b[key]


def run():
    start = time.perf_counter(); passed = []; negatives = 0
    processes = matrix_checks()
    passed.append('four_cells_e21_e30_same_group_scene_private_seeds_costs_and_exact_counters')
    passed.append('shared_seven_fields_nonalias_personal_rng_frequency_and_fresh_packet_AGC')
    identity_order_checks()
    passed.append('eq_first_ids_large_sig_not_TX_extra_TX_ignored_order_independence_global_RNG_and_input_immutability')

    raw, meta = fixture(); ns = 'FcrossRX/train/e21/step4441/L_S'
    def reject(fn):
        nonlocal negatives
        try: fn()
        except ValueError: negatives += 1
        else: raise AssertionError('Invalid source/physical/channel contract accepted')
    for bad in (raw.double(), raw[:63], raw[:, :, :255], raw*float('nan')):
        reject(lambda bad=bad: channel.grouped_view(Primary(), bad, meta, ns, True))
    for bad_ns in ('source_eval/RX6/final30', ns+'/extra', 'FcrossRX/train/e20/step4440/L_S',
                   'FcrossRX/train/e30/step6659/L_S', 'FcrossRX/train/e21/step4440/L_S'):
        # 6659 is inside E30 and valid, so use the mismatching E29 instead.
        if 'step6659' in bad_ns: bad_ns = 'FcrossRX/train/e29/step6660/L_S'
        reject(lambda bad_ns=bad_ns: channel.grouped_view(Primary(), raw, meta, bad_ns, True))
    reject(lambda: channel.grouped_view(Primary(), raw, meta, ns, 1))
    for key, bad in (('eq_i', torch.zeros(64, dtype=torch.long)), ('rx_i', torch.full((64,), 99)),
                     ('day_i', torch.zeros(64, dtype=torch.long)), ('sig_i', torch.full((64,), -1)),
                     ('base_index', torch.arange(63)), ('day_i', meta['day_i'].float())):
        reject(lambda key=key, bad=bad: channel.grouped_view(Primary(), raw, dict(meta, **{key:bad}), ns, True))
    reject(lambda: channel.grouped_view(Primary(), raw, {k:v for k,v in meta.items() if k!='eq_i'}, ns, True))
    for key, value in (('boundary', 'edge'), ('equalization_enabled', True), ('mode', 'raw'),
                       ('reference_snr_db', 31.), ('processing_route', 'full')):
        p = Primary(); p.configs = {s: replace(cfg, **{key:value}) for s,cfg in p.configs.items()}
        reject(lambda p=p: channel.grouped_view(p, raw, meta, ns, False))
        assert not p.made and not p.hardware_calls
    passed.append('invalid_shape_dtype_source_namespace_roles_integer_metadata_and_physical_config_rejected')

    for flag in ('used_on_create', 'missing_stream', 'bad_boundary', 'bad_metadata'):
        p = Primary(); setattr(p, flag, True)
        reject(lambda p=p: channel.grouped_view(p, raw, meta, ns, True))
        assert p.calls == p.packets == 0 and not p.coverage
    p = Primary(); cfg = next(iter(p.configs.values())); hw = Hardware('rx1:day1:eq1', 2.)
    personal = p.constructor(cfg, 1, hw); environment = p.constructor(cfg, 2, hw)
    personal.used = True; reject(lambda: channel.mix_propagation(personal, environment)); personal.used = False
    environment.used = True; reject(lambda: channel.mix_propagation(personal, environment)); environment.used = False
    environment.cfg = replace(cfg, boundary='edge'); reject(lambda: channel.mix_propagation(personal, environment))
    p = Primary(); p.fail = True
    constructor, normalize, receiver = p.residual.ResidualChannel, p.ch.normalize_input, p.ch.receiver_for_session
    before_raw, before_meta = raw.clone(), deepcopy(meta)
    before_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    try: channel.grouped_view(p, raw, meta, ns, True)
    except RuntimeError as error: assert str(error) == 'fake-native-process-failure'
    else: raise AssertionError('Native process failure hidden')
    assert p.residual.ResidualChannel is constructor and p.ch.normalize_input is normalize
    assert p.ch.receiver_for_session is receiver and p.calls == p.packets == 0 and not p.coverage
    assert torch.equal(raw, before_raw) and all(torch.equal(meta[k], before_meta[k]) for k in meta)
    assert_rng_equal(before_rng)
    passed.append('used_mismatched_snapshots_missing_stream_bad_native_metadata_and_process_failure_leave_no_patch')
    return dict(status='PASS', checks=len(passed), passed=passed, negatives=negatives,
        wall_seconds=time.perf_counter()-start, fake_native_valid_matrix_processes=processes,
        real_channel_process_calls=0, real_CVS_NN=0, IQ_reads=0, checkpoint_reads=0)


if __name__ == '__main__':
    print(json.dumps(run(), allow_nan=False))
