"""Focused fake-native channel checks; zero real IQ, CP, NN or physics calls."""
from copy import deepcopy
from dataclasses import asdict, dataclass
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
class Hardware:
    session_id: str
    image: float = .027


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


def stable_seed(seed, *parts):
    value = json.dumps([int(seed), *parts], ensure_ascii=False, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little')


class Primary:
    """Toy process preserves the native noise→IQ/rotation→signal+noise→AGC order."""
    def __init__(self):
        self.sessions = {}; self.made = []; self.fail = False; self.drift = False; self.no_noise_rng = False
        self.configs = {scene: Config(scene) for scene in channel.TRAIN_SCENES}
        self.ch = SimpleNamespace(stable_seed=stable_seed, normalize_input=self.normalize)
        self.residual = SimpleNamespace(ResidualChannel=self.constructor)
        self.normalize_calls = 0

    def normalize(self, x):
        self.normalize_calls += 1
        return x / max(float(np.sqrt(np.mean(np.abs(x)**2))), 1e-12), {}

    def constructor(self, cfg, seed, hardware):
        owner = self
        class Residual:
            def __init__(self):
                self.cfg, self.seed, self.receiver = cfg, seed, hardware
                names = ('geometry', 'states', 'shadow', 'scatter', 'phase', 'noise', 'sync', 'tracking', 'equalizer')
                self.latent = SimpleNamespace(rngs={k: np.random.default_rng(stable_seed(seed, k)) for k in names})
                if owner.no_noise_rng: del self.latent.rngs['noise']
                self.initial_streams = dict(self.latent.rngs)
                self.used = False; self.agc_calls = self.noise_chain_calls = 0
                owner.made.append(self)

            def process(self, raw):
                if owner.fail: raise RuntimeError('native-process-failure')
                if self.used: raise AssertionError('snapshot reused')
                rng = self.latent.rngs
                geometry = float(rng['geometry'].uniform(.3, 1.2))
                state = int(rng['states'].integers(0, 3))
                shadow = float(rng['shadow'].normal())
                scatter = float(rng['scatter'].normal())
                frequency = float(rng['sync'].normal())
                phase = float(rng['phase'].uniform(-np.pi, np.pi)) + rng['phase'].normal(0, .01, len(raw))
                tracking = rng['tracking'].normal(0, .001, len(raw))
                rotation = np.exp(1j*(phase+tracking+frequency*np.arange(len(raw))/25000000.))
                signal = (.3+geometry+.02*shadow)*raw*rotation + hardware.image*raw.conj()*rotation.conj()
                draws = rng['noise'].normal(size=(len(raw), 2))
                white = math.sqrt(.001/2)*(draws[:, 0]+1j*draws[:, 1])
                noise = white*rotation + hardware.image*white.conj()*rotation.conj()
                self.noise_chain_calls += 1
                y = signal+noise
                requested = -20*np.log10(max(float(np.sqrt(np.mean(abs(y)**2))), 1e-300))
                gain = float(np.clip(requested, -30, 30)); y *= 10**(gain/20)
                self.agc_calls += 1; self.used = True
                self.raw = raw.copy(); self.signal = signal.copy(); self.noise = noise.copy()
                self.changed_streams = {key for key, value in self.initial_streams.items() if rng[key] is not value}
                ps, pn = float(np.mean(abs(signal)**2)), float(np.mean(abs(noise)**2))
                output = dict(rms=float(np.sqrt(np.mean(abs(y)**2))), peak=float(max(abs(y))))
                metadata = dict(seed=seed, config_hash=cfg.scenario, scenario=cfg.scenario,
                    processing_route='residual', receiver=asdict(hardware), geometry=dict(scale=geometry),
                    state_end=('LOS', 'PARTIAL', 'BLOCKED')[state], receiver_lock_status='locked',
                    quality_snr_db=30+20*np.log10(geometry), quality_source='white_input_expected_power_proxy',
                    residual_scale=1., output_frequency_hz=frequency, phase_tracking_applied=True,
                    iq_compensation_applied=True, channel_equalization_applied=False,
                    residual_kernel_energy=geometry**2+scatter**2, input_metrics=dict(power=float(np.mean(abs(raw)**2))),
                    added_noise_variance_reference=.001, measured_added_noise_power_pre_agc=pn,
                    signal_to_added_noise_db=10*np.log10(ps/pn), agc_gain_db=gain,
                    agc_limited=abs(requested)>30, output_metrics=output)
                if owner.drift: metadata['output_frequency_hz'] += 1
                return y.astype(np.complex64), metadata
        return Residual()


def fixture(epoch=21, step=4441):
    primary = Primary()
    raw = torch.arange(64*2*256, dtype=torch.float32).reshape(64, 2, 256).sin() + .1
    meta = dict(eq_i=torch.ones(64, dtype=torch.long),
        rx_i=torch.tensor([channel.FIT[i % 5] for i in range(64)]),
        day_i=torch.arange(64) % 3+1, sig_i=torch.arange(64), base_index=torch.arange(64)+81)
    namespace = f'FcrossRX/train/e{epoch}/step{step}/L_S'; first = []; hardware = []; waves = []
    for i in range(64):
        pid = tuple(int(meta[key][i]) for key in channel.PHYSICAL_FIELDS)
        scene = str(np.random.default_rng(stable_seed(channel.TRAIN_SEED, namespace, 'scene', pid)).choice(channel.TRAIN_SCENES))
        cfg = primary.configs[scene]; session = f'rx{pid[1]}:day{pid[2]}:eq1'
        hw = Hardware(session); primary.sessions[session] = asdict(hw); hardware.append(hw)
        a = np.asarray(raw[i].tolist(), dtype=np.float64)
        normalized, _ = primary.ch.normalize_input(a[0]+1j*a[1])
        signal, record = primary.constructor(cfg, stable_seed(channel.TRAIN_SEED, namespace, scene, pid), hw).process(normalized)
        first.append(channel.serial(dict(record, scene=scene, sample_id=pid, session_id=session, namespace=namespace)))
        waves.append(np.stack((signal.real, signal.imag)))
    return primary, raw, meta, first, hardware, channel.tensor_iq(waves, 'cpu')


def run():
    start = time.perf_counter(); passed = []; fake_processes = 0
    for epoch, step in ((21, 4441), (30, 6660)):
        p, raw, meta, first, hw, first_iq = fixture(epoch, step)
        saved_raw = raw.clone(); saved_meta = deepcopy(meta); saved_first = deepcopy(first); saved_sessions = deepcopy(p.sessions)
        rng_np = np.random.get_state(); rng_python = random.getstate(); rng_torch = torch.get_rng_state().clone()
        output, records = channel.second_noise_view(p, raw, meta, first, hw)
        assert output.shape == (64, 2, 256) and output.dtype == torch.float32 and bool(torch.isfinite(output).all())
        assert not torch.equal(output, first_iq) and all(r['physics_seed'] == f['seed'] for r, f in zip(records, first))
        assert {r['sample_id'][1] for r in records} == set(channel.FIT)
        assert {r['scene'] for r in records} == set(channel.TRAIN_SCENES)
        assert len(p.made) == p.normalize_calls == 128
        for index, (a, b) in enumerate(zip(p.made[:64], p.made[64:])):
            assert b.cfg is a.cfg and b.receiver is hw[index] and b.seed == a.seed
            assert b.changed_streams == {'noise'} and a.changed_streams == set()
            assert np.array_equal(b.raw, a.raw) and np.array_equal(b.signal, a.signal)
            assert not np.array_equal(b.noise, a.noise)
            assert b.noise_chain_calls == b.agc_calls == 1 and b.used
            assert records[index]['native_agc_recomputed'] and not records[index]['first_agc_reused']
            assert records[index]['native_noise_chain_retained'] and records[index]['non_noise_metadata_equal']
            assert records[index]['namespace'] == first[index]['namespace'] + channel.SECOND_SUFFIX
        assert any(a['agc_gain_db'] != b['agc_gain_db'] for a, b in zip(first, records))
        assert len({round(r['measured_added_noise_power_pre_agc'], 10) for r in records}) > 1
        assert any(abs(r['measured_added_noise_power_pre_agc']-.001) > 1e-6 for r in records)
        assert torch.equal(raw, saved_raw) and all(torch.equal(meta[k], saved_meta[k]) for k in meta)
        assert first == saved_first and p.sessions == saved_sessions
        assert np.array_equal(np.random.get_state()[1], rng_np[1]) and np.random.get_state()[2:] == rng_np[2:]
        assert random.getstate() == rng_python and torch.equal(torch.get_rng_state(), rng_torch)
        fake_processes += len(p.made)
    passed.append('e21_e30_same_full_physics_exact_hardware_only_noise_rng_native_noise_chain_and_fresh_AGC_all_RX_scenes')
    passed.append('private_RNG_input_metadata_sessions_unchanged_and_noise_energy_not_exactly_calibrated')

    p, raw, meta, first, hw, _ = fixture()
    def reject(fn):
        try: fn()
        except ValueError: pass
        else: raise AssertionError('Invalid physical/noise-only contract accepted')
    reject(lambda: channel.second_noise_view(p, raw.double(), meta, first, hw))
    reject(lambda: channel.second_noise_view(p, raw[:63], meta, first, hw))
    reject(lambda: channel.second_noise_view(p, raw*float('nan'), meta, first, hw))
    reject(lambda: channel.second_noise_view(p, raw, dict(meta, rx_i=torch.zeros(64, dtype=torch.long)), first, hw))
    reject(lambda: channel.second_noise_view(p, raw, dict(meta, day_i=meta['day_i'].float()), first, hw))
    reject(lambda: channel.second_noise_view(p, raw, meta, first[:63], hw))
    reject(lambda: channel.second_noise_view(p, raw, meta, first, hw[:63]))
    corrupt = deepcopy(first); corrupt[0]['seed'] += 1
    reject(lambda: channel.second_noise_view(p, raw, meta, corrupt, hw))
    corrupt = deepcopy(first); corrupt[0]['namespace'] = 'FcrossRX/train/e20/step4440/L_S'
    reject(lambda: channel.second_noise_view(p, raw, meta, corrupt, hw))
    corrupt = deepcopy(first); corrupt[0]['session_id'] = 'different-session'
    reject(lambda: channel.second_noise_view(p, raw, meta, corrupt, hw))
    corrupt = deepcopy(first); corrupt[0]['scene'] = 'practical_high'
    reject(lambda: channel.second_noise_view(p, raw, meta, corrupt, hw))
    bad_hw = list(hw); bad_hw[0] = Hardware(hw[0].session_id, .5)
    reject(lambda: channel.second_noise_view(p, raw, meta, first, bad_hw))
    p.drift = True; reject(lambda: channel.second_noise_view(p, raw, meta, first, hw)); p.drift = False
    p.no_noise_rng = True; reject(lambda: channel.second_noise_view(p, raw, meta, first, hw)); p.no_noise_rng = False
    passed.append('bad_shape_dtype_physical_roles_scene_seed_session_hardware_nonnoise_replay_and_missing_noise_interface_rejected')

    process = p.residual.ResidualChannel; normalize = p.ch.normalize_input
    failure_raw = raw.clone(); failure_first = deepcopy(first); failure_sessions = deepcopy(p.sessions)
    p.fail = True
    try: channel.second_noise_view(p, raw, meta, first, hw)
    except RuntimeError as error: assert str(error) == 'native-process-failure'
    else: raise AssertionError('Native process failure hidden')
    assert p.residual.ResidualChannel is process and p.ch.normalize_input is normalize
    assert first == failure_first and torch.equal(raw, failure_raw) and p.sessions == failure_sessions
    passed.append('native_failure_propagates_without_monkeypatch_or_global_scope_to_restore')
    return dict(status='PASS', checks=len(passed), passed=passed,
        wall_seconds=time.perf_counter()-start, fake_native_processes_successful_valid_cases=fake_processes,
        real_CVS_NN=0, IQ_reads=0, checkpoint_reads=0, real_channel_process_calls=0)


if __name__ == '__main__':
    print(json.dumps(run(), allow_nan=False))
