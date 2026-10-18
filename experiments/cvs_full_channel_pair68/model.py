"""Replace the second existing As64 IQ; keep all six calls and all other losses."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from weakref import WeakKeyDictionary
import numpy as np
import torch
from . import design as d
from experiments.cvs_satellite_dropout18 import model as original
from experiments.cvs_crossview_supcon41 import model as base
from experiments.cvs_clock_factorial30 import design as physics
from experiments.cvs_clock_factorial30.channel import serial

classifier, source_domain_count = original.classifier, original.source_domain_count
PHYSICAL_FIELDS = ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')
SECOND_SUFFIX = '/view1-full-independent'
_PHYSICS_SINKS = WeakKeyDictionary()


def physics_sink(model, sink=None):
    """A training-only writer outside model state and EMA deepcopy graphs."""
    if sink is None:
        _PHYSICS_SINKS.pop(model, None)
    else:
        _PHYSICS_SINKS[model] = sink


def as_ce_terms(z0, z1, labels, mask, terms):
    """Mean both CE128, plus only the ordinary64 half of mixed symmetric KL."""
    if (mask.shape != (128,) or mask.dtype != torch.bool or int(mask.sum()) != 64
            or z0.shape != (128, 6) or z1.shape != (128, 6)):
        raise ValueError('Only the actual64 As slots in the existing mixed128 pair')
    _, ce, kl, ce0, ce1 = terms(z0, z1, labels, 1.)
    ordinary_kl = terms(z0[~mask], z1[~mask], labels[~mask], 1.)[2]
    satellite_kl = terms(z0[mask], z1[mask], labels[mask], 1.)[2]
    used = .5 * ordinary_kl
    return (ce + used, ce, used, ce0, ce1), dict(
        original_mixed_KL=float(kl.detach()), ordinary_half_KL=float(ordinary_kl.detach()),
        satellite_half_KL=float(satellite_kl.detach()), used_mixed_KL=float(used.detach()),
        removed_satellite_KL=float((.5 * satellite_kl).detach()),
        satellite_CE0=float(torch.nn.functional.cross_entropy(z0[mask], labels[mask], label_smoothing=.01).detach()),
        satellite_CE1=float(torch.nn.functional.cross_entropy(z1[mask], labels[mask], label_smoothing=.01).detach()))


def independent_view(primary, raw, meta, first, hardware):
    """One full new physical stream, conditional on the same raw and RX objects.

    The first channel has already generated every receiver object. Its objects
    are passed directly to the second ResidualChannel, with no receiver redraw.
    No global NumPy/Torch RNG is consumed by this deterministic named stream.
    """
    if (raw.shape != (64, 2, 256) or len(first) != 64 or len(hardware) != 64
            or not bool(torch.isfinite(raw).all())):
        raise ValueError('Exactly64 finite raw L packets and first-view hardware required')
    values = {k: meta[k].detach().cpu().tolist() if torch.is_tensor(meta[k]) else list(meta[k]) for k in PHYSICAL_FIELDS}
    if any(len(v) != 64 for v in values.values()):
        raise ValueError('Complete physical L metadata required')
    waves, records = [], []
    raw_cpu = np.asarray(raw.detach().cpu().tolist(), dtype=np.float64)
    for i, observation in enumerate(first):
        pid = tuple(int(values[k][i]) for k in PHYSICAL_FIELDS)
        session = f'rx{pid[1]}:day{pid[2]}:eq{pid[0]}'
        namespace = observation['namespace']
        if (tuple(observation['sample_id']) != pid or observation['session_id'] != session
                or not namespace.startswith('FcrossRX/train/e') or not namespace.endswith('/L_S')
                or pid[1] not in d.FIT or pid[2] not in (1, 2, 3) or pid[0] != 1
                or asdict(hardware[i]) != primary.sessions[session]):
            raise ValueError('First/second physical IDs, original namespace or stable RX hardware differ')
        namespace += SECOND_SUFFIX
        rng = np.random.default_rng(primary.ch.stable_seed(physics.CHANNEL_SEEDS['train'], namespace, 'scene', pid))
        scene = str(rng.choice(physics.TRAIN_SCENES))
        cfg = primary.configs[scene]
        if cfg.equalization_enabled or cfg.processing_route != 'residual' or cfg.mode != 'post_sync':
            raise ValueError('Original full residual/post_sync/EQoff must remain')
        seed = primary.ch.stable_seed(physics.CHANNEL_SEEDS['train'], namespace, scene, pid)
        normalized, _ = primary.ch.normalize_input(raw_cpu[i, 0] + 1j * raw_cpu[i, 1])
        signal, actual = primary.residual.ResidualChannel(cfg, seed, hardware[i]).process(normalized)
        if not np.isfinite(signal).all() or actual['channel_equalization_applied']:
            raise ValueError('Full finite EQoff view required; no packet dropping')
        waves.append(np.stack((signal.real, signal.imag)))
        records.append(serial(dict(actual, scene=scene, sample_id=pid, session_id=session,
            namespace=namespace, hardware_object_reused=True, from_same_raw=True)))
    array = np.ascontiguousarray(waves, dtype=np.float32)
    return torch.frombuffer(bytearray(array.tobytes()), dtype=torch.float32).reshape(array.shape).to(raw.device), records


def hook(self, **kw):
    labels, step = kw['labels'], kw['optimizer_step']
    mask = (torch.arange(128, device=labels.device) + step).remainder(2) != 0
    raw = kw['labeled_inputs'].detach()
    meta = {k: v[mask.to(v.device)] if torch.is_tensor(v) and v.ndim > 0 and len(v) == 128 else v
            for k, v in kw['extra'][1].items()}
    channel = self._full_channel
    primary = channel.obj
    forward, apply, terms = self.forward_identity_only, channel.apply, base.terms
    had_forward, had_apply = 'forward_identity_only' in vars(self), 'apply' in vars(channel)
    old_forward, old_apply = vars(self).get('forward_identity_only'), vars(channel).get('apply')
    frames, term_calls, hardware, observations, diagnostics = 0, 0, [], [], {}
    first_satellite = None

    def record_channel(*args, **kwargs):
        if observations:
            raise ValueError('Exactly one original channel call per native L step')
        receiver = primary.ch.receiver_for_session
        def capture_receiver(*a, **k):
            result = receiver(*a, **k)
            hardware.append(result)
            return result
        primary.ch.receiver_for_session = capture_receiver
        try:
            value = apply(*args, **kwargs)
        finally:
            primary.ch.receiver_for_session = receiver
        if len(value[1]) != 64 or len(hardware) != 64:
            raise ValueError('Every first As packet must supply its actual hardware object')
        observations.append(value[1])
        return value

    def record_forward(m, x, *args, **kwargs):
        nonlocal frames, first_satellite
        frames += 1
        engine.frame += 1
        if frames == 3:
            if len(observations) != 1 or first_satellite is None:
                raise ValueError('Independent second view must follow original first As')
            independent, second = independent_view(primary, raw[mask], meta, observations[0], hardware)
            # Never mutate the IQ tensor already used by the first mixed pass.
            replaced = x.clone()
            replaced[mask] = independent
            if not torch.equal(replaced[~mask], x[~mask]):
                raise ValueError('Ordinary64 positions must be exactly preserved')
            diagnostics['As_IQ_pair_MSE'] = float((independent - first_satellite).square().mean())
            x = replaced
            observations.append(second)
        elif frames == 2:
            first_satellite = x[mask].detach().clone()
        return forward(x, *args, **kwargs)

    def scoped_terms(z0, z1, y, alpha):
        nonlocal term_calls
        term_calls += 1
        if alpha != 1.:
            raise ValueError('Original RDrop alpha must remain one')
        if term_calls == 2:
            value, stats = as_ce_terms(z0, z1, y, mask, terms)
            diagnostics.update(stats)
            return value
        return terms(z0, z1, y, alpha)

    engine = self._sat_engine
    try:
        with engine.scope(mask):
            self.forward_identity_only = MethodType(record_forward, self)
            channel.apply = record_channel
            base.terms = scoped_terms
            delta, info = original.hook(self, **kw)
            if frames != 5 or term_calls != 3 or len(observations) != 2 or engine.calls == 0:
                raise ValueError('Exactly five extra L forwards/three terms/two complete As64 required')
            if not delta.requires_grad:
                raise ValueError('All native CE/default-head gradients must stay live')
            record = dict(step=step, cumulative_step=step + 4440, method='INDEP_CE6',
                mode='AS_CE', dropout_calls=engine.calls, bypassed_elements=engine.bypassed_elements,
                As_supervised=128, extra_NN=0, parameter_updates_extra=0,
                forward_count=frames, L_NN=768, As_forward_packets=128,
                generated_As0=64, generated_As1=64, same_hardware=True,
                ordinary_raw_same_IQ=True, As_same_IQ=False, all_labeled_CE_retained=True,
                two_separate_CE_mean=True, cross_channel_KL=False,
                first_namespaces=sorted({r['namespace'] for r in observations[0]}),
                second_namespaces=sorted({r['namespace'] for r in observations[1]}), **diagnostics)
            self._sat_steps.append(record)
            self._pair_records.append(record)
            for y, rx, actual in zip(labels[mask].tolist(), torch.as_tensor(meta['rx_i']).tolist(), observations[1]):
                self._pair_coverage[(int(y), int(rx), actual['scene'], actual['state_end'], actual['receiver_lock_status'])] += 1
            for actual in observations[1]:
                self._pair_strata[(int(actual['sample_id'][1]), int(actual['sample_id'][2]), actual['scene'], actual['state_end'], actual['receiver_lock_status'])] += 1
            self._pair_generated += 64
            self._pair_sessions.update(primary.sessions)
            sink = _PHYSICS_SINKS.get(self)
            if sink is not None:
                sink(dict(step=step, cumulative_step=step + 4440, view0=observations[0], view1=observations[1]))
            native = self._full_records[-1]
            native.update(satellite_subnetwork=record, full_as_pair=record,
                same_IQ_each_pair=False, same_IQ_ordinary_raw_pairs=True,
                As_generated_packets=128, As_forward_packets=128)
            return delta, info
    finally:
        base.terms = terms
        if had_forward:
            self.forward_identity_only = old_forward
        else:
            delattr(self, 'forward_identity_only')
        if had_apply:
            channel.apply = old_apply
        else:
            delattr(channel, 'apply')


hook._accepts_labeled_inputs = True


def attach(m, c):
    d.validate(c)
    m._pair_records, m._pair_coverage, m._pair_strata = [], Counter(), Counter()
    m._pair_generated, m._pair_sessions = 0, {}
    m._source_labeled_feature_hook = MethodType(hook, m)
    m._feature_learning_report['full_channel_pair68'] = dict(c['full_as_pair'])
    return m


def install(m, c):
    return attach(original.install(m, d.old_config(c['arm'])), c)


def build(c, device='cpu'):
    return attach(original.build(d.old_config(c['arm']), device), c)


def own_model(c, device='cpu'):
    from .source import parent_payload
    from .protocol import final_metadata, checkpoint
    d.validate(c)
    parent = parent_payload(c)
    out = Path(c['output_root'])
    final_metadata(c, d.read(out / 'completion.json'), d.read(out / 'source_frozen.json'),
        d.read(out / 'initialization.json'), d.read(out / 'source_contract.json'), d.read(d.SOURCE)['role_ids'])
    cp = torch.load(out / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    checkpoint(c, cp)
    if set(cp['model']) != set(parent['model']) or set(cp['ema_model']) != set(parent['ema_model']):
        raise ValueError('Own default-head/student/EMA keys differ')
    m = build(c, device)
    m.load_state_dict(cp['model'], strict=True)
    return m.eval().requires_grad_(False)
