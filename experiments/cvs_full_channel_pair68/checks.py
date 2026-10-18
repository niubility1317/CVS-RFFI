"""Fixed-scope negatives and real loss/routing helpers with tensor-only toys."""
import copy
from dataclasses import dataclass, asdict
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from . import design as d, model
from .protocol import final_metadata, checkpoint
from experiments.cvs_satellite_dropout18.dropout import Engine


@dataclass
class Receiver:
    session_id: str
    amplitude: float


class ToyPrimary:
    """No real channel or network; exercises actual namespace/object plumbing."""
    def __init__(self):
        self.sessions, self.processed, self.packets = {}, [], 0
        self.configs = {s: SimpleNamespace(scenario=s, equalization_enabled=False,
            processing_route='residual', mode='post_sync') for s in d.TRAIN_SCENES}
        self.ch = SimpleNamespace(stable_seed=self.seed, receiver_for_session=self.receiver,
            normalize_input=lambda x: (x / max(float(np.sqrt(np.mean(abs(x)**2))), 1e-12), {}))
        self.residual = SimpleNamespace(ResidualChannel=self.processor)

    @staticmethod
    def seed(seed, *parts):
        return int.from_bytes(hashlib.sha256(repr((seed, parts)).encode()).digest()[:8], 'big')

    @staticmethod
    def receiver(cfg, seed, session):
        return Receiver(session, .1)

    def processor(self, cfg, seed, hw):
        def process(raw):
            self.processed.append(dict(seed=seed, hardware=hw, raw=raw.copy(), scene=cfg.scenario))
            offset = (seed % 100003) / 100003
            return raw * (.8 + .1 * offset) + .01j * offset, dict(
                channel_equalization_applied=False, state_end='LOS', receiver_lock_status='locked')
        return SimpleNamespace(process=process)

    def apply(self, raw, meta, epoch, step):
        waves, records = [], []
        namespace = f'FcrossRX/train/e{epoch}/step{step}/L_S'
        for i in range(len(raw)):
            pid = tuple(int(meta[k][i]) for k in model.PHYSICAL_FIELDS)
            session = f'rx{pid[1]}:day{pid[2]}:eq{pid[0]}'
            rng = np.random.default_rng(self.seed(20261016421, namespace, 'scene', pid))
            scene = str(rng.choice(d.TRAIN_SCENES))
            cfg = self.configs[scene]
            hw = self.ch.receiver_for_session(cfg, 20261016423, session)
            self.sessions[session] = asdict(hw)
            x = np.asarray(raw[i].detach().cpu().tolist(), dtype=np.float64)
            normalized, _ = self.ch.normalize_input(x[0] + 1j*x[1])
            signal, actual = self.processor(cfg, self.seed(20261016421, namespace, scene, pid), hw).process(normalized)
            waves.append(np.stack((signal.real, signal.imag)))
            records.append(dict(actual, scene=scene, sample_id=pid, session_id=session, namespace=namespace))
        self.packets += len(raw)
        return torch.tensor(np.asarray(waves).tolist(), dtype=raw.dtype), records


class ToyShifted:
    def __init__(self):
        self.obj = ToyPrimary()

    def apply(self, x, meta, epoch, step):
        return self.obj.apply(x, meta, epoch + 20, step + 4440)


class ToyModel:
    def __init__(self, c):
        self._full_channel, self._sat_engine = ToyShifted(), Engine('AS_CE')
        self._sat_steps, self._full_records = [], []
        self._feature_learning_report = {}
        self.parameter = torch.tensor([.3, -.2, .7, -.5, .9, -.8], requires_grad=True)
        self.inputs = []
        model.attach(self, c)

    def forward_identity_only(self, x, *args, **kwargs):
        self.inputs.append(x.detach().clone())
        logits = x.mean((1, 2))[:, None] * self.parameter[None, :]
        logits = self._sat_engine.apply(SimpleNamespace(p=.2, training=True, inplace=False), logits)
        return dict(tx_logits=logits)


def toy_original(self, **kw):
    """The real hook's five extra call and three terms order, without CVS NN."""
    raw, labels, domains = kw['labeled_inputs'], kw['labels'], kw['domains']
    mask = (torch.arange(128) + kw['optimizer_step']).remainder(2) != 0
    ordinary = raw + .2
    second = self.forward_identity_only(ordinary, domain_labels=domains)['tx_logits']
    common = model.base.terms(kw['labeled_output']['tx_logits'], second, labels, 1.)[0]
    meta = {k: v[mask] for k, v in kw['extra'][1].items()}
    sat, _ = self._full_channel.apply(raw[mask], meta, epoch=1, step=kw['optimizer_step'])
    mixed = ordinary.clone()
    mixed[mask] = sat
    z0 = self.forward_identity_only(mixed, domain_labels=domains)['tx_logits']
    z1 = self.forward_identity_only(mixed, domain_labels=domains)['tx_logits']
    extra = model.base.terms(z0, z1, labels, 1.)[0]
    r0 = self.forward_identity_only(raw, domain_labels=domains)['tx_logits']
    r1 = self.forward_identity_only(raw, domain_labels=domains)['tx_logits']
    shape = model.base.terms(r0, r1, labels, 1.)[0]
    total = .5 * common + .5 * extra + .25 * shape
    self._full_records.append(dict(total_L=float(total.detach())))
    return total - kw['identity_loss'], {}


def run():
    negative = 0
    def bad(fn):
        nonlocal negative
        try:
            fn()
        except (ValueError, TypeError, KeyError, AttributeError):
            negative += 1
        else:
            raise AssertionError('Invalid fixed full-channel pairing accepted')

    assert len(d.rows()) == 1 and d.rows()[0]['arm'] == 'B01'
    c = d.config(d.rows()[0])
    d.validate(c)
    assert c['parent_source'] == d.reference_config('AS_CE')['parent_source']
    assert c['parent_source'] != d.reference_config('B01')['parent_source']
    for key, value in [('target_access', True), ('As', False), ('new_Ag', True), ('epochs', 20),
        ('updates', 4440), ('model_seed', 2026092702), ('parent_source', 'unknown'),
        ('label_epochs', 13), ('Ag_mode', 'WHITE'), ('method', 'SAME_CE6'), ('satellite_mode', 'AS_DET')]:
        changed = copy.deepcopy(c)
        changed[key] = value
        bad(lambda changed=changed: d.validate(changed))
    for key, value in [('channel_seed', 1), ('receiver_seed', 2), ('As_KL', True),
        ('second_from_same_raw', False), ('L_NN_per_step', 896), ('exact_hardware_object_reused', False)]:
        changed = copy.deepcopy(c)
        changed['full_as_pair'][key] = value
        bad(lambda changed=changed: d.validate(changed))
    bad(lambda: d.config(d.rows()[0], 20))
    bad(lambda: d.reference_config('AS_DET'))
    expected, actual = d.old.make_args(d.old_config(), 'cpu'), d.make_args(c, 'cpu')
    def equal(a, b):
        return a == b or (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b))
    differences = {k for k in set(vars(expected)) | set(vars(actual))
        if not equal(getattr(expected, k, None), getattr(actual, k, None))}
    assert differences <= {'run_id', 'candidate_id', 'output_dir'}
    assert actual.pseudo_temporal_mode == 'batch_neighbor' and actual.pseudo_temporal_gate

    roles = {'L': ['Ltoy'], 'U': ['Utoy'], 'V': ['Vtoy']}
    done = dict(config=c, epoch=10, optimizer_steps=2220, target_access=False, head_fit_updates=0)
    frozen = dict(config=c, selection='final_only', target_access=False)
    initial = dict(scratch_only=False, target_contact=False, checkpoint_sources=c['checkpoint_sources'], source_roles='EXACT_MATCH')
    final_metadata(c, done, frozen, initial, dict(role_ids=roles), roles)
    for key, value in [('target_access', True), ('epoch', 20), ('optimizer_steps', 4440), ('head_fit_updates', 1)]:
        changed = dict(done, **{key: value})
        bad(lambda changed=changed: final_metadata(c, changed, frozen, initial, dict(role_ids=roles), roles))
    cp = dict(epoch=10, a1_ema_successful_updates=2220, run_id=d.RUN, candidate_id=c['row_id'],
        checkpoint_selection='final_only', ema_model={}, args=dict(baseline_ckpt=c['parent_source']+'/final_ssdg.pth',
            teacher_ckpt='', pseudo_temporal_mode='batch_neighbor', pseudo_temporal_gate=True))
    checkpoint(c, cp)
    for key, value in [('run_id', d.REFERENCE_RUN), ('epoch', 20), ('ema_model', None)]:
        changed = dict(cp, **{key: value})
        bad(lambda changed=changed: checkpoint(c, changed))
    changed = copy.deepcopy(cp)
    changed['args']['pseudo_temporal_mode'] = 'epoch_bank'
    bad(lambda: checkpoint(c, changed))

    torch.manual_seed(6801)
    z0, z1 = (torch.randn(128, 6, dtype=torch.float64, requires_grad=True) for _ in range(2))
    labels = torch.arange(128) % 6
    mask = torch.arange(128) % 2 != 0
    terms = model.base.terms
    (loss, ce, kl, _, _), stats = model.as_ce_terms(z0, z1, labels, mask, terms)
    assert torch.allclose(ce, .5 * (torch.nn.functional.cross_entropy(z0, labels, label_smoothing=.01)
        + torch.nn.functional.cross_entropy(z1, labels, label_smoothing=.01)))
    sat_ce = .125 * (torch.nn.functional.cross_entropy(z0[mask], labels[mask], label_smoothing=.01)
        + torch.nn.functional.cross_entropy(z1[mask], labels[mask], label_smoothing=.01))
    g0, g1 = torch.autograd.grad(.5 * loss, (z0, z1), retain_graph=True)
    e0, e1 = torch.autograd.grad(sat_ce, (z0, z1))
    assert torch.allclose(g0[mask], e0[mask], atol=1e-12) and torch.allclose(g1[mask], e1[mask], atol=1e-12)
    assert torch.allclose(kl, .5 * terms(z0[~mask], z1[~mask], labels[~mask], 1.)[2])
    assert stats['removed_satellite_KL'] > 0
    bad(lambda: model.as_ce_terms(z0, z1, labels, ~torch.zeros(128, dtype=torch.bool), terms))

    toy = ToyModel(c)
    raw = torch.randn(128, 2, 256)
    metadata = dict(eq_i=torch.ones(128, dtype=torch.long), rx_i=torch.ones(128, dtype=torch.long),
        day_i=torch.arange(128) % 3 + 1, sig_i=torch.arange(128) + 10, base_index=torch.arange(128))
    first = toy.forward_identity_only(raw + .2)['tx_logits']
    toy.inputs.clear()
    ce = torch.nn.functional.cross_entropy(first, labels, label_smoothing=.01)
    kw = dict(labels=labels, optimizer_step=1, labeled_inputs=raw, extra=(None, metadata),
        domains=torch.zeros(128, dtype=torch.long), labeled_output=dict(tx_logits=first), identity_loss=ce)
    original_forward = toy.forward_identity_only
    with patch.object(model.original, 'hook', toy_original):
        delta, _ = model.hook(toy, **kw)
    (ce + delta).backward()
    assert toy.parameter.grad is not None and bool(torch.isfinite(toy.parameter.grad).all())
    assert len(toy.inputs) == 5
    # Actual step1 selects even indices; derive its mask rather than reuse loss-test parity.
    actual_mask = (torch.arange(128)+1).remainder(2) != 0
    assert torch.equal(toy.inputs[1][~actual_mask], toy.inputs[2][~actual_mask])
    assert not torch.equal(toy.inputs[1][actual_mask], toy.inputs[2][actual_mask])
    assert torch.equal(toy.inputs[3], raw) and torch.equal(toy.inputs[4], raw)
    processed = toy._full_channel.obj.processed
    assert len(processed) == 128
    assert all(a['hardware'] is b['hardware'] and a['seed'] != b['seed']
        and np.array_equal(a['raw'], b['raw']) for a, b in zip(processed[:64], processed[64:]))
    assert toy._pair_generated == 64 and sum(toy._pair_coverage.values()) == 64
    assert toy._pair_records[0]['generated_As0'] == 64 and toy._pair_records[0]['generated_As1'] == 64
    assert not toy._sat_engine.active and model.base.terms is terms
    assert 'forward_identity_only' not in vars(toy) and 'apply' not in vars(toy._full_channel)
    assert toy.forward_identity_only == original_forward

    def failure(*args, **kwargs):
        raise ValueError('Synthetic native failure')
    with patch.object(model.original, 'hook', failure):
        bad(lambda: model.hook(toy, **kw))
    assert not toy._sat_engine.active and model.base.terms is terms
    assert 'forward_identity_only' not in vars(toy) and 'apply' not in vars(toy._full_channel)
    first_meta = [dict(sample_id=tuple(int(metadata[k][i]) for k in model.PHYSICAL_FIELDS),
        namespace='FcrossRX/train/e21/step4441/L_S', session_id=f'rx1:day{int(metadata["day_i"][i])}:eq1')
        for i in range(0,128,2)]
    small_meta = {k: v[actual_mask] for k, v in metadata.items()}
    hardware = [r['hardware'] for r in processed[:64]]
    invalid = copy.deepcopy(first_meta)
    invalid[0]['sample_id'] = (1, 9, 1, 10, 0)
    bad(lambda: model.independent_view(toy._full_channel.obj, raw[actual_mask], small_meta, invalid, hardware))
    bad(lambda: model.independent_view(toy._full_channel.obj, raw[actual_mask], small_meta, first_meta, hardware[:-1]))
    for cfg in toy._full_channel.obj.configs.values():
        cfg.equalization_enabled = True
    bad(lambda: model.independent_view(toy._full_channel.obj, raw[actual_mask], small_meta, first_meta, hardware))
    return dict(status='PASS', protocol_negatives=negative, tensor_loss_gradient=True,
        As_CE_coefficient=.25, As_per_pass_CE_coefficient=.125,
        As_KL_gradient_absent=True, ordinary_raw_KL_retained=True,
        five_existing_extra_calls=True, independent_fullstream_same_hardware_objects=True,
        same_raw_normalized_input=True, ordinary_raw_IQ_preserved=True,
        scoped_failure_restores=True, native_U_arguments_unchanged=True,
        CVS_NN=0, real_IQ=0, checkpoint_reads=0, optimizer_updates=0, target_access=False)


if __name__ == '__main__':
    print(json.dumps(run()))
