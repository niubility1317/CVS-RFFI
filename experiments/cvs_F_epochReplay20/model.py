"""Pure F: ordinary CE plus epoch-local physical-ID replay of full As CE."""
from collections import Counter
from pathlib import Path
from types import MethodType
import torch
from torch.nn import functional as F
from . import design as d
from .replay import EpochReplay
from experiments.cvs_rdrop_sinc_units import model as af
from experiments.cvs_fullsource_concat20.channel import Practical

classifier, source_domain_count = af.classifier, af.source_domain_count


def pair(first, second, labels):
    if (first.shape != second.shape or first.shape != (len(labels), 6)
            or len(labels) == 0 or labels.dtype != torch.long
            or not bool(((labels >= 0) & (labels < 6)).all())
            or not bool(torch.isfinite(first).all())
            or not bool(torch.isfinite(second).all())):
        raise ValueError('Two finite native six-class true-label logits required')
    ordinary = F.cross_entropy(first, labels, label_smoothing=.01)
    satellite = F.cross_entropy(second, labels, label_smoothing=.01)
    return .5 * (ordinary + satellite), ordinary, satellite


def meta_summary(records):
    return dict(packets=len(records),
        scenes=dict(Counter(str(r['scene']) for r in records)),
        states=dict(Counter(str(r.get('state_end', 'UNKNOWN')) for r in records)),
        locks=dict(Counter(str(r.get('receiver_lock_status', 'UNKNOWN')) for r in records)),
        RX=dict(Counter(str(r['sample_id'][1]) for r in records)),
        days=dict(Counter(str(r['sample_id'][2]) for r in records)))


class ReplayPractical:
    """Always draw the original full channel before choosing the cached L IQ."""
    def __init__(self):
        self.fresh = Practical('train')
        self.replay = EpochReplay()
        self.calls = 0
        self.packets = 0

    def __deepcopy__(self, memo):
        copied = type(self)()
        memo[id(self)] = copied
        return copied

    def apply(self, raw, meta, *, epoch, step):
        fresh, fresh_records = self.fresh.apply(raw, meta, epoch=epoch, step=step)
        used, used_records, summary = self.replay.apply(
            fresh, fresh_records, raw, epoch=epoch, step=step)
        if (len(fresh_records) != 128 or len(used_records) != 128
                or used.shape != raw.shape or not bool(torch.isfinite(used).all())):
            raise ValueError('Every visible L128 requires finite fresh and used full As')
        self.calls += 1
        self.packets += 128
        return used, used_records, fresh_records, summary

    def audit(self):
        return dict(physical_channel=self.fresh.audit(),
            replay='First full As per physical ID within epoch; refresh each epoch',
            fresh_fullchannel_always_generated=True, eval_uses_cache=False,
            cache_in_state_dict=False, extra_NN=0, new_parameters=0)

    def summary(self):
        return dict(calls=self.calls, packets=self.packets,
                    fresh=self.fresh.summary(), replay=self.replay.stats())


def counted(self, x):
    key = '_full_train_packets' if self.training else '_full_eval_packets'
    setattr(self, key, getattr(self, key) + len(x))
    witness = self._epochreplay_witness
    if self.training and witness['raw'] is not None:
        phase = witness['phase']
        expected = witness['used_As'] if phase == 'extra' else witness['ordinary']
        if expected is None or x.shape != expected.shape or not torch.equal(x, expected):
            raise ValueError('Actual encoder input differs from ordinary or used cached As')
        key = 'extra_seen' if phase == 'extra' else 'first_seen'
        if phase not in ('first', 'extra') or witness[key]:
            raise ValueError('Exactly one encoder forward per declared L slot required')
        witness[key] += 1
    features = self._counted_features(x)
    if features.shape != (len(x), 160) or not bool(torch.isfinite(features).all()):
        raise ValueError('Every actual deployment encoder output must be finite joint160')
    self._joint160_seen_calls += 1
    self._joint160_seen_packets += len(x)
    self._joint160_last_shape = list(features.shape)
    return features


def hook(self, *, model, labeled_output, labels, extra, domains,
         optimizer_step, identity_loss, labeled_count, labeled_inputs):
    if (self is not model or not self.training or not torch.is_grad_enabled()
            or optimizer_step != self._feature_l_steps
            or optimizer_step != self._epochreplay_steps + 1 or labeled_count != 128
            or labeled_inputs.shape != (128, 2, 256) or labels.shape != (128,)):
        raise ValueError('Actual ordered native visible L128 update required')
    witness = self._epochreplay_witness
    if (witness['raw'] is None or witness['step'] != optimizer_step
            or witness['phase'] != 'first' or witness['first_seen'] != 1
            or witness['extra_seen'] or not torch.equal(labels, witness['labels'])
            or not torch.equal(labeled_inputs, witness['raw'])):
        raise ValueError('Captured raw, real first ordinary encoder and native hook must agree')
    if not isinstance(extra, (tuple, list)) or len(extra) < 2:
        raise ValueError('Original visible L physical metadata required')
    first = labeled_output['tx_logits']
    first_ce = F.cross_entropy(first, labels, label_smoothing=.01)
    if not torch.allclose(first_ce.detach(), identity_loss.detach(), atol=1e-6, rtol=1e-5):
        raise ValueError('Native ordinary CE normalization differs')
    epoch = (optimizer_step - 1) // 222 + 1
    used, used_records, fresh_records, replay = self._epochreplay_channel.apply(
        labeled_inputs.detach(), extra[1], epoch=epoch, step=optimizer_step)
    witness.update(phase='extra', used_As=used.detach())
    second = self.forward_identity_only(used, domain_labels=domains)['tx_logits']
    if witness['extra_seen'] != 1:
        raise ValueError('Exactly one used-As L encoder forward required')
    total, ordinary_ce, As_ce = pair(first, second, labels)
    prediction = second.detach().argmax(1).cpu().tolist()
    true = labels.detach().cpu().tolist()
    packet_records = []
    online = {group: dict(correct=0, n=0) for group in
              ('all', 'unique', 'hit', 'cross_step_hit', 'within_step_hit')}
    for slot, (y, pred, fresh, actual) in enumerate(zip(true, prediction, fresh_records, used_records)):
        hit = bool(actual['cache_hit'])
        source_step, source_slot = int(actual['replay_source_step']), int(actual['replay_source_slot'])
        groups = ['all', 'hit' if hit else 'unique']
        if hit:
            groups.append('cross_step_hit' if source_step < optimizer_step else 'within_step_hit')
        correct = int(pred == y)
        for group in groups:
            online[group]['correct'] += correct
            online[group]['n'] += 1
        packet_records.append(dict(slot=slot, sample_id=actual['sample_id'], label=y,
            prediction=pred, correct=correct, cache_hit=hit,
            source_step=source_step, source_slot=source_slot,
            first_observation=not hit, fresh_scene=fresh['scene'], used_scene=actual['scene'],
            fresh_namespace=fresh['namespace'], used_namespace=actual['namespace']))
    self._epochreplay_steps += 1
    self._epochreplay_L_packets += 256
    self._epochreplay_replay_records.append(dict(replay, step=optimizer_step, epoch=epoch))
    record = dict(step=optimizer_step, epoch=epoch,
        ordinary_CE=float(ordinary_ce.detach()), As_CE=float(As_ce.detach()),
        extra_CE=float(As_ce.detach()), total_L=float(total.detach()),
        first_input='native ordinary L', extra_input='epoch-local cached full practical(raw)',
        L_forwards=2, As_packets=128, fresh_As_packets=128, used_As_packets=128,
        teacher_packets=0, ordinary_first_forward_verified=True,
        encoder_ordinary_MSE=0., hook_raw_MSE=0., encoder_used_As_MSE=0.,
        first_encoder_calls=witness['first_seen'], extra_encoder_calls=witness['extra_seen'],
        online_As=online, fresh_meta_summary=meta_summary(fresh_records),
        used_meta_summary=meta_summary(used_records), packets=packet_records)
    self._epochreplay_records.append(record)
    witness.update(raw=None, ordinary=None, labels=None, used_As=None, phase=None)
    return total - identity_loss, {
        'epochReplay/ordinary_CE': ordinary_ce.detach(),
        'epochReplay/As_CE': As_ce.detach(), 'epochReplay/total_L': total.detach(),
        'epochReplay/As_correct': online['all']['correct'],
        'epochReplay/As_n': online['all']['n'],
        'epochReplay/As_hit_correct': online['hit']['correct'],
        'epochReplay/As_hit_n': online['hit']['n'],
        'epochReplay/As_unique_correct': online['unique']['correct'],
        'epochReplay/As_unique_n': online['unique']['n'],
    }


hook._accepts_labeled_inputs = True


def attach(m, c):
    d.validate(c)
    if c['arm'] != 'EPOCH_REPLAY':
        raise ValueError('Only the fixed epoch-local full As replay row is allowed')
    m._source_labeled_feature_hook = MethodType(hook, m)
    m._epochreplay_config = c
    m._epochreplay_steps = 0
    m._epochreplay_L_packets = 0
    m._epochreplay_records = []
    m._epochreplay_replay_records = []
    m._epochreplay_channel = ReplayPractical()
    m._epochreplay_witness = dict(raw=None, ordinary=None, labels=None, used_As=None,
                                phase=None, step=0, first_seen=0, extra_seen=0)
    encoder = m.id_backbone.encoder
    encoder._counted_features = encoder.features
    encoder._full_train_packets = 0
    encoder._full_eval_packets = 0
    encoder._joint160_seen_calls = 0
    encoder._joint160_seen_packets = 0
    encoder._joint160_last_shape = None
    encoder._epochreplay_witness = m._epochreplay_witness
    encoder.features = MethodType(counted, encoder)
    m._feature_learning_report.update(
        fixed_F='SIG_TIME_X3+CENTER_25K+CENTER_UNITS', CROSS_RX=False,
        RDrop=False, epochReplay=c['training_mechanism'], new_deployment_parameters=0,
        external_teacher=False, KD=False, head_fit_updates=0)
    return m


def install(m, c):
    return attach(af.configure(af.original.old.install(
        m, af.original.d.legacy_config(c['model_seed'])), d.model_config()), c)


def build(c, device='cpu'):
    return attach(af.configure(af.original.old.build(
        af.original.d.legacy_config(c['model_seed']), 'cpu'), d.model_config()), c).to(device)


def own_model(c, device='cpu'):
    d.validate(c)
    out = Path(c['output_root'])
    done, frozen = d.read(out / 'completion.json'), d.read(out / 'source_frozen.json')
    initial, contract = d.read(out / 'initialization.json'), d.read(out / 'source_contract.json')
    if (done['config'] != c or done['optimizer_steps'] != 4440 or done['epoch'] != 20
            or done['target_access'] or done['head_fit_updates']
            or frozen['config'] != c or frozen['selection'] != 'final_only'
            or frozen['target_access'] or not initial['scratch_only']
            or initial['checkpoint_sources'] or initial['ancestors']
            or initial['target_contact'] or contract['role_ids'] != d.read(d.SOURCE)['role_ids']):
        raise ValueError('Own final scratch20/default-head source lineage differs')
    cp = torch.load(out / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    if (cp['epoch'] != 20 or cp['a1_ema_successful_updates'] != 4440
            or cp['run_id'] != c['run_id'] or cp['candidate_id'] != c['row_id']
            or cp['args']['baseline_ckpt'] or cp['args']['teacher_ckpt']
            or cp['checkpoint_selection'] != 'final_only'):
        raise ValueError('Own final checkpoint identity differs')
    model = build(c, device)
    model.load_state_dict(cp['model'], strict=True)
    return model.eval().requires_grad_(False)
