"""Pure F/default head: raw CE plus one raw or complete-practical CE."""
from pathlib import Path
from types import MethodType
import torch
from torch.nn import functional as F
from . import design as d
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
    a = F.cross_entropy(first, labels, label_smoothing=.01)
    b = F.cross_entropy(second, labels, label_smoothing=.01)
    return .5 * (a + b), a, b


def counted(self, x):
    key = '_full_train_packets' if self.training else '_full_eval_packets'
    setattr(self, key, getattr(self, key) + len(x))
    witness = self._rawas_witness
    if self.training and witness['raw'] is not None:
        expected = witness['extra'] if witness['phase'] == 'extra' else witness['raw']
        if (expected is None or x.shape != expected.shape or not torch.equal(x, expected)):
            raise ValueError('Actual encoder L input differs from declared raw/extra IQ')
        if witness['phase'] == 'first':
            if witness['first_seen']:
                raise ValueError('Native first L encoder forward repeated')
            witness['first_seen'] += 1
        elif witness['phase'] == 'extra':
            if witness['extra_seen']:
                raise ValueError('Extra L encoder forward repeated')
            witness['extra_seen'] += 1
        else:
            raise ValueError('Unregistered raw L forward phase')
    return self._counted_features(x)


def hook(self, *, model, labeled_output, labels, extra, domains,
         optimizer_step, identity_loss, labeled_count, labeled_inputs):
    if (self is not model or not self.training or not torch.is_grad_enabled()
            or optimizer_step != self._feature_l_steps
            or optimizer_step != self._rawas_steps + 1 or labeled_count != 128
            or labeled_inputs.shape != (128, 2, 256) or labels.shape != (128,)):
        raise ValueError('Actual ordered legal L128 raw update required')
    witness = self._rawas_witness
    if (witness['raw'] is None or witness['step'] != optimizer_step
            or witness['phase'] != 'first' or witness['first_seen'] != 1
            or witness['extra_seen'] != 0 or not torch.equal(labels, witness['labels'])
            or not torch.equal(labeled_inputs, witness['raw'])):
        raise ValueError('Native first L forward/hook must consume the captured raw input')
    first = labeled_output['tx_logits']
    first_ce = F.cross_entropy(first, labels, label_smoothing=.01)
    if not torch.allclose(first_ce.detach(), identity_loss.detach(), atol=1e-6, rtol=1e-5):
        raise ValueError('Native raw CE normalization differs')
    records = []
    if self._rawas_config['arm'] == 'RAW_AS':
        if not isinstance(extra, (tuple, list)) or len(extra) < 2:
            raise ValueError('Original L physical metadata required for As')
        view, records = self._rawas_channel.apply(
            labeled_inputs.detach(), extra[1],
            epoch=(optimizer_step - 1) // 222 + 1, step=optimizer_step)
        if len(records) != 128:
            raise ValueError('Every legal labeled full As observation must be retained')
    else:
        view = labeled_inputs.detach()
    witness['phase'] = 'extra'
    witness['extra'] = view.detach()
    second = self.forward_identity_only(view, domain_labels=domains)['tx_logits']
    if witness['extra_seen'] != 1:
        raise ValueError('Exactly one extra raw/As L encoder forward required')
    total, raw_ce, extra_ce = pair(first, second, labels)
    self._rawas_steps += 1
    self._rawas_L_packets += 256
    record = dict(
        step=optimizer_step, epoch=(optimizer_step - 1) // 222 + 1,
        raw_CE=float(raw_ce.detach()), ordinary_CE=float(raw_ce.detach()),
        extra_CE=float(extra_ce.detach()), total_L=float(total.detach()),
        first_input='raw', extra_input='complete practical(raw)' if records else 'raw',
        L_forwards=2, As_packets=128 if records else 0, teacher_packets=0,
        raw_first_forward_verified=True, encoder_raw_MSE=0., hook_raw_MSE=0.,
    )
    self._rawas_records.append(record)
    witness.update(raw=None, labels=None, extra=None, phase=None)
    return total - identity_loss, {
        'rawAs/raw_CE': raw_ce.detach(), 'rawAs/extra_CE': extra_ce.detach(),
        'rawAs/total_L': total.detach(), 'rawAs/raw_first_forward_verified': 1,
    }


hook._accepts_labeled_inputs = True


def attach(m, c):
    d.validate(c)
    if c['arm'] not in ('RAW_NOAS', 'RAW_AS'):
        raise ValueError('Only registered raw/raw or raw/As condition allowed')
    m._source_labeled_feature_hook = MethodType(hook, m)
    m._rawas_config = c
    m._rawas_steps = 0
    m._rawas_L_packets = 0
    m._rawas_records = []
    m._rawas_channel = Practical('train') if c['arm'] == 'RAW_AS' else None
    m._rawas_witness = dict(raw=None, labels=None, extra=None, phase=None,
                           step=0, first_seen=0, extra_seen=0)
    encoder = m.id_backbone.encoder
    encoder._counted_features = encoder.features
    encoder._full_train_packets = 0
    encoder._full_eval_packets = 0
    encoder._rawas_witness = m._rawas_witness
    encoder.features = MethodType(counted, encoder)
    m._feature_learning_report.update(
        fixed_F='SIG_TIME_X3+CENTER_25K+CENTER_UNITS', CROSS_RX=False,
        RDrop=False, rawAs=c['training_mechanism'], new_deployment_parameters=0,
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
    done = d.read(out / 'completion.json')
    frozen = d.read(out / 'source_frozen.json')
    initial = d.read(out / 'initialization.json')
    contract = d.read(out / 'source_contract.json')
    if (done['config'] != c or done['optimizer_steps'] != 4440 or done['epoch'] != 20
            or done['target_access'] or done['head_fit_updates']
            or frozen['config'] != c or frozen['selection'] != 'final_only'
            or frozen['target_access'] or not initial['scratch_only']
            or initial['checkpoint_sources'] or initial['ancestors']
            or initial['target_contact']
            or contract['role_ids'] != d.read(d.SOURCE)['role_ids']):
        raise ValueError('Own final scratch20 source/default-head lineage differs')
    cp = torch.load(out / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    if (cp['epoch'] != 20 or cp['a1_ema_successful_updates'] != 4440
            or cp['run_id'] != c['run_id'] or cp['candidate_id'] != c['row_id']
            or cp['args']['baseline_ckpt'] or cp['args']['teacher_ckpt']
            or cp['checkpoint_selection'] != 'final_only'):
        raise ValueError('Own final checkpoint identity differs')
    model = build(c, device)
    model.load_state_dict(cp['model'], strict=True)
    return model.eval().requires_grad_(False)
