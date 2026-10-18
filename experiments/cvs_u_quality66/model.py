"""Matched quality weighting for full-As U; one external frozen As20 teacher."""
import gzip
import json
from copy import deepcopy
from pathlib import Path
from types import MethodType
from weakref import WeakKeyDictionary
import torch
from . import design as d
from .channel import UPractical
from .rng import private_rng
from .teaching import teaching_loss
from experiments.cvs_noise_shape_factorial43 import model as original

classifier, source_domain_count = original.classifier, original.source_domain_count
# A deepcopy of the native student/EMA cannot copy this external association.
_FROZEN_TEACHERS = WeakKeyDictionary()


class FrozenTeachers:
    """Not an nn.Module. No teacher can enter a student's EMA/optimizer/checkpoint."""
    def __init__(self, teachers):
        if set(teachers) != {'B01'}:
            raise ValueError('Exactly one own As20 EMA teacher required')
        self.teachers = dict(teachers)
        self.forward_NN = {k: 0 for k in teachers}
        self.snapshots = {}
        for key, model in self.teachers.items():
            model.eval().requires_grad_(False)
            self.snapshots[key] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def forward(self, x, domains):
        result = {}
        with torch.no_grad():
            for key in ('B01',):
                model = self.teachers[key]
                if model.training or any(p.requires_grad for p in model.parameters()):
                    raise ValueError('Frozen EMA teacher must remain eval and no-gradient')
                z = model.forward_identity_only(x, domain_labels=domains)['tx_logits']
                if z.shape != (len(x), 6) or not bool(torch.isfinite(z).all()) or z.requires_grad:
                    raise ValueError('Frozen default head must return finite detached six-class logits')
                result[key] = z.detach()
                self.forward_NN[key] += len(x)
        return result

    def assert_frozen(self):
        for key, model in self.teachers.items():
            state = model.state_dict()
            if (model.training or any(p.requires_grad for p in model.parameters())
                    or set(state) != set(self.snapshots[key])
                    or any(not torch.equal(v.detach().cpu(), self.snapshots[key][k]) for k, v in state.items())):
                raise ValueError('Training-only teacher parameter or persistent buffer changed')
        return True

    def summary(self):
        return dict(forward_NN=dict(self.forward_NN), total_forward_NN=sum(self.forward_NN.values()),
                    parameters_each={k: sum(p.numel() for p in m.parameters()) for k, m in self.teachers.items()},
                    state_keys_each={k: len(m.state_dict()) for k, m in self.teachers.items()},
                    extra_training_parameters=sum(p.numel() for m in self.teachers.values() for p in m.parameters()),
                    trainable_parameters=0, registered_in_student=False, EMA_updated=False,
                    teacher_checkpoint_key='ema_model', teacher_epoch=20, target_access=False)


def teacher_bank(model):
    if model not in _FROZEN_TEACHERS:
        raise ValueError('Training-only frozen teachers have not been prepared for this student')
    return _FROZEN_TEACHERS[model]


def prepare_frozen_teachers(model, config, payloads=None):
    """Load audited own20 EMA payloads, without registering any Module on student.

    Explicitly called by the training observer, never by build/own_model/formal
    prediction. Clone the already constructed student architecture directly:
    invoking a model builder here would reenter the native training builder
    while its initialization callback is active. The teacher parameter set
    is then replaced strictly by its audited EMA state.
    """
    d.validate(config)
    if model in _FROZEN_TEACHERS:
        raise ValueError('Frozen teachers may be initialized only once per student')
    if payloads is None:
        from .source import teacher_payload
        payloads = {'B01': teacher_payload(config)}
    if set(payloads) != {'B01'}:
        raise ValueError('Only the exact own As20 teacher checkpoint payload is allowed')
    before_keys = tuple(model.state_dict())
    before_parameters = tuple(id(p) for p in model.parameters())
    probe = next(model.parameters()).detach()
    teachers = {}
    for arm in ('B01',):
        cp = payloads[arm]
        ref = d.reference_config(arm)
        if (cp.get('epoch') != 20 or cp.get('a1_ema_successful_updates') != 4440
                or cp.get('run_id') != original.d.RUN or cp.get('candidate_id') != ref['row_id']
                or cp.get('checkpoint_selection') != 'final_only' or cp.get('ema_model') is None
                or cp['args']['baseline_ckpt'] or cp['args']['teacher_ckpt']):
            raise ValueError('Only exact same-seed P43 scratch20 final EMA teacher payloads allowed')
        with private_rng(probe, 4440):
            teacher = deepcopy(model)
        # The copied bound inference methods point at the copied modules.
        # Extra-U callbacks have no role in a fixed teacher and must not retain
        # a training output path or attempt to create a nested teacher bank.
        teacher._source_unlabeled_selection_hook = None
        teacher._source_unlabeled_feature_hook = None
        for name in ('_clean_teacher_config', '_clean_teacher_pending', '_clean_teacher_records',
                     '_clean_teacher_packets', '_clean_teacher_physics', '_clean_teacher_channel'):
            if hasattr(teacher, name):
                delattr(teacher, name)
        if hasattr(teacher, '_feature_learning_report'):
            teacher._feature_learning_report.pop('u_quality66', None)
        teacher.load_state_dict(cp['ema_model'], strict=True)
        teacher.eval().requires_grad_(False)
        if sum(p.numel() for p in teacher.parameters()) != 900722 or len(teacher.state_dict()) != 206:
            raise ValueError('Frozen teacher must be the same original F/default-head shape')
        teachers[arm] = teacher
    bank = FrozenTeachers(teachers)
    _FROZEN_TEACHERS[model] = bank
    if tuple(model.state_dict()) != before_keys or tuple(id(p) for p in model.parameters()) != before_parameters:
        raise ValueError('Frozen training-only teachers entered the student Module tree')
    return bank


def _metadata(extra, n, domains):
    if not isinstance(extra, (tuple, list)) or len(extra) < 2 or not isinstance(extra[1], dict):
        raise ValueError('Original source U physical metadata required')
    keys = ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')
    if any(k not in extra[1] or len(extra[1][k]) != n for k in keys):
        raise ValueError('Physical metadata and original U batch must match')
    from experiments.cvs_fullsource_concat20.supcon import RX_BY_DOMAIN
    rx = torch.as_tensor(extra[1]['rx_i'], device=domains.device, dtype=torch.long)
    if not torch.equal(rx, torch.tensor(RX_BY_DOMAIN, device=domains.device)[domains]):
        raise ValueError('Physical U RX and original domain differ')
    return {k: extra[1][k][:min(n, 128)] for k in keys}


def select(self, *, model, teacher, x_u, teacher_logits, student_logits, pseudo,
           base_mask, strong_mask, extra, domains, optimizer_step):
    if (model is not self or not self.training or teacher is self or teacher.training
            or self._clean_teacher_pending is not None or optimizer_step != self._feature_l_steps
            or optimizer_step not in range(1, 2221)):
        raise ValueError('Ordered original hidden-U own-EMA training context required')
    n = len(x_u)
    if (not 0 < n <= 256 or x_u.shape != (n, 2, 256)
            or teacher_logits.shape != (n, 6) or student_logits.shape != (n, 6)
            or domains is None or domains.shape != (n,) or domains.dtype != torch.long
            or not bool(((domains >= 0) & (domains < 15)).all())
            or base_mask.shape != (n,) or strong_mask.shape != (n,)
            or base_mask.dtype != torch.bool or strong_mask.dtype != torch.bool):
        raise ValueError('Original source U/default-head dimensions and boolean masks required')
    meta = _metadata(extra, n, domains)
    k = min(128, n)
    mask = base_mask & strong_mask
    self._clean_teacher_pending = dict(x=x_u[:k].detach().clone(), meta=meta,
        domains=domains[:k].detach().clone(), local_step=optimizer_step,
        teacher=teacher_logits[:k].detach().clone(), original_mask=mask.detach().clone(), native_n=n)
    return mask, {'u_quality66/original_mask_unchanged': 1.}


def extra_hook(self, *, model, teacher, student_output, teacher_output, extra, domains,
               optimizer_step, identity_loss, pseudo_loss, entropy_loss):
    p = self._clean_teacher_pending
    if (model is not self or not self.training or not torch.is_grad_enabled() or p is None
            or p['local_step'] != optimizer_step or teacher is self or teacher.training):
        raise ValueError('Matching live original U step and own EMA required')
    n = len(p['x'])
    if (not torch.equal(p['teacher'], teacher_output['tx_logits'][:n].detach())
            or not torch.equal(p['domains'], domains[:n])
            or not student_output['tx_logits'].requires_grad):
        raise ValueError('Native teacher/domain/live student context changed')
    try:
        step, epoch = optimizer_step + 4440, (optimizer_step - 1) // 222 + 21
        view, observations = self._clean_teacher_channel.apply(p['x'], p['meta'], epoch, step)
        if len(observations) != n:
            raise ValueError('Every fixed raw-U slice must receive a complete full-As view')
        for session, hw in self._clean_teacher_channel.sessions.items():
            lhw = self._full_channel.sessions.get(session)
            if lhw is not None and lhw != hw:
                raise ValueError('L/U hardware must match the same physical RX/day/eq session')
        with private_rng(view, step):
            z = self.forward_identity_only(view, domain_labels=p['domains'])['tx_logits']
        if not z.requires_grad:
            raise ValueError('The extra U student/default head must retain live gradients')
        frozen = teacher_bank(self).forward(p['x'], p['domains'])
        local_epoch = (optimizer_step - 1) // 222 + 1
        loss, record = teaching_loss(z, frozen['B01'],
            torch.full((n,), -1, device=z.device, dtype=torch.long), observations,
            self._clean_teacher_config['arm'], local_epoch, self._clean_teacher_config['U_teaching'])
        ids = [[int(p['meta'][k][i]) for k in ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')]
               for i in range(n)]
        record.update(step=optimizer_step, cumulative_step=step, epoch=epoch, local_epoch=local_epoch,
                      packets=n, native_packets=p['native_n'], native_n=p['native_n'], additional_n=n,
                      original_selected=int(p['original_mask'].sum()),
                      extra_original_selected=int(p['original_mask'][:n].sum()),
                      selected_physical_ids=ids, original_mask_unchanged=True,
                      extra_student_NN=n, extra_frozen_teacher_NN=n,
                      frozen_teacher_NN=dict(B01=n),
                      dropout_rng_restored=True, subset_rule='first_min_128_native_U_batch_no_padding',
                      source_health_gate=False)
        root = Path(self._clean_teacher_config['output_root'])
        with gzip.open(root / 'u_quality_physics.jsonl.gz', 'at' if self._clean_teacher_physics else 'xt',
                       encoding='utf8', compresslevel=3) as stream:
            stream.write(json.dumps(dict(step=step, observations=observations), allow_nan=False) + '\n')
        self._clean_teacher_physics = True
        self._clean_teacher_packets += n
        self._clean_teacher_records.append(record)
        return loss, {'u_quality66/' + k: record[k] for k in
                      ('weight', 'KL', 'weighted_KL', 'q_mass', 'q_mean', 'positive_q', 'packets')}
    finally:
        self._clean_teacher_pending = None


def attach(model, config):
    d.validate(config)
    if (getattr(model, '_source_unlabeled_selection_hook', None) is not None
            or getattr(model, '_source_unlabeled_feature_hook', None) is not None):
        raise ValueError('Original native U must have no competing extra method')
    model._clean_teacher_config = config
    model._clean_teacher_pending = None
    model._clean_teacher_records = []
    model._clean_teacher_packets = 0
    model._clean_teacher_physics = False
    model._clean_teacher_channel = UPractical('train')
    model._source_unlabeled_selection_hook = MethodType(select, model)
    model._source_unlabeled_feature_hook = MethodType(extra_hook, model)
    model._feature_learning_report['u_quality66'] = dict(config['U_teaching'])
    return model


def install(model, config):
    return attach(original.install(model, d.old_config()), config)


def build(config, device='cpu'):
    return attach(original.build(d.old_config(), device), config)


def own_model(config, device='cpu'):
    from .source import parent_payload
    from .protocol import final_metadata, checkpoint
    d.validate(config)
    parent = parent_payload(config)
    root = Path(config['output_root'])
    final_metadata(config, d.read(root / 'completion.json'), d.read(root / 'source_frozen.json'),
                   d.read(root / 'initialization.json'), d.read(root / 'source_contract.json'),
                   d.read(d.SOURCE)['role_ids'])
    cp = torch.load(root / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    checkpoint(config, cp)
    if set(cp['model']) != set(parent['model']) or set(cp['ema_model']) != set(parent['ema_model']):
        raise ValueError('Own default-head/student/EMA keys differ')
    model = build(config, device)
    model.load_state_dict(cp['model'], strict=True)
    # No training-only teacher payload is loaded or prepared for deployment.
    return model.eval().requires_grad_(False)
