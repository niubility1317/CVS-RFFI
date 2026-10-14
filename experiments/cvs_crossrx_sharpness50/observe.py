"""Native L-only sharpness callback; no sampler, teacher or optimizer mutation.

The caller builds a validated visible source-L bank and installs this callable
after the native teacher deepcopy. No forward hook is installed by this module.
"""
from collections.abc import Mapping
from pathlib import Path
import json

import torch

from .objective import MODES, SOURCE_RX, objective, select_names
from experiments.cvs_sigtime_heldcredit.held_credit import physical_ids


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _integers(value, name, count, *, allow_extra=False):
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError(f'{name} must contain one integer per physical packet')
        values = value.detach().cpu().tolist()
    elif isinstance(value, (list, tuple)) and all(type(v) is int for v in value):
        values = list(value)
    else:
        raise ValueError(f'{name} must contain one integer per physical packet')
    if len(values) < count or (not allow_extra and len(values) != count):
        raise ValueError(f'{name} metadata length differs')
    return values[:count]


class Observer:
    """Observer(native, config, bank, *, default_weight_name=None, metrics_path=None).

    config requires mode (ZERO/CLEAN_CE/GLOBAL_GAP/CROSS_GAP), warmup_steps and
    frequency. Active iff global optimizer_step > warmup_steps and divisible by
    frequency. BASE arm names must be mapped to ZERO by the caller.
    bank contains x[N,2,256], y[N], rx[N], complete physical_ids (eq,rx,day,sig,
    base_index), with optional role='L_s'. Bank provenance is caller-validated.
    """
    def __init__(self, native, config, bank, *, default_weight_name=None, metrics_path=None):
        if not callable(getattr(native, '_meta_from_extra', None)):
            raise ValueError('Native physical metadata decoder is required')
        if not isinstance(config, Mapping) or config.get('mode') not in MODES:
            raise ValueError('Caller must provide a registered objective mode')
        self.mode = config['mode']
        self.warmup_steps = _integer(config.get('warmup_steps'), 'warmup_steps', 0)
        self.frequency = _integer(config.get('frequency'), 'frequency', 1)
        if not isinstance(bank, Mapping) or bank.get('role', 'L_s') != 'L_s':
            raise ValueError('Only a caller-validated visible source L_s bank is permitted')
        x = bank.get('x')
        if (not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1:] != (2, 256)
                or not x.is_floating_point() or not bool(torch.isfinite(x).all())):
            raise ValueError('Validated bank must contain finite real IQ[N,2,256]')
        count = len(x)
        y = _integers(bank.get('y'), 'bank y', count)
        rx = _integers(bank.get('rx'), 'bank rx', count)
        if any(label not in range(6) for label in y) or any(r not in SOURCE_RX for r in rx):
            raise ValueError('Bank labels/RX must be visible source identities and receivers')
        ids = physical_ids(bank.get('physical_ids', []), count)
        if any(identity[1] != r or identity[2] not in (1, 2, 3) for identity, r in zip(ids, rx)):
            raise ValueError('Bank complete physical IDs disagree with source RX/day')
        self.native, self.bank_x = native, x.detach()
        self.bank_y, self.bank_rx = torch.tensor(y, dtype=torch.long), torch.tensor(rx, dtype=torch.long)
        self.bank_ids = ids
        self.default_weight_name = default_weight_name
        self.metrics_path = None if metrics_path is None else Path(metrics_path)
        self.model = None
        self.selected_names = None
        self.last_step = None
        self.last_record = None
        self.counts = dict(calls=0, L_packets=0, inactive_calls=0, scheduled_calls=0,
                           eligible_calls=0, skipped_missing_current_RX=0,
                           skipped_missing_held_RX=0, skipped_empty_bank=0,
                           excluded_bank_packets=0, held_L_packets=0, extra_IQ_forwards=0)

    def _bind(self, model):
        if self.model is not None:
            if model is not self.model:
                raise ValueError('L callback cannot bind a different student')
            return
        names = select_names(model)
        named = dict(model.named_parameters())
        wname = self.default_weight_name
        if wname is None:
            from experiments.cvs_crossrx_credit50.model import classifier
            weight = classifier(model).weight
            matching = [name for name, p in named.items() if p is weight]
            if len(matching) != 1:
                raise ValueError('Native default head weight name is not unique')
            wname = matching[0]
        if wname not in named or named[wname].shape != (6, 160) or wname in names:
            raise ValueError('Unchanged native default W6x160 outside feature scope required')
        if not all(named[name].requires_grad for name in names):
            raise ValueError('Selected live feature parameters must remain trainable')
        self.model, self.selected_names, self.default_weight_name = model, names, wname

    def _zero(self):
        # Selected-only connected zero. Never use current logits or identity CE:
        # those zeros would leave auxiliary W/gain derivatives present.
        p = dict(self.model.named_parameters())[self.selected_names[0]]
        return p.reshape(-1)[0] * 0.

    def _record(self, record, loss):
        record['weighted_loss'] = float(loss.detach())
        self.last_record = record
        if self.metrics_path is not None:
            with self.metrics_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
        eligible = record['status'] == 'executed'
        metrics = {'crossRX_sharpness/active': float(record['scheduled']),
                   'crossRX_sharpness/eligible': float(eligible),
                   'crossRX_sharpness/loss': record['weighted_loss'],
                   'crossRX_sharpness/extra_IQ_forwards': float(record['extra_IQ_forwards'])}
        for receiver, row in record.get('by_RX', {}).items():
            metrics[f'crossRX_sharpness/RX{receiver}_positive_gap'] = row['positive_gap']
        return loss, metrics

    def __call__(self, *, model, labeled_output, labels, extra, domains, optimizer_step,
                 identity_loss, labeled_count, labeled_inputs=None, role='L_s', **compat_kwargs):
        if role != 'L_s' or not model.training:
            raise ValueError('Only an actual training student visible L_s callback is permitted')
        step = _integer(optimizer_step, 'optimizer_step', 1)
        count = _integer(labeled_count, 'labeled_count', 1)
        if self.last_step is not None and step <= self.last_step:
            raise ValueError('Repeated or reordered native L callback step')
        self._bind(model)
        self.last_step = step
        self.counts['calls'] += 1
        self.counts['L_packets'] += count
        scheduled = step > self.warmup_steps and step % self.frequency == 0
        record = dict(step=step, mode=self.mode, source_count=count, bank_count=len(self.bank_ids),
                      scheduled=scheduled, extra_IQ_forwards=0, by_RX={})
        if not scheduled:
            self.counts['inactive_calls'] += 1
            record['status'] = 'inactive'
            return self._record(record, self._zero())
        self.counts['scheduled_calls'] += 1
        logits = labeled_output.get('tx_logits') if isinstance(labeled_output, Mapping) else None
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 6
                or len(logits) < count or not logits.requires_grad
                or not bool(torch.isfinite(logits[:count]).all())):
            raise ValueError('Actual attached original L tx_logits first clean-count slice required')
        current_y = _integers(labels, 'current visible labels', count)
        if any(label not in range(6) for label in current_y):
            raise ValueError('Hidden/non-source current TX labels are prohibited')
        meta = self.native._meta_from_extra(extra)
        keys = ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')
        if not isinstance(meta, Mapping) or any(key not in meta for key in keys):
            raise ValueError('Actual L complete physical metadata is required')
        columns = [_integers(meta[key], key, count, allow_extra=True) for key in keys]
        ids = physical_ids(list(zip(*columns)), count)
        receivers = columns[1]
        if any(r not in SOURCE_RX for r in receivers) or any(day not in (1, 2, 3) for day in columns[2]):
            raise ValueError('Current RX/day are outside legal source L')
        bases = {identity[-1] for identity in ids}
        keep = [i for i, identity in enumerate(self.bank_ids) if identity[-1] not in bases]
        held_ids = [self.bank_ids[i] for i in keep]
        held_rx = self.bank_rx[keep]
        missing_current = sorted(set(SOURCE_RX) - set(receivers))
        missing_held = sorted(set(SOURCE_RX) - set(held_rx.tolist()))
        excluded = len(self.bank_ids) - len(keep)
        self.counts['excluded_bank_packets'] += excluded
        record.update(excluded_bank_count=excluded, held_count=len(keep),
                      current_RX_counts={str(r):receivers.count(r) for r in SOURCE_RX},
                      held_RX_counts={str(r):int((held_rx == r).sum()) for r in SOURCE_RX},
                      missing_current_RX=missing_current, missing_held_RX=missing_held,
                      full_current_physical_batch_excluded=True)
        if not keep:
            self.counts['skipped_empty_bank'] += 1
            record['status'] = 'skip_empty_bank'
            return self._record(record, self._zero())
        if missing_current:
            self.counts['skipped_missing_current_RX'] += 1
            record['status'] = 'skip_missing_current_RX'
            return self._record(record, self._zero())
        if missing_held:
            self.counts['skipped_missing_held_RX'] += 1
            record['status'] = 'skip_missing_held_RX'
            return self._record(record, self._zero())
        device = logits.device
        result = objective(model, logits[:count], torch.tensor(current_y, dtype=torch.long, device=device),
                           torch.tensor(receivers, dtype=torch.long, device=device),
                           self.bank_x[keep].to(device=device), self.bank_y[keep].to(device=device),
                           held_rx.to(device=device), ids, held_ids, self.selected_names,
                           self.default_weight_name, self.mode, role='L_s')
        loss, facts = result['loss'], result['summary']
        self.counts['eligible_calls'] += 1
        self.counts['held_L_packets'] += len(keep)
        self.counts['extra_IQ_forwards'] += facts['extra_IQ_forwards']
        record.update(status='executed', extra_IQ_forwards=facts['extra_IQ_forwards'],
                      functional_forwards=facts['functional_forwards'], by_RX=facts['by_RX'])
        return self._record(record, loss)

    __call__._accepts_labeled_inputs = False

    def summary(self):
        return dict(self.counts, mode=self.mode, warmup_steps=self.warmup_steps, frequency=self.frequency,
                    bank_count=len(self.bank_ids), selected_count=None if self.selected_names is None else len(self.selected_names),
                    default_weight_name=self.default_weight_name, head_fit_updates=0,
                    extra_actual_optimizer_steps=0, EMA_updates=0, permanent_forward_hooks=0,
                    original_sampler_unchanged=True, default_head_joint_main_training=True,
                    U_truth_read=False, target_access=False, validated_source_bank_supplied_by_caller=True)
