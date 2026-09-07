"""Small integration boundary for revision objectives and resumable state.

The trainer owns baseline objectives, augmentation, optimizer/scaler success,
and checkpoint persistence. This object adds no automatic training forwards,
updates, sampling resets, or validation mutations outside explicit method calls.
"""
from __future__ import annotations

import copy

import torch
from torch.nn import functional as F

from .ecrs_sampling import RXDayBalancedQueue
from .ecrs_training import (EffectiveStepLR, ResponseEMA, assemble_ecrs_losses,
                            capture_rng_state, masked_response_ce, restore_rng_state)


def _raw_model(model):
    return getattr(model, '_orig_mod', model)


class ECRSRuntime:
    """Explicit caller-managed V1R/V2 training state; never owns the model.

    Call ``set_epoch(epoch, configure_ecrs_for_epoch(...))`` before batches.
    Add ``revision_losses(...)["total"]`` once to baseline loss. After an actual
    optimizer/scaler attempt, call ``after_optimizer_step`` BEFORE zero_grad.
    Save this state beside model/optimizer/scaler state at an epoch boundary;
    restoring RNG must occur after model/optimizer construction and loading.
    """
    def __init__(self, model, *, weights=None, ema_decay=.99, margin=.2,
                 optimizer=None, group_configs=None, u_rx=None, u_day=None, seed=0,
                 compute_only=None):
        raw = _raw_model(model)
        self.version = str(raw.ecrs_version)
        self.compute_only = bool(getattr(raw, 'ecrs_compute_only', False) if compute_only is None else compute_only)
        if self.version not in ('v1r', 'v2'):
            raise ValueError('ECRSRuntime is for v1r/v2, not historical v1')
        self.weights = dict({'resp_ce': .15, 'cross_rx': .05, 'u_pair': .03,
                             'fused_ce': 1.}, **(weights or {}))
        self.margin = float(margin)
        self.ema = ResponseEMA(raw.response_encoder(), decay=ema_decay) if self.version == 'v2' and not self.compute_only else None
        if group_configs is not None and optimizer is None and not self.compute_only:
            raise ValueError('Effective update clock requires an optimizer')
        self.clock = EffectiveStepLR(optimizer, group_configs) if group_configs is not None and not self.compute_only else None
        if (u_rx is None) != (u_day is None):
            raise ValueError('U queue requires both RX and day metadata')
        self.u_queue = RXDayBalancedQueue(u_rx, u_day, seed=seed) if u_rx is not None and not self.compute_only else None
        self.epoch, self.batch_index = 0, 0
        self.stage = {'enabled': True, 'resp_cls': True, 'cross_rx': False,
                      'u_pair': False, 'fusion': False}

    def set_epoch(self, epoch, stage_state=None):
        if int(epoch) != self.epoch:
            self.batch_index = 0
        self.epoch = int(epoch)
        if stage_state is not None:
            self.stage = dict(stage_state)

    def next_u_indices(self, count):
        if self.u_queue is None:
            raise ValueError('No U RX/day queue was configured')
        return self.u_queue.take(count)

    @staticmethod
    def _clean_response(model, x):
        # The clean target must not advance student BN/dropout state. Preserve
        # all per-submodule modes, including intentionally frozen eval children.
        branch = model.ecrs
        modes = [(module, module.training) for module in branch.modules()]
        try:
            branch.eval()
            with torch.no_grad():
                return model.forward_response(x)
        finally:
            for module, training in modes:
                module.training = training

    def revision_losses(self, model, out_clean, x_clean, y, meta,
                        x_leo=None, out_leo=None, u_clean=None, u_leo=None):
        """Return total, named tensors, JSON-safe telemetry for one L/U batch.

        ``meta`` uses label_mask, receiver_id, day_id. U IQ pairs must be in
        identical physical-sample order; no U transmitter labels are accepted.
        Reuse supplied supervised clean/LEO forwards. When only x_leo exists,
        V2 runs response-only and therefore fused CE uses clean outputs alone.
        The total contains no raw CE or historical V1R physical objective.
        """
        raw = _raw_model(model)
        if not raw.training:
            raise RuntimeError('revision_losses is training-only; validation must use model inference')
        active = not self.compute_only and bool(self.stage.get('enabled', True))
        if not active:
            reference = x_clean if x_clean is not None else next(raw.parameters())
            zero = reference.detach().new_zeros(())
            names = ('resp_ce', 'cross_rx', 'u_pair', 'fused_ce')
            self.batch_index += 1
            return {'total': zero, 'losses': {name: zero for name in names},
                    'telemetry': {name: {'configured': False, 'executed': False,
                        'valid_count': 0, 'raw_loss': 0., 'weighted_loss': 0.,
                        'reason': 'compute_only' if self.compute_only else 'disabled'} for name in names}}
        enabled = {'resp_ce': active and bool(self.stage.get('resp_cls', False)),
                   'cross_rx': active and self.version == 'v2' and bool(self.stage.get('cross_rx', False)),
                   'u_pair': active and self.version == 'v2' and bool(self.stage.get('u_pair', False))}
        if out_clean is None:
            # Pure-U callers may omit L completely. Empty views preserve the
            # result structure without invoking either full backbone.
            device = u_leo.device if u_leo is not None else next(raw.parameters()).device
            dim = raw.ecrs_response_head.in_features
            classes = raw.ecrs_response_head.out_features
            z = torch.empty((0, dim), device=device)
            logits = torch.empty((0, classes), device=device)
            labels = torch.empty(0, dtype=torch.long, device=device)
            mask = torch.empty(0, dtype=torch.bool, device=device)
            receivers = days = views = torch.empty(0, dtype=torch.long, device=device)
            outputs = []
        else:
            outputs = [out_clean]
            if out_leo is None and x_leo is not None and self.version == 'v2' and any(enabled.values()):
                out_leo = raw.forward_response(x_leo)
            if out_leo is not None:
                outputs.append(out_leo)
            z = torch.cat([o['z_resp'] for o in outputs], dim=0)
            logits = torch.cat([o['resp_tx_logits'] for o in outputs], dim=0)
            y = y.to(device=z.device, dtype=torch.long).reshape(-1)
            base_mask = (y >= 0) & (y < logits.shape[-1])
            if meta.get('label_mask') is not None:
                base_mask &= torch.as_tensor(meta['label_mask'], device=z.device, dtype=torch.bool).reshape(-1)
            labels, mask = y.repeat(len(outputs)), base_mask.repeat(len(outputs))
            if enabled['cross_rx'] and self.weights['cross_rx'] != 0:
                if meta.get('receiver_id') is None or meta.get('day_id') is None:
                    raise ValueError('crossRX requires receiver_id and day_id metadata')
                receivers = torch.as_tensor(meta['receiver_id'], device=z.device).reshape(-1).repeat(len(outputs))
                days = torch.as_tensor(meta['day_id'], device=z.device).reshape(-1).repeat(len(outputs))
                views = torch.arange(len(outputs), device=z.device).repeat_interleave(len(y))
            else:
                receivers = days = views = None
        student, teacher = None, None
        if enabled['u_pair'] and self.weights['u_pair'] != 0:
            if (u_clean is None) != (u_leo is None):
                raise ValueError('U consistency requires both aligned clean and LEO IQ')
            if u_clean is not None:
                if u_clean.shape != u_leo.shape:
                    raise ValueError('U clean/LEO IQ pair shapes differ')
                clean_response = self._clean_response(raw, u_clean)
                student_response = raw.forward_response(u_leo)
                student = student_response['z_resp']
                teacher = F.normalize(self.ema(clean_response['encoder_input'].detach()).float(), dim=-1)
                if 'quality_valid' in clean_response:
                    # Degenerate responses remain counted physical samples;
                    # quality is not used to delete difficult U examples.
                    teacher = torch.where(clean_response['quality_valid'][:, None], teacher, torch.zeros_like(teacher))
        result = assemble_ecrs_losses(logits=logits, z_resp=z, labels=labels,
            label_mask=mask, rx=receivers, day=days, view=views,
            student_leo=student, teacher_clean=teacher, weights={k: self.weights[k] for k in enabled},
            enabled=enabled, margin=self.margin)
        fused_enabled = active and bool(self.stage.get('fusion', False))
        fused_weight = float(self.weights['fused_ce'])
        fused = z.detach().new_zeros(())
        count = 0
        if fused_enabled and fused_weight != 0:
            fused_outputs = [o['tx_logits_fused'] for o in outputs if 'tx_logits_fused' in o]
            if not fused_outputs and outputs:
                raise ValueError('Active fusion requires tx_logits_fused from a supervised forward')
            if fused_outputs:
                fused_logits = torch.cat(fused_outputs)
                fused, count = masked_response_ce(fused_logits, y.repeat(len(fused_outputs)), base_mask.repeat(len(fused_outputs)))
        result['losses']['fused_ce'] = fused
        result['total'] = result['total'] + fused_weight * (fused if count else fused.detach())
        result['telemetry']['fused_ce'] = {'configured': fused_enabled, 'executed': bool(count),
            'valid_count': count, 'raw_loss': float(fused.detach()),
            'weighted_loss': float((fused_weight * fused).detach()),
            'reason': '' if count else 'disabled' if not fused_enabled else 'empty_legal_set'}
        self.batch_index += 1
        return result

    def after_optimizer_step(self, model, successful):
        """Update clocks/teacher only after successful student optimizer updates."""
        raw = _raw_model(model)
        if not raw.training or self.compute_only or not bool(self.stage.get('enabled', True)):
            return {'ema_updated': False, 'advanced_groups': []}
        grads = [p.grad for p in raw.response_encoder().parameters() if p.grad is not None]
        encoder_updated = bool(successful) and bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
        ema_updated = self.ema.update(raw.response_encoder(), successful=encoder_updated) if self.ema is not None else False
        advanced = self.clock.step(successful=successful) if self.clock is not None else []
        return {'ema_updated': ema_updated, 'advanced_groups': advanced}

    def state_dict(self):
        return copy.deepcopy({'training_semantics': 'ecrs_revision_20260907',
            'version': self.version, 'compute_only': self.compute_only, 'weights': self.weights, 'margin': self.margin,
            'epoch': self.epoch, 'batch_index': self.batch_index, 'stage': self.stage,
            'ema': self.ema.state_dict() if self.ema is not None else None,
            'u_queue': self.u_queue.state_dict() if self.u_queue is not None else None,
            'clock': self.clock.state_dict() if self.clock is not None else None,
            'rng': capture_rng_state()})

    def load_state_dict(self, state, restore_rng=True):
        if state.get('training_semantics') != 'ecrs_revision_20260907':
            raise ValueError('ECRS training semantics changed; use model-only initialization in a new run, not exact resume')
        if state['version'] != self.version or state.get('compute_only', False) != self.compute_only or state['weights'] != self.weights or state['margin'] != self.margin:
            raise ValueError('Runtime revision/objective configuration differs from checkpoint')
        for name, value in [('ema', self.ema), ('u_queue', self.u_queue), ('clock', self.clock)]:
            if (state[name] is None) != (value is None):
                raise ValueError(f'Runtime {name} configuration differs from checkpoint')
            if value is not None:
                value.load_state_dict(state[name])
        self.epoch, self.batch_index = int(state['epoch']), int(state['batch_index'])
        self.stage = copy.deepcopy(state['stage'])
        if restore_rng:
            restore_rng_state(state['rng'])
