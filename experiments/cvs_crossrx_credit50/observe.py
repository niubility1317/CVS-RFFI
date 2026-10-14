"""Native late-loss observer for the fixed scratch SIG_TIME credit matrix.

No data builder, sampler, teacher, pseudo-label gate or real extra update is
installed here. Only the existing main L forward and current full base loss are
consumed; credit evaluates other RX in that same visible source L batch.
"""
from contextlib import contextmanager
from types import MethodType
from pathlib import Path
import json
import sys
import torch

from .capture import capture
from .credit import assess, add_gradients
from .objective import objective, combine, SOURCE_RX
from experiments.cvs_u_schedule50.counterfactual import (
    simulate, _live_snapshot, _live_readback, _rng_snapshot, _equal, _preserve_rng,
)


@contextmanager
def labeled_context(model):
    """Top-level MethodType: deepcopy binds only the cloned model, no observer."""
    original = model._credit_original_context_function
    if not model.training:
        with original(model):
            yield
        return
    if model._credit_pre_gain is not None:
        raise ValueError('Unconsumed previous L feature capture')
    try:
        with original(model), capture(model) as observed:
            yield
        model._credit_pre_gain = observed['z']
    except BaseException:
        model._credit_pre_gain = None
        raise


class Observer:
    def initialize(self, model, c, out):
        if c['arm'] not in ('BASE', 'SUPCON', 'ORDER', 'CREDIT'):
            raise ValueError('Unregistered fixed SIG_TIME credit arm')
        if any(hasattr(model, name) for name in ('_credit_original_context_function', '_credit_pre_gain',
                                                '_source_labeled_feature_hook', '_source_optimizer_update_probe_hook')):
            raise ValueError('Credit observer requires an unbound own scratch student')
        original = model._phase1_labeled_feature_context
        if not isinstance(original, MethodType) or original.__self__ is not model:
            raise ValueError('Existing top-level bound L context required')
        self.model, self.c, self.out = model, c, Path(out)
        self.original_context = original
        model._credit_original_context_function = original.__func__
        model._credit_pre_gain = None
        model._phase1_labeled_feature_context = MethodType(labeled_context, model)
        self.pending = None
        self.pending_update = None
        self.optimizer = None
        self.last_l_step = None
        self.last_late_step = None
        self.bound = False
        self.counts = dict(L_packets=0, U_packets=0, late_calls=0, actual_optimizer_steps=0,
                           active_calls=0, eligible_calls=0, skipped_missing_RX_calls=0,
                           eligible_RX_groups=0, accepted_RX_groups=0,
                           temporary_AdamW_steps=0, extra_eval_IQ_packets=0,
                           full_cohort_packets=0, active_ORDER_relations=0)
        self.epoch_U_count = 0
        self.probe_records = []
        self.active_gradient_probe_steps = []
        self.records = []

    def _design(self):
        from . import design
        return design

    def _bind(self):
        if self.bound:
            return
        # First core call is after EMA deepcopy and after the L forward. The
        # capture was already installed by the safe top-level context wrapper.
        self.model._source_labeled_feature_hook = self.auxiliary
        self.model._source_optimizer_update_probe_hook = self.probe
        self.bound = True

    def auxiliary(self, *, model, labeled_output, labels, extra, domains, optimizer_step,
                  identity_loss, labeled_count, labeled_inputs):
        z = getattr(model, '_credit_pre_gain', None)
        model._credit_pre_gain = None
        if (model is not self.model or not model.training or z is None
                or model._feature_l_steps != optimizer_step or labeled_count != 128
                or len(labels) != 128 or z.shape != (128, 160)):
            raise ValueError('Only the captured actual complete clean L_s main forward')
        if self.pending is not None or self.last_l_step == optimizer_step:
            raise ValueError('Duplicate/unconsumed L callback')
        self.last_l_step = optimizer_step
        self.counts['L_packets'] += labeled_count
        zero = (z * 0.).sum()
        if not self._design().active(optimizer_step):
            return zero, {'SIGTIME_credit/active': 0.}
        if (not isinstance(labeled_inputs, torch.Tensor) or labeled_inputs.shape != (128, 2, 256)
                or not labeled_inputs.is_floating_point() or not bool(torch.isfinite(labeled_inputs).all())):
            raise ValueError('Explicit original clean labeled_inputs[128,2,256] required')
        meta = self.native._meta_from_extra(extra)
        if any(key not in meta for key in ('rx_i', 'day_i', 'eq_i', 'sig_i', 'base_index')):
            raise ValueError('Actual visible L physical metadata incomplete')
        receivers, days = meta['rx_i'], meta['day_i']
        mode = {'BASE': 'ZERO', 'SUPCON': 'SUPCON', 'ORDER': 'ORDER', 'CREDIT': 'ORDER'}[self.c['arm']]
        result = objective(z, labels, receivers, days, meta['base_index'], mode)
        self.pending = dict(step=optimizer_step, result=result,
                            x=labeled_inputs.detach().clone(), y=labels.detach().clone().long(),
                            rx=receivers.detach().to(device=labeled_inputs.device, dtype=torch.long).clone(),
                            day=days.detach().clone(), physical_ids=meta['base_index'].detach().clone())
        self.counts['active_calls'] += 1
        self.counts['full_cohort_packets'] += result['summary']['unique_eligible_count']
        self.counts['active_ORDER_relations'] += result['summary']['unsatisfied_margin_count']
        # Auxiliary enters only via the later post_core return, never here.
        return zero, {'SIGTIME_credit/active': 1.}

    auxiliary._accepts_labeled_inputs = True

    def probe(self, *, model, optimizer, epoch, batch_idx, phase, optimizer_step,
              total_loss, unlabeled_loss, labeled_count, unlabeled_count):
        if (model is not self.model or self.last_l_step != optimizer_step
                or model._feature_l_steps != optimizer_step or self.last_late_step == optimizer_step
                or (self.last_late_step is not None and optimizer_step != self.last_late_step + 1)
                or labeled_count != 128):
            raise ValueError('Native L/late global step alignment differs or late loss repeated')
        expected_phase = 'label' if epoch <= 33 else 'pseudo'
        expected_U = 0 if epoch <= 33 else 124 if batch_idx == 222 else 256
        if phase != expected_phase or unlabeled_count != expected_U:
            raise ValueError('Original 33L+17U packet schedule differs')
        if self.optimizer is not None and self.optimizer is not optimizer:
            raise ValueError('Changed actual optimizer')
        self.optimizer = optimizer
        self.last_late_step = optimizer_step
        self.counts['late_calls'] += 1
        self.counts['U_packets'] += int(unlabeled_count)
        self.epoch_U_count += int(unlabeled_count)
        if not self._design().active(optimizer_step):
            if self.pending is not None:
                raise ValueError('Unexpected inactive pending auxiliary')
            return None
        pending = self.pending
        state_guard = None
        try:
            if pending is None or pending['step'] != optimizer_step or self.pending_update is not None:
                raise ValueError('Late callback lacks the current attached L graph')
            result = pending['result']
            zero = result['weighted_loss'] * 0.
            record = dict(step=optimizer_step, epoch=epoch, arm=self.c['arm'],
                          objective=result['summary'], base_includes_actual_U=True,
                          U_packets=int(unlabeled_count), U_loss=float(unlabeled_loss.detach()),
                          original_L_loader_unchanged=True, original_U_sampler_unchanged=True,
                          target_access=False, U_truth_read=False)
            if set(pending['rx'].tolist()) != set(SOURCE_RX):
                self.counts['skipped_missing_RX_calls'] += 1
                record.update(skipped='missing_source_RX', accepted_by_RX={str(rx): False for rx in SOURCE_RX},
                              selected_auxiliary_loss=0., temporary_AdamW_steps=0, extra_eval_IQ_packets=0)
                self._shadow(optimizer, total_loss, zero, optimizer_step, record)
                self._record(record)
                return {'post_core_auxiliary_loss': zero,
                        'post_core_auxiliary_metrics': {'SIGTIME_credit/skipped_missing_RX': 1., 'SIGTIME_credit/loss': 0.}}
            parameters = [p for group in optimizer.param_groups for p in group['params']]
            named = dict(model.named_parameters())
            by_id = {id(p): name for name, p in named.items()}
            if any(id(p) not in by_id for p in parameters) or any(not p.requires_grad for p in parameters):
                raise ValueError('Optimizer trainable parameter traversal differs')
            names = [by_id[id(p)] for p in parameters]
            weights = [name for name, p in named.items() if name.endswith('.cls_head.weight') and p.shape == (6, 160)]
            if len(weights) != 1:
                raise ValueError('One exact native default classifier W required')
            default_w = weights[0]
            protected = [name for name in names if name == default_w or name.endswith('sequence_pool.gain_raw')
                         or name.endswith('.cls_head.gain') or name.endswith('.response_gain')]
            if not any(name.endswith('sequence_pool.gain_raw') for name in protected):
                raise ValueError('Actual SIG_TIME gain parameter required')
            state_guard = self._guard(model, optimizer)
            state_guard.__enter__()
            before = _live_snapshot(optimizer)
            buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
            flags = {module: module.training for module in model.modules()}
            rng = _rng_snapshot()
            # Full current original total includes all real L/U/entropy terms.
            base = torch.autograd.grad(total_loss, parameters, retain_graph=True, allow_unused=True)
            gates, utilities, grad_audit = {}, {}, {}
            for receiver in SOURCE_RX:
                aux = torch.autograd.grad(result['loss_by_RX'][str(receiver)], parameters,
                                          retain_graph=True, allow_unused=True)
                if any(aux[names.index(name)] is not None for name in protected):
                    raise ValueError('Pre-gain ORDER auxiliary has W/gain gradient')
                grad_audit[str(receiver)] = dict(W_gain_gradients_None=True,
                    gradient_norm=sum(float(g.detach().double().square().sum()) for g in aux if g is not None)**.5)
                utility = assess(model, optimizer, base, aux, pending['x'], pending['y'].to(pending['x'].device),
                                 pending['rx'], receiver, default_weight_name=default_w,
                                 role='L_s', tolerance=1e-6)['summary']
                gates[str(receiver)] = bool(utility['accepted'])
                utilities[str(receiver)] = utility
                self.counts['temporary_AdamW_steps'] += utility['temporary_AdamW_steps']
                self.counts['extra_eval_IQ_packets'] += utility['extra_IQ_forwards']
                self.counts['eligible_RX_groups'] += 1
                self.counts['accepted_RX_groups'] += int(utility['accepted'])
            self.counts['eligible_calls'] += 1
            selected = (combine(result['loss_by_RX'], gates) if self.c['arm'] == 'CREDIT'
                        else result['weighted_loss'])
            selected_grads = torch.autograd.grad(selected, parameters, retain_graph=True, allow_unused=True)
            if any(selected_grads[names.index(name)] is not None for name in protected):
                raise ValueError('Selected pre-gain auxiliary has W/gain gradient')
            preservation = _live_readback(optimizer, before)
            preservation.update(live_buffers_unchanged=all(torch.equal(buffers[name], value) for name, value in model.named_buffers()),
                                training_flags_unchanged=all(module.training == flag for module, flag in flags.items()),
                                RNG_unchanged=_equal(rng, _rng_snapshot()))
            if not all(preservation.values()):
                raise RuntimeError('Late credit observation changed live training state')
            record.update(accepted_by_RX=gates, utility_by_RX=utilities, auxiliary_gradients=grad_audit,
                          protected_parameter_names=protected, selected_W_gain_gradients_None=True,
                          selected_auxiliary_gradient_norm=sum(float(g.detach().double().square().sum()) for g in selected_grads if g is not None)**.5,
                          selected_auxiliary_loss=float(selected.detach()), preservation=preservation,
                          temporary_AdamW_steps=10, extra_eval_IQ_packets=sum(u['extra_IQ_forwards'] for u in utilities.values()))
            self._shadow(optimizer, total_loss, selected, optimizer_step, record, base, selected_grads)
            self._record(record)
            return {'post_core_auxiliary_loss': selected,
                    'post_core_auxiliary_metrics': {'SIGTIME_credit/loss': selected.detach(),
                        'SIGTIME_credit/accepted_RX': float(sum(gates.values())),
                        'SIGTIME_credit/cohort_coverage': result['summary']['cohort_coverage']}}
        finally:
            try:
                if state_guard is not None:
                    state_guard.__exit__(*sys.exc_info())
            finally:
                self.pending = None
                model._credit_pre_gain = None

    @contextmanager
    def _guard(self, model, optimizer):
        params = [(p, p.detach().clone(), p.grad, None if p.grad is None else p.grad.detach().clone())
                  for p in model.parameters()]
        buffers = [(b, b.detach().clone()) for b in model.buffers()]
        flags = [(module, module.training) for module in model.modules()]
        before = _live_snapshot(optimizer)
        with _preserve_rng():
            try:
                yield
            finally:
                # Never copy unchanged live tensors: their autograd versions
                # must remain unchanged for the original once-only backward.
                with torch.no_grad():
                    for p, value, grad_ref, grad_value in params:
                        if not torch.equal(p.detach(), value):
                            p.copy_(value)
                        if p.grad is not grad_ref:
                            p.grad = grad_ref
                        if grad_ref is not None and not torch.equal(grad_ref, grad_value):
                            grad_ref.copy_(grad_value)
                    for b, value in buffers:
                        if not torch.equal(b, value):
                            b.copy_(value)
                if not _equal(before['optimizer_state'], optimizer.state_dict()):
                    optimizer.load_state_dict(before['optimizer_state'])
                for module, training in flags:
                    module.training = training

    def _shadow(self, optimizer, total_loss, selected, step, record, base=None, auxiliary=None):
        if step not in self._design().PROBE_STEPS:
            return
        params = [p for group in optimizer.param_groups for p in group['params']]
        by_id = {id(p): name for name, p in self.model.named_parameters()}
        names = [by_id[id(p)] for p in params]
        if base is None:
            base = torch.autograd.grad(total_loss, params, retain_graph=True, allow_unused=True)
        if auxiliary is None:
            auxiliary = torch.autograd.grad(selected, params, retain_graph=True, allow_unused=True)
        protected = [i for i, name in enumerate(names) if name.endswith('.cls_head.weight')
                     or name.endswith('sequence_pool.gain_raw') or name.endswith('.cls_head.gain')
                     or name.endswith('.response_gain')]
        if any(auxiliary[i] is not None for i in protected):
            raise ValueError('Actual probe auxiliary connects to W/gain')
        norm = sum(float(g.detach().double().square().sum()) for g in auxiliary if g is not None)**.5
        record.update(selected_auxiliary_gradient_norm=norm, selected_W_gain_gradients_None=True)
        record.setdefault('protected_parameter_names', [names[i] for i in protected])
        if norm > 1e-8:
            self.active_gradient_probe_steps.append(step)
        shadow = simulate(optimizer, base, add_gradients(base, auxiliary), names=names,
                          max_grad_norm=5., step_fn=self.original_step)
        self.counts['temporary_AdamW_steps'] += 2
        record['temporary_AdamW_steps'] += 2
        record['selected_update_shadow'] = shadow['summary']
        self.pending_update = dict(optimizer=optimizer, step=step, shadow=shadow, params=params, record=record)

    def _record(self, record):
        if record['step'] in self._design().PROBE_STEPS:
            self.records.append(record)
        with (self.out / 'credit_training_steps.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')

    def _write_report(self, epoch):
        value = dict(epoch=epoch, arm=self.c['arm'], counts=self.counts,
                     actual_probe_steps=self.probe_records, actual_probes=self.records,
                     auxiliary_calls=self.counts['active_calls'], eligible_calls=self.counts['eligible_calls'],
                     accepted_groups=self.counts['accepted_RX_groups'], eligible_groups=self.counts['eligible_RX_groups'],
                     optimizer_probe_steps=self.probe_records, active_gradient_probe_steps=self.active_gradient_probe_steps,
                     original_L_loader_unchanged=True, original_U_sampler_unchanged=True,
                     main_base_includes_actual_U=True, post_core_auxiliary_added_once=True,
                     default_head_joint_training=True, teacher_callbacks_installed=False,
                     head_fit_updates=0, extra_actual_optimizer_steps=0, U_truth_read=False,
                     target_access=False, caller_credit='same-batch seen other-RX L; not unseen RX or V')
        self._design().write(self.out / 'source_training_mechanism.json', value)

    @contextmanager
    def native_scope(self, native):
        self.native = native
        core, original_step = native.compute_core_losses, torch.optim.AdamW.step
        self.original_step = original_step
        def losses(*args, **kwargs):
            result = core(*args, **kwargs)
            self._bind()
            return result
        def actual_step(optimizer, *args, **kwargs):
            result = original_step(optimizer, *args, **kwargs)
            if optimizer is self.optimizer:
                self.counts['actual_optimizer_steps'] += 1
            pending = self.pending_update
            if pending is not None and pending['optimizer'] is optimizer:
                expected = pending['shadow']['total_after']
                error = max(float((p.detach() - q).abs().max()) for p, q in zip(pending['params'], expected))
                scale = max(1., max(float(p.detach().abs().max()) for p in pending['params']))
                tolerance = 1e-6 * scale
                if error > tolerance:
                    raise ValueError('Selected late auxiliary shadow differs from the actual once-only AdamW update')
                pending['record']['actual_update'] = dict(actual_shadow_max_abs_error=error,
                    actual_shadow_FP32_tolerance=tolerance, actual_optimizer_steps=1,
                    bitwise_equality_claimed=False, main_base_includes_actual_U=True)
                with (self.out / 'actual_credit_optimizer_increment.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(pending['record'], allow_nan=False) + '\n')
                self.probe_records.append(pending['step'])
                self.pending_update = None
            return result
        native.compute_core_losses, torch.optim.AdamW.step = losses, actual_step
        try:
            yield
        finally:
            native.compute_core_losses, torch.optim.AdamW.step = core, original_step
            if hasattr(self, 'model'):
                self.model._phase1_labeled_feature_context = self.original_context
                for name in ('_credit_pre_gain', '_credit_original_context_function',
                             '_source_labeled_feature_hook', '_source_optimizer_update_probe_hook'):
                    if hasattr(self.model, name):
                        delattr(self.model, name)
            self.pending = self.pending_update = None
            self.bound = False
            self.native = None

    def epoch(self, model, ctx, rows, out, device):
        epoch = len(rows)
        if model is not self.model or self.pending is not None or self.pending_update is not None:
            raise ValueError('Unconsumed L/late/actual optimizer observation')
        if self.counts['actual_optimizer_steps'] != self.counts['late_calls']:
            raise ValueError('Real optimizer update count differs from native late count')
        expected_U = 56700 if epoch >= 34 else 0
        if self.epoch_U_count != expected_U:
            raise ValueError('Original complete U epoch exposure differs')
        self._write_report(epoch)
        self.epoch_U_count = 0
