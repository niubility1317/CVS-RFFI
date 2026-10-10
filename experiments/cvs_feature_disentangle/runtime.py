"""Source-only real action proposals and detached receiver-risk training.

The frozen reference is periodically copied from this run's student. It is
neither an EMA nor a pseudo-label teacher. Learned L/T shifts never train E.
"""
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from experiments.cvs_phase1_stack.runtime import installed as native_installed
from experiments.cvs_multi_disentangle.runtime import deterministic_forward, finite_grad_norm
from experiments.cvs_multi_state_action.runtime import physical_audit_role, gradient_comparison
from experiments.cvs_multi_state_action.model import (intermediate, classify_intermediate,
    identity_from_intermediate, reference_response)


def gradient_vector(loss, parameters):
    if not parameters:
        return torch.zeros(1, device=loss.device)
    values = (torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
              if loss.requires_grad else [None] * len(parameters))
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                      for p, g in zip(parameters, values)])


def parameter_groups(identity):
    groups = {'E': [], 'G_C': []}
    for name, p in identity.named_parameters():
        if p.requires_grad:
            backend = '.cls_head.' in name or '.response_projection.' in name or name.endswith('response_gain')
            groups['G_C' if backend else 'E'].append(p)
    return groups


class Observations:
    """Each scalar has its own denominator; absent observations stay null."""
    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)
        self.known = set()

    def add(self, values):
        for key, value in values.items():
            if value is None or isinstance(value, (float, int)):
                self.known.add(key)
            if isinstance(value, (float, int)) and math.isfinite(value):
                self.sums[key] += value
                self.counts[key] += 1

    def finish(self):
        result = {k: self.sums[k] / self.counts[k] if self.counts[k] else None for k in sorted(self.known)}
        result['metric_observation_counts'] = dict(self.counts)
        self.sums.clear(); self.counts.clear()
        return result


class ActionTrainer:
    def __init__(self, c, model):
        from experiments.cvs_multi_action_risk.actions import StateAction, SharedActionCore
        from experiments.cvs_multi_action_risk.receiver import OnlineReceiver
        self.config = c; self.recipe = c['risk_recipe']; self.plan = c['arm_plan']
        self.arm = c['arm']; self.paths = tuple(self.plan['paths'])
        self.output_root = Path(c['output_root']); self.device = next(model.parameters()).device
        self.generator = torch.Generator().manual_seed(int(c['model_seed']) + 49071)
        self.auxiliary = nn.ModuleDict(); self.receiver = None; self.reference = None
        self.receiver_available = 0
        self.reference_version = 0; self.reference_step = None; self.updates = 0
        self.last = {}; self.observations = Observations(); self.audit_snapshots = []
        self.fit_steps = defaultdict(int); self.identity_steps = defaultdict(int)
        self.attempted_identity_steps = defaultdict(int); self.exposure_totals = defaultdict(int)
        self.calibration = {}; self.seconds = 0.; self.epoch_count = 0
        self.stage_seconds = defaultdict(float)
        self.measured_cpu_seconds = defaultdict(float); self.measured_gpu_seconds = defaultdict(float)
        self.timing_events = []; self.timing_calls = defaultdict(int)
        self.u_loader = None; self.u_iterator = None; self.update_before = None
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(c['model_seed']) + 21939)
            shared = SharedActionCore(hidden=int(self.recipe.get('hidden', 64))) if self.plan.get('shared') else None
            for kind in self.paths:
                if kind != 'receiver':
                    self.auxiliary[kind] = StateAction(kind, hidden=int(self.recipe.get('hidden', 64)), shared_core=shared)
            if 'receiver' in self.paths:
                kwargs = dict(target_mode=self.plan.get('r_target', 'distribution'),
                              max_age_steps=int(self.recipe.get('receiver_max_age_steps', 888)))
                from experiments.cvs_multi_action_risk.receiver import ReceiverDistributionAction
                kwargs['action'] = ReceiverDistributionAction(shared_core=shared if shared is not None else SharedActionCore(hidden=int(self.recipe.get('hidden', 64))))
                self.receiver = OnlineReceiver(**kwargs)
                self.auxiliary['receiver'] = self.receiver.action
        self.auxiliary.to(self.device)
        self.optimizer = (torch.optim.AdamW(self.auxiliary.parameters(), lr=self.recipe['auxiliary_lr'], weight_decay=1e-4)
                          if len(self.auxiliary) else None)
        self.cost = dict(native_model_parameters=sum(p.numel() for p in model.parameters()),
            native_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            identity_parameters=sum(p.numel() for p in model.id_backbone.parameters()),
            auxiliary_parameters=sum(p.numel() for p in self.auxiliary.parameters()),
            inference_auxiliary_parameters=0, identity_forward_unchanged=True, target_access=False,
            reference_origin='periodic own scratch student copy; not EMA',
            auxiliary_data='visible_source_L; optional metadata-hidden U action fit',
            native_optimizer_excludes_auxiliaries=True, virtual_updates='R and exact control G+C only')

    def _refresh(self, model, step):
        if self.reference_step is not None and step - self.reference_step < int(self.recipe.get('reference_refresh_steps', 888)):
            return
        self.reference = deepcopy(model.id_backbone).eval()
        for p in self.reference.parameters(): p.requires_grad_(False)
        self.reference_version += 1; self.reference_step = step; self.calibration = {}
        self.last['reference_refreshed'] = 1

    def _ref(self, x):
        self.exposure_totals['exact29_calls'] += 1; self.exposure_totals['exact29_packets'] += len(x)
        timer = self._start_timer('exact29_uncached')
        value = reference_response(self.reference, x)
        self._end_timer(timer)
        return value

    def _reference_endpoint(self, x):
        from experiments.cvs_multi_disentangle.model import _encoder
        timers = []
        def before(module, inputs): timers.append(self._start_timer('exact29_cached_endpoint_creation'))
        def after(module, inputs, output):
            self._end_timer(timers.pop())
            self.exposure_totals['exact29_calls'] += 1
            self.exposure_totals['exact29_packets'] += len(x)
        response = _encoder(self.reference).response
        hooks = [response.register_forward_pre_hook(before), response.register_forward_hook(after)]
        try: return intermediate(self.reference, x).detach()
        finally:
            for hook in hooks: hook.remove()

    def _start_timer(self, label):
        event = None
        if self.device.type == 'cuda':
            event = torch.cuda.Event(enable_timing=True); event.record()
        return label, time.perf_counter(), event

    def _end_timer(self, timer):
        label, start, event = timer
        self.measured_cpu_seconds[label] += time.perf_counter() - start
        self.timing_calls[label] += 1
        if event is not None:
            end = torch.cuda.Event(enable_timing=True); end.record()
            self.timing_events.append((label, event, end))

    def _predict(self, kind, h, x, p, endpoint, noise=None, endpoint_h=None):
        # h's final 29 values and cached endpoint avoid recomputing known response.
        self.exposure_totals['exact29_source_cache_hit_packets'] += len(x)
        self.exposure_totals['exact29_endpoint_cache_' + ('hit_packets' if endpoint_h is not None else 'miss_packets')] += len(x)
        er = endpoint_h[:, -29:] if endpoint_h is not None else self._ref(endpoint)
        return self.auxiliary[kind].predict(h, x, p, self._ref, phase_noise=noise,
            cached_reference=h[:, -29:], cached_endpoint_reference=er, endpoint_x=endpoint)

    def _fit_digital(self, batch, anchors, fit, y, label_free=False):
        from experiments.cvs_multi_action_risk.actions import action_loss
        losses = []
        order = str(batch['order'])
        second = 'temporal' if order in ('LT', 'linear_temporal') else 'linear'
        for kind in self.paths:
            if kind == 'receiver' or not bool(fit.any()): continue
            base, end = 'x00', 'x10' if kind == 'linear' else 'x01'
            if self.plan.get('conditional_edges', True) and kind == second:
                base, end = ('x10' if kind == 'temporal' else 'x01'), 'x11'
                self.exposure_totals[kind + '_conditional_edge_packets'] += int(fit.sum())
            else:
                self.exposure_totals[kind + '_main_edge_packets'] += int(fit.sum())
            h = anchors[base][fit]; target = anchors[end][fit] - h
            p = batch[kind + '_parameters'][fit]
            noise = batch.get('phase_noise'); noise = noise[fit] if noise is not None else None
            pred = self._predict(kind, h, batch[base][fit], p, batch[end][fit], noise, anchors[end][fit])
            parts = action_loss(self.reference, h, target, pred, None if label_free else y[fit], label_free=label_free)
            losses.append(parts['loss']); self.fit_steps[kind] += 1
            self.last.update({kind + '_fit_' + k: float(v.detach()) if torch.is_tensor(v) else v for k, v in parts.items()})
        return losses

    def _quality_bins(self, x):
        from experiments.cvs_multi_action_risk.physics import quality_proxies
        # Fixed observable coherence bins, never TX/RX labels or inferred SNR.
        return torch.clamp((quality_proxies(x)[:, 0] * 4).long(), 0, 3)

    def _calibrate(self, kind, h, target, pred, ids, x):
        bins = self._quality_bins(x)
        for bucket in bins.unique().tolist():
            mask = bins == bucket
            self._calibrate_bucket((kind, bucket), h[mask], target[mask], pred[mask],
                                   [pid for pid, keep in zip(ids, mask.tolist()) if keep])

    def _calibrate_bucket(self, key, h, target, pred, ids):
        with torch.no_grad():
            z0 = F.normalize(identity_from_intermediate(self.reference, h), dim=1)
            z1 = F.normalize(identity_from_intermediate(self.reference, h + target), dim=1)
            zp = F.normalize(identity_from_intermediate(self.reference, h + pred), dim=1)
            entry = self.calibration.setdefault(key, dict(error=0., zero=0., ids=set()))
            entry['error'] = .95 * entry['error'] + float((zp - z1).square().sum())
            entry['zero'] = .95 * entry['zero'] + float((z0 - z1).square().sum())
            entry['ids'].update(ids)
            skill = 1 - entry['error'] / max(entry['zero'], 1e-12)
            entry['weight'] = max(0., min(1., skill)) if len(entry['ids']) >= 16 else 0.
            prefix = key[0] + '_quality' + str(key[1])
            self.last[prefix + '_holdout_skill'] = skill
            self.last[prefix + '_proposal_reliability'] = entry['weight']

    def _proposal(self, kind, x, h, y, batch, anchors, identity, slot=None):
        from experiments.cvs_multi_action_risk.physics import sample_parameters, phase_process, apply_action
        # Every candidate is bounded by the exact same sampler as random control.
        candidates = []; risks = []; predicted = []
        search_timer = self._start_timer('candidate_search_inclusive_exact29')
        for i in range(int(self.recipe.get('proposal_candidates', 4))):
            p = (batch[kind + '_parameters'] if i == 0 else sample_parameters(kind, len(x), self.generator, x))
            noise = batch.get('phase_noise') if i == 0 else phase_process(len(x), self.generator, x)
            endpoint = (batch['x10' if kind == 'linear' else 'x01'] if i == 0 else apply_action(x, p, kind, phase_noise=noise))
            with torch.no_grad():
                delta = self._predict(kind, h, x, p, endpoint, noise, anchors['x10' if kind == 'linear' else 'x01'] if i == 0 else None)
                risks.append(F.cross_entropy(classify_intermediate(identity, h + delta), y, reduction='none'))
            candidates.append(endpoint); predicted.append(delta)
        self._end_timer(search_timer)
        with torch.no_grad():
            risk = torch.stack(risks); pick = risk.argmax(0)
            # Reliability controls cheap proposal selection before observing its
            # true endpoint. Prediction error must never discard a legal hard IQ.
            bins = self._quality_bins(x)
            tolerance = float(self.recipe.get('proposal_risk_relative_tolerance', .25))
            reliability = x.new_tensor([self.calibration.get((kind, int(b)), {}).get('weight', 0.) *
                max(0., 1. - self.calibration.get((kind, int(b)), {}).get('risk_relative_error', 0.) / tolerance)
                for b in bins.tolist()])
            proposal_mask = torch.rand(len(x), generator=self.generator).to(x.device) < reliability
            random_mask = torch.zeros(len(x), dtype=torch.bool, device=x.device)
            random_indices = (torch.arange(math.ceil(.25 * len(x)), device=x.device) + self.epoch_count) % len(x)
            random_mask[random_indices] = True
            pick[random_mask | ~proposal_mask] = 0
            ix = torch.arange(len(x), device=x.device)
            selected = torch.stack(candidates)[pick, ix].detach()
            predicted_risk = risk[pick, ix]
            verify_timer = self._start_timer('proposal_true_IQ_verification')
            actual_h = intermediate(identity, selected)
            actual_risk = F.cross_entropy(classify_intermediate(identity, actual_h), y, reduction='none')
            self._end_timer(verify_timer)
            relative_gap = (predicted_risk - actual_risk).abs() / (1. + actual_risk)
            bins = self._quality_bins(x); tolerance = float(self.recipe.get('proposal_risk_relative_tolerance', .25))
            for bucket in bins.unique().tolist():
                entry = self.calibration.setdefault((kind, int(bucket)), dict(error=0., zero=0., ids=set()))
                error = float(relative_gap[bins == bucket].mean())
                entry['risk_relative_error'] = .9 * entry.get('risk_relative_error', error) + .1 * error
                self.last[kind + '_quality' + str(bucket) + '_risk_relative_error'] = entry['risk_relative_error']
            weight = x.new_tensor([self.calibration.get((kind, int(bucket)), {}).get('weight', 0.) *
                max(0., 1 - self.calibration[(kind, int(bucket))]['risk_relative_error'] / tolerance) for bucket in bins.tolist()])
            accept = (~random_mask) & proposal_mask
            # No true-IQ rejection here: verified legal endpoints remain in CE.
            for bucket in bins.unique().tolist():
                self.calibration[(kind, int(bucket))]['virtual_weight'] = float(weight[bins == bucket].mean())
            prefix = slot or kind
            self.last.update({prefix + '_proposal_predicted_ce': float(predicted_risk.mean()),
                prefix + '_proposal_real_ce': float(actual_risk.mean()), prefix + '_proposal_risk_relative_gap': float(relative_gap.mean()),
                prefix + '_proposal_accepted': int(accept.sum()), prefix + '_random_retained': int((~accept).sum()),
                prefix + '_mispredicted_legal_IQ_retained': int((accept & (relative_gap > tolerance)).sum())})
            self.exposure_totals[prefix + '_candidate_packets'] += len(x) * len(candidates)
            self.exposure_totals[prefix + '_verification_packets'] += len(x)
        return selected

    def loss(self, model, ema_model, x, y, receivers, epoch, step, *, days=None, physical_ids=None, native_loss=None, **unused):
        self.last = dict(active=0., epoch=epoch, native_update=step, reference_refreshed=0.,
                         source_L_only=not self.plan.get('source_u_fit', False), identity_source_L_only=True)
        if ema_model is not None: raise ValueError('Pure CE experiment must not construct EMA')
        if self.arm == 'native' or epoch <= self.recipe['warmup_epochs'] or (step + 1) % self.recipe['auxiliary_every_steps']:
            return None
        if days is None or physical_ids is None or len(physical_ids) != len(x): raise ValueError('Visible source L metadata required')
        if bool((y < 0).any() or (receivers < 0).any() or (days < 0).any()): raise ValueError('U metadata forbidden in L/R endpoint')
        from experiments.cvs_multi_action_risk.physics import factorial_batch, apply_action
        start = time.perf_counter(); n = min(len(x), int(self.recipe['auxiliary_batch_size']), 32)
        ix = torch.randperm(len(x), generator=self.generator)[:n].to(x.device)
        x, y, receivers, days = x[ix], y[ix], receivers[ix], days[ix]
        ids = [physical_ids[i] for i in ix.cpu().tolist()]
        audit = torch.tensor([physical_audit_role(p) for p in ids], device=x.device)
        # Auxiliary TX0 is held out; still participates in all identity CE.
        fit = (~audit) & (y != 0)
        if len(self.auxiliary): self._refresh(model, step)
        identity = model.id_backbone
        devices = [x.device.index] if x.is_cuda else []
        with torch.random.fork_rng(devices=devices), deterministic_forward(identity):
            b = factorial_batch(x, self.generator, step=self.epoch_count)
            needed = {'x00'} if self.reference is not None else set()
            for kind in set(self.paths) - {'receiver'}:
                needed.add('x10' if kind == 'linear' else 'x01')
                second = 'temporal' if b['order'] == 'LT' else 'linear'
                if self.plan.get('conditional_edges', True) and kind == second:
                    needed.update(('x10' if kind == 'temporal' else 'x01', 'x11'))
            if {'linear', 'temporal'} <= set(self.paths): needed.update(('x10', 'x01', 'x11'))
            with torch.no_grad():
                a = {k: self._reference_endpoint(b[k]) for k in sorted(needed)}
            self.exposure_totals['reference_endpoint_packets'] += len(a) * n
            stage_start = time.perf_counter(); self.stage_seconds['reference_endpoints'] += stage_start - start
            if self.receiver is not None:
                self.receiver.update(x, a['x00'], y, receivers, days, ids, 'clean', step, self.reference_version, self.reference)
            fit_losses = self._fit_digital(b, a, fit, y, self.plan.get('action_label_free', False))
            if self.plan.get('source_u_fit'):
                if self.u_loader is None: raise ValueError('Missing metadata-hidden source U action loader')
                if self.u_iterator is None: self.u_iterator = iter(self.u_loader)
                try: ub = next(self.u_iterator)
                except StopIteration: self.u_iterator = iter(self.u_loader); ub = next(self.u_iterator)
                ux = ub[0][:n].to(x.device)
                if len(ub) > 1 and bool((torch.as_tensor(ub[1]) >= 0).any()): raise ValueError('Unmasked U label')
                if len(ub) > 3 and ub[3]: raise ValueError('U metadata must be fully hidden')
                ubatch = factorial_batch(ux, self.generator, step=self.epoch_count)
                with torch.no_grad(): ua = {k: self._reference_endpoint(ubatch[k]) for k in ('x00', 'x10', 'x01', 'x11')}
                self.exposure_totals['reference_endpoint_packets'] += len(ua) * len(ux)
                fit_losses += self._fit_digital(ubatch, ua, torch.ones(len(ux), device=x.device, dtype=torch.bool), None, True)
                self.exposure_totals['source_U_action_only_packets'] += len(ux)
            if self.receiver is not None:
                rfit_timer = self._start_timer('R_distribution_fit')
                rfit, stats = self.receiver.fit_loss(self.reference, step, self.reference_version)
                self._end_timer(rfit_timer)
                self.receiver_available = int(stats.get('available', 0))
                self.last.update({'R_fit_' + k: v for k, v in stats.items()})
                if rfit is not None and rfit.requires_grad: fit_losses.append(rfit); self.fit_steps['receiver'] += 1
            if fit_losses:
                self.optimizer.zero_grad(set_to_none=True)
                fl = torch.stack(fit_losses).mean()
                if not torch.isfinite(fl): raise FloatingPointError('Nonfinite action fit')
                fl.backward(); self.last['auxiliary_gradient_norm'] = finite_grad_norm(self.auxiliary)
                torch.nn.utils.clip_grad_norm_(self.auxiliary.parameters(), 1.)
                fraction = min(1., max(0., (epoch - self.recipe['warmup_epochs']) / (self.config.get('epochs', 200) - self.recipe['warmup_epochs'])))
                lr = self.recipe.get('auxiliary_lr_min', 1e-6) + .5 * (self.recipe['auxiliary_lr'] - self.recipe.get('auxiliary_lr_min', 1e-6)) * (1 + math.cos(math.pi * fraction))
                for group in self.optimizer.param_groups: group['lr'] = lr
                self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True); self.updates += 1
                self.last.update(auxiliary_fit_loss=float(fl.detach()), auxiliary_lr=lr)
            now = time.perf_counter(); self.stage_seconds['action_R_fit'] += now - stage_start; stage_start = now
            for kind in self.paths:
                if kind == 'receiver': continue
                with torch.no_grad():
                    target_key = 'x10' if kind == 'linear' else 'x01'
                    pred = self._predict(kind, a['x00'], x, b[kind + '_parameters'], b[target_key], b.get('phase_noise'), a[target_key])
                    if bool(audit.any()): self._calibrate(kind, a['x00'][audit], (a[target_key] - a['x00'])[audit], pred[audit], [pid for pid, keep in zip(ids, audit.tolist()) if keep], x[audit])
            if {'linear', 'temporal'} <= set(self.paths):
                from experiments.cvs_multi_action_risk.actions import chain_metrics
                with torch.no_grad(): chain = chain_metrics(self.auxiliary, b, a, self._ref)
                self.last.update({'chain_' + k: v for k, v in chain.items()})
            slots = set(self.paths) - {'receiver'} | set(self.plan.get('replacements', [])) - {'receiver'}
            if {'linear', 'temporal'} <= set(self.paths): slots.add('joint')
            selected = {}
            for kind in ('linear', 'temporal'):
                if kind in slots:
                    selected[kind] = (self._proposal(kind, x, a['x00'], y, b, a, identity)
                        if kind in self.paths and self.plan.get('action_mode') == 'proposal' else b['x10' if kind == 'linear' else 'x01'])
            # Joint follows same alternating order; selected component parameters
            # are realized IQ, so composing a fresh known second action is exact.
            if 'joint' in slots:
                selected['joint'] = b['x11']
                if {'linear', 'temporal'} <= set(self.paths) and self.plan.get('action_mode') == 'proposal':
                    # Propose the second action from the actual accepted first state.
                    first, second = ('linear', 'temporal') if str(b['order']) in ('LT', 'linear_temporal') else ('temporal', 'linear')
                    mid = selected[first]
                    jb = factorial_batch(mid, self.generator, step=self.epoch_count)
                    with torch.no_grad(): ja = {k: self._reference_endpoint(jb[k]) for k in ('x00', 'x10' if second == 'linear' else 'x01')}
                    self.exposure_totals['reference_endpoint_packets'] += len(ja) * n
                    selected['joint'] = self._proposal(second, mid, ja['x00'], y, jb, ja, identity, slot='joint')
            now = time.perf_counter(); self.stage_seconds['digital_calibration_proposal_verification'] += now - stage_start; stage_start = now
            weights = self.recipe.get('branch_weights', dict(linear=.015, temporal=.015, joint=.02, receiver=.01))
            terms = {}; logits_all = []
            for kind, iq in selected.items():
                real_timer = self._start_timer('real_IQ_identity_' + kind)
                if self.plan.get('action_mode') == 'exact_g':
                    with torch.no_grad(): h = intermediate(identity, iq)
                else: h = intermediate(identity, iq.detach())
                logits = classify_intermediate(identity, h); logits_all.append(logits)
                terms[kind] = float(weights[kind]) * F.cross_entropy(logits, y)
                self.identity_steps[kind] += 1; self.attempted_identity_steps[kind] += 1
                self.exposure_totals[kind + '_identity_real_packets'] += n
                self.last[kind + '_weighted_loss'] = float(terms[kind].detach())
                self._end_timer(real_timer)
            cons = float(self.plan.get('consistency_weight', 0.))
            if cons and logits_all:
                with torch.no_grad(): clean_logits = classify_intermediate(identity, intermediate(identity, x))
                terms['consistency'] = cons * torch.stack([F.kl_div(F.log_softmax(z, 1), F.softmax(clean_logits, 1), reduction='batchmean') for z in logits_all]).mean()
            if self.receiver is not None:
                calibration_timer = self._start_timer('R_reference_calibration')
                cal = self.receiver.calibrate(self.reference, step, self.reference_version)
                self._end_timer(calibration_timer)
                with torch.no_grad(): h = intermediate(identity, x)
                live_timer = self._start_timer('R_live_calibration')
                live = self.receiver.calibrate_live(identity, h, y, receivers, days, ids, 'clean', step, self.reference_version)
                self._end_timer(live_timer)
                replayed = int(live.get('reencoded_audit_packets', 0))
                self.last['R_live_reencoded_audit_packets'] = replayed
                self.exposure_totals['R_live_cached_audit_IQ_packets'] += replayed
                weight = min(cal.get('weights', {}).get('clean', 0.), live.get('weights', {}).get('clean', 0.))
                risk_timer = self._start_timer('R_identity_risk')
                if self.config['feature_plan'].get('r_identity', True):
                    rloss, stats = self.receiver.identity_loss(identity, h.detach(), y, receivers, days, ids, 'clean', step, self.reference_version, reliability=weight)
                else:
                    rloss, stats = None, dict(disabled_by_control=True, selected=0, used=0)
                self._end_timer(risk_timer)
                self.last.update({'R_identity_' + k: v for k, v in stats.items()})
                self.last['R_reliability'] = weight
                self.attempted_identity_steps['receiver'] += int(self.config['feature_plan'].get('r_identity', True))
                if rloss is not None and self.config['feature_plan'].get('r_identity', True):
                    terms['receiver'] = float(weights['receiver']) * weight * rloss
                    self.identity_steps['receiver'] += int(weight > 0)
            elif 'receiver' in self.plan.get('replacements', []):
                with torch.no_grad(): h = intermediate(identity, x)
                terms['receiver_replacement'] = float(weights['receiver']) * F.cross_entropy(classify_intermediate(identity, h), y)
            result = sum(terms.values()) if terms else native_loss * 0.
            now = time.perf_counter(); self.stage_seconds['real_identity_R_calibration_risk'] += now - stage_start; stage_start = now
            if native_loss is not None and (step + 1) % int(self.recipe.get('gradient_audit_every_steps', 888)) == 0:
                for partition, parameters in parameter_groups(identity).items():
                    reference = gradient_vector(native_loss, parameters)
                    for kind in ('linear', 'temporal', 'joint', 'receiver', 'total'):
                        loss = result if kind == 'total' else terms.get(kind)
                        if loss is None:
                            self.last[kind + '_' + partition + '_weighted_norm'] = None
                            self.last[kind + '_' + partition + '_weighted_cosine'] = None
                            self.last[kind + '_' + partition + '_weighted_ratio'] = None
                            continue
                        metrics = gradient_comparison(reference, gradient_vector(loss, parameters))
                        if not metrics['both_nonzero']: metrics['cosine'] = None
                        self.last.update({kind + '_' + partition + '_weighted_' + k: v for k, v in metrics.items()})
                        self.last[kind + '_' + partition + '_weighted_ratio'] = metrics['norm'] / max(metrics['reference_norm'], 1e-12)
                self.audit_snapshots.append(dict(self.last))
            self.stage_seconds['weighted_gradient_audit'] += time.perf_counter() - stage_start
            self.last.update(active=1., auxiliary_total_loss=float(result.detach()), physical_source_L_packets=n,
                fit_packets=int(fit.sum()), audit_packets=int(audit.sum()), reference_version=self.reference_version,
                auxiliary_successful_updates=self.updates)
        elapsed = time.perf_counter() - start; self.seconds += elapsed; self.epoch_count += 1
        self.last['auxiliary_step_seconds'] = elapsed
        self.exposure_totals['source_L_physical_packets'] += n
        self.observations.add(self.last)
        return result

    def optimizer_snapshot(self, model, before, step):
        if (step + 1) % int(self.recipe.get('gradient_audit_every_steps', 888)): return
        groups = parameter_groups(model.id_backbone)
        if before:
            self.update_before = {k: [p.detach().clone() for p in ps] for k, ps in groups.items()}
        elif self.update_before is not None:
            values = {k + '_optimizer_actual_update_norm': math.sqrt(sum(float((p.detach() - old).square().sum()) for p, old in zip(ps, self.update_before[k]))) for k, ps in groups.items()}
            self.last.update(values); self.observations.add(values)
            self.audit_snapshots.append(dict(native_update=step, **values)); self.update_before = None

    def execution(self):
        counters = self.receiver.counts(available=self.receiver_available) if self.receiver is not None else {}
        return dict(fit_steps=dict(self.fit_steps), identity_steps=dict(self.identity_steps),
            attempted_identity_steps=dict(self.attempted_identity_steps), exposure_totals=dict(self.exposure_totals),
            receiver_counters=counters, reference_versions=self.reference_version, target_access=False,
            calibration_scope='physical source L audit; TX0 excluded from visible-L auxiliary fits only; U identity metadata unknown and never inspected; formal six-TX holdouts are separate source diagnostics', gated_off_is_valid_negative=True)

    def annotate(self, rows):
        if not rows: return
        epoch = int(rows[-1]['epoch']); values = self.observations.finish()
        counts = values.pop('metric_observation_counts')
        values.update(auxiliary_active_steps_epoch=self.epoch_count, auxiliary_training_seconds=self.seconds,
                      auxiliary_parameters=self.cost['auxiliary_parameters'], auxiliary_successful_updates=self.updates)
        rows[-1].update({'risk_action_' + k: v for k, v in values.items()})
        rows[-1]['risk_action_observation_counts'] = counts
        self.output_root.mkdir(parents=True, exist_ok=True)
        with (self.output_root / 'gradient_audit_steps.jsonl').open('a', encoding='utf-8') as f:
            for row in self.audit_snapshots: f.write(json.dumps(dict(row, epoch=epoch), default=str) + '\n')
        self.audit_snapshots.clear()
        if self.timing_events:
            torch.cuda.synchronize(self.device)
            for label, start, end in self.timing_events:
                self.measured_gpu_seconds[label] += start.elapsed_time(end) / 1000.
            self.timing_events.clear()
        self.cost.update(mechanism_execution=self.execution(), process_peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else 0,
                         auxiliary_training_seconds=self.seconds, stage_seconds=dict(self.stage_seconds), additional_inference_transmission_bytes=0,
                         timing_scope='shared device unsynchronized wall intervals; not isolated kernel time',
                         measured_cpu_seconds=dict(self.measured_cpu_seconds), measured_cuda_event_seconds=dict(self.measured_gpu_seconds), measured_section_calls=dict(self.timing_calls),
                         measured_timing_scope='inclusive nested sections on shared hardware; CUDA events read after one epoch-end synchronize; do not sum nested exact29 and candidate times',
                         reference_parameter_buffer_bytes=0 if self.reference is None else sum(t.numel()*t.element_size() for t in list(self.reference.parameters())+list(self.reference.buffers())),
                         auxiliary_parameter_buffer_bytes=sum(t.numel()*t.element_size() for t in list(self.auxiliary.parameters())+list(self.auxiliary.buffers())),
                         auxiliary_optimizer_state_bytes=0 if self.optimizer is None else sum(v.numel()*v.element_size() for state in self.optimizer.state.values() for v in state.values() if torch.is_tensor(v)),
                         receiver_contribution_bytes=0 if self.receiver is None else sum(v.numel()*v.element_size() for entry in self.receiver.entries.values() for v in entry.values() if torch.is_tensor(v)))
        try:
            import psutil
            self.cost['process_resident_memory_bytes'] = psutil.Process().memory_info().rss
        except ImportError: self.cost['process_resident_memory_bytes'] = None
        self.cost['FLOPs'] = None
        state = dict(schema='risk_action_training_only_v1', epoch=epoch, config=self.config, auxiliary=self.auxiliary.state_dict(),
            optimizer=None if self.optimizer is None else self.optimizer.state_dict(), reference_version=self.reference_version,
            private_generator=self.generator.get_state(), mechanism_execution=self.execution(), checkpoint_sources=[],
            resume_supported=False, inference_uses_auxiliary=False, target_access=False, cost=self.cost,
            feature_state=self.feature_state())
        torch.save(state, self.output_root / ('auxiliary_final.pth' if epoch == self.config.get('epochs', 200) else 'auxiliary_latest.pth'))
        (self.output_root / 'auxiliary_cost.json').write_text(json.dumps(self.cost, indent=2), encoding='utf-8')
        with (self.output_root / 'auxiliary_epoch_counts.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(epoch=epoch, counts=counts)) + '\n')
        print('RISK_ACTION ' + json.dumps(dict(values, epoch=epoch, arm=self.arm)), flush=True)
        self.epoch_count = 0


def feature_parameter_groups(identity):
    """Disjoint E/G/C groups, plus named branch subsets for dependence audits."""
    groups = {k: [] for k in ('E', 'G', 'C', 'time', 'frequency', 'PA')}
    for name, p in identity.named_parameters():
        if not p.requires_grad: continue
        if name.endswith('cls_head.weight'): partition = 'C'
        elif '.cls_head.' in name or '.response_projection.' in name or name.endswith('response_gain'): partition = 'G'
        else: partition = 'E'
        groups[partition].append(p)
        parts = name.split('.')
        if any(s.startswith('time_') or s in ('t1','t2','t3','t_pool','t_proj') for s in parts): groups['time'].append(p)
        if any(s.startswith('freq_') or s in ('f1','f2','f3','f_pool','f_proj') for s in parts): groups['frequency'].append(p)
        if any(s.startswith('pa_') for s in parts): groups['PA'].append(p)
    return groups


class OnlineTrainer(ActionTrainer):
    """Independent B48 relation stream; only native L can enter weak style."""
    def __init__(self, c, model):
        super().__init__(c, model)
        from .fishr import ConditionalFishr, GradientRatioCalibrator
        from .style import WeakConditionalMixStyle
        self.feature_plan = c['feature_plan']
        self.feature_recipe = c.get('feature_recipe', {})
        self.relation = None
        self.feature_counts = defaultdict(int)
        self.feature_last = {}
        self._last_annotated_epoch = None
        self.style = WeakConditionalMixStyle(seed=int(c['model_seed']) + 77917) if self.feature_plan['style'] else None
        if self.style is not None: self.style.attach(model.id_backbone)
        self.fishr = (ConditionalFishr(mode=self.feature_plan['fishr'], beta=.9)
                      if self.feature_plan['fishr'] != 'none' else None)
        self.fishr_calibrator = (GradientRatioCalibrator(float(self.feature_plan['fishr_ratio']))
                                 if self.fishr is not None else None)
        self.cost.update(relation_scene='clean only; satellite explicitly disabled',
            style_scope='native labeled clean identity forward time_down only',
            fishr_definition='cosine six-row direction/raw gradient variance; 960 coordinates',
            relation_bank_separate_from_R=True)

    def set_source(self, loader):
        if self.feature_plan['relation'] == 'none': return
        from .sampling import RelationStream
        self.relation = RelationStream(loader.dataset, seed=int(self.config['model_seed']) + 88739,
                                       mode=self.feature_plan['relation'])

    def _refresh(self, model, step):
        # A deepcopy must not clone the live style hook into reference fitting.
        refresh = self.reference_step is None or step - self.reference_step >= int(self.recipe.get('reference_refresh_steps', 888))
        if refresh and self.style is not None: self.style.detach()
        try: super()._refresh(model, step)
        finally:
            if refresh and self.style is not None: self.style.attach(model.id_backbone)

    def native_forward(self, model, x, *, y, rx, day, pid, epoch, kwargs):
        timer = self._start_timer('native_L_with_optional_style')
        context = (self.style.context(y=y, rx=rx, day=day, pid=pid, role='native_L_clean', epoch=epoch)
                   if self.style is not None else nullcontext())
        try:
            with context: return model(x, **kwargs)
        finally:
            self._end_timer(timer)

    def feature_state(self):
        return dict(schema='feature_disentangle_training_only_v1',
            fishr=None if self.fishr is None else self.fishr.state_dict(),
            calibration=None if self.fishr_calibrator is None else self.fishr_calibrator.state_dict(),
            style=None if self.style is None else self.style.state_dict(),
            relation=None if self.relation is None else self.relation.state_dict(),
            counts=dict(self.feature_counts), target_access=False)

    def _feature_audit(self, identity, native_loss, terms, step):
        if step % int(self.recipe.get('gradient_audit_every_steps', 888)): return
        for partition, parameters in feature_parameter_groups(identity).items():
            reference = gradient_vector(native_loss, parameters)
            self.feature_last['native_' + partition + '_gradient_norm'] = float(reference.norm())
            for kind, loss in terms.items():
                metrics = gradient_comparison(reference, gradient_vector(loss, parameters))
                if not metrics['both_nonzero']: metrics['cosine'] = None
                self.feature_last.update({kind + '_' + partition + '_weighted_' + k: v for k, v in metrics.items()})
                self.feature_last[kind + '_' + partition + '_weighted_ratio'] = metrics['norm'] / max(metrics['reference_norm'], 1e-12)
        self.audit_snapshots.append(dict(native_update=step, feature=True, **self.feature_last))

    @torch.no_grad()
    def _quality_fishr_audit(self, x, z, weight, y, receivers, step):
        if step % int(self.recipe.get('gradient_audit_every_steps', 888)): return
        from experiments.cvs_multi_action_risk.physics import quality_proxies
        from .fishr import cosine_sample_gradients, sample_variance, pair_penalty
        qualities = quality_proxies(x)
        gradients = cosine_sample_gradients(z, weight, y, self.fishr.mode)
        receiver_ids = receivers.unique().tolist()
        rows = []
        for qi, name in ((0, 'coherence'), (1, 'repeat_residual')):
            bins = torch.bucketize(qualities[:, qi].contiguous(), qualities.new_tensor([.25, .5, .75]))
            for bucket in range(4):
                subsets = [(receivers == r) & (bins == bucket) for r in receiver_ids]
                counts = [int(m.sum()) for m in subsets]
                gap = penalty = None
                if min(counts) >= 2:
                    variances = [sample_variance(gradients[m]) for m in subsets]
                    gap = float((variances[0] - variances[1]).abs().mean())
                    penalty = float(pair_penalty(*variances))
                prefix = 'fishr_quality_' + name + str(bucket)
                self.feature_last[prefix + '_gap'] = gap
                self.feature_last[prefix + '_penalty'] = penalty
                rows.append(dict(proxy=name, bin=bucket, bounds=[.25,.5,.75], receivers=receiver_ids,
                    support=counts, variance_gap=gap, pair_penalty=penalty,
                    class_counts=[torch.bincount(y[m],minlength=6).tolist() for m in subsets]))
        self.audit_snapshots.append(dict(native_update=step, fishr_quality=rows,
            scope='instantaneous real clean strata; diagnostic only, per-stratum class mix may differ; no EMA update'))

    def loss(self, model, ema_model, x, y, receivers, epoch, step, *, days=None, physical_ids=None, native_loss=None, **unused):
        if ema_model is not None: raise ValueError('Pure CE must never construct an EMA teacher')
        if self.paths or self.plan.get('replacements'):
            actions = super().loss(model, ema_model, x, y, receivers, epoch, step, days=days,
                                   physical_ids=physical_ids, native_loss=native_loss)
        else:
            self.last = dict(active=0., epoch=epoch, native_update=step)
            actions = None
        self.feature_last = dict(relation_active=0., fishr_active=0., feature_epoch=epoch)
        if self.feature_plan['relation'] == 'none' or (step + 1) % 8:
            self._feature_audit(model.id_backbone, native_loss, {} if actions is None else {'joint_auxiliary': actions}, step + 1)
            self.last.update(self.feature_last)
            self.observations.add(self.feature_last)
            return actions
        if self.relation is None: raise RuntimeError('Source L relation stream was not attached')
        timer = self._start_timer('relation_clean_CE_and_Fishr')
        try:
            rx, ry, _, meta = self.relation.next_batch(step=step + 1, device=self.device)
            if len(rx) != 48 or bool((ry < 0).any()): raise ValueError('Expected exactly 48 labeled source packets')
            identity = model.id_backbone
            devices = [rx.device.index] if rx.is_cuda else []
            with torch.random.fork_rng(devices=devices), deterministic_forward(identity):
                h = intermediate(identity, rx)
                z = identity_from_intermediate(identity, h)
                from experiments.cvs_multi_disentangle.model import _encoder
                encoder = _encoder(identity)
                logits = encoder.classify_features(z)
                relation_ce = F.cross_entropy(logits, ry)
                terms = {'relation': .1 * relation_ce}
                self.feature_counts['relation_calls'] += 1
                self.feature_counts['relation_clean_forward_packets'] += 48
                self.feature_last.update(relation_active=1., relation_CE=float(relation_ce.detach()),
                    relation_weighted_CE=float(terms['relation'].detach()), relation_packets=48)
                if self.fishr is not None:
                    fishr_loss, info = self.fishr(z, encoder.core.id_backbone.cls_head.weight, ry,
                        meta['rx_i'].to(self.device), meta['day_i'].to(self.device), step=step + 1, scene='clean')
                    if not info.get('updated'): raise RuntimeError('Registered balanced B48 failed Fishr support: '+str(info))
                    self.feature_counts['fishr_statistics_updates'] += 1
                    # Mature E20 source batches set one median scale. Freeze at
                    # update4440 and never adapt it after warmup.
                    if epoch == 20:
                        calibration = self.fishr_calibrator.observe(native_loss, fishr_loss,
                            model.id_backbone.parameters(), info['bucket_counts'], epoch=20)
                        self.feature_last['fishr_calibration'] = calibration
                        if step + 1 == 20 * int(self.recipe['steps_per_epoch']): self.fishr_calibrator.freeze()
                    weight = self.fishr_calibrator.weight(epoch)
                    if weight:
                        terms['fishr'] = weight * fishr_loss
                        self.feature_counts['fishr_positive_weight_calls'] += 1
                    self.feature_last.update(fishr_active=float(weight > 0), fishr_penalty=float(fishr_loss.detach()),
                        fishr_weight=weight, fishr_weighted_loss=weight * float(fishr_loss.detach()),
                        fishr_variance_gap=info['variance_gap'], fishr_bucket_min_count=min(info['bucket_counts']),
                        fishr_bucket_max_count=max(info['bucket_counts']),
                        fishr_weight_row_norm_mean=float(encoder.core.id_backbone.cls_head.weight.detach().norm(dim=1).mean()))
                    self.audit_snapshots.append(dict(native_update=step + 1, fishr_buckets=info['buckets'],
                        fishr_weight_row_norms=info['weight_row_norms']))
                    self._quality_fishr_audit(rx, z, encoder.core.id_backbone.cls_head.weight, ry,
                        meta['rx_i'].to(self.device), step + 1)
                if actions is not None: terms['actions'] = actions
                result = sum(terms.values())
                # Relation calls use step+1; D audits use existing step clock.
                audit_step = step + 1
                self._feature_audit(identity, native_loss, dict(terms, joint_auxiliary=result), audit_step)
                self.last.update(self.feature_last)
                self.observations.add(self.feature_last)
                return result
        finally: self._end_timer(timer)

    def execution(self):
        result = super().execution()
        result.update(feature_plan=self.feature_plan, feature_counts=dict(self.feature_counts),
            relation=None if self.relation is None else self.relation.summary(),
            style=None if self.style is None else self.style.snapshot(),
            fishr_calibration=None if self.fishr_calibrator is None else self.fishr_calibrator.state_dict())
        return result

    def annotate(self, rows):
        if not rows or self._last_annotated_epoch == int(rows[-1]['epoch']): return
        if self.style is not None:
            self.cost['style_execution'] = self.style.snapshot()
            if rows:
                for role, values in self.cost['style_execution']['roles'].items():
                    for key, value in values.items():
                        rows[-1]['feature_style_' + role + '_' + key + '_cumulative'] = value
        if self.fishr is not None:
            self.cost['fishr_statistics_bytes'] = sum(v['m'].numel()*v['m'].element_size() for v in self.fishr.buckets.values())
            self.cost['fishr_bucket_count'] = len(self.fishr.buckets)
        self.cost['feature_execution'] = self.execution()
        super().annotate(rows)
        self._last_annotated_epoch = int(rows[-1]['epoch'])


def instrument_train(native, callback, optimizer_callback, forward_callback):
    original = native.train; code = inspect.getsource(original)
    marker = '            loss_is_finite = bool(torch.isfinite(loss.detach()).item())'
    if code.count(marker) != 1: raise RuntimeError('Native objective insertion not unique')
    insertion = ('            _risk_loss = _risk_action_step(model, ema_model, r3_x_l,\n'
        '                y_l[:labeled_clean_count], receiver_l_base, epoch, r3_optimizer_steps,\n'
        '                days=day_l_base, physical_ids=stable_sample_keys(_meta_from_extra(extra_l) or {}), native_loss=loss)\n'
        '            if _risk_loss is not None: loss = loss + _risk_loss\n')
    step_marker = '                    scaler.step(optimizer)'
    if code.count(step_marker) != 1: raise RuntimeError('Native optimizer insertion not unique')
    code = code.replace(marker, insertion + marker).replace(step_marker,
        '                    _risk_optimizer_snapshot(model, True, r3_optimizer_steps)\n' + step_marker + '\n'
        '                    _risk_optimizer_snapshot(model, False, r3_optimizer_steps)')
    forward_marker = '                out_l = forward_model(\n                    x_l_main,'
    if code.count(forward_marker) != 1: raise RuntimeError('Native style forward insertion not unique')
    code = code.replace(forward_marker,
        '                out_l = _feature_native_forward(forward_model,\n                    x_l_main,\n'
        '                    _feature_metadata=dict(y=y_l, rx=receiver_l_base, day=day_l_base,\n'
        '                        pid=stable_sample_keys(_meta_from_extra(extra_l) or {}), epoch=epoch),')
    native.__dict__['_risk_action_step'] = callback; native.__dict__['_risk_optimizer_snapshot'] = optimizer_callback
    native.__dict__['_feature_native_forward'] = forward_callback
    exec(compile(code, inspect.getsourcefile(original) + '::risk_action', 'exec'), native.__dict__)
    return original


@contextmanager
def installed(c, training=True):
    if c.get('checkpoint_sources') or c['risk_recipe'].get('resume'): raise ValueError('Scratch identity only')
    with native_installed(c) as native:
        if not training: yield native; return
        names = ['build_baseline_model', '_write_ssdg_epoch_telemetry', '_detach_log_mapping', '_build_ssdg_wisig_data', 'WiSigSubsetDataset', '_muse_epoch_pairs']
        originals = {k: getattr(native, k) for k in names}; state = {}
        old_hooks = {k: getattr(native, k, None) for k in ('_risk_action_step', '_risk_optimizer_snapshot', '_risk_action_annotate', '_feature_native_forward', '_feature_training_state')}
        class MetadataHiddenU(originals['WiSigSubsetDataset']):
            def __getitem__(self, i):
                item = super().__getitem__(i)
                return (item[0], -1, -1, {}) if self.hide else item
        native.WiSigSubsetDataset = MetadataHiddenU
        def pairs(labeled_loader, unlabeled_loader, *, use_muse, use_unlabeled_step_budget=False):
            if use_muse: raise ValueError('Pure CE cannot enter MUSE')
            iterator = iter(labeled_loader)
            for _ in range(int(c['risk_recipe']['steps_per_epoch'])):
                try: batch = next(iterator)
                except StopIteration: iterator = iter(labeled_loader); batch = next(iterator)
                yield batch, None
        native._muse_epoch_pairs = pairs
        def data(*a, **kw):
            context = originals['_build_ssdg_wisig_data'](*a, **kw)
            state['u_loader'] = context['unlabeled_loader'] if c['arm_plan'].get('source_u_fit') else None
            state['l_loader'] = context['train_loader']
            if 'trainer' in state:
                state['trainer'].u_loader = state['u_loader']
                state['trainer'].set_source(state['l_loader'])
            return context
        def build(args, device):
            model = originals['build_baseline_model'](args, device)
            state['trainer'] = OnlineTrainer(c, model)
            state['trainer'].output_root = Path(getattr(args, 'output_dir', c['output_root']))
            state['trainer'].u_loader = state.get('u_loader')
            if 'l_loader' in state: state['trainer'].set_source(state['l_loader'])
            print('FEATURE_DISENTANGLE_CONFIG ' + json.dumps(dict(recipe=c['risk_recipe'], plan=c['arm_plan'], feature_recipe=c.get('feature_recipe'), feature_plan=c['feature_plan'], **state['trainer'].cost)), flush=True)
            return model
        def annotate(rows):
            if 'trainer' in state: state['trainer'].annotate(rows)
        def telemetry(cp, jp, rows):
            annotate(rows); return originals['_write_ssdg_epoch_telemetry'](cp, jp, rows)
        def metrics(values):
            if 'train/loss' in values and 'trainer' in state:
                values = dict(values, **{'train/risk_action_' + k: v for k, v in state['trainer'].last.items()})
            return originals['_detach_log_mapping'](values)
        native.build_baseline_model = build; native._build_ssdg_wisig_data = data
        native._write_ssdg_epoch_telemetry = telemetry; native._detach_log_mapping = metrics; native._risk_action_annotate = annotate
        native._feature_training_state = lambda: state['trainer']
        def forward(model, x, _feature_metadata, **kwargs):
            return state['trainer'].native_forward(model, x, kwargs=kwargs, **_feature_metadata)
        original_train = instrument_train(native, lambda *a, **kw: state['trainer'].loss(*a, **kw), lambda *a: state['trainer'].optimizer_snapshot(*a), forward)
        try: yield native
        finally:
            if 'trainer' in state and state['trainer'].style is not None: state['trainer'].style.detach()
            native.train = original_train
            for key, value in originals.items(): setattr(native, key, value)
            for key, value in old_hooks.items():
                if value is None: delattr(native, key)
                else: setattr(native, key, value)
