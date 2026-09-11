"""Reachable CORE90 training integration and explicit activation evidence."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import hashlib
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from .schema import records_from_dataset
from .sampler import CrossBlockSampler, CrossResponseDataset, extract_batch_plan
from .scheduler import FeedbackScheduler
from .roles import mixstyle_allowed_mask
from .statistics import WaveformStatistics, FixedSourceNormalizer, query_cell_targets
from .readouts import ResponseReadouts
from .predictor import SharedResponsePredictor
from .cache import FixedStatisticCache
from .tensor_ops import response_loss, identity_interaction_loss, double_center, unit_identity_geometry, normalized_identity_interaction_loss, decomposed_response_losses
from .gate_v2 import SourceMechanismGate
from .gradient_audit import audit_decision_parameter_gradients
from .decision import decision_margin_loss, decision_margin_loss_vectorized
from .decision_calibration import calibrated_decision_tensors, validate_decision_calibration
from .training import SourceResponseGate, parameter_roles, response_backward, IndependentAuxiliaryTransaction
from .source_eval import raw_received_iq, event_metadata_from_records, evaluate_source_response, evaluate_source_response_v2
from .source_mechanism import (paired_actual_optimizer_update, joint_objective_contract,
    source_prediction_measurements, prepare_source_query, aggregate_source_observation)
from .replay_audit import capture_rng_state, restore_rng_state


def _scalar(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = float(value.detach())
    if isinstance(value, (int, float, bool)) and math.isfinite(value):
        return float(value)
    return None


def _json_evidence(value):
    if torch.is_tensor(value):
        return _json_evidence(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): _json_evidence(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_evidence(v) for v in value]
    if isinstance(value, set):
        return [_json_evidence(v) for v in sorted(value, key=repr)]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def input_artifact_identity(path):
    """Bind dataset-local record IDs to the actual input artifact for resume.

    This is checkpoint data-contract identity, not a new data validation pass.
    A copied/changed input requires an explicit new scratch run contract.
    """
    resolved = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    size = 0
    with resolved.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    return {'resolved_path': str(resolved), 'sha256': digest.hexdigest(), 'size_bytes': size}


class CrossResponseRuntime:
    def __init__(self, model, data_ctx, config, args, device):
        self.config, self.device = deepcopy(config), device
        self._model = model
        self._args = args
        self._domain_label_map = data_ctx.get('domain_label_map', {})
        self.v2 = config.get("implementation_version", 1) == 2
        self.auxiliary_transaction = None
        self.statistics = self.normalizer = None
        self.statistic_cache = FixedStatisticCache() if self.v2 else None
        self.variant = str(args.cross_response_variant)
        self.active = bool(config["enabled"])
        self.source_dataset = data_ctx["train_loader"].dataset
        self.validation_dataset = data_ctx["val_loader"].dataset
        self.records = records_from_dataset(self.source_dataset)
        self.validation_records = records_from_dataset(self.validation_dataset,
            allowed_split_sources=("ssdg_source_v_select", "ssdg_source_v_cal"))
        unlabeled_loader = data_ctx.get("unlabeled_loader")
        unlabeled_dataset = unlabeled_loader.dataset if unlabeled_loader is not None else None
        # Metadata-only lineage check: U_s labels/IQ never enter auxiliary targets.
        unlabeled_records = records_from_dataset(unlabeled_dataset,
            allowed_split_sources=("ssdg_unlabeled_tx_hidden",)) if unlabeled_dataset is not None else ()
        groups = {"train": self.records, "unlabeled": unlabeled_records, "validation": self.validation_records}
        source_rxs = {int(v.strip()) for v in str(args.wisig_train_rxs).split(',') if v.strip()}
        source_days = {int(v.strip()) for v in str(args.wisig_train_days).split(',') if v.strip()}
        seen = set()
        for role, records in groups.items():
            ids = {r.physical_sample_id for r in records}
            if seen & ids:
                raise ValueError("source L_s/U_s/V physical records overlap")
            if any(r.rx_id not in source_rxs or r.day_id not in source_days for r in records):
                raise ValueError(f"{role} physical records are outside declared source RX/day roles")
            seen.update(ids)
        # Persist actual source roles, not just the dataset name or count.
        self.source_contract = {"input_artifact": input_artifact_identity(args.wisig_pkl),
                                "train": tuple(r.physical_sample_id for r in self.records),
                                "unlabeled": tuple(r.physical_sample_id for r in unlabeled_records),
                                "validation": tuple(r.physical_sample_id for r in self.validation_records),
                                "split_roles": {"train": self.source_dataset.split_source,
                                    "unlabeled": unlabeled_dataset.split_source if unlabeled_dataset is not None else None,
                                    "validation": self.validation_dataset.split_source},
                                "source_receivers": str(args.wisig_train_rxs),
                                "target_receivers": str(args.wisig_test_rxs),
                                "source_days": str(args.wisig_train_days),
                                "target_days": str(args.wisig_test_days)}
        self.decision_calibration = None
        if config.get("decision_mode") in ("delta", "delta_pairs"):
            self.decision_calibration = torch.load(config["decision_calibration"], map_location="cpu", weights_only=False)
            validate_decision_calibration(self.decision_calibration, mode=config["decision_mode"],
                source_contract=self.source_contract, num_classes=model.id_backbone.cls_head.head.weight.shape[0])
        self.output = Path(args.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.auxiliary = None
        self.sampler = None
        self.roles = None
        self.gate = SourceResponseGate(config["gate_min_blocks"], config["gate_error_ratio"], config["gate_stable_checks"])
        self.mechanism_gate = SourceMechanismGate(**config["mechanism_gate"]) if self.v2 and config.get("mechanism_gate") else None
        self.extended_identity_module = None
        self.mechanism_audit_state = dict(window=[], observations=[], attempts=0, paired_updates=0,
                                         phase=None, scope='cls_head', last_status=None)
        self.pending_source_update = None
        self.mechanism_audit_due = False
        self.joint_objective = joint_objective_contract(config,
            (config.get('mechanism_audit') or {}).get('identity_scope', 'cls_head'))
        self.counts = {"batches": 0, "successful_steps": 0, "effective_blocks": 0,
                       "response_batches": 0, "decision_batches": 0, "cross_batches": 0,
                       "joint_gradient_steps": 0, "domain_gradient_steps": 0, "head_gradient_steps": 0,
                       "decision_gradient_steps": 0, "cross_gradient_steps": 0, "forward_calls": 0}
        self.counts.update(audited_steps=0, audited_successful_steps=0, forced_audit_steps=0)
        self.maxima = {}
        self.source_evaluations = []
        self.rotation_counts = {}
        self.pending_blocks = []
        self.logs = {}
        self.loader_generator = torch.Generator().manual_seed(int(config["data_seed"]) + 811)
        if not self.active:
            return
        if int(args.batch_size) < config["P"] * config["Q"] * min(config["k_menu"]):
            raise ValueError("batch record budget cannot form any configured complete block")
        scheduler = FeedbackScheduler(mode="guided" if config["scheduler_mode"] == "feedback" else "uniform",
            exploration=config["exploration"], alpha=config["scheduler_alpha"], beta=config["scheduler_beta"],
            error_clip=config["feedback_clip"], update_interval=config["update_interval"], views=1,
            feedback_version="transaction_v2" if self.v2 else "legacy_v1",
            role_policy_version="balanced_partitions_v2" if self.v2 else "legacy_halves_v1",
            direction_shrinkage=config.get("direction_shrinkage", 0.),
            gain_strategy=config.get("gain_strategy", "legacy_gain"), evidence_config=config.get("evidence_config"))
        if config.get('gain_strategy') == 'reliable_evidence':
            if self.mechanism_gate is None:
                raise ValueError('reliable feedback requires a live joint mechanism gate')
            if config['evidence_config']['source_evidence']['gate_result'].get('source_freeze_id') != self.mechanism_gate.config['source_freeze_id']:
                raise ValueError('feedback gate source freeze does not match the current joint objective')
            scheduler.bind_qualification(source_contract=self.source_contract, joint_objective=self.joint_objective,
                provider=lambda: dict(authorized=self.joint_open, scope=self.extended_identity_module or 'cls_head'))
        old_loader = data_ctx["train_loader"]
        self.sampler = CrossBlockSampler(self.records, P=config["P"], Q=config["Q"],
            k_menu=config["k_menu"], batch_size=int(args.batch_size), seed=int(config["data_seed"]),
            max_candidates=config["scheduler_candidate_limit"], scheduler=scheduler,
            steps_per_epoch=len(old_loader), role_rotation=config["role_rotation"],
            role_policy="balanced_partitions_v2" if self.v2 else "legacy_halves_v1")
        self.reachable_coverage = self.sampler.scheduler.coverage.reachable_coverage(self.sampler.candidates,
            "balanced_partitions_v2" if self.v2 else "legacy_halves_v1") if self.v2 else None
        data_ctx["train_loader"] = DataLoader(CrossResponseDataset(self.source_dataset),
            batch_sampler=self.sampler, num_workers=0, pin_memory=device.type == "cuda", generator=self.loader_generator)
        if config["response_enabled"] or not self.v2 or config.get("statistics_audit", False):
            self._initialize_statistics(data_ctx, args)
        if config["response_enabled"]:
            # Auxiliary initialization cannot perturb paired baseline dropout/EMA RNG.
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(args.seed) + 913)
                id_dim = model.id_backbone.cls_head.head.weight.shape[1]
                dom_dim = model.dom_backbone.cls_head.head.weight.shape[1]
                self.auxiliary = nn.ModuleDict({
                    "readouts": ResponseReadouts(id_dim, dom_dim, config["readout_dim"]),
                    "predictor": SharedResponsePredictor(config["readout_dim"], config["target_dim"],
                        mode=config["predictor_mode"], rank=config["rank"]),
                }).to(device)
        self.roles = parameter_roles(model, self.auxiliary, config["gradient_tail_prefixes"])
        if self.v2 and config["head_only"]:
            self.auxiliary_transaction = IndependentAuxiliaryTransaction(self.parameters(),
                lr=float(args.lr), weight_decay=float(args.weight_decay),
                amp=bool(args.amp and device.type == "cuda"),
                max_grad_norm=float(getattr(args, "max_grad_norm", 0.)))
        self.normalization_layers = [name for name, m in model.named_modules()
                                    if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        for name, m in model.named_modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and not m.track_running_stats:
                raise ValueError(f"fixed-source normalization unavailable: {name}")
        self.write_report()

    def _initialize_statistics(self, data_ctx, args):
        config, device = self.config, self.device
        # Statistics never use the dataset's RMS-normalized model input.
        self.statistics = WaveformStatistics(config["target_family"], bands=config["target_dim"],
            lags=config["lags"], eps=config["statistics_epsilon"], input_length=int(data_ctx["input_len"])).to(device)
        self.normalizer = FixedSourceNormalizer(config["target_dim"]).to(device)
        fit_indices = random.Random(config["data_seed"]).sample(range(len(self.records)),
            min(len(self.records), config["source_fit_max_records"]))
        values = []
        for start in range(0, len(fit_indices), int(args.eval_batch_size)):
            indices = fit_indices[start:start + int(args.eval_batch_size)]
            values.append(self._record_statistics([self.records[i] for i in indices], normalized=False))
        self.normalizer.fit(torch.cat(values), source_role="source_train")

    def _record_statistics(self, records, *, normalized=True):
        def compute(selected):
            raw = torch.stack([raw_received_iq(self.source_dataset, r.index) for r in selected]).to(self.device)
            return self.statistics(raw, event_metadata=self._event(selected))
        if self.statistic_cache is None:
            values = compute(records)
        else:
            config = {k: self.config.get(k) for k in ("target_family", "target_dim", "lags", "statistics_epsilon", "event_windows")}
            keys = [self.statistic_cache.key(r.physical_sample_id,
                crop=("raw_received_iq", 0, int(self.statistics.input_length)), statistics_config=config) for r in records]
            by_key = dict(zip(keys, records))
            values = self.statistic_cache.get_many(keys, lambda missing: compute([by_key[k] for k in missing]), device=self.device)
        return self.normalizer(values) if normalized else values

    def _event(self, records):
        return event_metadata_from_records(records, self.config.get("event_windows", [])) if self.config["target_family"] == "event" else None

    def parameters(self):
        return list(self.auxiliary.parameters()) if self.auxiliary is not None else []

    def main_optimizer_parameters(self):
        return [] if self.auxiliary_transaction is not None else self.parameters()

    def audit_event(self, stage, **values):
        """Optional source-only replay observer; no copies or RNG work in production."""
        observer = getattr(self, "replay_observer", None)
        if observer is not None:
            observer(self.counts["batches"], stage, values)

    @property
    def joint_open(self):
        qualified = bool(self.mechanism_gate and self.mechanism_gate.last_result
            and self.mechanism_gate.last_result.get("authorized")
            and self.mechanism_gate.last_result.get("scope") == (self.extended_identity_module or "cls_head")) if self.v2 else self.gate.opened
        return qualified and not (self.config["head_only"] or self.config["permanent_detach"])

    def observe_source_mechanism(self, model, evidence, *, extension=None):
        if not self.v2 or self.mechanism_gate is None:
            raise ValueError("source mechanism thresholds are not frozen")
        scope = extension or 'cls_head'
        try:
            proposed_roles = self._response_roles_for_scope(model, scope)
            if not isinstance(evidence, dict):
                raise ValueError('structured source mechanism evidence required')
            # All measurements are generated against this run's established data
            # roles, not merely a caller-supplied source_role string.
            update = evidence.get('update_value', {})
            train_ids, query_ids = set(update.get('training_physical_ids', ())), set(update.get('query_physical_ids', ()))
            if not train_ids <= set(self.source_contract['train']) | set(self.source_contract['unlabeled']) or not query_ids <= set(self.source_contract['validation']):
                raise ValueError('source mechanism evidence does not match current physical data roles')
            result = (self.mechanism_gate.observe(evidence) if extension is None
                      else self.mechanism_gate.request_extension(extension, evidence))
        except (ValueError, TypeError, KeyError):
            self._close_source_mechanism('invalid_source_measurement', scope=scope)
            raise
        self.extended_identity_module = extension if result['authorized'] else None
        self.roles = proposed_roles if result['authorized'] else self._response_roles_for_scope(model, 'cls_head')
        return result

    def _response_roles_for_scope(self, model, scope):
        prefixes = list(self.config['gradient_tail_prefixes'])
        if scope != 'cls_head':
            if (scope not in ('time_fuse', 'freq_fuse') or self.mechanism_gate is None
                    or scope != self.mechanism_gate.config['terminal_identity_module']):
                raise ValueError('only one registered terminal identity feature module is eligible')
            parameters = list(model.id_backbone.get_submodule(scope).parameters())
            domain_ids = {id(p) for p in model.dom_backbone.parameters()}
            if not parameters or any(id(p) in domain_ids for p in parameters):
                raise ValueError('extension must have exclusive identity parameters')
            prefixes.append('id_backbone.' + scope + '.')
        return parameter_roles(model, self.auxiliary, prefixes)

    def _close_source_mechanism(self, reason, *, scope='cls_head'):
        if self.mechanism_gate is not None:
            self.mechanism_gate.invalidate(reason, scope=scope)
            self.extended_identity_module = None
            self.roles = self._response_roles_for_scope(self._model, 'cls_head')

    def prepare_training_phase(self, *, epoch=None, phase=None):
        """Invalidate old qualification before the loader draws its first ticket."""
        self.current_epoch = epoch
        if phase is not None and self.mechanism_audit_state['phase'] != phase:
            if self.mechanism_audit_state['phase'] is not None:
                self._close_source_mechanism('training_phase_changed_requires_fresh_source_measurement')
            self.mechanism_audit_state.update(phase=phase, scope='cls_head', window=[])

    def begin_batch(self, extra, *, epoch=None, phase=None):
        self.logs, self.pending_blocks = {}, []
        self.detailed_audits, self.decision_audit_inputs = [], []
        self.pending_source_update = None
        self.audit_response_term = self.audit_response_group_terms = None
        self.prepare_training_phase(epoch=epoch, phase=phase)
        self.counts["batches"] += 1
        self.audit_due = not self.v2 or (self.counts["batches"] - 1) % self.config.get("diagnostic_interval", 20) == 0
        audit_config = self.config.get('mechanism_audit')
        self.mechanism_audit_due = bool(self.v2 and self.mechanism_gate is not None and audit_config
            and (self.counts['batches'] - 1) % audit_config['interval_steps'] == 0)
        if not self.active:
            return None
        plan = extract_batch_plan(extra)
        self.current_plan = plan
        self.logs.update(effective_blocks=len(plan.blocks), independent_records=sum(len(b.records) for b in plan.blocks),
                         skipped_blocks=float(not plan.blocks), role_rotation_enabled=float(self.config["role_rotation"]))
        self.logs["selection_probability"] = [b.selection_probability for b in plan.blocks]
        self.logs["physical_k"] = [b.candidate.k for b in plan.blocks]
        physical = [r.physical_sample_id for b in plan.blocks for r in b.records]
        self.logs["duplicate_physical_fraction"] = 1 - len(set(physical)) / len(physical) if physical else 0.
        for block in plan.blocks:
            block.validate()
            self.sampler.scheduler.coverage.expose(block)
        self.batch_started = time.perf_counter()
        return plan

    @contextmanager
    def forward_context(self, model, plan, total_count):
        if not self.active or plan is None:
            yield
            return
        clean = len(plan.indices)
        if total_count % clean:
            raise ValueError("unknown clean/LEO view ordering")
        views = total_count // clean
        self.sampler.scheduler.views = views
        allowed = mixstyle_allowed_mask(plan, views=views, device=self.device) if self.config["mixstyle_role_policy"] == "donor_only" else None
        saved = []
        try:
            for module in model.modules():
                if module.__class__.__name__ == "MixStyle1D":
                    saved.append((module, "_cross_response_allowed", getattr(module, "_cross_response_allowed", None)))
                    module._cross_response_allowed = allowed
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    saved.append((module, "training", module.training))
                    module.training = False
            self.counts["forward_calls"] += 1
            yield
        finally:
            for module, key, value in reversed(saved):
                setattr(module, key, value)

    def losses(self, model, out, labels, plan):
        zero = out["tx_logits"].sum() * 0.
        if not self.active or plan is None:
            return zero, zero, zero
        c = self.config
        resp, dec, cross = [], [], []
        response_groups = {key: [] for key in ("predictor_full", "identity", "domain")}
        audit_responses = []
        audit_groups = {key: [] for key in response_groups}
        inference_logits = None
        if c["decision_enabled"]:
            feat = out["aux_id"]["feat_joint"]
            inference_logits = model.id_backbone.cls_head.head(feat.float(), labels=None)
        if self.v2:
            checked = {key: out[key] for key in ('z_id', 'z_dom', 'tx_logits')}
            if inference_logits is not None:
                checked['inference_logits'] = inference_logits
            invalid = [key for key, value in checked.items() if not torch.isfinite(value).all()]
            if invalid:
                self.force_diagnostic('nonfinite_forward', tensors=invalid)
                # Preserve the main finite-loss transaction instead of failing
                # inside a pure geometry diagnostic before it can record the fault.
                invalid_loss = sum(value.sum() for key, value in checked.items() if key in invalid)
                self.response_term = self.decision_term = self.cross_term = invalid_loss
                self.response_group_terms = None
                return invalid_loss, invalid_loss, invalid_loss
        for block in plan.blocks:
            cand, role = block.candidate, block.roles
            p, q, k = len(cand.tx_ids), len(cand.rx_ids), cand.k
            idx = torch.tensor(block.batch_positions, device=self.device)
            zi = out["z_id"].index_select(0, idx).float().reshape(p, q, k, -1)
            zd = out["z_dom"].index_select(0, idx).float().reshape(p, q, k, -1)
            qt = [cand.tx_ids.index(v) for v in role.query_tx]
            dt = [cand.tx_ids.index(v) for v in role.donor_tx]
            qr = [cand.rx_ids.index(v) for v in role.query_rx]
            dr = [cand.rx_ids.index(v) for v in role.donor_rx]
            noise = 0.
            if self.statistics is not None:
                record_stats = self._record_statistics(block.records).reshape(p, q, k, -1)
                target = record_stats.mean(2)
                noise = float(record_stats.var(2, unbiased=False).mean())
            reliability = 1. / (1. + noise)
            r_value, d_value, valid_dec = 0., 0., 0
            if c["response_enabled"]:
                t, d = self.auxiliary["readouts"](zi, zd, qt, qr, dt, dr, detach_identity=not self.joint_open)
                prediction = self.auxiliary["predictor"](t, d)
                response_finite = bool(torch.isfinite(prediction).all() and torch.isfinite(target[qt][:, qr]).all())
                if self.v2 and not response_finite:
                    self.force_diagnostic('nonfinite_response', candidate=repr(cand.key))
                    term = prediction.sum() + target[qt][:, qr].sum()
                    diagnostic = {key: float('nan') for key in ('total_error', 'interaction_error',
                                                              'tx_main_effect_error', 'rx_main_effect_error')}
                else:
                    term, diagnostic = response_loss(prediction, target[qt][:, qr], c["interaction_weight"])
                if c.get("response_routing") == "decomposed" and response_finite:
                    routed, decomposition = decomposed_response_losses(prediction, target[qt][:, qr],
                        **{key: c["response_decomposition"][key] for key in
                           ("alpha", "beta", "interaction_reliability", "weight_bounds")})
                    term = routed["predictor_full"]
                    for key, value in routed.items():
                        response_groups[key].append(value)
                    for key, value in decomposition["component_energies"].items():
                        self._accumulate("response_component_" + key, value)
                if self.mechanism_audit_due and response_finite:
                    if self.joint_open:
                        audit_prediction = prediction
                    else:
                        # A disposable hypothetical identity update needs a live
                        # identity graph, even while the actual main gate is closed.
                        at, ad = self.auxiliary['readouts'](zi, zd, qt, qr, dt, dr, detach_identity=False)
                        audit_prediction = self.auxiliary['predictor'](at, ad)
                        self._accumulate('source_audit_extra_predictor_forwards', 1)
                    audit_term, _ = response_loss(audit_prediction, target[qt][:, qr], c['interaction_weight'])
                    if c.get('response_routing') == 'decomposed':
                        routed_audit, _ = decomposed_response_losses(audit_prediction, target[qt][:, qr],
                            **{key: c['response_decomposition'][key] for key in
                               ('alpha', 'beta', 'interaction_reliability', 'weight_bounds')})
                        audit_term = routed_audit['predictor_full']
                        for key, value in routed_audit.items():
                            audit_groups[key].append(value)
                    audit_responses.append(audit_term)
                resp.append(term)
                r_value = float(diagnostic["total_error"])
                self._accumulate("response_mse", r_value)
                self._accumulate("interaction_error", diagnostic["interaction_error"])
                self._accumulate("tx_main_effect_error", diagnostic["tx_main_effect_error"])
                self._accumulate("rx_main_effect_error", diagnostic["rx_main_effect_error"])
                self._accumulate("response_tx_variance", t.detach().var(0, unbiased=False).mean())
                self._accumulate("response_rx_variance", d.detach().var(0, unbiased=False).mean())
            if c["decision_enabled"]:
                decision_fn = decision_margin_loss_vectorized if self.v2 else decision_margin_loss
                decision_kwargs = dict(delta=c["decision_delta"], min_records=c["decision_min_records"])
                if self.decision_calibration is not None:
                    calibrated = self.decision_calibration
                    delta, weights, calibration_log = calibrated_decision_tensors(calibrated["noise"],
                        calibrated.get("competitors") if c["decision_mode"] == "delta_pairs" else None,
                        labels.index_select(0, idx), [r.day_id for r in block.records],
                        [r.condition_id for r in block.records], device=self.device)
                    decision_kwargs.update(delta=delta, pair_weights=weights)
                    self._accumulate("calibrated_available_comparisons", calibration_log["available_comparisons"])
                term, diagnostic = decision_fn(inference_logits.index_select(0, idx), labels.index_select(0, idx),
                    [r.rx_id for r in block.records], [r.physical_sample_id for r in block.records],
                    [r.condition_id for r in block.records], [r.day_id for r in block.records],
                    **decision_kwargs)
                dec.append(term)
                valid_dec = diagnostic["valid_comparisons"]
                d_value = float(diagnostic["mean_gap"])
                self._accumulate("decision_valid_comparisons", valid_dec)
                self._accumulate("decision_valid_fraction", diagnostic["valid_fraction"])
                self._accumulate("decision_gap", d_value)
                if self.v2:
                    for key, value in diagnostic.items():
                        self._accumulate("decision_" + key, value)
                    if self.audit_due:
                        self.decision_audit_inputs.append((inference_logits.index_select(0, idx),
                            labels.index_select(0, idx), diagnostic, decision_kwargs))
            center = zi.mean(2)
            if not torch.isfinite(zi).all() and not self.audit_due:
                self.audit_due = True
                self.counts["forced_audit_steps"] += 1
            if self.audit_due:
                self._accumulate("identity_interaction", identity_interaction_loss(center).detach())
                self._accumulate("identity_scale", zi.detach().norm(dim=-1).mean())
                self._accumulate("identity_between_tx", center.detach().mean(1).var(0, unbiased=False).mean())
                if self.v2 and torch.isfinite(zi).all() and torch.isfinite(out["tx_logits"]).all():
                    geometry = unit_identity_geometry(zi, min_norm=c.get("geometry_min_norm", 1e-8),
                        labels=labels.index_select(0, idx).reshape(p,q,k),
                        logits=out["tx_logits"].index_select(0, idx).reshape(p,q,k,-1))
                    for key, value in geometry.items():
                        self._accumulate("geometry_" + key, value)
                    self.detailed_audits.append({"kind": "identity_geometry", "candidate": repr(cand.key),
                        "role_plan_id": role.plan_id, "geometry": {key:value for key,value in geometry.items() if key != "normalized_grid"}})
            if c["identity_interaction_enabled"]:
                if c.get("interaction_mode") == "normalized":
                    normalized_loss, geometry = normalized_identity_interaction_loss(zi, min_norm=c.get("geometry_min_norm", 1e-8))
                    if normalized_loss is not None:
                        cross.append(normalized_loss)
                    else:
                        self._accumulate("normalized_interaction_unavailable", 1.)
                else:
                    cross.append(identity_interaction_loss(center))
            self.pending_blocks.append((block, r_value, d_value, reliability, noise, valid_dec))
        self.response_term = torch.stack(resp).mean() if resp else zero
        self.decision_term = torch.stack(dec).mean() if dec else zero
        self.cross_term = torch.stack(cross).mean() if cross else zero
        self.response_group_terms = {key: torch.stack(values).mean() for key,values in response_groups.items()} if response_groups["predictor_full"] else None
        if self.mechanism_audit_due and len(audit_responses) == len(plan.blocks) and audit_responses:
            self.audit_response_term = torch.stack(audit_responses).mean()
            self.audit_response_group_terms = {key: torch.stack(values).mean() for key,values in audit_groups.items()} if audit_groups['predictor_full'] else None
        self.logs.update(loss_response=float(self.response_term.detach()), loss_decision=float(self.decision_term.detach()),
                         loss_cross=float(self.cross_term.detach()))
        self.logs["diagnostics_audited"] = float(self.audit_due)
        return self.response_term, self.decision_term, self.cross_term

    def force_diagnostic(self, reason, **details):
        """Record numerical faults even outside the low-frequency audit schedule."""
        if not self.v2 or not self.active:
            return
        if not self.audit_due:
            self.counts['forced_audit_steps'] += 1
        self.audit_due = True
        self.logs['diagnostics_audited'] = 1.
        self.detailed_audits.append(dict(kind='nonfinite_diagnostic', reason=reason, **details))

    def audit_source_update(self, model, optimizer, scaler, baseline, identity, terms, *, training_context):
        """Called by the real trainer before backward; only copied states step."""
        if not self.mechanism_audit_due:
            return
        started = time.perf_counter()
        rng = capture_rng_state()
        state, cfg = self.mechanism_audit_state, self.config['mechanism_audit']
        state['attempts'] += 1
        step = self.counts['batches'] - 1
        pending = dict(step=step, scope=state['scope'], status='UNAVAILABLE',
                       training_context=_json_evidence(training_context))
        try:
            if self.audit_response_term is None or not torch.isfinite(baseline.detach()):
                raise ValueError('finite complete baseline and complete response audit blocks required')
            source = evaluate_source_response_v2(model, self.auxiliary['readouts'], self.auxiliary['predictor'],
                self.statistics, self.normalizer, self.validation_dataset, self.config, self.device,
                self._domain_label_map, source_role='source_validation', source_train_dataset=self.source_dataset)
            pending['prediction_audit'] = source
            measurements = source_prediction_measurements(source)
            evaluate, query_ids, query_cost = prepare_source_query(self.validation_dataset,
                self.validation_records, per_cell=cfg['query_records_per_cell'], seed=cfg['seed'] + step,
                num_classes=model.id_backbone.cls_head.head.weight.shape[0], device=self.device,
                domain_label_map=self._domain_label_map, args=self._args, batch_size=int(self._args.eval_batch_size))
            train_ids = {r.physical_sample_id for b in self.current_plan.blocks for r in b.records}
            if train_ids & set(query_ids):
                raise ValueError('actual training and source query physical IDs overlap')
            c = self.config
            updates = paired_actual_optimizer_update(model, self.auxiliary, optimizer, scaler,
                baseline_loss=baseline, identity_loss=identity, response_loss=self.audit_response_term,
                decision_loss=terms[1], cross_loss=terms[2],
                roles=self._response_roles_for_scope(model, state['scope']),
                lambda_resp=c['lambda_resp'], lambda_dec=c['lambda_dec'] if c['decision_enabled'] else 0.,
                lambda_cross=c['lambda_cross'] if c['identity_interaction_enabled'] else 0.,
                gradient_cap=c['gradient_cap'], max_grad_norm=float(getattr(self._args, 'max_grad_norm', 0.)),
                evaluate=evaluate, response_group_losses=self.audit_response_group_terms)
            pending.update(updates=updates, query_cost=query_cost)
            if not all(updates[key]['step_applied'] for key in ('base', 'base_response')):
                raise ValueError('source counterfactual has a nonfinite or unapplied optimizer update')
            pending.update(status='MEASURED', measurement=dict(measurements, step=step, scope=state['scope'],
                training_physical_ids=sorted(train_ids, key=repr), query_physical_ids=query_ids,
                risks={key: updates[key]['risk'] for key in ('initial', 'base', 'base_response')}))
        except ValueError as error:
            pending['reason'] = str(error)
        finally:
            restore_rng_state(rng)
            pending['audit_seconds'] = time.perf_counter() - started
            self.pending_source_update = pending

    def _finish_source_update_audit(self, success):
        pending = self.pending_source_update
        if pending is None:
            if not success and self.mechanism_gate is not None:
                self.mechanism_audit_state['window'] = []
                self._close_source_mechanism('main_update_not_applied', scope=self.mechanism_audit_state['scope'])
            return
        state = self.mechanism_audit_state
        pending['main_step_applied'] = bool(success)
        state['last_status'] = pending['status']
        if success and pending['status'] == 'MEASURED':
            state['paired_updates'] += 1
            state['window'].append(pending['measurement'])
            if len(state['window']) == self.config['mechanism_audit']['pairs_per_observation']:
                try:
                    evidence = aggregate_source_observation(state['window'], scope=state['scope'],
                        source_freeze_id=self.mechanism_gate.config['source_freeze_id'])
                    result = self.observe_source_mechanism(self._model, evidence,
                        extension=None if state['scope'] == 'cls_head' else state['scope'])
                    observation = dict(evidence=evidence, result=result, applied_from_step=pending['step'] + 1)
                    state['observations'].append(observation)
                    pending['gate_observation'] = observation
                    # A feature extension never inherits base-head stability.
                    if result['authorized'] and state['scope'] == 'cls_head':
                        state['scope'] = self.config['mechanism_audit']['identity_scope']
                except ValueError as error:
                    pending.update(status='UNAVAILABLE', reason=str(error))
                    state['last_status'] = 'UNAVAILABLE'
                    self._close_source_mechanism(str(error), scope=state['scope'])
                finally:
                    state['window'] = []
        else:
            state['window'] = []
            self._close_source_mechanism(pending.get('reason', 'main_update_not_applied'), scope=state['scope'])
        self.detailed_audits.append(dict(kind='actual_source_update', audit=pending))
        self.logs['source_mechanism_audit_attempts'] = state['attempts']
        self.logs['source_mechanism_paired_updates'] = state['paired_updates']
        self.logs['source_mechanism_authorized_next_step'] = float(self.joint_open)
        self.pending_source_update = None
        self.audit_response_term = self.audit_response_group_terms = None

    def _accumulate(self, key, value):
        number = _scalar(value)
        if number is not None:
            self.logs.setdefault(key, []).append(number)

    def backward(self, model, baseline, identity, terms, scaler):
        response, decision, cross = terms
        c = self.config
        if not self.active:
            scaler.scale(baseline).backward()
            return
        if self.auxiliary_transaction is not None:
            self.logs.update(self.auxiliary_transaction.step(float(c["lambda_resp"]) * response))
            if self.logs.get('auxiliary_nonfinite'):
                self.force_diagnostic('nonfinite_auxiliary_transaction')
            main_loss = baseline
            if c["decision_enabled"]:
                main_loss = main_loss + float(c["lambda_dec"]) * decision
            if c["identity_interaction_enabled"]:
                main_loss = main_loss + float(c["lambda_cross"]) * cross
            scaler.scale(main_loss).backward()
            return
        if self.v2 and self.audit_due and self.decision_audit_inputs:
            named = self.roles["identity_tail"] + self.roles["identity_front"]
            classifier = [(n,p) for n,p in named if n.startswith("id_backbone.cls_head.head.")]
            representation = [(n,p) for n,p in named if not n.startswith("id_backbone.cls_head.head.")]
            compared = {"identity_supervision": identity}
            if c["response_enabled"]:
                compared["weighted_response"] = float(c["lambda_resp"]) * response
            for logits, labels, diagnostic, kwargs in self.decision_audit_inputs:
                audit = audit_decision_parameter_gradients(logits, labels, diagnostic["reference"],
                    diagnostic["valid_mask"], representation, named_classifier_parameters=classifier,
                    compared_losses=compared, delta=kwargs["delta"], pair_weights=kwargs.get("pair_weights"),
                    decision_weight=float(c["lambda_dec"])/len(self.decision_audit_inputs))
                self.detailed_audits.append({"kind": "weighted_decision_parameter_gradients", "audit": audit})
        # Independent gradient evidence distinguishes valid comparisons from an
        # already-satisfied one-sided constraint and an unreachable classifier.
        for name, term, enabled in (("decision", decision, c["decision_enabled"]),
                                    ("cross", cross, c["identity_interaction_enabled"])):
            if enabled and term.requires_grad and self.audit_due:
                params = [p for _, p in self.roles["identity_tail"]]
                diagnostic_scale = float(scaler.get_scale())
                gradients = torch.autograd.grad(term * diagnostic_scale, params, allow_unused=True, retain_graph=True)
                norm = math.sqrt(sum(float((g.detach().float() / diagnostic_scale).square().sum()) for g in gradients if g is not None))
                self.logs[f"{name}_identity_grad_norm"] = norm
        logs = response_backward(baseline_loss=baseline, identity_loss=identity,
            response_loss=response, decision_loss=decision, cross_loss=cross, roles=self.roles, scaler=scaler,
            lambda_resp=c["lambda_resp"] if c["response_enabled"] else 0.,
            lambda_dec=c["lambda_dec"] if c["decision_enabled"] else 0.,
            lambda_cross=c["lambda_cross"] if c["identity_interaction_enabled"] else 0.,
            joint_open=self.joint_open, gradient_cap=c["gradient_cap"], head_only=c["head_only"],
            response_group_losses=self.response_group_terms)
        self.logs.update(logs)

    def finite_gradients(self):
        if self.auxiliary_transaction is not None:
            return None
        for name, p in self.auxiliary.named_parameters() if self.auxiliary is not None else ():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return {"parameter_name": "cross_response." + name}
        return None

    def commit(self, success):
        self.counts["successful_steps"] += int(success)
        if not self.active:
            return
        if not success:
            self.force_diagnostic('main_update_not_applied')
        if self.v2:
            self.counts["audited_steps"] += int(self.audit_due)
            self.counts["audited_successful_steps"] += int(self.audit_due and success)
        for key, value in list(self.logs.items()):
            if isinstance(value, list):
                self.logs[key] = sum(value) / max(1, len(value))
        valid_blocks = 0
        valid_comparisons = 0
        for block, response, decision, reliability, noise, valid_dec in self.pending_blocks:
            block_success = bool(success and all(math.isfinite(float(x)) for x in (response, decision, reliability, noise)))
            collect = self.sampler.scheduler.collect_block_feedback if self.v2 else self.sampler.scheduler.commit
            collect(block, self.counts["batches"] - 1,
                response=response, decision=decision, reliability=reliability, noise=noise,
                valid_count=len(block.records), success=block_success)
            if block_success:
                valid_blocks += 1
                valid_comparisons += valid_dec
                self.counts["effective_blocks"] += 1
                key = str(block.roles.plan_id if self.v2 else block.roles.rotation)
                self.rotation_counts[key] = self.rotation_counts.get(key, 0) + 1
        if self.v2:
            self.sampler.scheduler.finish_step_and_maybe_flush(self.counts["batches"] - 1, success=success)
        self.logs["invalid_feedback_blocks"] = len(self.pending_blocks) - valid_blocks if success else 0
        if success and valid_blocks:
            c = self.config
            self.counts["response_batches"] += int(c["response_enabled"] and bool(self.pending_blocks))
            self.counts["decision_batches"] += int(c["decision_enabled"] and valid_comparisons > 0)
            self.counts["cross_batches"] += int(c["identity_interaction_enabled"] and bool(self.pending_blocks))
            self.counts["joint_gradient_steps"] += int(self.logs.get("response_grad_identity_tail", 0) > 0)
            self.counts["domain_gradient_steps"] += int(self.logs.get("response_grad_domain", 0) > 0)
            self.counts["head_gradient_steps"] += int((self.logs.get("response_grad_auxiliary", 0) or 0) > 0)
            self.counts["decision_gradient_steps"] += int(self.logs.get("decision_identity_grad_norm", 0) > 0)
            self.counts["cross_gradient_steps"] += int(self.logs.get("cross_identity_grad_norm", 0) > 0)
        self.logs["auxiliary_step_seconds"] = time.perf_counter() - self.batch_started
        self._finish_source_update_audit(success)
        if self.v2 and self.detailed_audits:
            with (self.output / "cross_response_diagnostics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_json_evidence({"step": self.counts["batches"] - 1,
                    "success": bool(success), "audited_steps": self.counts["audited_steps"],
                    "records": self.detailed_audits}), ensure_ascii=False, allow_nan=False) + "\n")
        self.logs["peak_memory_bytes"] = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        for key, value in self.logs.items():
            number = _scalar(value)
            if number is not None:
                self.maxima[key] = max(self.maxima.get(key, number), number)
        self.pending_blocks = []

    def evaluate_source(self, model, data_ctx, epoch):
        if self.auxiliary is None or epoch % self.config["gate_interval_epochs"]:
            return None
        result = evaluate_source_response(model, self.auxiliary["readouts"], self.auxiliary["predictor"],
            self.statistics, self.normalizer, self.validation_dataset, self.config, self.device,
            data_ctx["domain_label_map"], source_role="source_validation")
        result["epoch"] = epoch
        if self.v2 and self.config.get("source_audit_enabled"):
            result["v2_source_mechanisms"] = evaluate_source_response_v2(model,
                self.auxiliary["readouts"], self.auxiliary["predictor"], self.statistics, self.normalizer,
                self.validation_dataset, self.config, self.device, data_ctx["domain_label_map"],
                source_role="source_validation", source_train_dataset=self.source_dataset)
        legacy_result = self.gate.observe(response_error=result["response_mse"] if result["response_mse"] is not None else math.inf,
            constant_error=result["constant_mse"] if result["constant_mse"] is not None else 0.,
            blocks=result["valid_blocks"], source_role="source_validation")
        result.update({'legacy_' + key: value for key, value in legacy_result.items()} if self.v2 else legacy_result)
        if self.v2:
            result['response_joint_open'] = self.joint_open
            result['source_mechanism_gate'] = deepcopy(self.mechanism_gate.last_result) if self.mechanism_gate else None
        self.source_evaluations.append(result)
        self.write_report()
        return result

    def state_dict(self):
        return {"schema": "core90_cross_response_v2" if self.v2 else "core90_cross_response_v1", "initialization": "scratch",
                "config": deepcopy(self.config), "variant": self.variant,
                "source_contract": self.source_contract, "counts": deepcopy(self.counts),
                "maxima": deepcopy(self.maxima), "rotation_counts": deepcopy(self.rotation_counts),
                "source_evaluations": deepcopy(self.source_evaluations), "gate": self.gate.state_dict(),
                "mechanism_gate": self.mechanism_gate.state_dict() if self.mechanism_gate else None,
                "source_mechanism_audit": deepcopy(self.mechanism_audit_state),
                "extended_identity_module": self.extended_identity_module,
                "auxiliary": self.auxiliary.state_dict() if self.auxiliary is not None else None,
                "auxiliary_transaction": self.auxiliary_transaction.state_dict() if self.auxiliary_transaction else None,
                "statistics": self.statistics.state_dict() if self.statistics is not None else None,
                "normalizer": self.normalizer.state_dict() if self.normalizer is not None else None,
                "sampler": self.sampler.state_dict() if self.sampler is not None else None,
                "loader_generator": self.loader_generator.get_state()}

    def load_state_dict(self, state):
        for key, value in (("schema", "core90_cross_response_v2" if self.v2 else "core90_cross_response_v1"), ("initialization", "scratch"),
                           ("config", self.config), ("variant", self.variant), ("source_contract", self.source_contract)):
            if state.get(key) != value:
                raise ValueError(f"CHECKPOINT_DATA_CONTRACT_MISMATCH: cross_response {key}")
        self.gate.load_state_dict(state["gate"])
        if self.mechanism_gate is not None:
            self.mechanism_gate.load_state_dict(state["mechanism_gate"])
            if 'source_mechanism_audit' not in state:
                raise ValueError('missing source mechanism audit window on resume')
        if 'source_mechanism_audit' in state:
            self.mechanism_audit_state = deepcopy(state['source_mechanism_audit'])
        self.extended_identity_module = state.get("extended_identity_module")
        if self.extended_identity_module is not None:
            if self.mechanism_gate is None or self.extended_identity_module != self.mechanism_gate.config["terminal_identity_module"]:
                raise ValueError("invalid restored identity extension")
            self.roles = parameter_roles(self._model, self.auxiliary,
                list(self.config["gradient_tail_prefixes"]) + ["id_backbone." + self.extended_identity_module + "."])
        if self.active:
            if self.statistics is not None:
                self.statistics.load_state_dict(state["statistics"], strict=True)
                self.normalizer.load_state_dict(state["normalizer"], strict=True)
            # Actual view count is schedule state, not a fixed constructor option.
            self.sampler.scheduler.views = int(state["sampler"]["scheduler"]["config"][-1])
            self.sampler.load_state_dict(state["sampler"])
        if self.auxiliary is not None:
            self.auxiliary.load_state_dict(state["auxiliary"], strict=True)
        if self.auxiliary_transaction is not None:
            self.auxiliary_transaction.load_state_dict(state["auxiliary_transaction"])
        for key in ("counts", "maxima", "rotation_counts", "source_evaluations"):
            setattr(self, key, deepcopy(state[key]))
        self.loader_generator.set_state(state["loader_generator"].cpu())

    def validate_checkpoint(self, payload, args):
        if payload.get("cross_response", {}).get("initialization") != "scratch":
            raise ValueError("CHECKPOINT_PROVENANCE_UNVERIFIED: not a cross-response scratch lineage")
        if payload.get("checkpoint_selection") != "final_only" or payload.get("baseline_ckpt"):
            raise ValueError("CHECKPOINT_TARGET_CONTAMINATED: incompatible checkpoint selection/ancestor")
        if payload.get("checkpoint_role") != "cross_response_epoch_resume":
            raise ValueError("resume requires the epoch training state, not a selected/evaluated model")
        from scripts.core90_cross_response_matrix import load_config
        old_args = payload.get("args", {})
        variable = {"run_id", "candidate_id", "output_dir", "num_workers", "prefetch_factor",
                    "metrics_csv", "metrics_jsonl", "device", "safe_best_path", "safe_latest_path"}
        for key in load_config(args.cross_response_config)["baseline_args"]:
            if key not in variable and old_args.get(key) != getattr(args, key, None):
                raise ValueError(f"CHECKPOINT_DATA_CONTRACT_MISMATCH: training argument {key}")
        for key in ("config", "variant", "source_contract"):
            expected = self.config if key == "config" else getattr(self, key)
            if payload["cross_response"].get(key) != expected:
                raise ValueError(f"CHECKPOINT_DATA_CONTRACT_MISMATCH: {key}")

    def activation_report(self):
        c, n = self.config, self.counts
        missing = []
        if self.active and not n["effective_blocks"]:
            missing.append("NO_EFFECTIVE_CROSS_BLOCKS")
        if self.active and c["role_rotation"] and len(self.rotation_counts) < 2:
            missing.append("ROLE_ROTATION_NOT_OBSERVED")
        if self.active and c["scheduler_mode"] == "feedback" and not self.sampler.scheduler.history:
            missing.append("FEEDBACK_HISTORY_NOT_UPDATED")
        if c["response_enabled"]:
            if not n["head_gradient_steps"]:
                missing.append("RESPONSE_HEAD_GRADIENT_NOT_OBSERVED")
            if not c["head_only"] and not n["domain_gradient_steps"]:
                missing.append("DOMAIN_RESPONSE_GRADIENT_NOT_OBSERVED")
            if not (c["head_only"] or c["permanent_detach"]) and not n["joint_gradient_steps"]:
                missing.append("JOINT_IDENTITY_GRADIENT_NOT_OBSERVED")
        if c["decision_enabled"] and not n["decision_gradient_steps"]:
            missing.append("DECISION_GRADIENT_NOT_OBSERVED")
        if c["identity_interaction_enabled"] and not n["cross_gradient_steps"]:
            missing.append("IDENTITY_INTERACTION_GRADIENT_NOT_OBSERVED")
        return {"variant": self.variant, "status": "ACTIVE_VERIFIED" if not missing else "ACTIVATION_INCOMPLETE",
                "missing": missing, "counts": deepcopy(n), "maxima": deepcopy(self.maxima),
                "gate": self.gate.state_dict(), "rotation_counts": deepcopy(self.rotation_counts),
                "mechanism_gate": self.mechanism_gate.state_dict() if self.mechanism_gate else {"status": "SOURCE_PARAMETERS_UNFROZEN"},
                "source_mechanism_audit": deepcopy(self.mechanism_audit_state),
                "joint_objective": deepcopy(self.joint_objective),
                "extended_identity_module": self.extended_identity_module,
                "config": deepcopy(c), "source_evaluations": deepcopy(self.source_evaluations),
                "auxiliary_parameters": sum(p.numel() for p in self.parameters()),
                "fixed_statistics_cache": self.statistic_cache.report() if self.statistic_cache else None,
                "gradient_counter_semantics": {"decision_gradient_steps": "positive among audited_successful_steps",
                    "cross_gradient_steps": "positive among audited_successful_steps",
                    "response_gradient_steps": "every successful routed update; required routing never sampled"},
                "parameter_groups": {key:[name for name,_ in values] for key,values in self.roles.items()} if self.roles is not None else {},
                "optional_controls": {"target_family": c["target_family"],
                    "bilinear_rank_active": c["rank"] if c["response_enabled"] and c["predictor_mode"] == "bilinear" else None,
                    "interaction_reweighting_active": bool(c["response_enabled"] and c["interaction_weight"] > 0),
                    "adaptive_k_configured": bool(self.active and len(set(c["k_menu"])) > 1),
                    "response_identity_permanently_detached": bool(c["head_only"] or c["permanent_detach"])},
                "candidate_count": len(self.sampler.candidates) if self.sampler is not None else 0,
                "candidate_search_complete": self.sampler.candidate_search_complete if self.sampler is not None else None,
                "sampling_evidence": {
                    "feedback_history_entries": len(self.sampler.scheduler.history),
                    "pending_feedback_entries": len(self.sampler.scheduler.pending),
                    "joint_rectangles": len(self.sampler.scheduler.coverage.joint),
                    "directed_transfers": len(self.sampler.scheduler.coverage.directed),
                    "observed_rectangles": len(self.sampler.scheduler.coverage.observed_rectangles),
                    "response_query_rectangles": len(self.sampler.scheduler.coverage.response_query_rectangles),
                    "directed_response_transfers": len(self.sampler.scheduler.coverage.directed_response_transfers),
                    "actual_candidate_pool_reachable": {key:len(values) for key,values in self.reachable_coverage.items()} if self.reachable_coverage is not None else None,
                    "direction_history_entries": len(self.sampler.scheduler.direction_history),
                    "flush_count": self.sampler.scheduler.flush_count,
                    "last_step_event": self.sampler.scheduler.last_step_event,
                    "last_probability_audit": self.sampler.scheduler.last_probability_audit,
                    "unique_exposed_physical": len(self.sampler.scheduler.coverage.exposed_physical),
                    "skip_counts": dict(self.sampler.skip_counts),
                    "distribution_adaptation_testable": len(self.sampler.candidates) > 1,
                    "cost_views": self.sampler.scheduler.views,
                } if self.sampler is not None else None,
                "loader_workers": 0 if self.active else None,
                "normalization_layers_fixed_in_labeled_forward": getattr(self, "normalization_layers", []),
                "performance_claim": "NOT_EVALUATED"}

    def write_report(self):
        path = self.output / "cross_response_activation.json"
        path.write_text(json.dumps(_json_evidence(self.activation_report()), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def forward_labeled(model, runtime, plan, x, **kwargs):
    if runtime is None:
        return model(x, **kwargs)
    with runtime.forward_context(model, plan, x.shape[0]):
        if plan is not None:
            runtime.audit_event("input", physical_ids=tuple(r.physical_sample_id for b in plan.blocks for r in b.records),
                                roles=tuple(b.roles for b in plan.blocks), augmented_input=x)
        output = model(x, **kwargs)
        runtime.audit_event("logits", tx_logits=output["tx_logits"])
        return output
