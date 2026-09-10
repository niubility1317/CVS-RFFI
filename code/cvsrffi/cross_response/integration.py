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
from .tensor_ops import response_loss, identity_interaction_loss, double_center
from .decision import decision_margin_loss
from .training import SourceResponseGate, parameter_roles, response_backward
from .source_eval import raw_received_iq, event_metadata_from_records, evaluate_source_response


def _scalar(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = float(value.detach())
    if isinstance(value, (int, float, bool)) and math.isfinite(value):
        return float(value)
    return None


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
        self.output = Path(args.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.auxiliary = None
        self.sampler = None
        self.roles = None
        self.gate = SourceResponseGate(config["gate_min_blocks"], config["gate_error_ratio"], config["gate_stable_checks"])
        self.counts = {"batches": 0, "successful_steps": 0, "effective_blocks": 0,
                       "response_batches": 0, "decision_batches": 0, "cross_batches": 0,
                       "joint_gradient_steps": 0, "domain_gradient_steps": 0, "head_gradient_steps": 0,
                       "decision_gradient_steps": 0, "cross_gradient_steps": 0, "forward_calls": 0}
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
            error_clip=config["feedback_clip"], update_interval=config["update_interval"], views=1)
        old_loader = data_ctx["train_loader"]
        self.sampler = CrossBlockSampler(self.records, P=config["P"], Q=config["Q"],
            k_menu=config["k_menu"], batch_size=int(args.batch_size), seed=int(config["data_seed"]),
            max_candidates=config["scheduler_candidate_limit"], scheduler=scheduler,
            steps_per_epoch=len(old_loader), role_rotation=config["role_rotation"])
        data_ctx["train_loader"] = DataLoader(CrossResponseDataset(self.source_dataset),
            batch_sampler=self.sampler, num_workers=0, pin_memory=device.type == "cuda", generator=self.loader_generator)
        # Statistics never use the dataset's RMS-normalized model input.
        self.statistics = WaveformStatistics(config["target_family"], bands=config["target_dim"],
            lags=config["lags"], eps=config["statistics_epsilon"], input_length=int(data_ctx["input_len"])).to(device)
        self.normalizer = FixedSourceNormalizer(config["target_dim"]).to(device)
        fit_indices = random.Random(config["data_seed"]).sample(range(len(self.records)),
            min(len(self.records), config["source_fit_max_records"]))
        values = []
        for start in range(0, len(fit_indices), int(args.eval_batch_size)):
            indices = fit_indices[start:start + int(args.eval_batch_size)]
            raw = torch.stack([raw_received_iq(self.source_dataset, i) for i in indices]).to(device)
            event = self._event([self.records[i] for i in indices])
            values.append(self.statistics(raw, event_metadata=event))
        self.normalizer.fit(torch.cat(values), source_role="source_train")
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
        self.normalization_layers = [name for name, m in model.named_modules()
                                    if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        for name, m in model.named_modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm) and not m.track_running_stats:
                raise ValueError(f"fixed-source normalization unavailable: {name}")
        self.write_report()

    def _event(self, records):
        return event_metadata_from_records(records, self.config.get("event_windows", [])) if self.config["target_family"] == "event" else None

    def parameters(self):
        return list(self.auxiliary.parameters()) if self.auxiliary is not None else []

    @property
    def joint_open(self):
        return self.gate.opened and not (self.config["head_only"] or self.config["permanent_detach"])

    def begin_batch(self, extra):
        self.logs, self.pending_blocks = {}, []
        self.counts["batches"] += 1
        if not self.active:
            return None
        plan = extract_batch_plan(extra)
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
        allowed = mixstyle_allowed_mask(plan, views=views, device=self.device)
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
        inference_logits = None
        if c["decision_enabled"]:
            feat = out["aux_id"]["feat_joint"]
            inference_logits = model.id_backbone.cls_head.head(feat.float(), labels=None)
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
            raw = torch.stack([raw_received_iq(self.source_dataset, r.index) for r in block.records]).to(self.device)
            record_stats = self.normalizer(self.statistics(raw, event_metadata=self._event(block.records))).reshape(p, q, k, -1)
            target = record_stats.mean(2)
            noise = float(record_stats.var(2, unbiased=False).mean())
            reliability = 1. / (1. + noise)
            r_value, d_value, valid_dec = 0., 0., 0
            if c["response_enabled"]:
                t, d = self.auxiliary["readouts"](zi, zd, qt, qr, dt, dr, detach_identity=not self.joint_open)
                prediction = self.auxiliary["predictor"](t, d)
                term, diagnostic = response_loss(prediction, target[qt][:, qr], c["interaction_weight"])
                resp.append(term)
                r_value = float(diagnostic["total_error"])
                self._accumulate("response_mse", r_value)
                self._accumulate("interaction_error", diagnostic["interaction_error"])
                self._accumulate("tx_main_effect_error", diagnostic["tx_main_effect_error"])
                self._accumulate("rx_main_effect_error", diagnostic["rx_main_effect_error"])
                self._accumulate("response_tx_variance", t.detach().var(0, unbiased=False).mean())
                self._accumulate("response_rx_variance", d.detach().var(0, unbiased=False).mean())
            if c["decision_enabled"]:
                term, diagnostic = decision_margin_loss(inference_logits.index_select(0, idx), labels.index_select(0, idx),
                    [r.rx_id for r in block.records], [r.physical_sample_id for r in block.records],
                    [r.condition_id for r in block.records], [r.day_id for r in block.records],
                    delta=c["decision_delta"], min_records=c["decision_min_records"])
                dec.append(term)
                valid_dec = diagnostic["valid_comparisons"]
                d_value = float(diagnostic["mean_gap"])
                self._accumulate("decision_valid_comparisons", valid_dec)
                self._accumulate("decision_valid_fraction", diagnostic["valid_fraction"])
                self._accumulate("decision_gap", d_value)
            center = zi.mean(2)
            self._accumulate("identity_interaction", identity_interaction_loss(center).detach())
            self._accumulate("identity_scale", zi.detach().norm(dim=-1).mean())
            self._accumulate("identity_between_tx", center.detach().mean(1).var(0, unbiased=False).mean())
            if c["identity_interaction_enabled"]:
                cross.append(identity_interaction_loss(center))
            self.pending_blocks.append((block, r_value, d_value, reliability, noise, valid_dec))
        self.response_term = torch.stack(resp).mean() if resp else zero
        self.decision_term = torch.stack(dec).mean() if dec else zero
        self.cross_term = torch.stack(cross).mean() if cross else zero
        self.logs.update(loss_response=float(self.response_term.detach()), loss_decision=float(self.decision_term.detach()),
                         loss_cross=float(self.cross_term.detach()))
        return self.response_term, self.decision_term, self.cross_term

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
        # Independent gradient evidence distinguishes valid comparisons from an
        # already-satisfied one-sided constraint and an unreachable classifier.
        for name, term, enabled in (("decision", decision, c["decision_enabled"]),
                                    ("cross", cross, c["identity_interaction_enabled"])):
            if enabled and term.requires_grad:
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
            joint_open=self.joint_open, gradient_cap=c["gradient_cap"], head_only=c["head_only"])
        self.logs.update(logs)

    def finite_gradients(self):
        for name, p in self.auxiliary.named_parameters() if self.auxiliary is not None else ():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return {"parameter_name": "cross_response." + name}
        return None

    def commit(self, success):
        self.counts["successful_steps"] += int(success)
        if not self.active:
            return
        for key, value in list(self.logs.items()):
            if isinstance(value, list):
                self.logs[key] = sum(value) / max(1, len(value))
        valid_blocks = 0
        valid_comparisons = 0
        for block, response, decision, reliability, noise, valid_dec in self.pending_blocks:
            block_success = bool(success and all(math.isfinite(float(x)) for x in (response, decision, reliability, noise)))
            self.sampler.scheduler.commit(block, self.counts["batches"] - 1,
                response=response, decision=decision, reliability=reliability, noise=noise,
                valid_count=len(block.records), success=block_success)
            if block_success:
                valid_blocks += 1
                valid_comparisons += valid_dec
                self.counts["effective_blocks"] += 1
                key = str(block.roles.rotation)
                self.rotation_counts[key] = self.rotation_counts.get(key, 0) + 1
        self.logs["invalid_feedback_blocks"] = len(self.pending_blocks) - valid_blocks if success else 0
        if success and valid_blocks:
            c = self.config
            self.counts["response_batches"] += int(c["response_enabled"] and bool(self.pending_blocks))
            self.counts["decision_batches"] += int(c["decision_enabled"] and valid_comparisons > 0)
            self.counts["cross_batches"] += int(c["identity_interaction_enabled"] and bool(self.pending_blocks))
            self.counts["joint_gradient_steps"] += int(self.logs.get("response_grad_identity_tail", 0) > 0)
            self.counts["domain_gradient_steps"] += int(self.logs.get("response_grad_domain", 0) > 0)
            self.counts["head_gradient_steps"] += int(self.logs.get("response_grad_auxiliary", 0) > 0)
            self.counts["decision_gradient_steps"] += int(self.logs.get("decision_identity_grad_norm", 0) > 0)
            self.counts["cross_gradient_steps"] += int(self.logs.get("cross_identity_grad_norm", 0) > 0)
        self.logs["auxiliary_step_seconds"] = time.perf_counter() - self.batch_started
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
        result.update(self.gate.observe(response_error=result["response_mse"] if result["response_mse"] is not None else math.inf,
            constant_error=result["constant_mse"] if result["constant_mse"] is not None else 0.,
            blocks=result["valid_blocks"], source_role="source_validation"))
        self.source_evaluations.append(result)
        self.write_report()
        return result

    def state_dict(self):
        return {"schema": "core90_cross_response_v1", "initialization": "scratch",
                "config": deepcopy(self.config), "variant": self.variant,
                "source_contract": self.source_contract, "counts": deepcopy(self.counts),
                "maxima": deepcopy(self.maxima), "rotation_counts": deepcopy(self.rotation_counts),
                "source_evaluations": deepcopy(self.source_evaluations), "gate": self.gate.state_dict(),
                "auxiliary": self.auxiliary.state_dict() if self.auxiliary is not None else None,
                "statistics": self.statistics.state_dict() if self.active else None,
                "normalizer": self.normalizer.state_dict() if self.active else None,
                "sampler": self.sampler.state_dict() if self.sampler is not None else None,
                "loader_generator": self.loader_generator.get_state()}

    def load_state_dict(self, state):
        for key, value in (("schema", "core90_cross_response_v1"), ("initialization", "scratch"),
                           ("config", self.config), ("variant", self.variant), ("source_contract", self.source_contract)):
            if state.get(key) != value:
                raise ValueError(f"CHECKPOINT_DATA_CONTRACT_MISMATCH: cross_response {key}")
        self.gate.load_state_dict(state["gate"])
        if self.active:
            self.statistics.load_state_dict(state["statistics"], strict=True)
            self.normalizer.load_state_dict(state["normalizer"], strict=True)
            # Actual view count is schedule state, not a fixed constructor option.
            self.sampler.scheduler.views = int(state["sampler"]["scheduler"]["config"][-1])
            self.sampler.load_state_dict(state["sampler"])
        if self.auxiliary is not None:
            self.auxiliary.load_state_dict(state["auxiliary"], strict=True)
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
                "config": deepcopy(c), "source_evaluations": deepcopy(self.source_evaluations),
                "auxiliary_parameters": sum(p.numel() for p in self.parameters()),
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
        path.write_text(json.dumps(self.activation_report(), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def forward_labeled(model, runtime, plan, x, **kwargs):
    if runtime is None:
        return model(x, **kwargs)
    with runtime.forward_context(model, plan, x.shape[0]):
        return model(x, **kwargs)
