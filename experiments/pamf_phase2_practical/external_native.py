"""Native external families for the paired Phase2 three-pool experiment.

Only source bundle loading and explicitly supplied support enter this module.
No query reader, target data root, truth file, or validation selection is used.
Each B starts at the frozen source model; every C forks the *same* B state.
The implementation is the historical matched WiSig port, namespaced locally
to keep the unrelated Phase1 activation mechanism from changing its imports.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .vendor_native.comparison_suite import adaptation as da
from .vendor_native.comparison_suite import models as nets
from .vendor_native.comparison_suite import registration as cil


SOURCE_ROOT = Path("/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/runs/"
                   "20260930-phase1-native-baselines-practical-manysig-m5-r01")
NATIVE_MODEL_SEEDS = (2026092701, 2026092702, 2026092703, 2026092704)
_REGISTRATION = {"protonet", "csil", "mopc_hr", "orthogonal"}
_REPLAY = {"radionet_ada", "mrior", "dadda", "twostage"}
_ALIASES = {"legacy_feature_separation": "feature_separation_legacy",
            "feature_separation_ft": "feature_separation_legacy"}


def profiles():
    """Return concrete supported versions, budgets, and honest unavailable rows."""
    definitions = (
        ("radionet_ada_ft_knn", "radionet_ada", True, False, False,
         "ADA10000_iterations+support_FT30_epochs;cosine_distance_weighted_kNN_k=K"),
        ("radionet_ada_knn", "radionet_ada", True, False, True,
         "ADA10000_iterations;cosine_distance_weighted_kNN_k=K"),
        ("protonet_cda", "protonet", True, False, False, "B0;C0;support_squared_Euclidean_prototypes"),
        ("csil_corefix", "csil", True, False, False,
         "B10epochs_Adam0.001;C3epochs_masked_SGDM0.01;source_empirical_Fisher"),
        ("mopc_paper", "mopc_hr", True, False, False,
         "B10epochs_Adam0.001;C20epochs_per_5class_session_SGD0.01;paper_equations"),
        ("mopc_public_code", "mopc_hr", True, False, True,
         "B10epochs_Adam0.001;C20epochs_per_5class_session_SGD0.01;public_code_semantics"),
        ("orthogonal", "orthogonal", True, False, False, "B10epochs_SGD0.08;C50epochs_SGD0.08"),
        ("mrior_sda", "mrior", False, True, False, "B100_outer_updates;7_estimator_updates_per_outer;lr0.0006"),
        ("dadda_sda", "dadda", False, True, False, "B100_updates;support_true_label_conditional_alignment"),
        ("twostage_sda", "twostage", False, True, False, "B100_DANN+100_LMMD+100_support_FT_updates"),
        ("feature_separation_legacy", "feature_separation", False, True, True, "B100epochs_one_update_per_epoch_Adam0.001;legacy_TX_RX_port"),
    )
    result = {}
    for name, method, native_c, expand, auxiliary, budget in definitions:
        result[name] = dict(id=name, method=method, available=True,
            source_available=True, native_C=native_c, expand_head=expand,
            auxiliary=auxiliary, source_replay="L_s_only" if method in _REPLAY else False,
            source_budget="fixed_final_E200", B_budget=budget,
            source_path_pattern=str(SOURCE_ROOT / (method + "-s{seed}")),
            model_seeds=list(NATIVE_MODEL_SEEDS),
            claim="matched_WiSig_IQ_support_only_extension",
            registration="native_mechanism" if native_c else "closed_set_C_NA;explicit_output_expansion_extension_available")
    result["feature_separation_legacy"].update(
        claim="auxiliary_legacy_feature_separation;NOT_Hu2024", aliases=list(_ALIASES))
    result["mopc_public_code"].update(
        semantics="rawdot_softmax_correction;unsquared_layer_L2_HR;prototype_CE_temperature2;KD_not_optimized",
        claim="public_code_loss_and_correction_semantics_matched_IQ_task_extension;not_original_dataset_reproduction")
    result["csil_author"] = dict(id="csil_author", method="csil", available=False,
        source_available=False, native_C=False, auxiliary=True,
        reason="Original MATLAB workspace/author source Fisher is not a runnable Python version of the current capsule. "
               "The historical empirical-Fisher corefix source cannot be relabeled as original exp(gradient_squared) Fisher; "
               "that original numerical path overflowed even float64.",
        corefix_alternative="csil_corefix")
    return result


def _profile(value):
    name = value["id"] if isinstance(value, dict) else str(value)
    name = _ALIASES.get(name, name)
    if name not in profiles():
        raise ValueError("Unknown native external profile: " + name)
    profile = profiles()[name]
    if not profile["available"]:
        raise NotImplementedError(name + ": " + profile["reason"])
    return profile


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_path(rt, profile, seed):
    sources = rt.get("native_sources", {})
    value = sources.get(profile["id"], sources.get(profile["method"], {}))
    if isinstance(value, dict):
        value = value.get(str(seed), value.get(seed))
    if value:
        return Path(value["source_output"] if isinstance(value, dict) else value)
    return Path(rt.get("native_source_root", SOURCE_ROOT)) / f'{profile["method"]}-s{seed}'


def _source_provenance(rt, path, method, seed):
    contract = _read(path / "source_contract.json")
    actual = rt["source_contract"]
    actual = _read(actual) if isinstance(actual, (str, Path)) else actual
    complete, initial = _read(path / "completion.json"), _read(path / "initialization.json")
    if complete.get("status") != "SOURCE_TRAINED" or complete.get("epoch") != 200:
        raise ValueError("Native checkpoint is not fixed final source E200")
    for name in ("role_ids", "source_rxs", "source_days", "ratios", "split_seed", "classes", "num_classes"):
        if contract.get(name) != actual.get(name):
            raise ValueError("CHECKPOINT_DATA_CONTRACT_MISMATCH: " + name)
    if contract.get("equalized") != 1 or contract.get("out_len") != 256 or contract.get("normalize") is not True:
        raise ValueError("CHECKPOINT_DATA_CONTRACT_MISMATCH: native equalized1/center256/unitRMS source")
    if (initial.get("status") != "SCRATCH" or initial.get("scratch_only") is not True
            or initial.get("checkpoint") is not None or initial.get("checkpoint_sources")
            or initial.get("ancestors") or initial.get("target_contact")
            or initial.get("target_access") is not False or initial.get("model_seed") != seed):
        raise ValueError("CHECKPOINT_PROVENANCE_UNVERIFIED or CHECKPOINT_TARGET_CONTAMINATED")
    return contract, initial


def _logits(model, x):
    model.eval()  # query batches never update BN/dropout or any fitted state
    out = model(x)
    if torch.is_tensor(out):
        return out
    return out["tx_logits"] if "tx_logits" in out else out["logits"]


@dataclass(frozen=True)
class _Features:
    method: str

    def __call__(self, model, x):
        model.eval()
        if hasattr(model, "embedding"):
            return model.embedding(x)
        return nets.embedding(model, self.method, x)


@dataclass
class _Scores:
    """Decision data is copied with state; the actual model is always an argument."""
    method: str
    decision: str = "native"
    weights: Any = None
    support_features: Any = None
    support_labels: Any = None
    count: int = 0
    knn_k: int = 1

    def __call__(self, model, x):
        model.eval()
        if self.decision == "native":
            return _logits(model, x)[:, :self.count]
        z = _Features(self.method)(model, x)
        if self.decision == "protonet":
            return -torch.cdist(z, self.weights.to(z)).square()
        if self.decision == "orthogonal":
            return F.normalize(z, dim=1) @ F.normalize(model.classifier_weight, dim=1).t()
        if self.decision == "knn":
            distances = (1 - F.normalize(z, dim=1) @ F.normalize(self.support_features.to(z), dim=1).t()).clamp_min(0)
            d, indices = distances.topk(self.knn_k, dim=1, largest=False, sorted=True)
            zero = d == 0
            weights = torch.where(zero.any(1, keepdim=True), zero.to(d.dtype), d.clamp_min(1e-12).reciprocal())
            votes = z.new_zeros((len(z), self.count))
            votes.scatter_add_(1, self.support_labels.to(z.device)[indices], weights)
            return votes
        raise ValueError("Unknown native decision " + self.decision)


def load_source(rt, profile, seed, device):
    """Load a verified source-only checkpoint; target adaptation never occurs here."""
    profile = _profile(profile)
    seed = int(seed)
    if seed not in NATIVE_MODEL_SEEDS:
        raise ValueError("Native source seed must be one of the four audited seeds")
    source, method = _source_path(rt, profile, seed), profile["method"]
    contract, initial = _source_provenance(rt, source, method, seed)
    payload = torch.load(source / "last.pt", map_location=device, weights_only=False)
    if (payload.get("epoch") != 200 or payload.get("method") != method
            or payload.get("classes") != contract["classes"]
            or payload.get("source_contract") != contract or payload.get("initialization") != initial
            or payload.get("selection") != "fixed_last_epoch"
            or payload.get("num_classes") != contract["num_classes"]):
        raise ValueError("Native checkpoint content/provenance mismatch")
    model = nets.factory(method, payload["num_classes"], payload["num_receivers"],
                         **payload.get("model_kwargs", {})).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    summaries = None
    if method in _REGISTRATION:
        summaries = torch.load(source / "base_state.pt", map_location=device, weights_only=False)
    provenance = dict(status="VERIFIED", source=str(source), checkpoint=str(source / "last.pt"),
        model_seed=seed, method=method, source_role_ids="EXACT_MATCH", source_contract=rt["source_contract"],
        selection="fixed_last_epoch_200", scratch_ancestors=[], target_contact_before_B=False,
        audit_reference=rt.get("native_source_audit", "native_source_reuse_audit.json:VERIFIED36"),
        implementation="namespaced_historical_matched_native_port", data_revalidated=False)
    classes = tuple(range(contract["num_classes"]))
    return dict(profile=profile["id"], method=method, model=model,
        features_fn=_Features(method), score_fn=_Scores(method, count=len(classes)), classes=classes,
        resources={}, base_metadata=summaries, source_contract=contract,
        source_path=str(source), source_checkpoint=str(source / "last.pt"), provenance=provenance,
        source_only=True, native_state=None)


def _support(x, y, expected):
    if (not torch.is_tensor(x) or not torch.is_tensor(y) or y.dtype != torch.long
            or y.ndim != 1 or len(x) != len(y) or not len(x)
            or set(y.detach().cpu().tolist()) != set(expected) or not bool(torch.isfinite(x).all())):
        raise ValueError("Support must be finite tensors with compact long labels covering every current class")


def _source_L(rt, bundle):
    # This existing reader checks physical role IDs against the frozen contract.
    # It opens L_s only; U_s metadata and hidden TX labels are never requested.
    from experiments.cric_dura_comparison.f_data import SourceDataset
    dataset = SourceDataset(rt["source_cache"], "L_s", bundle["source_contract"])
    return dataset.x, torch.tensor(dataset.metadata["y"].tolist(), dtype=torch.long)


def _registration_config(rt, profile, seed):
    cfg = dict(seed=int(seed), old_adaptation_epochs=10, old_adaptation_lr=.001,
               old_adaptation_batch_size=128, csil_epochs=3, csil_old_fingerprint_trainable=True,
               mopc_epochs=20, mopc_stage_size=5, orthogonal_epochs=50,
               mopc_semantics="public_code" if profile == "mopc_public_code" else "paper")
    # Explicit recorded overrides serve focused source-only implementation tests;
    # the experiment runtime uses the complete_master_r05 defaults above.
    cfg.update(rt.get("native_registration_overrides", {}).get(profile, {}))
    return cfg


def _adaptation_config(rt, profile, seed, device, k, logger):
    cfg = dict(seed=int(seed), device=str(device), k=int(k), batch_size=128,
               epochs=100, steps_per_epoch=1, lr=.001, logger=logger)
    if profile.startswith("radionet_ada"):
        cfg.update(iterations=10000, lr=.00001, discriminator_lr=.00001,
                   finetune_epochs=30, finetune_lr=.001, class_weight=4., domain_weight=4.)
    elif profile == "mrior_sda":
        cfg.update(lr=.0006, estimate_steps=7)
    elif profile == "twostage_sda":
        cfg.update(dann_epochs=100, lmmd_epochs=100, finetune_epochs=100)
    elif profile == "feature_separation_legacy":
        # The immutable 20260930 feature_separation target config has E100,
        # steps_per_epoch=1. Keep that actual legacy budget separate from Hu2024.
        cfg.update(epochs=100)
    cfg.update(rt.get("native_adaptation_overrides", {}).get(profile, {}))
    return cfg


def _pack(native, profile, bundle):
    method = bundle["method"]
    if isinstance(native, cil.RegistrationState):
        decision = "protonet" if method == "protonet" else "orthogonal" if method == "orthogonal" else "native"
        scores = _Scores(method, decision=decision, weights=native.prototypes, count=len(native.classes))
        resources, metadata = dict(native.resource), dict(native.protocol_metadata)
    else:
        scores = _Scores(method, decision=native.decision, count=len(native.classes),
                         support_features=native.support_features, support_labels=native.support_labels,
                         weights=native.weight, knn_k=native.knn_k)
        resources, metadata = dict(native.resources), dict(native.metadata)
    result = dict(model=native.model, features_fn=_Features(method), score_fn=scores,
        classes=tuple(native.classes), resources=resources, native_state=native,
        profile=profile["id"], method=method, provenance=copy.deepcopy(bundle["provenance"]),
        source_path=bundle["source_path"], metadata=metadata)
    if profile["expand_head"]:
        result["expand_head"] = _ExpandHead(method)
    return result


def fit_B(rt, bundle, support_x, support_y, seed, logger=None):
    profile = _profile(bundle["profile"])
    classes = tuple(bundle["classes"])
    _support(support_x, support_y, range(len(classes)))
    if len(set(classes)) != len(classes):
        raise ValueError("Duplicate old class index")
    if bundle["method"] in _REGISTRATION:
        cfg = _registration_config(rt, profile["id"], seed)
        native = cil.prepare_registration_state(bundle["method"], bundle["model"], support_x,
            support_y, classes, config=cfg, base_metadata=copy.deepcopy(bundle["base_metadata"]),
            metadata=bundle["provenance"], logger=logger)
    else:
        counts = torch.bincount(support_y, minlength=len(classes))
        if not bool((counts == counts[0]).all()):
            raise ValueError("Fixed K-shot B requires equal support counts")
        cfg = _adaptation_config(rt, profile["id"], seed, support_x.device, int(counts[0]), logger)
        cfg["source_classes"] = list(classes)
        source_x = source_y = None
        if bundle["method"] in _REPLAY:
            source_x, source_y = _source_L(rt, bundle)
        adapter = "feature_separation_ft" if profile["id"] == "feature_separation_legacy" else profile["id"]
        native = da.fit_adaptation(adapter, bundle["model"], support_x, support_y, classes, cfg,
            source_x=source_x, source_y=source_y, metadata=bundle["provenance"])
    state = _pack(native, profile, bundle)
    if isinstance(native, da.AdaptationState):
        outer = len(native.history)
        if bundle["method"] == "mrior":
            optimizer_steps = sum(1 + int(row.get("estimate_steps", 0)) for row in native.history)
        elif bundle["method"] == "radionet_ada":
            optimizer_steps = sum(2 if row.get("stage") == "ADA" else 1 for row in native.history)
        else:
            optimizer_steps = outer
        state["resources"].update(optimizer_steps=optimizer_steps, outer_logged_updates=outer)
    state["resources"].update(query_fit_rows=0, U_s_truth_reads=0,
                              B_initialization="independent_from_frozen_source", fixed_budget=True)
    return state


def native_C(rt, bundle, Bstate, new_x, new_y, new_ids, seed, logger=None):
    profile = _profile(bundle["profile"])
    if not profile["native_C"]:
        raise NotImplementedError(profile["id"] + " has closed-set native C=N/A; use its explicitly named extension")
    if Bstate["profile"] != profile["id"] or tuple(Bstate["classes"]) != tuple(bundle["classes"]):
        raise ValueError("C must fork the matching old-only B state, not a preceding C prefix")
    new_ids = tuple(int(v) for v in new_ids)
    if not new_ids or len(set(new_ids)) != len(new_ids) or set(new_ids) & set(Bstate["classes"]):
        raise ValueError("New class IDs must be unique and disjoint from B")
    old_count = len(Bstate["classes"])
    _support(new_x, new_y, range(old_count, old_count + len(new_ids)))
    if bundle["method"] in _REGISTRATION:
        native = cil.register_new_classes(Bstate["native_state"], new_x, new_y, new_ids,
            config=_registration_config(rt, profile["id"], seed), logger=logger)
    else:
        # RadioNet's actual decision is its frozen cosine-distance kNN, not NCM.
        native = copy.deepcopy(Bstate["native_state"])
        started = time.perf_counter()
        with torch.no_grad():
            z = da.features(native.model.eval(), new_x)
        native.classes += new_ids
        native.support_features = torch.cat((native.support_features, z.detach().clone()))
        native.support_labels = torch.cat((native.support_labels, new_y.detach().clone()))
        native.metadata.update(C_method="frozen_ADA_encoder_cosine_distance_weighted_kNN_support_append",
                               C_inherits_B_state=True, independent_refit=False)
        native.resources = dict(optimizer_steps=0, trainable_parameters=0,
            fit_seconds=time.perf_counter() - started,
            resident_decision_bytes=sum(v.numel() * v.element_size() for v in
                                       (native.support_features, native.support_labels)),
            inherited_knn_k=native.knn_k)
    state = _pack(native, profile, bundle)
    state["resources"].update(query_fit_rows=0, U_s_truth_reads=0, C_initialization="fork_same_B",
        C_inherits_B=True, native_mechanism=True,
        initialization_corefix="generator_compatible_Kaiming_uniform_same_seed_same_distribution"
            if bundle["method"] in ("csil", "mopc_hr") else None)
    return state


def _head_location(model, method):
    if method in ("mrior", "dadda"):
        return model.classifier.net, str(len(model.classifier.net) - 1)
    if method == "twostage":
        return model.tx_classifier.net, str(len(model.tx_classifier.net) - 1)
    if method == "feature_separation":
        return model, "tx_classifier"
    raise NotImplementedError("No affine output extension for " + method)


@dataclass(frozen=True)
class _ExpandHead:
    method: str

    def __call__(self, state, all_support_x, all_support_y, class_count, seed):
        """Clone the native B model; replace only its final affine output layer.

        Hidden head layers, encoder, estimator/discriminator and all buffers
        are inherited and frozen. Only new rows use support initialization;
        inherited old weights/bias are bitwise copied in B column order.
        The common control runner performs the separately named H100 update.
        """
        class_count, old_count = int(class_count), len(state["classes"])
        if class_count <= old_count or state["method"] != self.method:
            raise ValueError("Output expansion requires this family's B state and new classes")
        _support(all_support_x, all_support_y, range(class_count))
        cloned = copy.deepcopy(state["model"])
        if isinstance(cloned, da._ActiveClasses):
            selected = cloned.columns.detach().clone()
            model, base = cloned, cloned.base
            mapping_keys = ["columns"]
        else:
            model, selected = cloned, torch.arange(old_count, device=all_support_x.device)
            base, mapping_keys = model, []
        model.eval()
        parent, name = _head_location(base, self.method)
        old = parent._modules[name]
        if not isinstance(old, nn.Linear) or len(selected) != old_count:
            raise ValueError("Native final affine layer or inherited B columns mismatch")
        captured = []
        handle = old.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        try:
            with torch.no_grad():
                # Batch extraction avoids materializing large convolution activations.
                for begin in range(0, len(all_support_x), 128):
                    _logits(model, all_support_x[begin:begin + 128])
                z = torch.cat(captured).detach()
        finally:
            handle.remove()
        selected = selected.to(old.weight.device)
        device = old.weight.device
        rng_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(int(seed))
            expanded = nn.Linear(old.in_features, class_count, bias=old.bias is not None).to(old.weight)
        with torch.no_grad():
            expanded.weight[:old_count].copy_(old.weight[selected])
            scale = old.weight[selected].norm(dim=1).median()
            for label in range(old_count, class_count):
                mean = z[all_support_y == label].mean(0)
                expanded.weight[label].copy_(F.normalize(mean, dim=0) * scale)
            if expanded.bias is not None:
                expanded.bias[:old_count].copy_(old.bias[selected])
                expanded.bias[old_count:].zero_()
        parent._modules[name] = expanded
        if mapping_keys:
            # Retain the registered buffer and wrapper/base.* state names.
            # Its columns now expose the expanded native head in compact order.
            model.columns = torch.arange(class_count, device=model.columns.device, dtype=model.columns.dtype)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in expanded.parameters():
            parameter.requires_grad_(True)
        if hasattr(base, "num_tx"):
            base.num_tx = class_count
        if hasattr(base, "num_classes"):
            base.num_classes = class_count
        return dict(model=model, logits_fn=_logits, trainable_parameters=list(expanded.parameters()),
            declared_class_mapping_keys=mapping_keys,
            classes=tuple(range(class_count)), metadata=dict(extension="native_final_affine_expand",
                prehead_inherited_frozen=True, old_output_rows_retained=True,
                new_output_init="normalized_support_prehead_mean_times_median_old_weight_norm;zero_bias",
                query_fit_rows=0, C_inherits_B=True, output_layer=name))
