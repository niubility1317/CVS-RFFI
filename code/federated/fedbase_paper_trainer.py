from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _CODE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.common.spectrogram import iq_to_complex, iq_to_log_spectrogram
from cvsrffi.eval import aggregate_named_stats, apply_sat_channel_for_scenario, resolve_sat_eval_loader_names
from cvsrffi.tensors import extract_domain_from_extra, make_torch_generator, safe_iq_tensor, unpack_batch
from Fedbase.FedFA import FedFAComplexCNN, pairwise_coral_alignment_loss, peer_coral_alignment_losses
from Fedbase.FedRIEI import (
    RIEIModel,
    apply_fedriei_server_gradient_step,
    compressed_gradient_from_states,
    fedriei_alternating_step,
    normalize_compression_name,
)
from Fedbase.FUCL import (
    FUCL1DModel,
    TDLChannelConfig,
    channel_independent_spectrogram,
    encoder_only_state_dict,
    make_two_channel_views,
    nt_xent_loss,
)
from Fedbase.RAFL import (
    RAFLPaperResNet2D,
    gradient_reverse as _grad_reverse,
    label_loss_driven_client_selection,
    receiver_agnostic_loss,
)
from federated.client_split import build_client_loaders, build_client_splits, get_sample_metadata
from federated.fed_aggregate import aggregate_state_dicts, resolve_exclude_keys


FEDBASE_PAPER_MODES = {"fedriei", "fedfa", "fucl", "rafl"}


def infer_num_receivers_from_dataset(dataset) -> int:
    values = set()
    for idx in range(len(dataset)):
        meta = get_sample_metadata(dataset, idx)
        rx = meta.get("rx_id", meta.get("rx_i", None))
        if rx is not None:
            values.add(int(rx))
    if not values:
        raise ValueError("Fedbase paper modes require rx_i/rx_id metadata in the training dataset.")
    return len(values)


def build_fedbase_paper_model(
    mode: str,
    *,
    num_classes: int,
    num_receivers: int,
    feature_dim: int = 512,
    rafl_input_channels: int = 1,
) -> nn.Module:
    normalized = str(mode).lower()
    if normalized == "fedriei":
        return RIEIModel(
            num_emitters=int(num_classes),
            num_receivers=int(num_receivers),
            feature_dim=int(feature_dim),
            classifier_hidden_dim=256,
            dropout=0.0,
            encoder_use_projection=True,
        )
    if normalized == "fedfa":
        return FedFAComplexCNN(num_classes=int(num_classes), embedding_dim=int(feature_dim), dropout=0.2)
    if normalized == "fucl":
        return FUCL1DModel(num_classes=int(num_classes), feature_dim=int(feature_dim), projection_dim=None)
    if normalized == "rafl":
        return RAFLPaperResNet2D(
            num_classes=int(num_classes),
            num_receivers=int(num_receivers),
            feature_dim=int(feature_dim),
            input_channels=int(rafl_input_channels),
        )
    raise ValueError(f"Unsupported Fedbase paper mode: {mode}")


def _forward_outputs(model: nn.Module, x: torch.Tensor, *, grl_lambda: float = 1.0) -> dict[str, torch.Tensor]:
    try:
        out = model(x, y_tx=None, grl_lambda=float(grl_lambda), return_aux=True)
    except TypeError:
        out = model(x)
    if "tx_logits" not in out:
        if "emitter_logits" in out:
            out = dict(out)
            out["tx_logits"] = out["emitter_logits"]
        elif "logits" in out:
            out = dict(out)
            out["tx_logits"] = out["logits"]
    if "dom_logits" not in out:
        if "receiver_logits" in out:
            out = dict(out)
            out["dom_logits"] = out["receiver_logits"]
        elif "rx_logits" in out:
            out = dict(out)
            out["dom_logits"] = out["rx_logits"]
        else:
            logits = out["tx_logits"]
            out = dict(out)
            out["dom_logits"] = logits.new_zeros(logits.size(0), 1)
    return out


def _tensor_to_int_list(value: Any) -> list[int]:
    if torch.is_tensor(value):
        return [int(v) for v in value.detach().cpu().view(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


class FedbasePaperTrainer:
    """Strict paper-method simulator adapted to the CVS-RFFI/WiSig data protocol."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_loader,
        named_test_loaders: Mapping[str, Any],
        cfg,
        *,
        device,
        split_info: Optional[Mapping[str, Any]] = None,
        named_test_meta: Optional[Mapping[str, Any]] = None,
        rafl_selection_dataset=None,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_loader = val_loader
        self.named_test_loaders = dict(named_test_loaders)
        self.cfg = cfg
        self.device = device
        self.mode = str(getattr(cfg, "train_mode", "")).lower()
        if self.mode not in FEDBASE_PAPER_MODES:
            raise ValueError(f"FedbasePaperTrainer only supports {sorted(FEDBASE_PAPER_MODES)}, got {self.mode}")
        self.split_info = dict(split_info or {})
        self.named_test_meta = dict(named_test_meta or {})
        self.output_dir = Path(str(getattr(cfg, "output_dir", "") or Path("runs") / "fedbase_paper" / self.mode))
        self.log_dir = Path(str(getattr(cfg, "log_dir", "") or self.output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logs_jsonl = self.log_dir / "logs.jsonl"
        self.metrics_csv = self.log_dir / "metrics.csv"
        self.summary_json = self.output_dir / "summary.json"
        self.checkpoint_path = self.output_dir / "best_model.pt"

        self.receiver_label_map = self._build_receiver_label_map()
        self.client_splits = build_client_splits(
            train_dataset,
            getattr(cfg, "fl_client_key", "receiver"),
            min_samples_per_client=int(getattr(cfg, "fl_min_samples_per_client", 1)),
            drop_small=bool(getattr(cfg, "fl_drop_small_clients", False)),
            verbose=bool(getattr(cfg, "fl_verbose_clients", False)),
        )
        self.fucl_validation_dataset = None
        self.fucl_validation_client_splits: dict[str, list[int]] = {}
        self.fucl_validation_client_loaders = {}
        self.fucl_client_states: dict[str, OrderedDict[str, torch.Tensor]] = {}
        if self.mode == "fucl":
            self.fucl_validation_dataset = train_dataset
            self.client_splits, self.fucl_validation_client_splits = self._split_train_and_validation_splits(
                self.client_splits,
                ratio=float(getattr(cfg, "fucl_local_validation_ratio", 0.1)),
                seed_offset=314159,
            )
        self.rafl_selection_source = ""
        self.rafl_selection_client_splits: dict[str, list[int]] = {}
        if self.mode == "rafl":
            if rafl_selection_dataset is not None:
                self.rafl_selection_dataset = rafl_selection_dataset
                self.rafl_selection_client_splits = build_client_splits(
                    rafl_selection_dataset,
                    getattr(cfg, "fl_client_key", "receiver"),
                    min_samples_per_client=1,
                    drop_small=False,
                    verbose=False,
                )
                self.rafl_selection_source = "external_heldout_E_rxj"
            else:
                self.rafl_selection_dataset = train_dataset
                self.client_splits, self.rafl_selection_client_splits = self._split_rafl_train_and_selection_splits(self.client_splits)
                self.rafl_selection_source = "internal_train_split_heldout_E_rxj"
        else:
            self.rafl_selection_dataset = None
        self.client_loaders = build_client_loaders(
            train_dataset,
            self.client_splits,
            int(getattr(cfg, "batch_size", 64)),
            int(getattr(cfg, "fl_num_workers", 0)),
            sampler_cfg={"shuffle": True, "drop_last": False},
        )
        if self.mode == "fucl":
            self.fucl_validation_client_loaders = build_client_loaders(
                self.fucl_validation_dataset,
                self.fucl_validation_client_splits,
                int(getattr(cfg, "batch_size", 64)),
                int(getattr(cfg, "fl_num_workers", 0)),
                sampler_cfg={"shuffle": False, "drop_last": False},
            )
        self.rafl_selection_client_loaders = {}
        if self.mode == "rafl":
            self.rafl_selection_client_loaders = build_client_loaders(
                self.rafl_selection_dataset,
                self.rafl_selection_client_splits,
                int(getattr(cfg, "batch_size", 64)),
                int(getattr(cfg, "fl_num_workers", 0)),
                sampler_cfg={"shuffle": False, "drop_last": False},
            )
        self.client_num_samples = {cid: len(ids) for cid, ids in self.client_splits.items()}
        self.global_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in self.model.state_dict().items())
        self._write_config()

    def _split_train_and_validation_splits(
        self,
        splits: Mapping[str, Sequence[int]],
        *,
        ratio: float,
        seed_offset: int = 0,
    ) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        ratio = min(0.9, max(0.0, ratio))
        seed = int(getattr(self.cfg, "seed", 0))
        train_splits: dict[str, list[int]] = {}
        validation_splits: dict[str, list[int]] = {}
        for client_offset, (client_id, indices) in enumerate(splits.items()):
            ordered = [int(idx) for idx in indices]
            rng = random.Random(seed + int(seed_offset) + 104729 * (client_offset + 1))
            rng.shuffle(ordered)
            if len(ordered) >= 2 and ratio > 0.0:
                n_eval = max(1, int(round(len(ordered) * ratio)))
                n_eval = min(n_eval, len(ordered) - 1)
            else:
                n_eval = 0
            selection = sorted(ordered[:n_eval])
            train = sorted(ordered[n_eval:] if n_eval > 0 else ordered)
            if train:
                train_splits[str(client_id)] = train
            if selection:
                validation_splits[str(client_id)] = selection
        return train_splits, validation_splits

    def _split_rafl_train_and_selection_splits(self, splits: Mapping[str, Sequence[int]]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        return self._split_train_and_validation_splits(
            splits,
            ratio=float(getattr(self.cfg, "rafl_selection_eval_ratio", 0.1)),
            seed_offset=0,
        )

    def _write_config(self) -> None:
        adaptations = {
            "fedriei": [
                "CVS-RFFI/WiSig receiver split and evaluation protocol replace the paper's original experimental corpus.",
                "FedRIEI exchanges reconstructed local gradients and applies the paper server gradient update; optional one-bit SignSGD variants are controlled by --fedriei_gradient_compression.",
            ],
            "fedfa": [
                "CVS-RFFI/WiSig receiver split and evaluation protocol replace the paper's original experimental corpus.",
                "Cross-client CORAL is implemented on synchronized participating-client batches for a single-process simulator.",
            ],
            "fucl": [
                "CVS/WiSig raw IQ is converted through the paper TDL channel views and channel-independent spectrogram before the FUCL CNN.",
                "The original LoRa packet acquisition corpus is unavailable; only the signal representation, model, losses, and training protocol are reproduced.",
                "FUCL fine-tuning keeps client-specific classifiers for the strict paper protocol; a CVS common aggregate classifier is retained only as an adapter diagnostic.",
                "Fine-tuning applies the same TDL channel simulator and channel-independent spectrogram representation as pretraining.",
            ],
            "rafl": [
                "CVS-RFFI/WiSig receiver split and evaluation protocol replace the paper's original LoRa receiver corpus.",
                "RAFL converts raw IQ to 2D STFT spectrograms; paper_52x126 uses [B,1,52,126] log-magnitude, while wisig_complex uses two-channel complex STFT for the 256-sample WiSig corpus.",
                "Label Loss Driven selection uses held-out local evaluation loaders E_rxj; those samples are separate from local training loaders.",
            ],
        }
        payload = {
            "fedbase_strict_reproduction_scope": "paper losses, heads, aggregation choices, federated training semantics, and RAFL spectrogram architecture where listed; CVS dataset/evaluation corpus remains explicit",
            "cvs_rffi_protocol_adaptation": True,
            "method_level_adaptations": adaptations.get(self.mode, []),
            "train_mode": self.mode,
            "fedbase_paper_profile": str(getattr(self.cfg, "fedbase_paper_profile", "cvs_adapter")),
            "fl_client_key": str(getattr(self.cfg, "fl_client_key", "")),
            "wisig_train_ratio": float(getattr(self.cfg, "wisig_train_ratio", float("nan"))),
            "epochs": int(getattr(self.cfg, "epochs", 0)),
            "fl_rounds": int(getattr(self.cfg, "fl_rounds", 0)),
            "batch_size": int(getattr(self.cfg, "batch_size", 0)),
            "lr": float(getattr(self.cfg, "lr", float("nan"))),
            "fl_local_epochs": int(getattr(self.cfg, "fl_local_epochs", 0)),
            "paper_method_marker": str(getattr(self.cfg, "fedbase_paper_method", "")),
            "fucl_finetune_epochs": int(getattr(self.cfg, "fucl_finetune_epochs", 1)),
            "fucl_local_validation": {
                "ratio": float(getattr(self.cfg, "fucl_local_validation_ratio", 0.1)),
                "lr_patience_epochs": int(getattr(self.cfg, "fucl_local_lr_patience", 10)),
                "lr_decay": float(getattr(self.cfg, "fucl_local_lr_decay", 0.1)),
                "early_stop_patience_epochs": int(getattr(self.cfg, "fucl_local_early_stop_patience", 20)),
                "max_epochs_override": int(getattr(self.cfg, "fucl_local_max_epochs", 0)),
                "client_num_validation_samples": {
                    str(cid): len(indices) for cid, indices in getattr(self, "fucl_validation_client_splits", {}).items()
                },
            },
            "fucl_signal_representation": {
                "tdl": {
                    "sample_rate_hz": float(getattr(self.cfg, "fucl_sample_rate_hz", 500_000.0)),
                    "rms_delay_ns": [
                        float(getattr(self.cfg, "fucl_tdl_rms_delay_min_ns", 5.0)),
                        float(getattr(self.cfg, "fucl_tdl_rms_delay_max_ns", 300.0)),
                    ],
                    "doppler_hz": [
                        float(getattr(self.cfg, "fucl_tdl_doppler_min_hz", 0.0)),
                        float(getattr(self.cfg, "fucl_tdl_doppler_max_hz", 5.0)),
                    ],
                    "snr_db": [
                        float(getattr(self.cfg, "fucl_tdl_snr_min_db", 0.0)),
                        float(getattr(self.cfg, "fucl_tdl_snr_max_db", 80.0)),
                    ],
                    "num_taps": int(getattr(self.cfg, "fucl_tdl_num_taps", 8)),
                },
                "channel_independent_spectrogram": {
                    "n_fft": int(getattr(self.cfg, "fucl_cis_n_fft", 64)),
                    "hop_length": int(getattr(self.cfg, "fucl_cis_hop_length", 32)),
                    "win_length": int(getattr(self.cfg, "fucl_cis_win_length", 64)),
                    "crop_fraction": float(getattr(self.cfg, "fucl_cis_crop_fraction", 0.30)),
                    "target_shape": [
                        int(getattr(self.cfg, "fucl_cis_freq_bins", 26)),
                        int(getattr(self.cfg, "fucl_cis_time_bins", 126)),
                    ],
                    "normalize": str(getattr(self.cfg, "fucl_cis_normalize", "none")),
                },
            },
            "fedriei_gradient_compression": normalize_compression_name(
                str(getattr(self.cfg, "fedriei_gradient_compression", "none"))
            ),
            "fedriei_compression_noise_std": float(getattr(self.cfg, "fedriei_compression_noise_std", 0.01)),
            "fedriei_server_lr": float(
                getattr(self.cfg, "fedriei_server_lr", 0.0)
                or getattr(self.cfg, "lr", 0.0001)
            ),
            "fedfa_profile": {
                "communication_rounds": int(getattr(self.cfg, "fl_rounds", 0)),
                "local_epochs": int(getattr(self.cfg, "fl_local_epochs", 0)),
                "batch_size": int(getattr(self.cfg, "batch_size", 0)),
                "lr": float(getattr(self.cfg, "lr", float("nan"))),
                "momentum": 0.5,
                "lambda_coral": float(getattr(self.cfg, "fedfa_align_lambda", 0.03)),
                "aggregation": "uniform",
                "stat_exchange_mode": "single_process_detached_peer_covariance",
                "fc_profile": "three_fc_layers_with_relu_dropout",
                "complex_block_profile": "explicit_real_imag_bn_relu",
            },
            "rafl_lambda_rx": float(getattr(self.cfg, "rafl_lambda_rx", float("nan"))),
            "rafl_momentum": float(getattr(self.cfg, "rafl_momentum", float("nan"))),
            "rafl_candidate_clients": int(getattr(self.cfg, "rafl_candidate_clients", 0)),
            "rafl_candidate_fraction": float(getattr(self.cfg, "rafl_candidate_fraction", 1.0)),
            "rafl_selected_clients": int(getattr(self.cfg, "rafl_selected_clients", 0)),
            "rafl_client_selection_profile": (
                "paper_strict_lld"
                if str(getattr(self.cfg, "fedbase_paper_profile", "cvs_adapter")) == "strict_paper"
                and self.mode == "rafl"
                else "adaptive_cvs"
            ),
            "rafl_resolved_selected_clients": (
                self._resolve_rafl_selected_count(len(self.client_splits)) if self.mode == "rafl" else None
            ),
            "rafl_resolved_candidate_clients": (
                self._resolve_rafl_candidate_count(
                    len(self.client_splits),
                    self._resolve_rafl_selected_count(len(self.client_splits)),
                )
                if self.mode == "rafl"
                else None
            ),
            "rafl_selected_fraction": float(getattr(self.cfg, "fl_clients_per_round", 1.0)),
            "rafl_spectrogram": {
                "input_version": str(getattr(self.cfg, "rafl_input_version", "wisig_native")),
                "representation": self._rafl_representation_name(),
                "input_channels": self._rafl_input_channels(),
                "n_fft": int(getattr(self.cfg, "rafl_spec_n_fft", 64)),
                "hop_length": int(getattr(self.cfg, "rafl_spec_hop_length", 32)),
                "win_length": int(getattr(self.cfg, "rafl_spec_win_length", getattr(self.cfg, "rafl_spec_n_fft", 64))),
                "normalize": str(getattr(self.cfg, "rafl_spec_normalize", "zscore")),
                "paper_target_shape_b_c_f_t": [
                    "B",
                    1,
                    int(getattr(self.cfg, "rafl_spec_freq_bins", 52)),
                    int(getattr(self.cfg, "rafl_spec_time_bins", 126)),
                ],
                "resize_to_paper_shape": str(getattr(self.cfg, "rafl_input_version", "wisig_native")) == "paper_52x126",
            },
            "rafl_selection_source": str(getattr(self, "rafl_selection_source", "")),
            "rafl_selection_eval_ratio": float(getattr(self.cfg, "rafl_selection_eval_ratio", 0.1)),
            "rafl_selection_dataset": str(getattr(self.cfg, "rafl_selection_dataset", "internal_train_split")),
            "rafl_train_client_num_samples": dict(self.client_num_samples),
            "rafl_selection_client_num_samples": {
                str(cid): len(indices) for cid, indices in getattr(self, "rafl_selection_client_splits", {}).items()
            },
            "receiver_label_map": dict(self.receiver_label_map),
            "split_info": self.split_info,
            "named_test_meta": self.named_test_meta,
        }
        with (self.output_dir / "fedbase_config.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _build_receiver_label_map(self) -> dict[int, int]:
        values = set()
        for idx in range(len(self.train_dataset)):
            meta = get_sample_metadata(self.train_dataset, idx)
            rx = meta.get("rx_id", meta.get("rx_i", None))
            if rx is not None:
                values.add(int(rx))
        if not values:
            raise ValueError("Fedbase paper trainer requires rx_i/rx_id metadata for receiver labels.")
        return {raw: mapped for mapped, raw in enumerate(sorted(values))}

    def _map_receiver_tensor(self, raw: torch.Tensor) -> torch.Tensor:
        raw_cpu = raw.detach().cpu().view(-1).long()
        mapped = [self.receiver_label_map.get(int(v), -1) for v in raw_cpu.tolist()]
        out = torch.as_tensor(mapped, device=self.device, dtype=torch.long)
        if (out < 0).any():
            missing = sorted({int(raw_cpu[i]) for i, value in enumerate(mapped) if value < 0})
            raise ValueError(f"Receiver labels not present in training map: {missing}")
        return out

    def _extract_rx_from_extra(self, extra, batch_size: int) -> torch.Tensor:
        meta = extra[1] if extra is not None and len(extra) > 1 else None
        if isinstance(meta, Mapping):
            for key in ("rx_i", "rx_id"):
                if key in meta:
                    raw = torch.as_tensor(_tensor_to_int_list(meta[key]), device=self.device).view(-1)
                    return self._map_receiver_tensor(raw)
        d_raw = extract_domain_from_extra(extra, self.device)
        if d_raw is None:
            raise KeyError("Batch lacks receiver metadata; expected meta['rx_i'] or meta['rx_id'].")
        if int(d_raw.numel()) != int(batch_size):
            d_raw = d_raw.view(-1)[:batch_size]
        return self._map_receiver_tensor(d_raw)

    def _extract_raw_rx_from_extra(self, extra, batch_size: int) -> torch.Tensor:
        meta = extra[1] if extra is not None and len(extra) > 1 else None
        if isinstance(meta, Mapping):
            for key in ("rx_i", "rx_id"):
                if key in meta:
                    raw = torch.as_tensor(_tensor_to_int_list(meta[key]), device=self.device).view(-1).long()
                    if int(raw.numel()) == 1 and int(batch_size) > 1:
                        raw = raw.expand(int(batch_size))
                    elif int(raw.numel()) != int(batch_size):
                        raw = raw[:batch_size]
                    return raw
        d_raw = extract_domain_from_extra(extra, self.device)
        if d_raw is None:
            raise KeyError("Batch lacks receiver metadata; expected meta['rx_i'] or meta['rx_id'].")
        d_raw = d_raw.view(-1).long()
        if int(d_raw.numel()) == 1 and int(batch_size) > 1:
            d_raw = d_raw.expand(int(batch_size))
        elif int(d_raw.numel()) != int(batch_size):
            d_raw = d_raw[:batch_size]
        return d_raw

    def _batch_to_xy_rx(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, extra = unpack_batch(batch)
        x = safe_iq_tensor(x.to(self.device, non_blocking=True))
        y = y.to(self.device, non_blocking=True).long().view(-1)
        rx = self._extract_rx_from_extra(extra, int(y.numel()))
        return x, y, rx

    def _batch_to_xy(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, y, _extra = unpack_batch(batch)
        x = safe_iq_tensor(x.to(self.device, non_blocking=True))
        y = y.to(self.device, non_blocking=True).long().view(-1)
        return x, y

    def _rafl_representation_name(self) -> str:
        input_version = str(getattr(self.cfg, "rafl_input_version", "wisig_native"))
        if input_version == "wisig_complex":
            return "complex_stft_real_imag"
        return "log_magnitude_stft"

    def _rafl_input_channels(self) -> int:
        return 2 if self._rafl_representation_name() == "complex_stft_real_imag" else 1

    def _rafl_complex_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        iq = safe_iq_tensor(x)
        z = iq_to_complex(iq)
        n_fft = int(getattr(self.cfg, "rafl_spec_n_fft", 64))
        hop_length = int(getattr(self.cfg, "rafl_spec_hop_length", 32))
        win_length = int(getattr(self.cfg, "rafl_spec_win_length", getattr(self.cfg, "rafl_spec_n_fft", 64)))
        window = torch.hann_window(win_length, device=z.device, dtype=z.real.dtype)
        stft = torch.stft(
            z,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True,
            onesided=False,
        )
        spec = torch.stack([stft.real, stft.imag], dim=1).float()
        norm = str(getattr(self.cfg, "rafl_spec_normalize", "zscore"))
        if norm == "zscore":
            mean = spec.mean(dim=(-2, -1), keepdim=True)
            std = spec.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            spec = (spec - mean) / std
        elif norm == "minmax":
            lo = spec.amin(dim=(-2, -1), keepdim=True)
            hi = spec.amax(dim=(-2, -1), keepdim=True)
            spec = (spec - lo) / (hi - lo).clamp_min(1e-8)
        elif norm != "none":
            raise ValueError(f"Unsupported spectrogram normalize={norm!r}")
        return spec

    def _rafl_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        target = (
            int(getattr(self.cfg, "rafl_spec_freq_bins", 52)),
            int(getattr(self.cfg, "rafl_spec_time_bins", 126)),
        )
        input_version = str(getattr(self.cfg, "rafl_input_version", "wisig_native"))
        if input_version == "wisig_complex":
            spec = self._rafl_complex_spectrogram(x)
        else:
            spec = iq_to_log_spectrogram(
                safe_iq_tensor(x),
                n_fft=int(getattr(self.cfg, "rafl_spec_n_fft", 64)),
                hop_length=int(getattr(self.cfg, "rafl_spec_hop_length", 32)),
                win_length=int(getattr(self.cfg, "rafl_spec_win_length", getattr(self.cfg, "rafl_spec_n_fft", 64))),
                normalize=str(getattr(self.cfg, "rafl_spec_normalize", "zscore")),
            )
        if input_version == "paper_52x126" and tuple(spec.shape[-2:]) != target:
            spec = F.interpolate(spec, size=target, mode="bilinear", align_corners=False)
        elif input_version not in {"paper_52x126", "wisig_native", "wisig_complex"}:
            raise ValueError(f"Unsupported rafl_input_version={input_version!r}")
        return torch.nan_to_num(spec.float(), nan=0.0, posinf=0.0, neginf=0.0)

    def _fucl_tdl_config(self) -> TDLChannelConfig:
        return TDLChannelConfig(
            sample_rate_hz=float(getattr(self.cfg, "fucl_sample_rate_hz", 500_000.0)),
            rms_delay_min_ns=float(getattr(self.cfg, "fucl_tdl_rms_delay_min_ns", 5.0)),
            rms_delay_max_ns=float(getattr(self.cfg, "fucl_tdl_rms_delay_max_ns", 300.0)),
            doppler_min_hz=float(getattr(self.cfg, "fucl_tdl_doppler_min_hz", 0.0)),
            doppler_max_hz=float(getattr(self.cfg, "fucl_tdl_doppler_max_hz", 5.0)),
            snr_min_db=float(getattr(self.cfg, "fucl_tdl_snr_min_db", 0.0)),
            snr_max_db=float(getattr(self.cfg, "fucl_tdl_snr_max_db", 80.0)),
            num_taps=int(getattr(self.cfg, "fucl_tdl_num_taps", 8)),
        )

    def _fucl_cis_kwargs(self) -> dict[str, Any]:
        return {
            "n_fft": int(getattr(self.cfg, "fucl_cis_n_fft", 64)),
            "hop_length": int(getattr(self.cfg, "fucl_cis_hop_length", 32)),
            "win_length": int(getattr(self.cfg, "fucl_cis_win_length", 64)),
            "crop_fraction": float(getattr(self.cfg, "fucl_cis_crop_fraction", 0.30)),
            "target_shape": (
                int(getattr(self.cfg, "fucl_cis_freq_bins", 26)),
                int(getattr(self.cfg, "fucl_cis_time_bins", 126)),
            ),
            "normalize": str(getattr(self.cfg, "fucl_cis_normalize", "none")),
        }

    def _fucl_signal_representation(self, x: torch.Tensor) -> torch.Tensor:
        return channel_independent_spectrogram(safe_iq_tensor(x), **self._fucl_cis_kwargs())

    def _model_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "rafl":
            return self._rafl_spectrogram(x)
        if self.mode == "fucl":
            return self._fucl_signal_representation(x)
        return x

    def _resolve_rafl_selected_count(self, num_clients: int) -> int:
        n = max(1, int(num_clients))
        explicit = int(getattr(self.cfg, "rafl_selected_clients", 0))
        if explicit > 0:
            return min(n, explicit)
        frac = float(getattr(self.cfg, "fl_clients_per_round", 1.0))
        if frac >= 1.0:
            return n
        return max(1, min(n, int(math.ceil(n * max(0.0, frac)))))

    def _resolve_rafl_candidate_count(self, num_clients: int, selected_count: int) -> int:
        n = max(1, int(num_clients))
        k = max(1, min(n, int(selected_count)))
        explicit = int(getattr(self.cfg, "rafl_candidate_clients", 0))
        if explicit > 0:
            return min(n, max(k, explicit))
        frac = float(getattr(self.cfg, "rafl_candidate_fraction", 1.0))
        return min(n, max(k, int(math.ceil(n * min(1.0, max(0.0, frac))))))

    def _selected_clients_random(self, round_idx: int) -> list[str]:
        client_ids = list(self.client_splits.keys())
        if self.mode == "rafl":
            k = self._resolve_rafl_selected_count(len(client_ids))
        else:
            frac = float(getattr(self.cfg, "fl_clients_per_round", 1.0))
            k = len(client_ids) if frac >= 1.0 else max(1, int(math.ceil(len(client_ids) * max(0.0, frac))))
        rng = random.Random(int(getattr(self.cfg, "seed", 0)) + int(round_idx) * 1009)
        return sorted(rng.sample(client_ids, k=k))

    def _selected_clients(self, round_idx: int) -> list[str]:
        if self.mode == "rafl" and str(getattr(self.cfg, "rafl_client_selection", "label_loss_driven")) == "all":
            return list(self.client_splits.keys())
        if self.mode == "rafl" and str(getattr(self.cfg, "rafl_client_selection", "label_loss_driven")) == "label_loss_driven":
            all_clients = list(self.client_splits.keys())
            k = self._resolve_rafl_selected_count(len(all_clients))
            candidate_size = self._resolve_rafl_candidate_count(len(all_clients), k)
            if candidate_size >= len(all_clients):
                candidate_clients = list(all_clients)
            else:
                rng = random.Random(int(getattr(self.cfg, "seed", 0)) + int(round_idx) * 7919)
                candidate_clients = sorted(rng.sample(all_clients, k=candidate_size))
            label_losses = self._rafl_label_losses(candidate_clients)
            result = label_loss_driven_client_selection(
                label_losses,
                clients_per_round=k,
                candidate_clients=candidate_clients,
                random_seed=int(getattr(self.cfg, "seed", 0)) + int(round_idx) * 1543,
            )
            result["num_available_clients"] = len(all_clients)
            result["resolved_selected_clients"] = k
            result["resolved_candidate_clients"] = len(candidate_clients)
            result["adaptive_selection"] = {
                "selected_clients_arg": int(getattr(self.cfg, "rafl_selected_clients", 0)),
                "candidate_clients_arg": int(getattr(self.cfg, "rafl_candidate_clients", 0)),
                "selected_fraction": float(getattr(self.cfg, "fl_clients_per_round", 1.0)),
                "candidate_fraction": float(getattr(self.cfg, "rafl_candidate_fraction", 1.0)),
            }
            selected = [cid for cid in result["selected_clients"] if cid in self.client_splits]
            self._last_rafl_selection = result
            return selected or self._selected_clients_random(round_idx)
        return self._selected_clients_random(round_idx)

    def _make_optimizer(self, model: nn.Module, method: Optional[str] = None):
        mode = str(method or self.mode)
        if mode == "fedfa":
            return torch.optim.SGD(
                model.parameters(),
                lr=float(getattr(self.cfg, "lr", 0.01)),
                momentum=0.5,
                weight_decay=float(getattr(self.cfg, "wd", 0.0)),
            )
        if mode == "rafl":
            return torch.optim.SGD(
                model.parameters(),
                lr=float(getattr(self.cfg, "lr", 0.001)),
                momentum=float(getattr(self.cfg, "rafl_momentum", 0.0)),
                weight_decay=float(getattr(self.cfg, "wd", 0.0)),
            )
        if mode == "fedriei":
            return torch.optim.SGD(
                model.parameters(),
                lr=float(getattr(self.cfg, "lr", 0.0001)),
                momentum=0.0,
                weight_decay=float(getattr(self.cfg, "wd", 0.0)),
            )
        lr = float(getattr(self.cfg, "lr", 0.001))
        if mode == "fucl_pretrain":
            lr = float(getattr(self.cfg, "fucl_pretrain_lr", 0.0003))
        elif mode == "fucl_finetune":
            lr = float(getattr(self.cfg, "fucl_finetune_lr", 0.001))
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(getattr(self.cfg, "wd", 0.0)))

    @torch.no_grad()
    def _evaluate_loader(self, loader, max_batches: int = 0) -> dict[str, Any]:
        self.model.eval()
        correct = total = 0
        for batch_idx, batch in enumerate(loader):
            x, y = self._batch_to_xy(batch)
            out = _forward_outputs(self.model, self._model_input(x))
            pred = out["tx_logits"].argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            if max_batches > 0 and batch_idx + 1 >= max_batches:
                break
        return {
            "tx_acc": 100.0 * correct / max(1, total),
            "dom_acc": float("nan"),
            "probe_dom_acc": float("nan"),
            "tx_correct": int(correct),
            "tx_total": int(total),
        }

    def _evaluate_named(self) -> dict[str, Any]:
        max_batches = int(getattr(self.cfg, "eval_max_batches", 0))
        return {name: self._evaluate_loader(loader, max_batches=max_batches) for name, loader in self.named_test_loaders.items()}

    def _load_fucl_client_specific_models(self) -> OrderedDict[str, nn.Module]:
        models: OrderedDict[str, nn.Module] = OrderedDict()
        for client_id in sorted(self.fucl_client_states):
            model = copy.deepcopy(self.model)
            model.load_state_dict(self.fucl_client_states[str(client_id)], strict=False)
            model.to(self.device)
            model.eval()
            models[str(client_id)] = model
        return models

    @torch.no_grad()
    def _evaluate_fucl_client_specific_loader(self, loader, max_batches: int = 0) -> dict[str, Any]:
        models = self._load_fucl_client_specific_models()
        if not models:
            return self._evaluate_loader(loader, max_batches=max_batches)
        client_ids = list(models.keys())
        correct = total = 0
        matched = ensembled = 0
        for batch_idx, batch in enumerate(loader):
            x, y, extra = unpack_batch(batch)
            x = safe_iq_tensor(x.to(self.device, non_blocking=True))
            y = y.to(self.device, non_blocking=True).long().view(-1)
            raw_rx = self._extract_raw_rx_from_extra(extra, int(y.numel()))
            sample_client_ids = [f"rx{int(v)}" for v in raw_rx.detach().cpu().view(-1).tolist()]
            needs_ensemble = any(cid not in models for cid in sample_client_ids)
            needed = client_ids if needs_ensemble else sorted({cid for cid in sample_client_ids if cid in models})
            x_in = self._fucl_signal_representation(x)
            logits_by_client = {cid: _forward_outputs(models[cid], x_in)["tx_logits"] for cid in needed}
            ensemble_stack = None
            if needs_ensemble:
                ensemble_stack = torch.stack([logits_by_client[cid] for cid in client_ids], dim=0)
            sample_logits: list[torch.Tensor] = []
            for sample_idx, client_id in enumerate(sample_client_ids):
                if client_id in logits_by_client:
                    sample_logits.append(logits_by_client[client_id][sample_idx])
                    matched += 1
                else:
                    if ensemble_stack is None:
                        ensemble_stack = torch.stack([_forward_outputs(models[cid], x_in)["tx_logits"] for cid in client_ids], dim=0)
                    sample_logits.append(ensemble_stack[:, sample_idx, :].mean(dim=0))
                    ensembled += 1
            logits = torch.stack(sample_logits, dim=0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            if max_batches > 0 and batch_idx + 1 >= max_batches:
                break
        return {
            "tx_acc": 100.0 * correct / max(1, total),
            "dom_acc": float("nan"),
            "probe_dom_acc": float("nan"),
            "tx_correct": int(correct),
            "tx_total": int(total),
            "fucl_eval_mode": "client_specific_seen_receiver_else_source_client_ensemble",
            "fucl_seen_receiver_matched": int(matched),
            "fucl_unseen_receiver_ensembled": int(ensembled),
        }

    def _evaluate_fucl_client_specific_named(self) -> dict[str, Any]:
        max_batches = int(getattr(self.cfg, "eval_max_batches", 0))
        return {
            name: self._evaluate_fucl_client_specific_loader(loader, max_batches=max_batches)
            for name, loader in self.named_test_loaders.items()
        }

    @torch.no_grad()
    def _evaluate_sat_loader(self, loader, scenario: str, *, seed: int, max_batches: int = 0) -> dict[str, Any]:
        self.model.eval()
        correct = total = 0
        gen = make_torch_generator(self.device, int(seed))
        for batch_idx, batch in enumerate(loader):
            x, y = self._batch_to_xy(batch)
            x_sat, _meta = apply_sat_channel_for_scenario(x, str(scenario), self.cfg, gen=gen, return_meta=False)
            out = _forward_outputs(self.model, self._model_input(safe_iq_tensor(x_sat)))
            pred = out["tx_logits"].argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            if max_batches > 0 and batch_idx + 1 >= max_batches:
                break
        return {
            "tx_acc": 100.0 * correct / max(1, total),
            "dom_acc": float("nan"),
            "probe_dom_acc": float("nan"),
            "tx_correct": int(correct),
            "tx_total": int(total),
        }

    @torch.no_grad()
    def _evaluate_fucl_client_specific_sat_loader(self, loader, scenario: str, *, seed: int, max_batches: int = 0) -> dict[str, Any]:
        models = self._load_fucl_client_specific_models()
        if not models:
            return self._evaluate_sat_loader(loader, scenario, seed=seed, max_batches=max_batches)
        client_ids = list(models.keys())
        correct = total = 0
        matched = ensembled = 0
        gen = make_torch_generator(self.device, int(seed))
        for batch_idx, batch in enumerate(loader):
            x, y, extra = unpack_batch(batch)
            x = safe_iq_tensor(x.to(self.device, non_blocking=True))
            y = y.to(self.device, non_blocking=True).long().view(-1)
            raw_rx = self._extract_raw_rx_from_extra(extra, int(y.numel()))
            sample_client_ids = [f"rx{int(v)}" for v in raw_rx.detach().cpu().view(-1).tolist()]
            x_sat, _meta = apply_sat_channel_for_scenario(x, str(scenario), self.cfg, gen=gen, return_meta=False)
            x_in = self._fucl_signal_representation(safe_iq_tensor(x_sat))
            needs_ensemble = any(cid not in models for cid in sample_client_ids)
            needed = client_ids if needs_ensemble else sorted({cid for cid in sample_client_ids if cid in models})
            logits_by_client = {cid: _forward_outputs(models[cid], x_in)["tx_logits"] for cid in needed}
            ensemble_stack = None
            if needs_ensemble:
                ensemble_stack = torch.stack([logits_by_client[cid] for cid in client_ids], dim=0)
            sample_logits: list[torch.Tensor] = []
            for sample_idx, client_id in enumerate(sample_client_ids):
                if client_id in logits_by_client:
                    sample_logits.append(logits_by_client[client_id][sample_idx])
                    matched += 1
                else:
                    if ensemble_stack is None:
                        ensemble_stack = torch.stack([_forward_outputs(models[cid], x_in)["tx_logits"] for cid in client_ids], dim=0)
                    sample_logits.append(ensemble_stack[:, sample_idx, :].mean(dim=0))
                    ensembled += 1
            logits = torch.stack(sample_logits, dim=0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            if max_batches > 0 and batch_idx + 1 >= max_batches:
                break
        return {
            "tx_acc": 100.0 * correct / max(1, total),
            "dom_acc": float("nan"),
            "probe_dom_acc": float("nan"),
            "tx_correct": int(correct),
            "tx_total": int(total),
            "fucl_eval_mode": "client_specific_seen_receiver_else_source_client_ensemble",
            "fucl_seen_receiver_matched": int(matched),
            "fucl_unseen_receiver_ensembled": int(ensembled),
        }

    def _evaluate_sat_named(self) -> dict[str, Any]:
        if not bool(getattr(self.cfg, "eval_sat_channel", False)):
            return {}
        scenario_names = list(getattr(self.cfg, "eval_sat_scenario_list", []) or [])
        if not scenario_names:
            return {}
        selected_names = resolve_sat_eval_loader_names(self.named_test_loaders, getattr(self.cfg, "eval_sat_on", "main"))
        max_batches = int(getattr(self.cfg, "sat_eval_max_batches", -1))
        if max_batches < 0:
            max_batches = int(getattr(self.cfg, "eval_max_batches", 0))
        out: dict[str, Any] = {}
        for scenario_idx, scenario in enumerate(scenario_names):
            named_stats = {}
            for loader_idx, name in enumerate(selected_names):
                named_stats[name] = self._evaluate_sat_loader(
                    self.named_test_loaders[name],
                    str(scenario),
                    seed=int(getattr(self.cfg, "sat_seed", 2027)) + scenario_idx * 1009 + loader_idx * 97,
                    max_batches=max_batches,
                )
            main_keys = [k for k in ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"] if k in named_stats]
            if not main_keys:
                main_keys = list(named_stats.keys())
            out[str(scenario)] = {
                "aggregate": aggregate_named_stats(named_stats, main_keys),
                "strict_udu": named_stats.get("test_unseen_day_unseen_rx", {}).get("tx_acc", float("nan")),
                "named": named_stats,
                "selected_names": list(selected_names),
            }
        return out

    def _evaluate_fucl_client_specific_sat_named(self) -> dict[str, Any]:
        if not bool(getattr(self.cfg, "eval_sat_channel", False)):
            return {}
        scenario_names = list(getattr(self.cfg, "eval_sat_scenario_list", []) or [])
        if not scenario_names:
            return {}
        selected_names = resolve_sat_eval_loader_names(self.named_test_loaders, getattr(self.cfg, "eval_sat_on", "main"))
        max_batches = int(getattr(self.cfg, "sat_eval_max_batches", -1))
        if max_batches < 0:
            max_batches = int(getattr(self.cfg, "eval_max_batches", 0))
        out: dict[str, Any] = {}
        for scenario_idx, scenario in enumerate(scenario_names):
            named_stats = {}
            for loader_idx, name in enumerate(selected_names):
                named_stats[name] = self._evaluate_fucl_client_specific_sat_loader(
                    self.named_test_loaders[name],
                    str(scenario),
                    seed=int(getattr(self.cfg, "sat_seed", 2027)) + scenario_idx * 1009 + loader_idx * 97,
                    max_batches=max_batches,
                )
            main_keys = [k for k in ["test_unseen_day_seen_rx", "test_seen_day_unseen_rx", "test_unseen_day_unseen_rx"] if k in named_stats]
            if not main_keys:
                main_keys = list(named_stats.keys())
            out[str(scenario)] = {
                "aggregate": aggregate_named_stats(named_stats, main_keys),
                "strict_udu": named_stats.get("test_unseen_day_unseen_rx", {}).get("tx_acc", float("nan")),
                "named": named_stats,
                "selected_names": list(selected_names),
                "fucl_eval_mode": "client_specific_seen_receiver_else_source_client_ensemble",
            }
        return out

    @staticmethod
    def _parse_eval_offsets(value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            raw_items = value
        else:
            raw_items = str(value).replace(";", ",").split(",")
        offsets = []
        for item in raw_items:
            text = str(item).strip()
            if not text:
                continue
            try:
                offset = int(text)
            except ValueError:
                continue
            if offset > 0:
                offsets.append(offset)
        return offsets

    def _should_run_eval(self, round_idx: int) -> bool:
        total_rounds = max(1, int(getattr(self.cfg, "fl_rounds", 1)))
        if int(round_idx) >= total_rounds:
            return True
        interval = int(getattr(self.cfg, "fl_test_eval_interval", 0))
        if interval > 0 and int(round_idx) % interval == 0:
            return True
        last_n = int(getattr(self.cfg, "fl_test_eval_last_n", 0))
        if last_n > 0 and int(round_idx) > total_rounds - last_n:
            return True
        for offset in self._parse_eval_offsets(getattr(self.cfg, "fl_test_eval_final_offsets", "")):
            if int(round_idx) == total_rounds - int(offset) + 1:
                return True
        return False

    def _primary_test_acc(self, named: Mapping[str, Mapping[str, Any]]) -> float:
        primary = str(self.split_info.get("primary_named_test", "test_unseen_day_unseen_rx"))
        if primary in named:
            return float(named[primary].get("tx_acc", float("nan")))
        if named:
            return float(next(iter(named.values())).get("tx_acc", float("nan")))
        return float("nan")

    @staticmethod
    def _tx_acc_json(stats: Mapping[str, Mapping[str, Any]]) -> str:
        summary = {}
        for name, item in (stats or {}).items():
            try:
                summary[str(name)] = float(item.get("tx_acc", float("nan")))
            except Exception:
                summary[str(name)] = float("nan")
        return json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _sat_tx_acc_json(stats: Mapping[str, Any]) -> str:
        summary = {}
        for scenario, payload in (stats or {}).items():
            scenario_summary = {}
            named = {}
            if isinstance(payload, Mapping) and isinstance(payload.get("named"), Mapping):
                aggregate = payload.get("aggregate", {}) or {}
                if isinstance(aggregate, Mapping):
                    try:
                        scenario_summary["aggregate"] = float(aggregate.get("tx_acc", float("nan")))
                    except Exception:
                        scenario_summary["aggregate"] = float("nan")
                try:
                    scenario_summary["strict_udu"] = float(payload.get("strict_udu", float("nan")))
                except Exception:
                    scenario_summary["strict_udu"] = float("nan")
                named = dict(payload.get("named", {}) or {})
            elif isinstance(payload, Mapping):
                named = dict(payload)
            named_summary = {}
            for name, item in (named or {}).items():
                try:
                    named_summary[str(name)] = float(item.get("tx_acc", float("nan")))
                except Exception:
                    named_summary[str(name)] = float("nan")
            if isinstance(payload, Mapping) and isinstance(payload.get("named"), Mapping):
                scenario_summary["named"] = named_summary
            else:
                scenario_summary.update(named_summary)
            summary[str(scenario)] = scenario_summary
        return json.dumps(summary, ensure_ascii=False, sort_keys=True, default=str)

    def _format_named_test_stdout(self, named: Mapping[str, Mapping[str, Any]], *, round_label: Any) -> list[str]:
        if not named:
            return []
        lines: list[str] = []
        label = f"R{round_label}" if str(round_label).isdigit() else str(round_label)
        primary = str(self.split_info.get("primary_named_test", "test_unseen_day_unseen_rx"))
        ordered_names = [primary] if primary in named else []
        ordered_names.extend(name for name in sorted(named) if name not in ordered_names)
        for name in ordered_names:
            stats = named.get(name, {}) or {}
            meta = self.named_test_meta.get(name, {}) or {}
            total = stats.get("tx_total", meta.get("size", "na"))
            tx_acc = float(stats.get("tx_acc", float("nan")))
            rx_acc = stats.get("rx_acc", float("nan"))
            rx_text = ""
            try:
                rx_text = f" rx_acc={float(rx_acc):.2f}"
            except Exception:
                rx_text = ""
            lines.append(
                f"[FEDBASE-TEST][{label}] mode={self.mode} split={name} "
                f"tx_acc={tx_acc:.2f}{rx_text} tx_total={total}"
            )
        return lines

    def _format_sat_test_stdout(self, sat_stats: Mapping[str, Any], *, round_label: Any) -> list[str]:
        if not sat_stats:
            return []
        lines: list[str] = []
        label = f"R{round_label}" if str(round_label).isdigit() else str(round_label)
        for scenario in sorted(sat_stats):
            payload = sat_stats.get(scenario, {}) or {}
            named: Mapping[str, Any] = {}
            if isinstance(payload, Mapping) and isinstance(payload.get("named"), Mapping):
                aggregate = payload.get("aggregate", {}) or {}
                if isinstance(aggregate, Mapping):
                    lines.append(
                        f"[FEDBASE-SAT][{label}] mode={self.mode} scenario={scenario} "
                        f"split=aggregate tx_acc={float(aggregate.get('tx_acc', float('nan'))):.2f} "
                        f"tx_total={aggregate.get('tx_total', 'na')}"
                    )
                if "strict_udu" in payload:
                    try:
                        strict_udu = float(payload.get("strict_udu", float("nan")))
                    except Exception:
                        strict_udu = float("nan")
                    lines.append(
                        f"[FEDBASE-SAT][{label}] mode={self.mode} scenario={scenario} "
                        f"split=strict_udu tx_acc={strict_udu:.2f} tx_total=na"
                    )
                named = payload.get("named", {}) or {}
            elif isinstance(payload, Mapping):
                named = payload
            for split in sorted(named):
                stats = named.get(split, {}) or {}
                tx_acc = float(stats.get("tx_acc", float("nan")))
                total = stats.get("tx_total", "na")
                lines.append(
                    f"[FEDBASE-SAT][{label}] mode={self.mode} scenario={scenario} "
                    f"split={split} tx_acc={tx_acc:.2f} tx_total={total}"
                )
        return lines

    @staticmethod
    def _mean_std(values: Sequence[float]) -> dict[str, float]:
        vals = []
        for value in values:
            try:
                fv = float(value)
            except Exception:
                continue
            if math.isfinite(fv):
                vals.append(fv)
        if not vals:
            return {"mean": float("nan"), "std": float("nan")}
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return {"mean": mean, "std": math.sqrt(var)}

    def _summarize_paper_eval_records(self, records: Sequence[Mapping[str, Any]], *, name: str) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "name": str(name),
            "n": len(records),
            "rounds": [int(r.get("round", -1)) for r in records],
            "primary_test_tx_acc": self._mean_std([float(r.get("primary_test_tx_acc", float("nan"))) for r in records]),
        }
        named_keys = sorted({k for r in records for k in (r.get("named_tests", {}) or {}).keys()})
        named: dict[str, Any] = {}
        for key in named_keys:
            vals = [
                float((r.get("named_tests", {}) or {}).get(key, {}).get("tx_acc", float("nan")))
                for r in records
            ]
            named[key] = {"tx_acc": self._mean_std(vals)}
        summary["named_tests"] = named
        return summary

    def _append_log(self, row: Mapping[str, Any]) -> None:
        with self.logs_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    def _append_metrics(self, row: Mapping[str, Any]) -> None:
        fieldnames = [
            "round",
            "train_mode",
            "selected_clients",
            "train_loss",
            "train_acc",
            "rafl_loss_tx",
            "rafl_loss_rx",
            "rafl_lambda_rx",
            "rafl_receiver_loss_weight",
            "fedriei_loss_ce",
            "fedriei_loss_mi",
            "fedriei_loss_ie",
            "fedriei_loss_dis",
            "val_tx_acc",
            "primary_test_tx_acc",
            "named_test_tx_acc_json",
            "fucl_common_aggregate_val_tx_acc",
            "fucl_common_aggregate_primary_test_tx_acc",
            "fucl_common_aggregate_named_test_tx_acc_json",
            "sat_channel_tx_acc_json",
            "test_trigger",
        ]
        exists = self.metrics_csv.exists()
        with self.metrics_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    def _load_state_to_model(self, model: nn.Module) -> None:
        model.load_state_dict(self.global_state, strict=False)
        model.to(self.device)

    def _aggregate_client_states(self, client_states: Mapping[str, Mapping[str, torch.Tensor]], selected: Sequence[str], *, agg_weight: str) -> None:
        new_state = aggregate_state_dicts(
            client_states,
            self.client_num_samples,
            exclude_keys=set(),
            agg_weight=str(agg_weight),
        )
        self.global_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in new_state.items())
        self.model.load_state_dict(self.global_state, strict=False)
        self.model.to(self.device)

    def _aggregate_encoder_only(self, client_states: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
        ref = next(iter(client_states.values()))
        exclude = resolve_exclude_keys(ref, prefixes=("classifier", "classification_head", "projection_head", "head"))
        aggregated = aggregate_state_dicts(
            client_states,
            self.client_num_samples,
            exclude_keys=exclude,
            agg_weight="num_samples",
        )
        current = OrderedDict((k, v.detach().cpu().clone()) for k, v in self.global_state.items())
        current.update(aggregated)
        self.global_state = current
        self.model.load_state_dict(self.global_state, strict=False)
        self.model.to(self.device)

    def _aggregate_fedriei_gradients(self, client_gradients: Mapping[str, Mapping[str, torch.Tensor]], selected: Sequence[str]) -> None:
        server_lr = float(getattr(self.cfg, "fedriei_server_lr", 0.0) or getattr(self.cfg, "lr", 0.0001))
        self.global_state = apply_fedriei_server_gradient_step(
            self.global_state,
            client_gradients,
            client_num_samples=self.client_num_samples,
            selected=selected,
            server_lr=server_lr,
        )
        self.model.load_state_dict(self.global_state, strict=False)
        self.model.to(self.device)

    def _local_train_fedriei(self, client_id: str, round_idx: int) -> dict[str, Any]:
        local_model = copy.deepcopy(self.model)
        self._load_state_to_model(local_model)
        start_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in self.global_state.items())
        parameter_keys = {str(name) for name, _param in local_model.named_parameters()}
        opt_ce = self._make_optimizer(local_model, "fedriei")
        opt_dis = torch.optim.SGD(local_model.fed.parameters(), lr=float(getattr(self.cfg, "lr", 0.0001)), momentum=0.0)
        totals: Dict[str, float] = {}
        seen = correct = 0
        for _epoch in range(max(1, int(getattr(self.cfg, "fl_local_epochs", 1)))):
            for batch in self.client_loaders[client_id]:
                x, y, rx = self._batch_to_xy_rx(batch)
                metrics = fedriei_alternating_step(
                    local_model,
                    {"iq": x, "label": y, "receiver": rx},
                    {"ce": opt_ce, "disentangle": opt_dis},
                    self.device,
                    lambda_mi=float(getattr(self.cfg, "fedriei_lambda_mi", 1.2)),
                    lambda_ie=float(getattr(self.cfg, "fedriei_lambda_ie", 1.2)),
                )
                out = _forward_outputs(local_model, x)
                pred = out["tx_logits"].argmax(dim=1)
                correct += int((pred == y).sum().item())
                seen += int(y.numel())
                for key, value in metrics.items():
                    if isinstance(value, (float, int)):
                        totals[key] = totals.get(key, 0.0) + float(value) * int(y.numel())
        local_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in local_model.state_dict().items())
        grad_gen = torch.Generator(device="cpu")
        grad_gen.manual_seed(int(getattr(self.cfg, "seed", 0)) + int(round_idx) * 7919 + sum(ord(ch) for ch in str(client_id)))
        compression = normalize_compression_name(str(getattr(self.cfg, "fedriei_gradient_compression", "none")))
        gradient = compressed_gradient_from_states(
            start_state,
            local_state,
            client_lr=float(getattr(self.cfg, "lr", 0.0001)),
            method=compression,
            noise_std=float(getattr(self.cfg, "fedriei_compression_noise_std", 0.01)),
            include_keys=parameter_keys,
            generator=grad_gen,
        )
        return {
            "state": local_state,
            "gradient": gradient,
            "seen": seen,
            "loss": totals.get("ce_phase_loss_ce", 0.0) / max(1, seen),
            "acc": 100.0 * correct / max(1, seen),
            "metrics": {
                **{key: value / max(1, seen) for key, value in totals.items()},
                "fedriei_gradient_num_tensors": float(len(gradient)),
            },
        }

    def _local_train_rafl(self, client_id: str, round_idx: int) -> dict[str, Any]:
        local_model = copy.deepcopy(self.model)
        self._load_state_to_model(local_model)
        opt = self._make_optimizer(local_model, "rafl")
        seen = correct = 0
        total_loss = 0.0
        component_sums: dict[str, float] = {}
        for _epoch in range(max(1, int(getattr(self.cfg, "fl_local_epochs", 1)))):
            for batch in self.client_loaders[client_id]:
                x, y, rx = self._batch_to_xy_rx(batch)
                opt.zero_grad(set_to_none=True)
                out = _forward_outputs(
                    local_model,
                    self._rafl_spectrogram(x),
                    grl_lambda=1.0,
                )
                terms = receiver_agnostic_loss(
                    out,
                    y,
                    rx,
                    lambda_rx=float(getattr(self.cfg, "rafl_lambda_rx", 0.1)),
                    scale_receiver_loss=True,
                )
                loss = terms["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), float(getattr(self.cfg, "grad_clip", 1.0)))
                opt.step()
                pred = out["tx_logits"].argmax(dim=1)
                correct += int((pred == y).sum().item())
                seen += int(y.numel())
                total_loss += float(loss.detach().cpu()) * int(y.numel())
                for key in ("loss_tx", "loss_rx", "lambda_rx", "receiver_loss_weight"):
                    value = terms.get(key)
                    if torch.is_tensor(value):
                        component_sums[f"rafl_{key}"] = component_sums.get(f"rafl_{key}", 0.0) + float(value.detach().cpu()) * int(y.numel())
        return {
            "state": OrderedDict((k, v.detach().cpu().clone()) for k, v in local_model.state_dict().items()),
            "seen": seen,
            "loss": total_loss / max(1, seen),
            "acc": 100.0 * correct / max(1, seen),
            "metrics": {key: value / max(1, seen) for key, value in component_sums.items()},
        }

    @torch.no_grad()
    def _rafl_label_losses(self, candidate_clients: Sequence[str] | None = None) -> dict[str, dict[int, float]]:
        self.model.eval()
        max_batches = int(getattr(self.cfg, "rafl_selection_max_batches", 0))
        result: dict[str, dict[int, float]] = {}
        source_loaders = self.rafl_selection_client_loaders
        if not source_loaders:
            raise ValueError("RAFL Label Loss Driven selection requires held-out local evaluation loaders E_rxj.")
        client_ids = [cid for cid in (candidate_clients or source_loaders.keys()) if cid in source_loaders]
        for client_id in client_ids:
            loader = source_loaders[client_id]
            sums: dict[int, float] = {}
            counts: dict[int, int] = {}
            for batch_idx, batch in enumerate(loader):
                x, y, _rx = self._batch_to_xy_rx(batch)
                logits = _forward_outputs(self.model, self._rafl_spectrogram(x))["tx_logits"]
                losses = F.cross_entropy(logits, y, reduction="none")
                for label, value in zip(y.detach().cpu().tolist(), losses.detach().cpu().tolist()):
                    label_i = int(label)
                    sums[label_i] = sums.get(label_i, 0.0) + float(value)
                    counts[label_i] = counts.get(label_i, 0) + 1
                if max_batches > 0 and batch_idx + 1 >= max_batches:
                    break
            result[str(client_id)] = {label: sums[label] / max(1, counts[label]) for label in sums.keys()}
        return result

    def _train_round_fedfa(self, selected: Sequence[str]) -> dict[str, Any]:
        local_models = {cid: copy.deepcopy(self.model).to(self.device) for cid in selected}
        for model in local_models.values():
            model.load_state_dict(self.global_state, strict=False)
            model.train()
        opts = {cid: self._make_optimizer(model, "fedfa") for cid, model in local_models.items()}
        total_loss = 0.0
        total_seen = 0
        total_correct = 0
        local_epochs = max(1, int(getattr(self.cfg, "fl_local_epochs", 4)))
        for _epoch in range(local_epochs):
            iterators = {cid: iter(self.client_loaders[cid]) for cid in selected}
            while True:
                batches = {}
                for cid, iterator in iterators.items():
                    try:
                        batches[cid] = next(iterator)
                    except StopIteration:
                        batches = {}
                        break
                if not batches:
                    break
                for opt in opts.values():
                    opt.zero_grad(set_to_none=True)
                cls_losses = []
                client_features = []
                batch_seen = 0
                for cid, batch in batches.items():
                    x, y, _rx = self._batch_to_xy_rx(batch)
                    out = _forward_outputs(local_models[cid], x)
                    log_probs = out.get("log_probs", F.log_softmax(out["tx_logits"], dim=1))
                    cls = F.nll_loss(log_probs, y)
                    cls_losses.append(cls)
                    client_features.append(out["embedding"])
                    pred = out["tx_logits"].argmax(dim=1)
                    total_correct += int((pred == y).sum().item())
                    batch_seen += int(y.numel())
                peer_align_losses = peer_coral_alignment_losses(client_features)
                if len(peer_align_losses) != len(cls_losses):
                    align = pairwise_coral_alignment_loss(client_features)
                    local_objectives = [cls + float(getattr(self.cfg, "fedfa_align_lambda", 0.03)) * align for cls in cls_losses]
                else:
                    local_objectives = [
                        cls + float(getattr(self.cfg, "fedfa_align_lambda", 0.03)) * align
                        for cls, align in zip(cls_losses, peer_align_losses)
                    ]
                loss = torch.stack(local_objectives).mean()
                loss.backward()
                for model in local_models.values():
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(getattr(self.cfg, "grad_clip", 1.0)))
                for opt in opts.values():
                    opt.step()
                total_loss += float(loss.detach().cpu()) * batch_seen
                total_seen += batch_seen
        client_states = {
            cid: OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())
            for cid, model in local_models.items()
        }
        self._aggregate_client_states(client_states, selected, agg_weight="uniform")
        return {
            "client_results": {
                cid: {"state": state, "seen": self.client_num_samples[cid]}
                for cid, state in client_states.items()
            },
            "train_loss": total_loss / max(1, total_seen),
            "train_acc": 100.0 * total_correct / max(1, total_seen),
        }

    def _make_fucl_views(self, x: torch.Tensor, round_idx: int, client_id: str, batch_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        gen1 = torch.Generator(device=x.device)
        gen2 = torch.Generator(device=x.device)
        client_hash = sum(ord(ch) for ch in str(client_id))
        seed = int(getattr(self.cfg, "seed", 0)) + int(round_idx) * 1009 + int(batch_idx) * 131 + client_hash * 17
        gen1.manual_seed(seed + 17)
        gen2.manual_seed(seed + 31)
        return make_two_channel_views(
            safe_iq_tensor(x),
            generator_a=gen1,
            generator_b=gen2,
            tdl_config=self._fucl_tdl_config(),
            **self._fucl_cis_kwargs(),
        )

    @torch.no_grad()
    def _fucl_contrastive_val_loss(self, round_idx: int) -> float:
        self.model.eval()
        losses = []
        max_batches = int(getattr(self.cfg, "eval_max_batches", 0))
        for batch_idx, batch in enumerate(self.val_loader):
            x, _y = self._batch_to_xy(batch)
            if int(x.size(0)) < 2:
                continue
            v1, v2 = self._make_fucl_views(x, round_idx, "val", batch_idx)
            z1 = _forward_outputs(self.model, v1)["projection"]
            z2 = _forward_outputs(self.model, v2)["projection"]
            loss = nt_xent_loss(z1, z2, temperature=float(getattr(self.cfg, "fucl_temperature", 0.05)))
            losses.append(float(loss.detach().cpu()))
            if max_batches > 0 and batch_idx + 1 >= max_batches:
                break
        if not losses:
            return float("nan")
        return sum(losses) / float(len(losses))

    @torch.no_grad()
    def _fucl_client_validation_loss(self, local_model: nn.Module, client_id: str, round_idx: int, epoch_idx: int) -> float:
        loader = self.fucl_validation_client_loaders.get(str(client_id))
        if loader is None:
            return float("nan")
        was_training = local_model.training
        local_model.eval()
        losses = []
        max_batches = int(getattr(self.cfg, "fucl_validation_max_batches", 0))
        if max_batches <= 0:
            max_batches = int(getattr(self.cfg, "eval_max_batches", 0))
        for batch_idx, batch in enumerate(loader):
            x, _y, _rx = self._batch_to_xy_rx(batch)
            if int(x.size(0)) < 2:
                continue
            v1, v2 = self._make_fucl_views(x, round_idx + int(epoch_idx), f"{client_id}_val", batch_idx)
            z1 = _forward_outputs(local_model, v1)["projection"]
            z2 = _forward_outputs(local_model, v2)["projection"]
            loss = nt_xent_loss(z1, z2, temperature=float(getattr(self.cfg, "fucl_temperature", 0.05)))
            losses.append(float(loss.detach().cpu()))
            if max_batches > 0 and batch_idx + 1 >= max_batches:
                break
        if was_training:
            local_model.train()
        if not losses:
            return float("nan")
        return sum(losses) / float(len(losses))

    @torch.no_grad()
    def _fucl_client_supervised_validation_loss(self, local_model: nn.Module, client_id: str, epoch_idx: int) -> float:
        loader = self.fucl_validation_client_loaders.get(str(client_id))
        if loader is None:
            return float("nan")
        was_training = local_model.training
        local_model.eval()
        losses = []
        max_batches = int(getattr(self.cfg, "fucl_validation_max_batches", 0))
        if max_batches <= 0:
            max_batches = int(getattr(self.cfg, "eval_max_batches", 0))
        for batch_idx, batch in enumerate(loader):
            x, y, _rx = self._batch_to_xy_rx(batch)
            x_val = self._fucl_signal_representation(x)
            logits = _forward_outputs(local_model, x_val)["tx_logits"]
            losses.append(float(F.cross_entropy(logits, y).detach().cpu()))
            if max_batches > 0 and batch_idx + 1 >= max_batches:
                break
        if was_training:
            local_model.train()
        if not losses:
            return float("nan")
        return sum(losses) / float(len(losses))

    def _local_train_fucl_pretrain(self, client_id: str, round_idx: int) -> dict[str, Any]:
        local_model = copy.deepcopy(self.model)
        self._load_state_to_model(local_model)
        opt = self._make_optimizer(local_model, "fucl_pretrain")
        max_epochs_override = int(getattr(self.cfg, "fucl_local_max_epochs", 0))
        local_epochs = max(1, max_epochs_override if max_epochs_override > 0 else int(getattr(self.cfg, "fl_local_epochs", 1)))
        lr_patience = max(1, int(getattr(self.cfg, "fucl_local_lr_patience", 10)))
        lr_decay = float(getattr(self.cfg, "fucl_local_lr_decay", 0.1))
        early_patience = max(1, int(getattr(self.cfg, "fucl_local_early_stop_patience", 20)))
        best_val_loss = float("inf")
        bad_epochs = 0
        lr_reductions = 0
        epochs_run = 0
        total_loss = 0.0
        seen = 0
        val_history: list[float] = []
        early_stopped = False
        for epoch_idx in range(1, local_epochs + 1):
            local_model.train()
            for batch_idx, batch in enumerate(self.client_loaders[client_id]):
                x, _y, _rx = self._batch_to_xy_rx(batch)
                if int(x.size(0)) < 2:
                    continue
                v1, v2 = self._make_fucl_views(x, round_idx, client_id, batch_idx + (epoch_idx - 1) * 100000)
                opt.zero_grad(set_to_none=True)
                z1 = _forward_outputs(local_model, v1)["projection"]
                z2 = _forward_outputs(local_model, v2)["projection"]
                loss = nt_xent_loss(z1, z2, temperature=float(getattr(self.cfg, "fucl_temperature", 0.05)))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), float(getattr(self.cfg, "grad_clip", 1.0)))
                opt.step()
                seen += int(x.size(0))
                total_loss += float(loss.detach().cpu()) * int(x.size(0))
            epochs_run = epoch_idx
            val_loss = self._fucl_client_validation_loss(local_model, client_id, round_idx, epoch_idx)
            val_history.append(float(val_loss))
            if math.isfinite(val_loss):
                if val_loss < best_val_loss - 1e-12:
                    best_val_loss = float(val_loss)
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs > 0 and bad_epochs % lr_patience == 0:
                        for group in opt.param_groups:
                            group["lr"] = float(group.get("lr", 0.0)) * lr_decay
                        lr_reductions += 1
                    if bad_epochs >= early_patience:
                        early_stopped = True
                        break
        local_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in local_model.state_dict().items())
        return {
            "state": local_state,
            "seen": seen,
            "loss": total_loss / max(1, seen),
            "metrics": {
                "epochs_run": int(epochs_run),
                "early_stopped": bool(early_stopped),
                "lr_reductions": int(lr_reductions),
                "best_val_loss": float(best_val_loss),
                "final_lr": float(opt.param_groups[0].get("lr", float("nan"))),
                "val_history": val_history,
                "validation_samples": len(self.fucl_validation_client_splits.get(str(client_id), [])),
            },
        }

    def _train_fucl_pretrain_round(self, selected: Sequence[str], round_idx: int) -> dict[str, Any]:
        client_states = {}
        client_metrics = {}
        total_loss = 0.0
        total_seen = 0
        for client_id in selected:
            result = self._local_train_fucl_pretrain(client_id, round_idx)
            client_states[client_id] = result["state"]
            client_metrics[str(client_id)] = result["metrics"]
            seen = int(result.get("seen", 0))
            total_loss += float(result.get("loss", 0.0)) * seen
            total_seen += seen
        self._aggregate_encoder_only(client_states)
        return {
            "train_loss": total_loss / max(1, total_seen),
            "train_acc": float("nan"),
            "components": {
                "fucl_client_pretrain": client_metrics,
                "fucl_local_validation_ratio": float(getattr(self.cfg, "fucl_local_validation_ratio", 0.1)),
                "fucl_lr_patience": int(getattr(self.cfg, "fucl_local_lr_patience", 10)),
                "fucl_early_stop_patience": int(getattr(self.cfg, "fucl_local_early_stop_patience", 20)),
            },
        }

    def _train_fucl_finetune(self) -> dict[str, Any]:
        selected = list(self.client_splits.keys())
        client_states = {}
        client_eval: dict[str, dict[str, Any]] = {}
        client_metrics: dict[str, dict[str, Any]] = {}
        total_loss = 0.0
        total_seen = total_correct = 0
        max_epochs_override = int(getattr(self.cfg, "fucl_local_max_epochs", 0))
        epochs = max(1, max_epochs_override if max_epochs_override > 0 else int(getattr(self.cfg, "fucl_finetune_epochs", 1)))
        lr_patience = max(1, int(getattr(self.cfg, "fucl_local_lr_patience", 10)))
        lr_decay = float(getattr(self.cfg, "fucl_local_lr_decay", 0.1))
        early_patience = max(1, int(getattr(self.cfg, "fucl_local_early_stop_patience", 20)))
        for client_id in selected:
            local_model = copy.deepcopy(self.model)
            self._load_state_to_model(local_model)
            opt = self._make_optimizer(local_model, "fucl_finetune")
            best_val_loss = float("inf")
            bad_epochs = 0
            lr_reductions = 0
            epochs_run = 0
            val_history: list[float] = []
            early_stopped = False
            for _epoch in range(1, epochs + 1):
                for batch_idx, batch in enumerate(self.client_loaders[client_id]):
                    x, y, _rx = self._batch_to_xy_rx(batch)
                    x_aug, _ = self._make_fucl_views(
                        x,
                        max(1, int(getattr(self.cfg, "fl_rounds", 1))) + _epoch + 1,
                        client_id,
                        batch_idx,
                    )
                    opt.zero_grad(set_to_none=True)
                    logits = _forward_outputs(local_model, x_aug)["tx_logits"]
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), float(getattr(self.cfg, "grad_clip", 1.0)))
                    opt.step()
                    pred = logits.argmax(dim=1)
                    total_correct += int((pred == y).sum().item())
                    total_seen += int(y.numel())
                    total_loss += float(loss.detach().cpu()) * int(y.numel())
                epochs_run = int(_epoch)
                val_loss = self._fucl_client_supervised_validation_loss(local_model, client_id, int(_epoch))
                val_history.append(float(val_loss))
                if math.isfinite(val_loss):
                    if val_loss < best_val_loss - 1e-12:
                        best_val_loss = float(val_loss)
                        bad_epochs = 0
                    else:
                        bad_epochs += 1
                        if bad_epochs > 0 and bad_epochs % lr_patience == 0:
                            for group in opt.param_groups:
                                group["lr"] = float(group.get("lr", 0.0)) * lr_decay
                            lr_reductions += 1
                        if bad_epochs >= early_patience:
                            early_stopped = True
                            break
            state = OrderedDict((k, v.detach().cpu().clone()) for k, v in local_model.state_dict().items())
            client_states[client_id] = state
            self.fucl_client_states[str(client_id)] = state
            client_metrics[str(client_id)] = {
                "epochs_run": int(epochs_run),
                "early_stopped": bool(early_stopped),
                "lr_reductions": int(lr_reductions),
                "best_val_loss": float(best_val_loss),
                "final_lr": float(opt.param_groups[0].get("lr", float("nan"))),
                "val_history": val_history,
                "validation_samples": len(self.fucl_validation_client_splits.get(str(client_id), [])),
            }
            local_correct = local_total = 0
            local_model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.client_loaders[client_id]):
                    x, y, _rx = self._batch_to_xy_rx(batch)
                    logits = _forward_outputs(local_model, self._fucl_signal_representation(x))["tx_logits"]
                    local_correct += int((logits.argmax(dim=1) == y).sum().item())
                    local_total += int(y.numel())
                    if int(getattr(self.cfg, "eval_max_batches", 0)) > 0 and batch_idx + 1 >= int(getattr(self.cfg, "eval_max_batches", 0)):
                        break
            client_eval[str(client_id)] = {
                "tx_acc": 100.0 * local_correct / max(1, local_total),
                "tx_correct": int(local_correct),
                "tx_total": int(local_total),
            }
        self._aggregate_client_states(client_states, selected, agg_weight="num_samples")
        return {
            "train_loss": total_loss / max(1, total_seen),
            "train_acc": 100.0 * total_correct / max(1, total_seen),
            "fucl_finetune_paper_eval_mode": "client_specific_classification_nns",
            "fucl_client_specific_eval": client_eval,
            "fucl_client_finetune": client_metrics,
            "fucl_finetune_adapter": "client_specific_classification_nns_plus_sample_weighted_global_classifier_for_cvs_adapter_diagnostic",
            "fucl_paper_protocol_deviation": "CVS common held-out evaluation uses an aggregate classifier only as an adapter diagnostic; strict FUCL paper result is client-specific.",
            "fucl_finetune_augmentation": "tdl_channel_simulator_plus_channel_independent_spectrogram",
        }

    def _train_standard_round(self, selected: Sequence[str], round_idx: int) -> dict[str, Any]:
        client_results = {}
        for client_id in selected:
            if self.mode == "fedriei":
                client_results[client_id] = self._local_train_fedriei(client_id, round_idx)
            elif self.mode == "rafl":
                client_results[client_id] = self._local_train_rafl(client_id, round_idx)
            else:
                raise ValueError(f"Unexpected standard Fedbase mode: {self.mode}")
        if self.mode == "fedriei":
            self._aggregate_fedriei_gradients({cid: result["gradient"] for cid, result in client_results.items()}, selected)
        else:
            self._aggregate_client_states({cid: result["state"] for cid, result in client_results.items()}, selected, agg_weight="num_samples")
        seen = sum(int(result.get("seen", 0)) for result in client_results.values())
        loss = sum(float(result.get("loss", 0.0)) * int(result.get("seen", 0)) for result in client_results.values()) / max(1, seen)
        acc = sum(float(result.get("acc", 0.0)) * int(result.get("seen", 0)) for result in client_results.values()) / max(1, seen)
        component_sums: dict[str, float] = {}
        for result in client_results.values():
            weight = int(result.get("seen", 0))
            for key, value in (result.get("metrics", {}) or {}).items():
                component_sums[key] = component_sums.get(key, 0.0) + float(value) * weight
        components = {key: value / max(1, seen) for key, value in component_sums.items()}
        if self.mode == "rafl" and hasattr(self, "_last_rafl_selection"):
            components["rafl_selection"] = getattr(self, "_last_rafl_selection")
        if self.mode == "fedriei":
            components["fedriei_gradient_compression"] = normalize_compression_name(
                str(getattr(self.cfg, "fedriei_gradient_compression", "none"))
            )
            components["fedriei_server_update"] = "weighted_compressed_gradient_step"
        return {"client_results": client_results, "train_loss": loss, "train_acc": acc, "components": components}

    def train(self) -> dict[str, Any]:
        best_val = -float("inf")
        best_primary = float("nan")
        best_round = 0
        final_primary = float("nan")
        best_pretrain_loss = float("inf")
        best_pretrain_round = 0
        best_pretrain_state: OrderedDict[str, torch.Tensor] | None = None
        rounds = max(1, int(getattr(self.cfg, "fl_rounds", 1)))
        paper_eval_last_n = max(0, int(getattr(self.cfg, "fl_test_eval_last_n", 0)))
        paper_eval_records: list[dict[str, Any]] = []
        last_row: dict[str, Any] = {}
        for round_idx in range(1, rounds + 1):
            selected = self._selected_clients(round_idx)
            if self.mode == "fedfa":
                train_result = self._train_round_fedfa(selected)
            elif self.mode == "fucl":
                train_result = self._train_fucl_pretrain_round(selected, round_idx)
            else:
                train_result = self._train_standard_round(selected, round_idx)

            val_stats = self._evaluate_loader(self.val_loader, max_batches=int(getattr(self.cfg, "eval_max_batches", 0)))
            val_acc = float(val_stats.get("tx_acc", float("nan")))
            components = dict(train_result.get("components", {}) or {})
            if self.mode == "fucl":
                pretrain_classifier_val_acc = val_acc
                pretrain_val_loss = self._fucl_contrastive_val_loss(round_idx)
                components["fucl_pretrain_classifier_val_tx_acc_diagnostic"] = pretrain_classifier_val_acc
                components["fucl_pretrain_val_loss"] = pretrain_val_loss
                val_stats = dict(val_stats)
                val_stats["tx_acc"] = float("nan")
                val_stats["fucl_pretrain_classifier_val_tx_acc_diagnostic"] = pretrain_classifier_val_acc
                val_acc = float("nan")
                val_improved = False
                if math.isfinite(pretrain_val_loss) and pretrain_val_loss < best_pretrain_loss:
                    best_pretrain_loss = pretrain_val_loss
                    best_pretrain_round = int(round_idx)
                    best_pretrain_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in self.global_state.items())
            else:
                val_improved = math.isfinite(val_acc) and val_acc > best_val
            is_final_round = int(round_idx) >= rounds
            scheduled_test = False if self.mode == "fucl" else self._should_run_eval(round_idx)
            test_reasons = []
            if val_improved:
                test_reasons.append("val_improved")
            if scheduled_test and not is_final_round:
                test_reasons.append("scheduled")
            if is_final_round and self.mode != "fucl":
                test_reasons.append("final")
            should_test = bool(test_reasons)
            named_stats = self._evaluate_named() if should_test else {}
            sat_stats = self._evaluate_sat_named() if should_test else {}
            primary_acc = self._primary_test_acc(named_stats) if should_test else float("nan")
            if is_final_round:
                final_primary = primary_acc
            row = {
                "round": int(round_idx),
                "train_mode": self.mode,
                "selected_clients": ",".join(selected),
                "train_loss": float(train_result.get("train_loss", float("nan"))),
                "train_acc": float(train_result.get("train_acc", float("nan"))),
                "rafl_loss_tx": float(components.get("rafl_loss_tx", float("nan"))),
                "rafl_loss_rx": float(components.get("rafl_loss_rx", float("nan"))),
                "rafl_lambda_rx": float(components.get("rafl_lambda_rx", float("nan"))),
                "rafl_receiver_loss_weight": float(components.get("rafl_receiver_loss_weight", float("nan"))),
                "fedriei_loss_ce": float(components.get("ce_phase_loss_ce", float("nan"))),
                "fedriei_loss_mi": float(components.get("disentangle_phase_loss_mi", float("nan"))),
                "fedriei_loss_ie": float(components.get("disentangle_phase_loss_ie", float("nan"))),
                "fedriei_loss_dis": float(components.get("disentangle_phase_loss", float("nan"))),
                "val_tx_acc": val_acc,
                "primary_test_tx_acc": float(primary_acc),
                "named_test_tx_acc_json": self._tx_acc_json(named_stats) if should_test else "{}",
                "sat_channel_tx_acc_json": self._sat_tx_acc_json(sat_stats) if should_test else "{}",
                "test_trigger": "+".join(test_reasons) if test_reasons else "none",
            }
            self._append_metrics(row)
            self._append_log({**row, "val": val_stats, "named_tests": named_stats, "sat_channel": sat_stats, "components": components})
            if should_test and paper_eval_last_n > 0 and int(round_idx) > rounds - paper_eval_last_n:
                paper_eval_records.append(
                    {
                        "round": int(round_idx),
                        "primary_test_tx_acc": float(primary_acc),
                        "named_tests": named_stats,
                    }
                )
            last_row = row
            if val_improved:
                best_val = val_acc
                best_primary = primary_acc
                best_round = round_idx
                torch.save(
                    {
                        "model": self.global_state,
                        "round": round_idx,
                        "mode": self.mode,
                        "best_selection_metric": "val_tx_acc",
                        "best_val_tx_acc": best_val,
                        "primary_test_tx_acc": primary_acc,
                    },
                    self.checkpoint_path,
                )
            loss_terms = ""
            if self.mode == "rafl":
                loss_terms = f" tx_loss={row['rafl_loss_tx']:.4f} rx_loss={row['rafl_loss_rx']:.4f}"
            elif self.mode == "fedriei":
                loss_terms = (
                    f" ce={row['fedriei_loss_ce']:.4f} mi={row['fedriei_loss_mi']:.4f} "
                    f"ie={row['fedriei_loss_ie']:.4f} dis={row['fedriei_loss_dis']:.4f}"
                )
            print(
                f"[FEDBASE][R{round_idx:03d}] mode={self.mode} clients={len(selected)}/{len(self.client_splits)} "
                f"loss={row['train_loss']:.4f}{loss_terms} train_tx={row['train_acc']:.2f} "
                f"val_tx={row['val_tx_acc']:.2f} primary_tx={row['primary_test_tx_acc']:.2f} "
                f"test_trigger={row['test_trigger']}",
                flush=True,
            )
            if should_test:
                round_label = f"{round_idx:03d}"
                for line in self._format_named_test_stdout(named_stats, round_label=round_label):
                    print(line, flush=True)
                for line in self._format_sat_test_stdout(sat_stats, round_label=round_label):
                    print(line, flush=True)

        if self.mode == "fucl":
            if best_pretrain_state is not None:
                self.global_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in best_pretrain_state.items())
                self.model.load_state_dict(self.global_state, strict=False)
                self.model.to(self.device)
            finetune = self._train_fucl_finetune()
            aggregate_val_stats = self._evaluate_loader(self.val_loader, max_batches=int(getattr(self.cfg, "eval_max_batches", 0)))
            val_stats = (
                self._evaluate_fucl_client_specific_loader(self.val_loader, max_batches=int(getattr(self.cfg, "eval_max_batches", 0)))
                if self.fucl_client_states
                else aggregate_val_stats
            )
            val_acc = float(val_stats.get("tx_acc", float("nan")))
            val_improved = math.isfinite(val_acc) and val_acc > best_val
            aggregate_named_stats = self._evaluate_named()
            named_stats = self._evaluate_fucl_client_specific_named() if self.fucl_client_states else aggregate_named_stats
            aggregate_sat_stats = self._evaluate_sat_named()
            sat_stats = self._evaluate_fucl_client_specific_sat_named() if self.fucl_client_states else aggregate_sat_stats
            primary_acc = self._primary_test_acc(named_stats)
            aggregate_primary_acc = self._primary_test_acc(aggregate_named_stats)
            final_primary = primary_acc
            row = {
                "round": "finetune",
                "train_mode": self.mode,
                "selected_clients": "all",
                "train_loss": float(finetune.get("train_loss", float("nan"))),
                "train_acc": float(finetune.get("train_acc", float("nan"))),
                "val_tx_acc": val_acc,
                "primary_test_tx_acc": float(primary_acc),
                "named_test_tx_acc_json": self._tx_acc_json(named_stats),
                "fucl_common_aggregate_val_tx_acc": float(aggregate_val_stats.get("tx_acc", float("nan"))),
                "fucl_common_aggregate_primary_test_tx_acc": float(aggregate_primary_acc),
                "fucl_common_aggregate_named_test_tx_acc_json": self._tx_acc_json(aggregate_named_stats),
                "sat_channel_tx_acc_json": self._sat_tx_acc_json(sat_stats),
                "test_trigger": "val_improved+final" if val_improved else "final",
            }
            self._append_metrics(row)
            self._append_log(
                {
                    **row,
                    "val": val_stats,
                    "named_tests": named_stats,
                    "sat_channel": sat_stats,
                    "fucl_stage": "supervised_finetune",
                    "fucl_adapter": finetune,
                    "fucl_common_aggregate_val": aggregate_val_stats,
                    "fucl_common_aggregate_named_tests": aggregate_named_stats,
                    "fucl_common_aggregate_sat_channel": aggregate_sat_stats,
                }
            )
            last_row = row
            if val_improved or best_round == 0:
                best_val = val_acc
                best_primary = primary_acc
                best_round = "finetune"
                torch.save(
                    {
                        "model": self.global_state,
                        "round": "finetune",
                        "mode": self.mode,
                        "best_selection_metric": "val_tx_acc",
                        "best_val_tx_acc": best_val,
                        "primary_test_tx_acc": primary_acc,
                        "fucl_common_aggregate_primary_test_tx_acc": aggregate_primary_acc,
                        "fucl_final_eval_mode": "client_specific_seen_receiver_else_source_client_ensemble",
                        "fucl_client_states": self.fucl_client_states,
                    },
                    self.checkpoint_path,
                )
            print(
                f"[FEDBASE][FINETUNE] mode={self.mode} clients=all loss={row['train_loss']:.4f} "
                f"train_tx={row['train_acc']:.2f} val_tx={row['val_tx_acc']:.2f} "
                f"primary_tx={row['primary_test_tx_acc']:.2f} "
                f"aggregate_primary_tx={row['fucl_common_aggregate_primary_test_tx_acc']:.2f} "
                f"test_trigger={row['test_trigger']}",
                flush=True,
            )
            for line in self._format_named_test_stdout(named_stats, round_label="FINETUNE"):
                print(line, flush=True)
            for name in sorted(aggregate_named_stats):
                stats = aggregate_named_stats.get(name, {}) or {}
                print(
                    f"[FEDBASE-FUCL-AGG][FINETUNE] split={name} "
                    f"tx_acc={float(stats.get('tx_acc', float('nan'))):.2f} "
                    f"tx_total={stats.get('tx_total', 'na')}",
                    flush=True,
                )
            for line in self._format_sat_test_stdout(sat_stats, round_label="FINETUNE"):
                print(line, flush=True)

        summary = {
            "train_mode": self.mode,
            "rounds": rounds,
            "num_clients": len(self.client_splits),
            "best_selection_metric": "val_tx_acc",
            "best_round": best_round,
            "best_val_tx_acc": best_val,
            "best_primary_test_tx_acc": best_primary,
            "final_primary_test_tx_acc": final_primary,
            "last": last_row,
            "output_dir": str(self.output_dir),
            "log_dir": str(self.log_dir),
            "checkpoint_path": str(self.checkpoint_path),
        }
        if self.mode == "fucl":
            summary["fucl_best_pretrain_round"] = best_pretrain_round
            summary["fucl_best_pretrain_val_loss"] = best_pretrain_loss
            summary["fucl_final_eval_mode"] = "client_specific_seen_receiver_else_source_client_ensemble"
            summary["fucl_client_state_checkpointed"] = bool(self.fucl_client_states)
            summary["fucl_client_ids"] = sorted(str(cid) for cid in self.fucl_client_states)
            summary["fucl_common_aggregate_primary_test_tx_acc"] = float(last_row.get("fucl_common_aggregate_primary_test_tx_acc", float("nan")))
            summary["fucl_common_aggregate_val_tx_acc"] = float(last_row.get("fucl_common_aggregate_val_tx_acc", float("nan")))
        if paper_eval_records:
            summary["paper_eval_window"] = self._summarize_paper_eval_records(
                paper_eval_records,
                name=f"final{paper_eval_last_n}",
            )
        with self.summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary
