from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch

from cvsrffi.phase1_fcr_types import FCRV2Metadata


@dataclass(frozen=True)
class FCRV2PairBuilder:
    crop_span: int = 256

    def build(self, metadata: FCRV2Metadata, *, epoch: int, seed: int) -> dict[str, torch.Tensor]:
        if int(self.crop_span) < 1:
            raise ValueError("crop_span must be >= 1")
        return {
            "nuisance": self._build_axis(metadata, axis="nuisance", epoch=epoch, seed=seed),
            "content": self._build_axis(metadata, axis="content", epoch=epoch, seed=seed),
            "fingerprint": self._build_axis(metadata, axis="fingerprint", epoch=epoch, seed=seed),
        }

    def _build_axis(self, metadata: FCRV2Metadata, *, axis: str, epoch: int, seed: int) -> torch.Tensor:
        pairs: list[list[int]] = []
        for source in range(metadata.batch_size):
            candidates = [target for target in range(metadata.batch_size) if self._eligible(metadata, axis, source, target)]
            if not candidates:
                continue
            ordered = sorted(candidates, key=lambda index: self._candidate_signature(metadata, index))
            pairs.append([source, ordered[self._anchor_hash(metadata, source, axis=axis, epoch=epoch, seed=seed) % len(ordered)]])
        if not pairs:
            return torch.empty((0, 2), dtype=torch.long)
        return torch.tensor(pairs, dtype=torch.long)

    def _anchor_hash(self, metadata: FCRV2Metadata, index: int, *, axis: str, epoch: int, seed: int) -> int:
        payload = (
            f"{seed}:{epoch}:{axis}:{metadata.physical_sample_id[index]}:{metadata.view_type[index]}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")

    @staticmethod
    def _candidate_signature(metadata: FCRV2Metadata, index: int) -> tuple[object, ...]:
        return (
            metadata.physical_sample_id[index],
            metadata.content_record_id[index],
            int(metadata.crop_offset[index]),
            metadata.common_preamble_id[index],
            int(metadata.tx_id[index]),
            int(metadata.rx_i[index]),
            int(metadata.day_i[index]),
            metadata.view_type[index],
            metadata.link_condition[index],
            int(metadata.excitation_bin[index]),
        )

    def _eligible(self, metadata: FCRV2Metadata, axis: str, source: int, target: int) -> bool:
        if source == target:
            return False
        if axis == "nuisance":
            return self._nuisance_pair(metadata, source, target)
        if axis == "content":
            return self._content_pair(metadata, source, target)
        if axis == "fingerprint":
            return self._fingerprint_pair(metadata, source, target)
        raise ValueError(f"unknown pairing axis: {axis}")

    @staticmethod
    def _nuisance_pair(metadata: FCRV2Metadata, source: int, target: int) -> bool:
        if int(metadata.tx_id[source]) != int(metadata.tx_id[target]):
            return False
        if metadata.content_record_id[source] != metadata.content_record_id[target]:
            return False
        if int(metadata.crop_offset[source]) != int(metadata.crop_offset[target]):
            return False
        source_clean = metadata.view_type[source] == "clean"
        target_clean = metadata.view_type[target] == "clean"
        if source_clean == target_clean:
            return False
        satellite = target if source_clean else source
        valid = metadata.eta_valid_mask[satellite]
        if not bool(valid.any()):
            return False
        return bool((metadata.eta[satellite][valid].abs() > 0).any())

    def _content_pair(self, metadata: FCRV2Metadata, source: int, target: int) -> bool:
        if int(metadata.tx_id[source]) < 0 or int(metadata.tx_id[target]) < 0:
            return False
        if int(metadata.tx_id[source]) != int(metadata.tx_id[target]):
            return False
        if int(metadata.rx_i[source]) != int(metadata.rx_i[target]):
            return False
        if int(metadata.day_i[source]) != int(metadata.day_i[target]):
            return False
        if metadata.link_condition[source] != metadata.link_condition[target]:
            return False
        if metadata.view_type[source] != metadata.view_type[target]:
            return False
        if metadata.content_record_id[source] != metadata.content_record_id[target]:
            overlap = 0.0
        else:
            delta = abs(int(metadata.crop_offset[source]) - int(metadata.crop_offset[target]))
            overlap = max(0, int(self.crop_span) - delta) / float(self.crop_span)
        return overlap <= 0.25

    @staticmethod
    def _fingerprint_pair(metadata: FCRV2Metadata, source: int, target: int) -> bool:
        if int(metadata.tx_id[source]) < 0 or int(metadata.tx_id[target]) < 0:
            return False
        if not metadata.common_preamble_id[source] or not metadata.common_preamble_id[target]:
            return False
        if int(metadata.excitation_bin[source]) < 0 or int(metadata.excitation_bin[target]) < 0:
            return False
        if metadata.common_preamble_id[source] != metadata.common_preamble_id[target]:
            return False
        if int(metadata.rx_i[source]) != int(metadata.rx_i[target]):
            return False
        if int(metadata.day_i[source]) != int(metadata.day_i[target]):
            return False
        if metadata.link_condition[source] != metadata.link_condition[target]:
            return False
        if metadata.view_type[source] != metadata.view_type[target]:
            return False
        if int(metadata.excitation_bin[source]) != int(metadata.excitation_bin[target]):
            return False
        return int(metadata.tx_id[source]) != int(metadata.tx_id[target])
