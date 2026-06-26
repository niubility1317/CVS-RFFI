from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import torch

from .style_packet import StylePacket


@dataclass
class StyleCentroid:
    style_id: int
    client_id: str
    round_idx: int
    count: int
    stats: dict[str, Any]
    vector: torch.Tensor = field(repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)
    age: int = 0

    def as_packet(self) -> StylePacket:
        meta = dict(self.metadata)
        meta.update({"age": self.age, "centroid": True})
        return StylePacket(
            client_id=self.client_id,
            round_idx=self.round_idx,
            count=self.count,
            stats=dict(self.stats),
            style_id=self.style_id,
            metadata=meta,
        )


class FederatedStyleBank:
    """Small EMA bank of class-marginalized client style centroids."""

    def __init__(self, *, momentum: float = 0.5, max_centroids: int = 64, merge_radius: float = 0.0):
        self.momentum = max(0.0, min(0.9999, float(momentum)))
        self.max_centroids = max(1, int(max_centroids))
        self.merge_radius = max(0.0, float(merge_radius))
        self._centroids: list[StyleCentroid] = []
        self._next_id = 0
        self._packets_seen = 0
        self._stat_keys: tuple[str, ...] = ()

    @property
    def centroids(self) -> tuple[StyleCentroid, ...]:
        return tuple(self._centroids)

    def update(self, packets: Iterable[StylePacket]) -> dict[str, Any]:
        packets = list(packets)
        added = 0
        merged = 0
        for centroid in self._centroids:
            centroid.age += 1
        for packet in packets:
            self._packets_seen += 1
            self._ensure_stat_keys(packet.stats)
            vec = self._vector_from_stats(packet.stats)
            target = self._nearest(vec, packet.metadata) if self.merge_radius > 0.0 else None
            if target is not None and float(torch.linalg.vector_norm(target.vector - vec).item()) <= self.merge_radius:
                target.stats = _blend_stats(target.stats, packet.stats, self.momentum)
                target.metadata = _merge_metadata(target.metadata, dict(packet.metadata or {}))
                self._ensure_stat_keys(target.stats)
                target.vector = self._vector_from_stats(target.stats)
                target.count += int(packet.count)
                target.round_idx = int(packet.round_idx)
                target.age = 0
                merged += 1
            else:
                self._centroids.append(
                    StyleCentroid(
                        style_id=self._next_id,
                        client_id=str(packet.client_id),
                        round_idx=int(packet.round_idx),
                        count=int(packet.count),
                        stats=dict(packet.stats),
                        vector=vec,
                        metadata=dict(packet.metadata or {}),
                        age=0,
                    )
                )
                self._next_id += 1
                added += 1
            self._trim()
        return {
            "num_centroids": len(self._centroids),
            "num_packets_seen": self._packets_seen,
            "added": added,
            "merged": merged,
            "size_bytes": self.size_bytes(),
        }

    def _nearest(self, vec: torch.Tensor, metadata: Optional[dict[str, Any]] = None) -> Optional[StyleCentroid]:
        if not self._centroids:
            return None
        vec = vec.detach().cpu().float()
        candidates = [c for c in self._centroids if _metadata_compatible(c.metadata, metadata or {})]
        if not candidates:
            return None
        return min(candidates, key=lambda c: float(torch.linalg.vector_norm(c.vector - vec).item()))

    def _trim(self) -> None:
        if len(self._centroids) <= self.max_centroids:
            return
        self._centroids.sort(key=lambda c: (int(c.count), -int(c.age), int(c.round_idx)), reverse=True)
        del self._centroids[self.max_centroids :]

    def _ensure_stat_keys(self, stats: dict[str, Any]) -> None:
        keys = set(self._stat_keys)
        for key, value in stats.items():
            try:
                float(value)
            except (TypeError, ValueError):
                continue
            keys.add(str(key))
        new_keys = tuple(sorted(keys))
        if new_keys == self._stat_keys:
            return
        self._stat_keys = new_keys
        for centroid in self._centroids:
            centroid.vector = self._vector_from_stats(centroid.stats)

    def _vector_from_stats(self, stats: dict[str, Any]) -> torch.Tensor:
        if not self._stat_keys:
            return torch.zeros(1, dtype=torch.float32)
        vals = []
        for key in self._stat_keys:
            try:
                vals.append(_style_distance_value(str(key), float(stats.get(key, 0.0))))
            except (TypeError, ValueError):
                vals.append(0.0)
        return torch.tensor(vals, dtype=torch.float32)

    def sample_remote_styles(
        self,
        *,
        exclude_client_id: str,
        k: int = 1,
        preferred_keys: Optional[Sequence[str]] = None,
        policy: str = "diverse",
    ) -> tuple[StylePacket, ...]:
        del preferred_keys
        candidates = [c for c in self._centroids if c.client_id != str(exclude_client_id)]
        if not candidates:
            return tuple()
        policy = str(policy or "diverse").lower()
        if policy in {"target_balanced", "balanced_target", "balanced_receiver", "receiver_balanced"}:
            return tuple(c.as_packet() for c in _select_target_balanced(candidates, max(1, int(k))))
        selected: list[StyleCentroid] = []
        remaining = sorted(candidates, key=lambda c: (c.count, -c.age, c.round_idx), reverse=True)
        while remaining and len(selected) < max(1, int(k)):
            if not selected:
                chosen = remaining.pop(0)
            else:
                chosen = max(
                    remaining,
                    key=lambda c: (
                        min(float(torch.linalg.vector_norm(c.vector - s.vector).item()) for s in selected),
                        c.count,
                        -c.age,
                    ),
                )
                remaining.remove(chosen)
            selected.append(chosen)
        return tuple(c.as_packet() for c in selected)

    def sample_remote_style(
        self,
        *,
        exclude_client_id: str,
        preferred_keys: Optional[Sequence[str]] = None,
        policy: str = "diverse",
    ) -> Optional[StylePacket]:
        styles = self.sample_remote_styles(
            exclude_client_id=exclude_client_id,
            k=1,
            preferred_keys=preferred_keys,
            policy=policy,
        )
        return styles[0] if styles else None

    def diagnostics(self) -> dict[str, Any]:
        if len(self._centroids) < 2:
            mean_l2 = 0.0
        else:
            dists = []
            for i, a in enumerate(self._centroids):
                for b in self._centroids[i + 1 :]:
                    dists.append(float(torch.linalg.vector_norm(a.vector - b.vector).item()))
            mean_l2 = float(sum(dists) / max(1, len(dists)))
        return {
            "num_centroids": len(self._centroids),
            "num_packets_seen": self._packets_seen,
            "mean_pairwise_l2": mean_l2,
            "size_bytes": self.size_bytes(),
            "num_clients": len({c.client_id for c in self._centroids}),
        }

    def size_bytes(self) -> int:
        return sum(c.as_packet().size_bytes() for c in self._centroids)


def _blend_stats(old: dict[str, Any], new: dict[str, Any], momentum: float) -> dict[str, Any]:
    out: dict[str, Any] = dict(old)
    for key, value in new.items():
        try:
            cur = float(value)
            prev = float(old.get(key, cur))
            blended = prev * float(momentum) + cur * (1.0 - float(momentum))
            out[key] = blended if math.isfinite(blended) else cur
        except (TypeError, ValueError):
            out[key] = value
    return out


def _style_distance_value(key: str, value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    key = str(key)
    v = float(value)
    if key == "phys_cfo_hz":
        return _clip(v / 35000.0, -1.0, 1.0)
    if key == "phys_cfo_cycles_per_sample":
        return _clip(v / 0.49, -1.0, 1.0)
    if key == "phys_sro_ppm":
        return _clip(v / 150.0, -1.0, 1.0)
    if key == "phys_agc_gain_db":
        return _clip(v / 8.0, -1.0, 1.0)
    if key == "phys_iq_gain_imbalance_db":
        return _clip(v / 3.0, -1.0, 1.0)
    if key == "phys_iq_phase_imbalance_deg":
        return _clip(v / 3.0, -1.0, 1.0)
    if key == "phys_phase_noise_std":
        return _clip(v / 0.002, 0.0, 1.0)
    if key == "phys_awgn_snr_db":
        return _clip((80.0 - v) / 70.0, 0.0, 1.0)
    if key == "phys_multipath_strength":
        return _clip(v, 0.0, 1.0)
    if key == "phys_lowpass_cutoff_frac":
        return _clip(1.0 - v, 0.0, 1.0)
    if key == "phys_lowpass_transition_frac":
        return _clip(v / 0.20, 0.0, 1.0)
    if key == "phys_softclip_level":
        return _clip((8.0 - v) / 8.0, 0.0, 1.0)
    return _clip(v, -4.0, 4.0)


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(value))))


def _target_balance_key(centroid: StyleCentroid) -> str:
    meta = centroid.metadata or {}
    for key in ("target_domain_label", "mapped_target_domain_label", "raw_target_domain_label", "raw_domain_label"):
        if key in meta:
            return f"{key}:{meta[key]}"
    return f"client:{centroid.client_id}"


def _select_target_balanced(candidates: Sequence[StyleCentroid], k: int) -> list[StyleCentroid]:
    groups: dict[str, list[StyleCentroid]] = {}
    for centroid in candidates:
        groups.setdefault(_target_balance_key(centroid), []).append(centroid)
    for items in groups.values():
        items.sort(key=lambda c: (c.count, -c.age, c.round_idx), reverse=True)
    ordered_keys = sorted(
        groups.keys(),
        key=lambda key: (groups[key][0].count, -groups[key][0].age, groups[key][0].round_idx),
        reverse=True,
    )
    selected: list[StyleCentroid] = []
    while ordered_keys and len(selected) < max(1, int(k)):
        for key in list(ordered_keys):
            if len(selected) >= max(1, int(k)):
                break
            items = groups[key]
            if items:
                selected.append(items.pop(0))
        ordered_keys = [item_key for item_key in ordered_keys if groups[item_key]]
    return selected


def _metadata_compatible(old: dict[str, Any], new: dict[str, Any]) -> bool:
    for key in ("target_domain_label", "mapped_target_domain_label", "raw_target_domain_label", "raw_domain_label"):
        if key in old and key in new:
            try:
                return int(old[key]) == int(new[key])
            except (TypeError, ValueError):
                return str(old[key]) == str(new[key])
    return True


def _merge_metadata(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    out = dict(old)
    for key, value in new.items():
        if key in ("target_domain_label", "mapped_target_domain_label", "raw_target_domain_label", "raw_domain_label"):
            if key not in out:
                out[key] = value
            continue
        out[key] = value
    return out
