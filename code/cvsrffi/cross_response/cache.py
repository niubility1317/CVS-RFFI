"""Only detached received-IQ statistics; never cache learned representations."""
from __future__ import annotations

import json
import torch


class FixedStatisticCache:
    def __init__(self):
        self.values = {}
        self.hits = self.misses = 0

    @staticmethod
    def key(physical_id, *, crop, statistics_config, scale_version="raw"):
        if crop is None or not scale_version:
            raise ValueError("cache requires explicit raw crop and scale version")
        return (str(physical_id), json.dumps(crop, sort_keys=True),
                json.dumps(statistics_config, sort_keys=True), str(scale_version))

    def get_many(self, keys, compute, *, device):
        missing = list(dict.fromkeys(key for key in keys if key not in self.values))
        if missing:
            values = compute(missing)
            if values.requires_grad or len(values) != len(missing) or not torch.isfinite(values).all():
                raise ValueError("cache accepts finite detached fixed statistics only")
            for key, value in zip(missing, values):
                self.values[key] = value.detach().float().cpu().clone()
        self.misses += len(missing)
        self.hits += len(keys) - len(missing)
        return torch.stack([self.values[key] for key in keys]).to(device)

    def report(self):
        return {"entries": len(self.values), "hits": self.hits, "misses": self.misses,
                "value_bytes": sum(v.numel()*v.element_size() for v in self.values.values()),
                "key_utf8_bytes": sum(len(repr(k).encode("utf-8")) for k in self.values),
                "python_container_overhead_included": False,
                "contents": "detached raw waveform statistics only"}
