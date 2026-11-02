"""LoRa kNN registration core on an explicitly supplied B-stage encoder.

This is a cross-method input adaptation, not an author LoRa source model. Only
old/new labelled support is accepted during construction; prediction is frozen.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


def _validate(x: torch.Tensor, y: torch.Tensor, classes: tuple[int, ...]) -> None:
    if not classes or x.ndim < 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
        raise ValueError('support must contain nonempty aligned tensors')
    if y.dtype != torch.long or tuple(torch.unique(y, sorted=True).tolist()) != classes:
        raise ValueError('support labels must cover exactly the expected compact classes')
    counts = torch.bincount(y, minlength=classes[-1] + 1)[list(classes)]
    if not bool((counts == counts[0]).all()) or not bool(torch.isfinite(x).all()):
        raise ValueError('support must be finite and have equal K per class')


def _extract(model: nn.Module, extractor: Callable, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        z = extractor(model, x)
    if z.ndim != 2 or len(z) != len(x) or not bool(torch.isfinite(z).all()):
        raise ValueError('B encoder must produce finite per-record features')
    return z.detach().clone()


@dataclass(frozen=True)
class Gallery:
    model: nn.Module
    extractor: Callable
    features: torch.Tensor
    labels: torch.Tensor
    old_count: int


@dataclass(frozen=True)
class LoRaState(Gallery):
    class_count: int
    k: int

    def score(self, x: torch.Tensor) -> torch.Tensor:
        z = _extract(self.model, self.extractor, x)
        if z.shape[1] != self.features.shape[1]:
            raise ValueError('query feature dimension differs from gallery')
        distances = torch.cdist(z, self.features.to(z.device))
        indices = distances.argsort(dim=1, stable=True)[:, :self.k]
        neighbours = self.labels.to(z.device)[indices]
        return torch.stack([(neighbours == c).sum(1).to(z.dtype)
                            for c in range(self.class_count)], 1) / self.k

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(x).argmax(dim=1)


def build_gallery(model: nn.Module, extractor: Callable, old_x: torch.Tensor,
                  old_y: torch.Tensor, *, old_count: int) -> Gallery:
    if old_count <= 0:
        raise ValueError('old_count must be positive')
    _validate(old_x, old_y, tuple(range(old_count)))
    frozen = copy.deepcopy(model).eval().requires_grad_(False)
    z = _extract(frozen, extractor, old_x)
    return Gallery(frozen, extractor, z, old_y.detach().clone(), old_count)


def register_lora(gallery: Gallery, new_x: torch.Tensor, new_y: torch.Tensor,
                  *, k: int = 15) -> LoRaState:
    if k <= 0 or len(new_y) == 0:
        raise ValueError('positive neighbour count and nonempty new support required')
    class_count = int(new_y.max().item()) + 1
    _validate(new_x, new_y, tuple(range(gallery.old_count, class_count)))
    if len(new_y) // (class_count - gallery.old_count) != len(gallery.labels) // gallery.old_count:
        raise ValueError('old and new classes must use the same K')
    new_z = _extract(gallery.model, gallery.extractor, new_x)
    if new_z.shape[1] != gallery.features.shape[1]:
        raise ValueError('new feature dimension differs from old gallery')
    features = torch.cat((gallery.features, new_z.to(gallery.features.device)))
    labels = torch.cat((gallery.labels, new_y.to(gallery.labels.device)))
    return LoRaState(gallery.model, gallery.extractor, features, labels,
                     gallery.old_count, class_count, min(k, len(labels)))
