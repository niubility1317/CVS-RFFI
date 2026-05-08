from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from baselines.common.augmentation import OnlineRFChannelAugment
from baselines.common.spectrogram import iq_to_log_spectrogram


class SpectrogramTransform:
    def __init__(self, n_fft: int = 128, hop_length: int = 64, win_length: int | None = None, normalize: str = "zscore"):
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length or n_fft)
        self.normalize = normalize

    def __call__(self, iq: torch.Tensor) -> torch.Tensor:
        return iq_to_log_spectrogram(
            iq,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            normalize=self.normalize,
        )[0]


class PretrainDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        augment: Callable[[torch.Tensor], torch.Tensor] | None = None,
        spec_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.base_dataset = base_dataset
        self.augment = augment or (lambda x: x)
        self.spec_transform = spec_transform or SpectrogramTransform()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = self.base_dataset[int(idx)]
        iq = sample["iq"] if isinstance(sample, dict) else sample[0]
        return self.spec_transform(self.augment(iq)), self.spec_transform(self.augment(iq))


class SiamesePairDataset(Dataset):
    """Positive same-DUT different-receiver pairs for Siamese fine-tuning."""

    def __init__(
        self,
        base_dataset: Dataset,
        receivers: Sequence[int] | None = None,
        augment: Callable[[torch.Tensor], torch.Tensor] | None = None,
        spec_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        seed: int = 0,
    ):
        self.base_dataset = base_dataset
        self.receivers = {int(r) for r in receivers} if receivers is not None else None
        self.augment = augment or (lambda x: x)
        self.spec_transform = spec_transform or SpectrogramTransform()
        self.generator = torch.Generator().manual_seed(int(seed))
        self.by_label_receiver: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        aligned_groups: Dict[tuple[int, int, int, int], Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        for idx in range(len(base_dataset)):
            sample = base_dataset[idx]
            label = int(sample["label"])
            receiver = int(sample["receiver"])
            if self.receivers is not None and receiver not in self.receivers:
                continue
            self.by_label_receiver[label][receiver].append(idx)
            meta = dict(sample.get("meta", {}))
            day = int(sample.get("day", meta.get("day_i", -1)))
            sig_i = int(sample.get("sig_i", meta.get("sig_i", -1)))
            eq_i = int(meta.get("eq_i", -1))
            if day >= 0 and sig_i >= 0:
                aligned_groups[(label, day, eq_i, sig_i)][receiver].append(idx)
        self.aligned_pairs: List[tuple[int, int, int]] = []
        for key, rx_map in sorted(aligned_groups.items()):
            receivers_for_key = sorted(rx_map)
            if len(receivers_for_key) < 2:
                continue
            label = int(key[0])
            for i, rx1 in enumerate(receivers_for_key):
                for rx2 in receivers_for_key[i + 1 :]:
                    n = min(len(rx_map[rx1]), len(rx_map[rx2]))
                    for j in range(n):
                        self.aligned_pairs.append((int(rx_map[rx1][j]), int(rx_map[rx2][j]), label))
        self.labels = [label for label, rx_map in self.by_label_receiver.items() if len(rx_map) >= 2]
        self.aligned_pair_count = len(self.aligned_pairs)
        self.pair_mode = "aligned" if self.aligned_pairs else "sampled"
        if not self.labels and not self.aligned_pairs:
            raise ValueError("SiamesePairDataset requires at least one label present in two receivers.")

    def __len__(self):
        if self.aligned_pairs:
            return len(self.aligned_pairs)
        return sum(sum(len(v) for v in self.by_label_receiver[label].values()) for label in self.labels)

    def _choice(self, values: Sequence[int]) -> int:
        j = torch.randint(len(values), (1,), generator=self.generator).item()
        return int(values[j])

    def __getitem__(self, idx: int):
        if self.aligned_pairs:
            i1, i2, label = self.aligned_pairs[int(idx) % len(self.aligned_pairs)]
            s1 = self.base_dataset[i1]
            s2 = self.base_dataset[i2]
            x1 = self.spec_transform(self.augment(s1["iq"]))
            x2 = self.spec_transform(self.augment(s2["iq"]))
            return x1, x2, int(label)
        label = self.labels[int(idx) % len(self.labels)]
        rx_values = sorted(self.by_label_receiver[label])
        r1_idx = int(torch.randint(len(rx_values), (1,), generator=self.generator).item())
        r2_idx = int(torch.randint(len(rx_values) - 1, (1,), generator=self.generator).item())
        if r2_idx >= r1_idx:
            r2_idx += 1
        rx1, rx2 = rx_values[r1_idx], rx_values[r2_idx]
        s1 = self.base_dataset[self._choice(self.by_label_receiver[label][rx1])]
        s2 = self.base_dataset[self._choice(self.by_label_receiver[label][rx2])]
        x1 = self.spec_transform(self.augment(s1["iq"]))
        x2 = self.spec_transform(self.augment(s2["iq"]))
        return x1, x2, int(label)
