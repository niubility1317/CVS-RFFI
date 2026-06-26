import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated.client_split import build_client_splits, infer_client_id


class MetaDataset(Dataset):
    def __init__(self):
        self.samples = [
            SimpleNamespace(tx_i=0, rx_i=1, day_i=0, channel_view="ground"),
            SimpleNamespace(tx_i=1, rx_i=1, day_i=0, channel_view="ground"),
            SimpleNamespace(tx_i=0, rx_i=2, day_i=1, channel_view="clear_leo"),
        ]
        self.index = self.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = {
            "tx_i": self.samples[idx].tx_i,
            "rx_i": self.samples[idx].rx_i,
            "day_i": self.samples[idx].day_i,
            "channel_view": self.samples[idx].channel_view,
        }
        return torch.zeros(2, 8), self.samples[idx].tx_i, 0, meta


class ClientSplitTest(unittest.TestCase):
    def test_builds_receiver_day_clients_from_wisig_metadata(self):
        splits = build_client_splits(MetaDataset(), client_key="receiver_day", min_samples_per_client=1)

        self.assertEqual(splits, {"rx1_day0": [0, 1], "rx2_day1": [2]})

    def test_infer_client_id_supports_channel_granularity(self):
        cid = infer_client_id({"rx_i": 3, "day_i": 2, "channel_view": "mixed_orbit"}, "receiver_day_channel")

        self.assertEqual(cid, "rx3_day2_chmixed_orbit")


if __name__ == "__main__":
    unittest.main()
