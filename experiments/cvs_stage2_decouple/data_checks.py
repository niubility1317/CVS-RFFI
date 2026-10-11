"""Synthetic native-IQ cache and validation boundary regression checks."""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import data
from . import design as d


class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(2)
        self.calls = 0
        self.samples = 0

    def forward(self, x):
        self.calls += 1
        self.samples += len(x)
        v = self.bn(x).mean(2)
        return torch.stack((v[:, 0], v[:, 1], -v[:, 0], -v[:, 1],
                            v[:, 0] + v[:, 1], v[:, 0] - v[:, 1]), 1) * 30


def run():
    torch.set_num_threads(2)
    native = Path(__file__).resolve().parents[1] / "adv3b02_xuc/code"
    if str(native) not in sys.path:
        sys.path.insert(0, str(native))
    from dataset_wisig import WiSigCompactDataset, WiSigSubsetDataset, _pad_or_crop_2t, _rms_normalize_iq
    from cvsrffi.game_tracking.data import opaque_id
    from cvsrffi.xuc_fusion.native import role_ids_from_native
    rng = np.random.default_rng(4512)
    cells = [[[[rng.normal(size=(5, 270, 2)).astype(np.float32)] for day in range(2)]
              for rx in range(2)] for tx in range(2)]
    base = WiSigCompactDataset(dict(data=cells, tx_list=["tx0", "tx1"], rx_list=["rx0", "rx1"],
                                    capture_date_list=["d0", "d1"], equalized_list=[0]),
                                out_len=256, crop_mode="center", normalize=True, center=False,
                                equalized=0, domain="rx_day")
    labeled_indices, val_indices, hidden_indices = [17, 0, 23, 8, 32, 5], [2, 15, 27, 36], [1, 7, 11]
    labeled = WiSigSubsetDataset(base, labeled_indices, "ssdg_labeled_tx_visible")
    validation = WiSigSubsetDataset(base, val_indices, "ssdg_source_v_select")

    class NeverSampleU(WiSigSubsetDataset):
        reads = 0
        def __getitem__(self, index):
            self.reads += 1
            raise AssertionError("U source labels/IQ were sampled")

    hidden = NeverSampleU(base, hidden_indices, "ssdg_unlabeled_tx_hidden")
    context = dict(train_loader=SimpleNamespace(dataset=labeled),
                   val_loader=SimpleNamespace(dataset=validation),
                   unlabeled_loader=SimpleNamespace(dataset=hidden), named_test_loaders={})
    expected = dict(role_ids=role_ids_from_native(context), fixture_only=True)
    parent = Parent()
    parent.train()
    before = copy.deepcopy(parent.state_dict())
    before_mode = parent.training
    lc = data._cache_dataset(labeled, "L_s", opaque_id, expected["role_ids"]["L_s"], parent, "cpu", batch_size=4)
    assert parent.calls == 2 and parent.samples == len(labeled)
    assert parent.training == before_mode
    assert all(torch.equal(v, parent.state_dict()[k]) for k, v in before.items())
    assert lc["ids"] == [opaque_id(item) for item in labeled.index]
    assert lc["x"].dtype == torch.float32 and lc["y"].dtype == torch.long
    assert lc["state"].shape == (6, 40)
    assert lc["parent_margin"].shape == lc["parent_correct"].shape == (6,)
    for i, original_index in enumerate(labeled_indices):
        item = base.index[original_index]
        raw = cells[item.tx_i][item.rx_i][item.day_i][item.eq_i][item.sig_i].T
        expected_x = torch.from_numpy(_rms_normalize_iq(_pad_or_crop_2t(raw, 256)))
        assert torch.equal(lc["x"][i], expected_x)
        assert int(lc["y"][i]) == item.tx_i
        assert int(lc["rx"][i]) == item.rx_i and int(lc["day"][i]) == item.day_i
    vc = data._cache_dataset(validation, "V", opaque_id, expected["role_ids"]["V"])
    assert parent.calls == 2 and parent.samples == 6
    assert not any(k in vc for k in ("state", "parent_margin", "parent_correct", "u0"))
    random_before = torch.get_rng_state().clone()
    loader = data.cached_loader(vc, batch_size=3)
    batches = list(loader)
    assert torch.equal(random_before, torch.get_rng_state())
    assert [len(batch[0]) for batch in batches] == [3, 1]
    assert torch.equal(torch.cat([batch[0] for batch in batches]), vc["x"])
    assert torch.equal(torch.cat([batch[1] for batch in batches]), vc["y"])
    assert torch.equal(torch.cat([batch[2] for batch in batches]), vc["domain"])
    assert torch.equal(torch.cat([batch[3]["rx_i"] for batch in batches]), vc["rx"])
    assert torch.equal(torch.cat([batch[3]["day_i"] for batch in batches]), vc["day"])
    first = vc["x"][0].clone()
    loader.dataset[0][0].zero_()
    assert torch.equal(first, vc["x"][0])
    assert hidden.reads == 0

    calls = []
    def construct(args, device):
        calls.append((args, str(device)))
        return context

    @contextmanager
    def installed(_c):
        yield SimpleNamespace(_build_ssdg_wisig_data=construct)

    source_args = SimpleNamespace(a1_source_screen_only=True, use_unlabeled=False,
                                 use_ema_teacher=False, use_concat_sat_channel_aug=False)
    c = d.config(d.rows()[0])
    assert data.SOURCE_COUNTS == (6300, 27000)
    with patch("experiments.cvs_phase1_stack.runtime.installed", installed), \
            patch.object(data.d, "read", return_value=expected), \
            patch.object(data.d, "make_args", return_value=source_args), \
            patch.object(data, "SOURCE_COUNTS", (6, 4)):
        cache = data.build_source_cache(c, parent, "cpu")
        assert cache["contract"] == dict(expected, native_role_comparison="EXACT_MATCH")
        assert cache["diagnostics"]["unlabeled_sample_reads"] == 0
        assert cache["diagnostics"]["V_parent_forwards"] == 0
        assert len(calls) == 1 and hidden.reads == 0
        assert not any(k in cache for k in ("unlabeled", "unlabeled_loader", "parent", "context"))
        context["named_test_loaders"] = {"target": object()}
        parent_calls = parent.calls
        try:
            data.build_source_cache(c, parent, "cpu")
            raise AssertionError("Target loaders accepted")
        except ValueError as exc:
            assert "target loaders" in str(exc)
        assert parent.calls == parent_calls
        context["named_test_loaders"] = {}
        expected["role_ids"]["L_s"] = ["wrong"]
        try:
            data.build_source_cache(c, parent, "cpu")
            raise AssertionError("Mismatched source role accepted")
        except ValueError as exc:
            assert "EXACT_MATCH" in str(exc)
        assert parent.calls == parent_calls
    return dict(status="PASS", native_center_crop_and_complex_rms_exact=True,
                actual_dataset_order_and_opaque_ids_exact=True,
                metadata_rx_day_domain_preserved=True, parent_state_and_mode_preserved=True,
                source_L_only_parent_statistics=True, source_V_no_parent_forward_or_state_fit=True,
                unlabeled_sample_reads=hidden.reads, target_and_role_mismatch_rejected_before_caching=True,
                cached_validation_is_sequential_and_no_drop=True,
                validation_item_cannot_mutate_cache=True, cached_validation_preserves_global_rng=True,
                production_counts=dict(L_s=data.SOURCE_COUNTS[0], V=data.SOURCE_COUNTS[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run()
    text = json.dumps(result, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
