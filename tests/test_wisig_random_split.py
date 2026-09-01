import unittest

import numpy as np


def _tiny_wisig(samples_per_combo: int = 32, *, num_tx: int = 2, num_rx: int = 2, num_days: int = 2):
    data = []
    for tx in range(num_tx):
        tx_rows = []
        for rx in range(num_rx):
            day_rows = []
            for day in range(num_days):
                samples = np.zeros((samples_per_combo, 16, 2), dtype=np.float32)
                samples[:, :, 0] = float(tx)
                samples[:, :, 1] = np.arange(samples_per_combo, dtype=np.float32)[:, None]
                day_rows.append([samples])
            tx_rows.append(day_rows)
        data.append(tx_rows)
    return {
        "data": data,
        "tx_list": [f"tx{i}" for i in range(num_tx)],
        "rx_list": [f"rx{i}" for i in range(num_rx)],
        "capture_date_list": [f"day{i}" for i in range(num_days)],
        "equalized_list": [1],
    }


def _sig_indices(dataset):
    return [int(dataset[i][3]["sig_i"]) for i in range(len(dataset))]


def _sample_keys(dataset):
    keys = []
    for i in range(len(dataset)):
        meta = dataset[i][3]
        keys.append(
            (
                int(meta["tx_i"]),
                int(meta["rx_i"]),
                int(meta["day_i"]),
                int(meta["eq_i"]),
                int(meta["sig_i"]),
            )
        )
    return keys


def _grouped_sig_indices(dataset):
    groups = {}
    for i in range(len(dataset)):
        meta = dataset[i][3]
        key = (
            int(meta["tx_i"]),
            int(meta["rx_i"]),
            int(meta["day_i"]),
            int(meta["eq_i"]),
        )
        groups.setdefault(key, []).append(int(meta["sig_i"]))
    return {key: sorted(values) for key, values in groups.items()}


def _label_counts(dataset):
    counts = {}
    for i in range(len(dataset)):
        meta = dataset[i][3]
        counts[int(meta["tx_i"])] = counts.get(int(meta["tx_i"]), 0) + 1
    return counts


def _label_domain_counts(dataset):
    counts = {}
    for i in range(len(dataset)):
        meta = dataset[i][3]
        key = (int(meta["tx_i"]), int(meta["rx_i"]), int(meta["day_i"]), int(meta["eq_i"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _combined_grouped_sig_indices(*datasets):
    groups = {}
    for dataset in datasets:
        for key, values in _grouped_sig_indices(dataset).items():
            groups.setdefault(key, set()).update(values)
    return {key: sorted(values) for key, values in groups.items()}


class WiSigRandomSplitTest(unittest.TestCase):
    def test_disjoint_receiver_release_keeps_all_requested_target_days_once(self):
        from dataset_wisig import make_wisig_trainval_test_by_day_rx

        train, _val, _test, named_tests, named_meta, info = make_wisig_trainval_test_by_day_rx(
            _tiny_wisig(samples_per_combo=8, num_tx=2, num_rx=6, num_days=4),
            out_len=16,
            train_ratio=0.5,
            guard_gap=0,
            train_days=[1, 2, 3],
            test_days=[0, 1, 2, 3],
            train_rxs=[1, 3, 4],
            test_rxs=[0, 2, 5],
            split_strategy="random",
            cap_strategy="random",
            seed=392005,
            allow_source_target_day_overlap=True,
        )

        self.assertGreater(len(train), 0)
        self.assertEqual(info["train_days_idx"], [1, 2, 3])
        self.assertEqual(named_meta["test_all_day_unseen_rx"]["days_idx"], [0, 1, 2, 3])
        self.assertEqual(named_meta["test_all_day_unseen_rx"]["rxs_idx"], [0, 2, 5])
        self.assertEqual(len(named_tests["test_all_day_unseen_rx"]), 2 * 3 * 4 * 8)
        self.assertEqual(len(named_tests["test_seen_day_unseen_rx"]), 2 * 3 * 3 * 8)
        self.assertEqual(len(named_tests["test_unseen_day_unseen_rx"]), 2 * 3 * 1 * 8)

    def test_random_split_and_cap_are_seed_reproducible(self):
        from dataset_wisig import make_wisig_trainval_test_by_day_rx

        kwargs = dict(
            ds=_tiny_wisig(),
            out_len=16,
            train_ratio=0.5,
            guard_gap=0,
            train_days=[0],
            test_days=[1],
            train_rxs=[0],
            test_rxs=[1],
            max_samples_per_combo_train=4,
            split_strategy="random",
            cap_strategy="random",
        )

        train_a, val_a, *_ = make_wisig_trainval_test_by_day_rx(seed=11, **kwargs)
        train_b, val_b, *_ = make_wisig_trainval_test_by_day_rx(seed=11, **kwargs)
        train_c, val_c, *_ = make_wisig_trainval_test_by_day_rx(seed=29, **kwargs)

        self.assertEqual(_sig_indices(train_a), _sig_indices(train_b))
        self.assertEqual(_sig_indices(val_a), _sig_indices(val_b))
        self.assertNotEqual(_sig_indices(train_a), _sig_indices(train_c))
        self.assertNotEqual(_sig_indices(val_a), _sig_indices(val_c))
        self.assertTrue(set(_sample_keys(train_a)).isdisjoint(set(_sample_keys(val_a))))

    def test_contiguous_front_mode_preserves_legacy_first_samples(self):
        from dataset_wisig import make_wisig_trainval_test_by_day_rx

        train, val, *_ = make_wisig_trainval_test_by_day_rx(
            _tiny_wisig(),
            out_len=16,
            train_ratio=0.5,
            guard_gap=0,
            train_days=[0],
            test_days=[1],
            train_rxs=[0],
            test_rxs=[1],
            max_samples_per_combo_train=4,
            split_strategy="contiguous",
            cap_strategy="front",
            seed=99,
        )

        self.assertEqual(_sig_indices(train)[:4], [0, 1, 2, 3])
        self.assertEqual(_sig_indices(val)[:4], [16, 17, 18, 19])

    def test_random_day123_pool_cap_is_not_front_n(self):
        from dataset_wisig import make_wisig_trainval_test_by_day_rx

        train, val, *_ = make_wisig_trainval_test_by_day_rx(
            _tiny_wisig(samples_per_combo=64),
            out_len=16,
            train_ratio=0.5,
            guard_gap=0,
            train_days=[0],
            test_days=[1],
            train_rxs=[0],
            test_rxs=[1],
            max_samples_per_combo_day123=8,
            split_strategy="random",
            cap_strategy="random",
            seed=17,
        )

        groups = _combined_grouped_sig_indices(train, val)
        self.assertTrue(groups)
        for sigs in groups.values():
            self.assertEqual(len(sigs), 8)
            self.assertNotEqual(sigs, list(range(8)))

    def test_random_test_cap_is_not_front_n(self):
        from dataset_wisig import make_wisig_trainval_test_by_day_rx

        _, _, _, named_tests, *_ = make_wisig_trainval_test_by_day_rx(
            _tiny_wisig(samples_per_combo=64),
            out_len=16,
            train_ratio=0.5,
            guard_gap=0,
            train_days=[0],
            test_days=[1],
            train_rxs=[0],
            test_rxs=[1],
            max_samples_per_combo_test=4,
            split_strategy="random",
            cap_strategy="random",
            seed=23,
        )

        groups = _grouped_sig_indices(named_tests["test_unseen_day_unseen_rx"])
        self.assertTrue(groups)
        for sigs in groups.values():
            self.assertEqual(len(sigs), 4)
            self.assertNotEqual(sigs, [0, 1, 2, 3])

    def test_train_shots_per_class_is_total_not_per_combo(self):
        from dataset_wisig import make_wisig_trainval_test_by_day_rx

        train_a, val_a, _, _, _, info_a = make_wisig_trainval_test_by_day_rx(
            _tiny_wisig(samples_per_combo=64, num_tx=3, num_rx=3, num_days=3),
            out_len=16,
            train_ratio=0.5,
            guard_gap=0,
            train_days=[0, 1],
            test_days=[2],
            train_rxs=[0, 1, 2],
            test_rxs=[],
            max_samples_per_class_train=5,
            train_class_cap_strategy="domain_balanced",
            split_strategy="random",
            cap_strategy="random",
            seed=31,
        )
        train_b, _, *_ = make_wisig_trainval_test_by_day_rx(
            _tiny_wisig(samples_per_combo=64, num_tx=3, num_rx=3, num_days=3),
            out_len=16,
            train_ratio=0.5,
            guard_gap=0,
            train_days=[0, 1],
            test_days=[2],
            train_rxs=[0, 1, 2],
            test_rxs=[],
            max_samples_per_class_train=5,
            train_class_cap_strategy="domain_balanced",
            split_strategy="random",
            cap_strategy="random",
            seed=31,
        )
        train_c, _, *_ = make_wisig_trainval_test_by_day_rx(
            _tiny_wisig(samples_per_combo=64, num_tx=3, num_rx=3, num_days=3),
            out_len=16,
            train_ratio=0.5,
            guard_gap=0,
            train_days=[0, 1],
            test_days=[2],
            train_rxs=[0, 1, 2],
            test_rxs=[],
            max_samples_per_class_train=5,
            train_class_cap_strategy="domain_balanced",
            split_strategy="random",
            cap_strategy="random",
            seed=43,
        )

        self.assertEqual(len(train_a), 15)
        self.assertEqual(_label_counts(train_a), {0: 5, 1: 5, 2: 5})
        self.assertEqual(_sample_keys(train_a), _sample_keys(train_b))
        self.assertNotEqual(_sample_keys(train_a), _sample_keys(train_c))
        self.assertTrue(set(_sample_keys(train_a)).isdisjoint(set(_sample_keys(val_a))))
        self.assertEqual(info_a["max_samples_per_class_train"], 5)
        self.assertEqual(info_a["train_class_cap_strategy"], "domain_balanced")

        domain_counts = _label_domain_counts(train_a)
        for tx_i in [0, 1, 2]:
            used_domains = [key for key in domain_counts if key[0] == tx_i]
            self.assertEqual(len(used_domains), 5)
            self.assertTrue(all(domain_counts[key] == 1 for key in used_domains))

    def test_rx_day_balanced_train_shots_spread_receiver_and_day(self):
        from dataset_wisig import make_wisig_trainval_test_by_day_rx

        train, *_ = make_wisig_trainval_test_by_day_rx(
            _tiny_wisig(samples_per_combo=16, num_tx=2, num_rx=4, num_days=3),
            out_len=16,
            train_ratio=0.5,
            guard_gap=0,
            train_days=[0, 1],
            test_days=[2],
            train_rxs=[0, 1, 2, 3],
            test_rxs=[],
            max_samples_per_class_train=5,
            train_class_cap_strategy="rx_day_balanced",
            split_strategy="random",
            cap_strategy="random",
            seed=7,
        )

        by_tx = {}
        for i in range(len(train)):
            meta = train[i][3]
            by_tx.setdefault(int(meta["tx_i"]), []).append((int(meta["rx_i"]), int(meta["day_i"])))

        self.assertEqual(_label_counts(train), {0: 5, 1: 5})
        for domains in by_tx.values():
            self.assertGreaterEqual(len({rx for rx, _ in domains}), 4)
            self.assertEqual({day for _, day in domains}, {0, 1})


if __name__ == "__main__":
    unittest.main()
