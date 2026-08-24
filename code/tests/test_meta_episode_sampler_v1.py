import numpy as np
import pytest
from dataclasses import replace
from collections import Counter

from dataset_wisig import WiSigIndex, wisig_capture_block_id, wisig_physical_sample_id
from dataset_wisig import WiSigCompactDataset
from cvsrffi.meta_episodes import (
    EpisodeKind,
    HierarchicalMetaEpisodeSampler,
    MetaEpisodeSamplerConfig,
    MetaSampleRef,
)


_LEO_VIEWS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _make_balanced_refs(*, role="L_s", class_ids=range(4), samples_per_domain=8):
    refs = []
    dataset_index = 0
    for tx_i in class_ids:
        for rx_i in range(3):
            for day_i in range(3):
                for capture_block_i in range(2):
                    for sig_i in range(samples_per_domain):
                        physical_id = (
                            f"tx{tx_i}|rx{rx_i}|day{day_i}|eq0|"
                            f"block{capture_block_i}|sig{sig_i}"
                        )
                        for view in ("clean",) + _LEO_VIEWS:
                            refs.append(
                                MetaSampleRef(
                                    dataset_index=dataset_index,
                                    tx_i=int(tx_i),
                                    rx_i=rx_i,
                                    day_i=day_i,
                                    eq_i=0,
                                    capture_block_i=capture_block_i,
                                    physical_sample_id=physical_id,
                                    role=role,
                                    view=view,
                                )
                            )
                            dataset_index += 1
    return refs


def _kind_config(kind, **kwargs):
    weights = {episode_kind: 0.0 for episode_kind in EpisodeKind}
    weights[kind] = 1.0
    params = dict(
        k_choices=(1,),
        query_per_class=2,
        episode_weights=weights,
    )
    params.update(kwargs)
    return MetaEpisodeSamplerConfig(**params)


def _episode_rows(episode):
    return episode.support + episode.query_adapt + episode.query_guard


def _assert_common_episode_invariants(episode, *, allowed_roles):
    rows = _episode_rows(episode)
    assert rows
    assert {row.role for row in rows}.issubset(set(allowed_roles))
    support_ids = {row.physical_sample_id for row in episode.support}
    query_ids = {
        row.physical_sample_id
        for row in episode.query_adapt + episode.query_guard
    }
    assert support_ids.isdisjoint(query_ids)
    assert len(support_ids) == len(episode.support)
    assert len(query_ids) == len(episode.query_adapt) + len(episode.query_guard)
    assert {row.tx_i for row in episode.support} == set(episode.adapt_class_ids)
    assert {row.tx_i for row in episode.query_adapt}.issubset(
        set(episode.adapt_class_ids)
    )
    assert {row.tx_i for row in episode.query_guard} == set(episode.guard_class_ids)
    assert set(episode.adapt_class_ids).isdisjoint(set(episode.guard_class_ids))


def test_hierarchical_sampler_interfaces_are_available():
    assert EpisodeKind.SAME_DOMAIN.value == "Q_SAME_DOMAIN"
    assert MetaSampleRef.__dataclass_fields__.keys() >= {
        "dataset_index",
        "tx_i",
        "rx_i",
        "day_i",
        "eq_i",
        "capture_block_i",
        "physical_sample_id",
        "role",
        "view",
    }
    assert MetaEpisodeSamplerConfig.__dataclass_fields__.keys() >= {
        "k_choices",
        "query_per_class",
        "allowed_roles",
        "training",
        "episode_weights",
    }
    assert hasattr(HierarchicalMetaEpisodeSampler, "sample")


def test_training_and_evaluation_role_combinations_are_strict():
    with pytest.raises(ValueError, match="training.*allowed_roles.*L_s"):
        MetaEpisodeSamplerConfig(training=True, allowed_roles=("V_cal",))
    with pytest.raises(ValueError, match="evaluation.*allowed_roles.*V_cal.*V_select"):
        MetaEpisodeSamplerConfig(training=False, allowed_roles=("L_s",))
    with pytest.raises(ValueError, match="outside allowed_roles"):
        HierarchicalMetaEpisodeSampler(
            _make_balanced_refs(role="V_cal"),
            MetaEpisodeSamplerConfig(),
        )


def test_training_sampler_never_admits_validation_roles():
    refs = _make_balanced_refs()
    refs.append(replace(refs[0], dataset_index=999999, role="V_cal"))
    with pytest.raises(ValueError, match="role"):
        HierarchicalMetaEpisodeSampler(refs, MetaEpisodeSamplerConfig())


def test_support_and_query_physical_ids_are_disjoint_with_views():
    sampler = HierarchicalMetaEpisodeSampler(
        _make_balanced_refs(),
        _kind_config(EpisodeKind.SAME_DOMAIN, partial_coverage_probability=0.0),
    )
    episode = sampler.sample(seed=73)
    _assert_common_episode_invariants(episode, allowed_roles=("L_s",))
    assert all(row.view == episode.support[0].view for row in episode.support)
    assert all(row.view == episode.query_adapt[0].view for row in episode.query_adapt)


def test_query_only_classes_are_guard_not_adapt():
    sampler = HierarchicalMetaEpisodeSampler(
        _make_balanced_refs(),
        _kind_config(
            EpisodeKind.SAME_DOMAIN,
            partial_coverage_probability=1.0,
        ),
    )
    episode = sampler.sample(seed=11)
    _assert_common_episode_invariants(episode, allowed_roles=("L_s",))
    assert episode.guard_class_ids
    assert set(episode.guard_class_ids).isdisjoint(set(episode.adapt_class_ids))
    assert all(row.tx_i in episode.guard_class_ids for row in episode.query_guard)
    assert all(row.tx_i not in episode.guard_class_ids for row in episode.support)
    assert all(row.tx_i not in episode.guard_class_ids for row in episode.query_adapt)


@pytest.mark.parametrize(
    "kind",
    [
        EpisodeKind.SAME_DOMAIN,
        EpisodeKind.RX_HOLDOUT,
        EpisodeKind.DAY_CHANNEL_HOLDOUT,
        EpisodeKind.CLEAN_TO_LEO,
        EpisodeKind.LEO_CROSS,
    ],
)
def test_each_episode_kind_is_constructible_and_respects_domain_relation(kind):
    sampler = HierarchicalMetaEpisodeSampler(
        _make_balanced_refs(),
        _kind_config(kind, partial_coverage_probability=0.0),
    )
    episode = sampler.sample(seed=17)
    _assert_common_episode_invariants(episode, allowed_roles=("L_s",))
    support = episode.support
    query = episode.query_adapt
    assert support and query

    if kind is EpisodeKind.SAME_DOMAIN:
        support_domain = {
            (row.rx_i, row.day_i, row.eq_i, row.view) for row in support
        }
        query_domain = {
            (row.rx_i, row.day_i, row.eq_i, row.view) for row in query
        }
        assert support_domain == query_domain
    elif kind is EpisodeKind.RX_HOLDOUT:
        assert {row.rx_i for row in support}.isdisjoint({row.rx_i for row in query})
        assert {row.tx_i for row in support} == {row.tx_i for row in query}
    elif kind is EpisodeKind.DAY_CHANNEL_HOLDOUT:
        assert {row.rx_i for row in support} == {row.rx_i for row in query}
        assert any(
            (left.day_i != right.day_i)
            or (left.capture_block_i != right.capture_block_i)
            for left in support
            for right in query
        )
    elif kind is EpisodeKind.CLEAN_TO_LEO:
        assert {row.view == "clean" for row in support + query} == {False, True}
        assert any(row.view == "clean" for row in support)
        assert any(row.view.startswith("leo_") for row in support + query)
    elif kind is EpisodeKind.LEO_CROSS:
        assert all(row.view.startswith("leo_") for row in support + query)
        assert {row.view for row in support}.isdisjoint({row.view for row in query})


def test_same_seed_is_exactly_reproducible_and_different_seed_changes_episode():
    sampler = HierarchicalMetaEpisodeSampler(
        _make_balanced_refs(),
        _kind_config(EpisodeKind.SAME_DOMAIN, partial_coverage_probability=0.0),
    )
    first = sampler.sample(seed=101)
    second = sampler.sample(seed=101)
    other = sampler.sample(seed=102)
    assert first == second
    assert first != other
    _assert_common_episode_invariants(other, allowed_roles=("L_s",))


def test_repeated_sampling_reuses_frozen_candidate_plans(monkeypatch):
    sampler = HierarchicalMetaEpisodeSampler(
        _make_balanced_refs(),
        _kind_config(EpisodeKind.SAME_DOMAIN, partial_coverage_probability=0.0),
    )
    original = sampler._candidate_plans
    calls = 0

    def counted(kind):
        nonlocal calls
        calls += 1
        return original(kind)

    monkeypatch.setattr(sampler, "_candidate_plans", counted)
    first = sampler.sample(seed=101)
    second = sampler.sample(seed=101)

    assert first == second
    assert calls == 1


def test_repeated_pool_lookup_reuses_frozen_class_spec_index(monkeypatch):
    sampler = HierarchicalMetaEpisodeSampler(
        _make_balanced_refs(),
        _kind_config(EpisodeKind.SAME_DOMAIN, partial_coverage_probability=0.0),
    )
    original = sampler._row_matches
    calls = 0

    def counted(row, spec):
        nonlocal calls
        calls += 1
        return original(row, spec)

    monkeypatch.setattr(sampler, "_row_matches", counted)
    spec = {"rx_i": 0, "day_i": 0, "eq_i": 0, "view": "clean"}
    first = sampler._pool(0, spec)
    first_scan_calls = calls
    second = sampler._pool(0, spec)

    assert first == second
    assert first_scan_calls == len(sampler._by_class[0])
    assert calls == first_scan_calls


def test_rx_holdout_plan_build_scans_descriptor_collection_once():
    sampler = HierarchicalMetaEpisodeSampler(
        _make_balanced_refs(),
        _kind_config(EpisodeKind.RX_HOLDOUT, partial_coverage_probability=0.0),
    )

    class CountingDescriptors:
        def __init__(self, values):
            self.values = tuple(values)
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return iter(self.values)

    descriptors = CountingDescriptors(sampler._descriptors)
    sampler._descriptors = descriptors
    plans = sampler._candidate_plans(EpisodeKind.RX_HOLDOUT)

    assert len(plans) == 144
    assert descriptors.iterations == 1


@pytest.mark.parametrize(
    ("kind", "expected_count"),
    [
        (EpisodeKind.SAME_DOMAIN, 36),
        (EpisodeKind.RX_HOLDOUT, 144),
        (EpisodeKind.DAY_CHANNEL_HOLDOUT, 360),
        (EpisodeKind.CLEAN_TO_LEO, 54),
        (EpisodeKind.LEO_CROSS, 108),
    ],
)
def test_grouped_candidate_plan_counts_match_complete_domain_product(
    kind, expected_count
):
    sampler = HierarchicalMetaEpisodeSampler(
        _make_balanced_refs(),
        _kind_config(kind, partial_coverage_probability=0.0),
    )
    assert len(sampler._candidate_plans(kind)) == expected_count


def test_label_permutation_preserves_task_counts_and_class_coverage():
    refs = _make_balanced_refs()
    permuted = [replace(row, tx_i={0: 20, 1: 10, 2: 30, 3: 40}[row.tx_i]) for row in refs]
    config = _kind_config(
        EpisodeKind.SAME_DOMAIN,
        partial_coverage_probability=1.0,
    )
    original = HierarchicalMetaEpisodeSampler(refs, config).sample(seed=7)
    renamed = HierarchicalMetaEpisodeSampler(permuted, config).sample(seed=7)
    assert len(original.support) == len(renamed.support)
    assert len(original.query_adapt) == len(renamed.query_adapt)
    assert len(original.query_guard) == len(renamed.query_guard)
    assert len(original.adapt_class_ids) == len(renamed.adapt_class_ids)
    assert len(original.guard_class_ids) == len(renamed.guard_class_ids)
    assert sorted(Counter(row.tx_i for row in _episode_rows(original)).values()) == sorted(
        Counter(row.tx_i for row in _episode_rows(renamed)).values()
    )


def test_evaluation_sampler_only_accepts_validation_roles():
    refs = _make_balanced_refs(role="V_cal")
    config = _kind_config(
        EpisodeKind.SAME_DOMAIN,
        training=False,
        allowed_roles=("V_cal", "V_select"),
        partial_coverage_probability=0.0,
    )
    episode = HierarchicalMetaEpisodeSampler(refs, config).sample(seed=5)
    assert {row.role for row in _episode_rows(episode)} == {"V_cal"}


def test_unsatisfiable_pool_fails_descriptively_instead_of_downgrading():
    refs = _make_balanced_refs(samples_per_domain=1)[:16]
    config = _kind_config(
        EpisodeKind.SAME_DOMAIN,
        k_choices=(2,),
        query_per_class=2,
        partial_coverage_probability=0.0,
    )
    with pytest.raises(ValueError, match="(cannot|not enough|unsatisfiable|physical)"):
        HierarchicalMetaEpisodeSampler(refs, config).sample(seed=3)


def test_wisig_physical_id_is_complete_and_stable():
    item = WiSigIndex(tx_i=2, rx_i=3, day_i=1, eq_i=0, sig_i=19)
    assert wisig_physical_sample_id(item) == "tx2|rx3|day1|eq0|sig19"


def test_capture_block_uses_sig_index_without_claiming_real_channel():
    item = WiSigIndex(tx_i=2, rx_i=3, day_i=1, eq_i=0, sig_i=19)
    assert wisig_capture_block_id(item, block_size=8) == 2


def test_capture_block_rejects_non_positive_block_size():
    item = WiSigIndex(tx_i=2, rx_i=3, day_i=1, eq_i=0, sig_i=19)
    for block_size in (0, -1):
        with pytest.raises(ValueError, match="capture block_size must be positive"):
            wisig_capture_block_id(item, block_size=block_size)


def test_compact_dataset_rejects_non_positive_capture_block_size():
    samples = np.zeros((1, 2, 2), dtype=np.float32)
    ds = {
        "data": [[[[samples]]]],
        "tx_list": ["tx"],
        "rx_list": ["rx"],
        "capture_date_list": ["day"],
        "equalized_list": [1],
    }
    with pytest.raises(ValueError, match="capture_block_size must be positive"):
        WiSigCompactDataset(ds, out_len=2, normalize=False, capture_block_size=0)


def test_compact_dataset_metadata_contains_stable_identity_and_proxy_block():
    samples = np.zeros((3, 2, 2), dtype=np.float32)
    ds = {
        "data": [[[[samples]]]],
        "tx_list": ["tx"],
        "rx_list": ["rx"],
        "capture_date_list": ["day"],
        "equalized_list": [1],
    }
    dataset = WiSigCompactDataset(
        ds,
        out_len=2,
        normalize=False,
        capture_block_size=2,
    )

    _, _, _, meta = dataset[2]

    assert meta["physical_sample_id"] == "tx0|rx0|day0|eq0|sig2"
    assert meta["capture_block_i"] == 1
    assert meta["capture_block_semantics"] == "sig_index_time_block_proxy"
