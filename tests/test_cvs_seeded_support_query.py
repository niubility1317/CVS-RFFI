from paper_reproduction.cvs_aligned.evaluate import _seeded_support_query_indices


def test_seeded_nested_split_is_deterministic_nested_and_query_fixed_across_k():
    common = {
        "available": 100,
        "query_count": 20,
        "support_pool_max_k": 20,
        "seed": 713101,
        "identity": "target_old|tx-a|rx-a|day0|eq1",
    }
    support_1, query_1 = _seeded_support_query_indices(k_shot=1, **common)
    support_5, query_5 = _seeded_support_query_indices(k_shot=5, **common)
    support_20, query_20 = _seeded_support_query_indices(k_shot=20, **common)

    assert support_1 == support_5[:1] == support_20[:1]
    assert support_5 == support_20[:5]
    assert query_1 == query_5 == query_20
    assert not set(support_20) & set(query_20)
    assert _seeded_support_query_indices(k_shot=5, **common) == (support_5, query_5)


def test_seeded_nested_split_changes_with_seed_or_identity():
    base = {
        "available": 100,
        "k_shot": 5,
        "query_count": 20,
        "support_pool_max_k": 20,
        "seed": 713101,
        "identity": "target_old|tx-a|rx-a|day0|eq1",
    }
    first = _seeded_support_query_indices(**base)
    second = _seeded_support_query_indices(**{**base, "seed": 713102})
    third = _seeded_support_query_indices(**{**base, "identity": "target_old|tx-b|rx-a|day0|eq1"})
    assert first != second
    assert first != third
