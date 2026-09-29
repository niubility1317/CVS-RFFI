"""Synthetic metadata-only pairing: no models, predictors, truth or scores."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from d92_adaptation_registration_pairing import SCHEMA, validate_pairs


def fixture():
    old = ['old-b', 'old-a']
    splits = []
    for k in (1, 2):
        for seed in (7, 8):
            for new_count in (0, 2, 5):
                classes = old + [f'new-{i}' for i in range(new_count)]
                support = [f'{cls}-s{seed}-{i}' for cls in classes for i in range(k)]
                query = [f'{cls}-q{i}' for cls in classes for i in range(2)]
                if new_count:
                    query.reverse()  # Physical pairing must not depend on row position.
                splits.append(dict(split_id=f'split-{k}-{seed}-{new_count}', capsule_id='capsule',
                    receiver='rx', scenario='scene', k=k, support_seed=seed,
                    registered_classes=classes, support_ids=support,
                    support_labels=[j for j in range(len(classes)) for _ in range(k)],
                    query_ids=query))
    dg = [dict(split_id='split-1-7-0', capsule_id='capsule', receiver='rx', scenario='scene',
        classes=old[::-1], query_ids=splits[0]['query_ids'][::-1], k=999, support_seed=123456,
        mode='frozen_dg')]
    return dict(splits=splits, frozen_dg=dg, old_classes=old,
        expected_new_counts=(0, 2, 5),
        expected_episode_keys={('rx', 'scene', k, seed) for k in (1, 2) for seed in (7, 8)},
        model_seed=2026092701, checkpoint_sha256='a' * 64)


def test_order_independent_pairing_ignores_incidental_dg_k_seed_and_preserves_inputs():
    args = fixture(); before = deepcopy(args)
    rows = validate_pairs(**args)
    assert args == before and len(rows) == 12
    assert all(r['schema'] == SCHEMA and r['status'] == 'VERIFIED' for r in rows)
    for row in rows:
        assert row['a_split_id'] == 'split-1-7-0'
        assert row['b_split_id'] == f"split-{row['k']}-{row['support_seed']}-0"
        assert row['a_query_count'] == row['b_query_count'] == 4
        assert row['c_query_count'] == (2 + row['new_count']) * 2
        assert row['dg_matching_ignores_k_and_support_seed'] is True
        assert row['state_inheritance_verified'] is False
        assert len(row['old_support_ids']) == 2 * row['k']
        if row['new_count'] == 0:
            assert row['b_split_id'] == row['c_split_id']
    shuffled = deepcopy(args)
    shuffled['splits'].reverse()
    for split in shuffled['splits']:
        split['query_ids'].reverse()
        split['support_ids'].reverse(); split['support_labels'].reverse()
    assert validate_pairs(**shuffled) == rows


def test_registry_permutation_with_matching_support_labels_is_legal():
    args = fixture()
    for split in args['splits']:
        original = split['registered_classes']
        permuted = original[::-1]
        split['support_labels'] = [permuted.index(original[i]) for i in split['support_labels']]
        split['registered_classes'] = permuted
    assert len(validate_pairs(**args)) == 12


@pytest.mark.parametrize('fault', [
    'old_query_missing', 'old_support_changed', 'class_label_mapping', 'old_class_missing',
    'dg_query_missing', 'dg_classes', 'dg_split', 'dg_duplicate', 'dg_missing',
    'split_duplicate', 'cell_duplicate', 'old_only_missing', 'expanded_missing',
    'episode_missing', 'capsule', 'query_duplicate', 'support_duplicate', 'overlap',
    'query_not_nested', 'classes_not_nested', 'boolean_label', 'labels_missing',
    'scores', 'truth', 'roles',
])
def test_mismatch_missing_duplicate_and_forbidden_values_are_rejected(fault):
    args = fixture(); row = args['splits'][1]
    if fault == 'old_query_missing': row['query_ids'].remove('old-a-q0')
    elif fault == 'old_support_changed': row['support_ids'][0] = 'different-old-support'
    elif fault == 'class_label_mapping': row['support_labels'][:2] = [1, 0]
    elif fault == 'old_class_missing': row['registered_classes'][0] = 'unexpected-class'
    elif fault == 'dg_query_missing': args['frozen_dg'][0]['query_ids'].pop()
    elif fault == 'dg_classes': args['frozen_dg'][0]['classes'][0] = 'unexpected-class'
    elif fault == 'dg_split': args['frozen_dg'][0]['split_id'] = row['split_id']
    elif fault == 'dg_duplicate': args['frozen_dg'].append(deepcopy(args['frozen_dg'][0]))
    elif fault == 'dg_missing': args['frozen_dg'].clear()
    elif fault == 'split_duplicate': args['splits'].append(deepcopy(row))
    elif fault == 'cell_duplicate':
        duplicate = deepcopy(row); duplicate['split_id'] = 'another-same-cell'; args['splits'].append(duplicate)
    elif fault == 'old_only_missing': args['splits'].pop(0)
    elif fault == 'expanded_missing': args['splits'].pop(1)
    elif fault == 'episode_missing': args['splits'] = args['splits'][3:]
    elif fault == 'capsule': row['capsule_id'] = 'other-capsule'
    elif fault == 'query_duplicate': row['query_ids'].append(row['query_ids'][0])
    elif fault == 'support_duplicate': row['support_ids'][0] = row['support_ids'][1]
    elif fault == 'overlap': row['query_ids'][0] = row['support_ids'][0]
    elif fault == 'query_not_nested': args['splits'][2]['query_ids'].remove('new-0-q0')
    elif fault == 'classes_not_nested': args['splits'][2]['registered_classes'][2] = 'different-new-class'
    elif fault == 'boolean_label': row['support_labels'][0] = False
    elif fault == 'labels_missing': row['support_labels'].pop()
    else: row[fault] = ['forbidden']
    with pytest.raises(ValueError):
        validate_pairs(**args)


def test_pairing_does_not_claim_semantic_query_labels_or_state_inheritance():
    rows = validate_pairs(**fixture())
    assert all(r['pairing_scope'] == 'PHYSICAL_IDENTITY_ONLY_NO_TRUTH_OR_SCORE_ACCESS'
               and r['state_inheritance_verified'] is False for r in rows)
    assert all(not ({'scores', 'truth', 'roles', 'accuracy'} & set(r)) for r in rows)
