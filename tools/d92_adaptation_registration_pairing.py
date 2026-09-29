"""Pure identity-only A/B/C pairing for independent, post-prediction analysis.

No files, features, scores, truth or model state are consumed here. A is frozen
DG; B is the method's old-only support fit; C is that method's expanded registry.
Identity pairing does not establish that C inherits B's fitted state.
"""
from collections import defaultdict


SCHEMA = 'd92_adaptation_registration_pairing_v1'
_SPLIT_FIELDS = {
    'split_id', 'capsule_id', 'receiver', 'scenario', 'k', 'support_seed',
    'registered_classes', 'support_ids', 'support_labels', 'query_ids',
}
_SPLIT_OPTIONAL = {'new_count', 'protocol_schema', 'phase2_data_status'}
_DG_FIELDS = {'split_id', 'capsule_id', 'receiver', 'scenario', 'classes', 'query_ids'}
_DG_OPTIONAL = {'mode', 'k', 'support_seed'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _strings(values, name):
    _require(isinstance(values, (list, tuple)) and len(values) > 0,
             name + ' must be a nonempty list')
    _require(all(isinstance(v, str) and v for v in values), name + ' must contain strings')
    _require(len(set(values)) == len(values), name + ' contains duplicate identities')
    return list(values)


def _record(value, required, optional, name):
    _require(isinstance(value, dict), name + ' must be an identity metadata object')
    _require(required <= set(value) <= required | optional,
             name + ' fields missing or unsupported; scores/truth/roles are forbidden')
    for key in ('split_id', 'capsule_id', 'receiver', 'scenario'):
        _require(isinstance(value[key], str) and value[key], name + ': invalid ' + key)


def _episode(value):
    return (value['receiver'], value['scenario'], value['k'], value['support_seed'])


def validate_pairs(*, splits, frozen_dg, old_classes,
                   expected_new_counts=(0, 2, 5, 10, 20), expected_episode_keys=None,
                   model_seed=None, checkpoint_sha256=None):
    """Return one traceable identity pair per episode and registered new count.

    ``splits`` contains normalized physical IDs, not numeric storage indices.
    ``frozen_dg`` contains only prediction *identity metadata*. Its incidental K
    and support seed are ignored. The caller independently binds model/checkpoint
    provenance before calling; these optional context values are only carried
    into the evidence. ``expected_episode_keys`` entries are (RX, scene, K, seed).

    Query membership is used only to establish paired analysis populations, never
    as a predictor input or a label. B's entire query set must occur in C. This
    cannot prove the semantic labels of C-minus-B samples without an independent
    scorer; it does not claim that proof. The function makes no performance gate.
    """
    old = _strings(old_classes, 'old_classes')
    old_set = set(old)
    _require(isinstance(expected_new_counts, (list, tuple)) and expected_new_counts,
             'Expected new-count matrix is required')
    counts = list(expected_new_counts)
    _require(all(type(n) is int and n >= 0 for n in counts)
             and len(set(counts)) == len(counts) and 0 in counts,
             'Expected new counts must be unique nonnegative integers including zero')
    counts.sort()
    _require(isinstance(splits, (list, tuple)) and splits, 'Missing split identities')
    _require(isinstance(frozen_dg, (list, tuple)) and frozen_dg, 'Missing frozen DG identities')
    _require(model_seed is None or type(model_seed) is int, 'Invalid model seed context')
    _require(checkpoint_sha256 is None or (isinstance(checkpoint_sha256, str)
             and len(checkpoint_sha256) == 64
             and all(v in '0123456789abcdef' for v in checkpoint_sha256)),
             'Invalid checkpoint digest context')

    groups = defaultdict(dict)
    split_ids, capsule_ids = set(), set()
    for split in splits:
        _record(split, _SPLIT_FIELDS, _SPLIT_OPTIONAL, 'split')
        _require(split['split_id'] not in split_ids, 'Duplicate split ID')
        split_ids.add(split['split_id']); capsule_ids.add(split['capsule_id'])
        _require(type(split['k']) is int and split['k'] > 0
                 and type(split['support_seed']) is int, 'Invalid K/support seed')
        if 'protocol_schema' in split:
            _require(split['protocol_schema'] == 'p2_min_v1', 'Unexpected protocol schema')
        if 'phase2_data_status' in split:
            _require(split['phase2_data_status'] == 'VALIDATED_ONCE', 'Unexpected capsule data status')
        classes = _strings(split['registered_classes'], 'registered_classes')
        _require(old_set <= set(classes), 'Registered classes omit an old class')
        new_count = len(classes) - len(old)
        _require(new_count in counts, 'Unexpected new count')
        if 'new_count' in split:
            _require(type(split['new_count']) is int and split['new_count'] == new_count,
                     'Declared new count disagrees with registry')
        support = _strings(split['support_ids'], 'support_ids')
        query = _strings(split['query_ids'], 'query_ids')
        _require(not set(support) & set(query), 'Support/query physical IDs overlap')
        labels = split['support_labels']
        _require(isinstance(labels, (list, tuple)) and len(labels) == len(support)
                 and all(type(v) is int and 0 <= v < len(classes) for v in labels),
                 'Invalid support label/class mapping')
        by_class = {cls: set() for cls in classes}
        for physical_id, label in zip(support, labels):
            by_class[classes[label]].add(physical_id)
        _require(all(len(v) == split['k'] for v in by_class.values()),
                 'Support class counts do not equal K')
        key = _episode(split)
        _require(new_count not in groups[key], 'Duplicate episode/new-count cell')
        groups[key][new_count] = dict(record=split, classes=classes,
            support=by_class, query=set(query))
    _require(len(capsule_ids) == 1, 'Mixed capsule identities')
    capsule_id = next(iter(capsule_ids))
    if expected_episode_keys is not None:
        expected = list(expected_episode_keys)
        _require(all(isinstance(k, (list, tuple)) and len(k) == 4 for k in expected),
                 'Expected episode keys must be RX/scene/K/seed tuples')
        expected = [tuple(k) for k in expected]
        _require(len(set(expected)) == len(expected), 'Duplicate expected episode key')
        _require(set(groups) == set(expected), 'Missing or unexpected episode group')
    for cells in groups.values():
        _require(set(cells) == set(counts), 'Missing old-only or expanded new-count cell')

    dg_by_scene = {}
    for record in frozen_dg:
        _record(record, _DG_FIELDS, _DG_OPTIONAL, 'frozen DG')
        _require(record['capsule_id'] == capsule_id, 'DG capsule mismatch')
        if 'mode' in record:
            _require(record['mode'] == 'frozen_dg', 'Expected frozen DG identity record')
        classes = _strings(record['classes'], 'DG classes')
        _require(set(classes) == old_set, 'DG class registry differs from old classes')
        query = set(_strings(record['query_ids'], 'DG query_ids'))
        key = (record['receiver'], record['scenario'])
        _require(key not in dg_by_scene, 'Duplicate DG RX/scene record')
        dg_by_scene[key] = dict(record=record, classes=classes, query=query)
    _require(set(dg_by_scene) == {(key[0], key[1]) for key in groups},
             'Missing or unexpected DG RX/scene record')

    result = []
    for key, cells in sorted(groups.items()):
        b = cells[0]
        a = dg_by_scene[key[:2]]
        _require(a['query'] == b['query'], 'DG and old-only query ID sets differ')
        old_only_split_ids = {entry[0]['record']['split_id'] for episode, entry in groups.items()
                              if episode[:2] == key[:2]}
        _require(a['record']['split_id'] in old_only_split_ids,
                 'DG split ID is not a bound old-only split')
        previous = b
        for new_count in counts:
            c = cells[new_count]
            _require(all(c['support'][cls] == b['support'][cls] for cls in old),
                     'Old support physical ID/class mapping changed')
            _require(set(previous['classes']) <= set(c['classes']),
                     'Registered class identities are not nested')
            _require(previous['query'] <= c['query'], 'Query ID sets are not nested')
            _require(b['query'] <= c['query'], 'Expanded split is missing old query IDs')
            row = dict(schema=SCHEMA, status='VERIFIED', capsule_id=capsule_id,
                model_seed=model_seed, checkpoint_sha256=checkpoint_sha256,
                receiver=key[0], scenario=key[1], k=key[2], support_seed=key[3],
                new_count=new_count, episode_key=list(key), pair_key=[*key, new_count],
                a_split_id=a['record']['split_id'], b_split_id=b['record']['split_id'],
                c_split_id=c['record']['split_id'], a_classes=list(a['classes']),
                b_classes=list(b['classes']), c_classes=list(c['classes']),
                a_query_count=len(a['query']), b_query_count=len(b['query']),
                c_query_count=len(c['query']), old_query_ids=sorted(b['query']),
                c_query_ids=sorted(c['query']),
                old_support_ids=sorted(pid for cls in old for pid in b['support'][cls]),
                old_support_by_class={cls: sorted(b['support'][cls]) for cls in old},
                dg_matching_ignores_k_and_support_seed=True,
                pairing_scope='PHYSICAL_IDENTITY_ONLY_NO_TRUTH_OR_SCORE_ACCESS',
                state_inheritance_verified=False)
            result.append(row)
            previous = c
    return result
