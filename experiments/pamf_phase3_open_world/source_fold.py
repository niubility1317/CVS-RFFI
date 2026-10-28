"""A source-only development fold; never formal target-unknown evidence.

The original L_s array may be loaded by the private selector, but only selected
L_s rows enter optimization.  The complete original V remains read-only
validation; U_s, original datasets and target inputs are never opened here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


SOURCE_RX = [1, 3, 4, 6, 8]
SOURCE_DAYS = [1, 2, 3]
METADATA_FIELDS = frozenset({
    'rx_i', 'day_i', 'eq_i', 'sig_i', 'base_index', 'domain', 'ids', 'y',
})
FORBIDDEN_CONTRACT_FIELDS = frozenset({
    'target_inputs', 'target_truth', 'truth', 'query', 'query_ids', 'views_root',
    'tests', 'dataset_path', 'checkpoint', 'teacher_checkpoint',
    'target_receivers', 'target_rxs', 'target_days',
})


def _spec_and_contract(contract: Mapping[str, Any], fold_spec: Mapping[str, Any]) -> None:
    if (not isinstance(fold_spec, dict)
            or set(fold_spec) != {'unknown_source_labels', 'heldout_source_rx'}):
        raise ValueError('The source fold requires exactly its two declared fields')
    unknown = fold_spec['unknown_source_labels']
    heldout = fold_spec['heldout_source_rx']
    if (not isinstance(unknown, list) or any(type(v) is not int for v in unknown)
            or len(set(unknown)) != len(unknown) or unknown != [0, 1]
            or type(heldout) is not int or heldout != 8):
        raise ValueError('This source fold requires unknown labels [0,1] and held-out RX8; bools are invalid')
    if not isinstance(contract, dict) or set(contract) & FORBIDDEN_CONTRACT_FIELDS:
        raise ValueError('Source contract contains target/query/checkpoint inputs')
    if (contract.get('source_rxs') != SOURCE_RX
            or any(type(v) is not int for v in contract.get('source_rxs', []))
            or contract.get('source_days') != SOURCE_DAYS
            or any(type(v) is not int for v in contract.get('source_days', []))
            or type(contract.get('num_classes')) is not int
            or contract['num_classes'] != 6
            or contract.get('target_access_before_freeze') is not False
            or contract.get('target_constructed', False) is not False):
        raise ValueError('Only the original six-class Main split source contract is accepted')
    classes = contract.get('classes')
    if (not isinstance(classes, list) or len(classes) != 6
            or any(not isinstance(value, str) or not value for value in classes)
            or len(set(classes)) != 6):
        raise ValueError('Source contract must preserve its ordered six-class map')
    counts = contract.get('counts', {})
    if (not isinstance(counts, dict)
            or any(type(counts.get(role)) is not int or counts[role] != expected
                   for role, expected in (('L_s', 6300), ('V', 27000)))):
        raise ValueError('Source contract L_s/V counts differ from 6300/27000')
    role_ids = contract.get('role_ids')
    if not isinstance(role_ids, dict) or any(role not in role_ids for role in ('L_s', 'V')):
        raise ValueError('Source contract is missing L_s/V physical role IDs')


def _metadata(
    values: Mapping[str, Any], role: str, contract: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if not isinstance(values, Mapping) or set(values) != METADATA_FIELDS:
        raise ValueError(f'{role} metadata fields must be source physical fields and y only')
    result = {key: np.asarray(value) for key, value in values.items()}
    count = contract['counts'][role]
    for key, array in result.items():
        if array.ndim != 1 or len(array) != count:
            raise ValueError(f'{role} metadata count/shape differs for {key}')
        if key != 'ids' and (array.dtype.kind not in 'iu' or (array < 0).any()):
            raise ValueError(f'{role} metadata {key} must be nonnegative integers; bools are invalid')
    ids = result['ids']
    expected = np.asarray(contract['role_ids'][role])
    if (ids.dtype.kind != 'U' or expected.dtype.kind != 'U'
            or expected.ndim != 1 or len(expected) != count
            or any(not str(value) for value in ids)
            or len(set(ids.tolist())) != count
            or len(set(expected.tolist())) != count):
        raise ValueError(f'{role} physical IDs must be unique Unicode IDs with the original count')
    if not np.array_equal(ids, expected):
        raise ValueError(f'{role} physical ID order differs from the existing contract')
    if (not np.isin(result['rx_i'], SOURCE_RX).all()
            or not np.isin(result['day_i'], SOURCE_DAYS).all()
            or not (result['eq_i'] == 1).all()):
        raise ValueError(f'{role} metadata contains a target RX/day or a different IQ representation')
    if set(result['y'].tolist()) != set(range(6)):
        raise ValueError(f'{role} source class label coverage must contain all six classes')
    expected_domain = (np.searchsorted(SOURCE_RX, result['rx_i']) * len(SOURCE_DAYS)
                       + np.searchsorted(SOURCE_DAYS, result['day_i']))
    if not np.array_equal(result['domain'], expected_domain):
        raise ValueError(f'{role} source RX/day domain metadata differs')
    # This fold is deliberately restricted to the existing balanced Main roles.
    per_cell = 70 if role == 'L_s' else 300
    for label in range(6):
        for receiver in SOURCE_RX:
            for day in SOURCE_DAYS:
                n = np.count_nonzero((result['y'] == label)
                    & (result['rx_i'] == receiver) & (result['day_i'] == day))
                if n != per_cell:
                    raise ValueError(f'{role} source class/RX/day cell count differs')
    return result


def select_fold(
    l_metadata: Mapping[str, Any],
    v_metadata: Mapping[str, Any],
    contract: Mapping[str, Any],
    fold_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Select legal training indices and remap labels with NumPy only."""
    _spec_and_contract(contract, fold_spec)
    labeled = _metadata(l_metadata, 'L_s', contract)
    validation = _metadata(v_metadata, 'V', contract)
    if set(labeled['ids'].tolist()) & set(validation['ids'].tolist()):
        raise ValueError('L_s and V physical roles overlap instead of being disjoint')
    unknown = list(fold_spec['unknown_source_labels'])
    heldout = fold_spec['heldout_source_rx']
    known = [label for label in range(6) if label not in unknown]
    training_receivers = [rx for rx in SOURCE_RX if rx != heldout]
    indices = np.flatnonzero(~np.isin(labeled['y'], unknown)
                            & (labeled['rx_i'] != heldout)).astype(np.int64)
    remap = np.empty(6, dtype=np.int64)
    for model_label, source_label in enumerate(known + unknown):
        remap[source_label] = model_label
    train_y = remap[labeled['y'][indices]]
    validation_y = remap[validation['y']]
    if (len(indices) != 3360 or set(train_y.tolist()) != set(range(4))
            or np.isin(labeled['y'][indices], unknown).any()
            or (labeled['rx_i'][indices] == heldout).any()):
        raise ValueError('Unknown TX or held-out RX entered the source training fold')
    heldout_mask = validation['rx_i'] == heldout
    unknown_mask = np.isin(validation['y'], unknown)
    counts = dict(source_L_s=6300, train=int(len(indices)), validation=27000,
        validation_heldout_receiver=int(heldout_mask.sum()),
        validation_known=int((~unknown_mask).sum()),
        validation_unknown=int(unknown_mask.sum()),
        validation_heldout_receiver_known=int((heldout_mask & ~unknown_mask).sum()),
        validation_heldout_receiver_unknown=int((heldout_mask & unknown_mask).sum()))
    if counts['validation_heldout_receiver'] != 5400:
        raise ValueError('The complete V held-out source RX must contain 5400 observations')
    manifest = dict(schema='pamf_phase3_source_fold_v1', source_only_development=True,
        formal_target_unknown_evidence=False, source_receivers=SOURCE_RX.copy(),
        source_days=SOURCE_DAYS.copy(), known_source_labels=known.copy(),
        unknown_source_labels=unknown.copy(), training_receivers=training_receivers.copy(),
        heldout_receiver=heldout, heldout_source_rx=heldout,
        training_role='selected_L_s_only',
        validation_policy='complete_V_readonly_no_updates', V_split=False,
        only_selected_train_ids_enter_optimization=True,
        full_L_loaded_for_private_selection_only=True, U_s_opened=False,
        target_access=False, model_label_to_source_label=known + unknown,
        source_class_names=list(contract['classes']), counts=counts.copy())
    return dict(train_indices=indices, train_y=train_y.copy(),
        validation_y=validation_y.copy(), known_source_labels=known,
        unknown_source_labels=unknown, training_receivers=training_receivers,
        heldout_receiver=heldout, train_ids=labeled['ids'][indices].copy(),
        validation_ids=validation['ids'].copy(),
        validation_receivers=validation['rx_i'].astype(np.int64, copy=True),
        counts=counts, manifest=manifest)


def load_fold(
    source_cache: str | Path,
    contract: Mapping[str, Any],
    fold_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Use the existing L_s/V loaders after metadata validation; never open U_s.

    The caller activates the existing base code before invoking this function.
    No checkpoint, original pickle, target data or query interface is accepted.
    """
    _spec_and_contract(contract, fold_spec)
    root = Path(source_cache)
    metadata = {}
    for role in ('L_s', 'V'):
        with np.load(root / f'{role}_metadata.npz', allow_pickle=False) as stored:
            metadata[role] = {name: stored[name].copy() for name in stored.files}
    result = select_fold(metadata['L_s'], metadata['V'], contract, fold_spec)
    from experiments.cric_dura_comparison.f_data import SourceDataset
    import torch

    labeled = SourceDataset(root, 'L_s', contract)
    validation = SourceDataset(root, 'V', contract)
    if (not isinstance(labeled.x, torch.Tensor)
            or not isinstance(validation.x, torch.Tensor)
            or tuple(labeled.x.shape) != (6300, 2, 256)
            or tuple(validation.x.shape) != (27000, 2, 256)):
        raise ValueError('Existing source loaders must return complete original L_s/V tensors')
    indices = torch.tensor(result['train_indices'].tolist(), dtype=torch.long,
                           device=labeled.x.device)
    result['train_x'] = labeled.x.index_select(0, indices)
    result['validation_x'] = validation.x
    return result
