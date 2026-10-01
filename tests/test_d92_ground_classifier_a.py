"""Synthetic original CosFaceHead oracle; no weights/cache/truth/package reads."""
import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys
from typing import Optional

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from cvsrffi.d92_ground_classifier_a import (
    GroundClassifierA, GroundFeatureContract, GroundHeadMetadata, WEIGHT_KEY,
)


def contract(**changes):
    return GroundFeatureContract(**(dict(checkpoint_sha256='a'*64, cache_key='z_id', source_tensor='feat_joint',
        representation='raw', dtype='float32', feature_dim=160) | changes))


def metadata(**changes):
    return GroundHeadMetadata(**(dict(checkpoint_sha256='a'*64, checkpoint_weight_key=WEIGHT_KEY,
        ordered_classes=('tx-z', 'tx-a', 'tx-q', 'tx-b', 'tx-m', 'tx-c'), scale=17.25, norm_eps=1e-4,
        feature_contract=contract(), source_only_verdict='MATCHED_SOURCE_ONLY_SCRATCH',
        target_access_before_freeze=False, checkpoint_inheritance=(), logit_corrections='none') | changes))


@pytest.fixture(scope='module')
def original_head_class():
    # Execute the unmodified source class, avoiding unrelated encoder imports.
    source = ast.parse((ROOT/'code/model.py').read_text(encoding='utf-8'))
    node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'CosFaceHead')
    scope = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, Optional=Optional)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT/'code/model.py'), 'exec'), scope)
    return scope['CosFaceHead']


def tensors():
    gen = torch.Generator().manual_seed(163)
    return torch.randn(6, 160, generator=gen), torch.randn(11, 160, generator=gen)


def test_scores_match_original_label_free_cosface_float32(original_head_class):
    weight, features = tensors(); meta = metadata()
    original = original_head_class(160, 6, s=meta.scale, m=.91).eval()
    with torch.no_grad(): original.weight.copy_(weight); expected = original(features)
    head = GroundClassifierA(weight=weight, metadata=meta)
    actual = head.score(z_id=features, feature_contract=contract())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (11, 6) and actual.dtype == torch.float32 and not actual.requires_grad
    assert head.classes == meta.ordered_classes
    assert head.predict(z_id=features, feature_contract=contract()) == tuple(meta.ordered_classes[i] for i in expected.argmax(1))


def test_original_eps_zero_tiny_and_nonunit_feature_and_weight_boundaries(original_head_class):
    weight, features = tensors(); weight[0].zero_(); weight[1] *= 1e-8
    features[0].zero_(); features[1] *= 1e-8; features[2] *= 123.
    meta = metadata(scale=3.75); original = original_head_class(160, 6, s=meta.scale).eval()
    with torch.no_grad(): original.weight.copy_(weight); expected = original(features)
    head = GroundClassifierA(weight=weight, metadata=meta)
    torch.testing.assert_close(head.score(z_id=features, feature_contract=contract()), expected, rtol=0, atol=0)
    assert head.predict(z_id=features[:1], feature_contract=contract()) == ('tx-z',)


def test_original_column_order_ties_and_permutation_are_not_sorted():
    weight = torch.zeros(6, 160); weight[:, 0] = 1.
    features = torch.ones(2, 160); head = GroundClassifierA(weight=weight, metadata=metadata())
    assert head.predict(z_id=features, feature_contract=contract()) == ('tx-z', 'tx-z')
    weight, features = tensors(); order = torch.tensor([3, 0, 5, 1, 4, 2]); meta = metadata()
    original = GroundClassifierA(weight=weight, metadata=meta)
    permuted = GroundClassifierA(weight=weight[order], metadata=replace(meta,
        ordered_classes=tuple(meta.ordered_classes[i] for i in order)))
    torch.testing.assert_close(permuted.score(z_id=features, feature_contract=contract()),
        original.score(z_id=features, feature_contract=contract())[:, order])
    assert permuted.predict(z_id=features, feature_contract=contract()) == original.predict(z_id=features, feature_contract=contract())


def test_recordwise_inference_empty_input_no_grad_and_autocast():
    weight, features = tensors(); features.requires_grad_(True)
    head = GroundClassifierA(weight=weight, metadata=metadata())
    scores = head.score(z_id=features, feature_contract=contract())
    separately = torch.cat([head.score(z_id=row[None], feature_contract=contract()) for row in features])
    torch.testing.assert_close(separately, scores)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        auto = head.score(z_id=features, feature_contract=contract())
    torch.testing.assert_close(auto, scores, rtol=0, atol=0)
    assert not scores.requires_grad and features.grad is None
    empty = torch.empty(0, 160)
    assert head.score(z_id=empty, feature_contract=contract()).shape == (0, 6)
    assert head.predict(z_id=empty, feature_contract=contract()) == ()


def test_frozen_weight_and_metadata_are_not_caller_mutable():
    weight, features = tensors(); classes = list(metadata().ordered_classes)
    meta = metadata(ordered_classes=classes, checkpoint_inheritance=[])
    head = GroundClassifierA(weight=weight, metadata=meta)
    before = head.score(z_id=features, feature_contract=contract())
    weight.zero_(); classes.reverse()
    torch.testing.assert_close(head.score(z_id=features, feature_contract=contract()), before, rtol=0, atol=0)
    assert head.classes[0] == 'tx-z' and head.metadata.checkpoint_inheritance == ()
    with pytest.raises(FrozenInstanceError): meta.scale = 30.
    with pytest.raises(AttributeError): head._metadata = metadata(scale=30.)
    assert not hasattr(head, 'fit') and not hasattr(head, 'train') and not hasattr(head, 'parameters')


@pytest.mark.parametrize('changes', [
    dict(checkpoint_sha256='unknown'), dict(cache_key='prototype'), dict(source_tensor='feat_id'),
    dict(representation='unit_normalized'), dict(dtype='float64'), dict(feature_dim=96), dict(feature_dim=True),
])
def test_invalid_feature_contract_rejected(changes):
    with pytest.raises(ValueError): contract(**changes)


@pytest.mark.parametrize('changes', [
    dict(checkpoint_sha256='unknown'), dict(checkpoint_weight_key='prototype'),
    dict(ordered_classes=('a',)*6), dict(ordered_classes=('a','b')), dict(ordered_classes=(1,2,3,4,5,6)),
    dict(scale=None), dict(scale=0.), dict(scale=True), dict(scale=float('inf')), dict(scale=float('nan')),
    dict(norm_eps=1e-12), dict(norm_eps=None), dict(feature_contract=contract(checkpoint_sha256='b'*64)),
    dict(source_only_verdict='UNKNOWN'), dict(target_access_before_freeze=True), dict(target_access_before_freeze=0),
    dict(checkpoint_inheritance=['unverified-parent']), dict(logit_corrections='UNKNOWN'), dict(logit_corrections='Dual'),
])
def test_illegal_or_unknown_head_declarations_rejected(changes):
    with pytest.raises((ValueError, TypeError)): metadata(**changes)


def test_illegal_tensor_inputs_and_different_checkpoint_binding_rejected():
    weight, features = tensors(); head = GroundClassifierA(weight=weight, metadata=metadata())
    for invalid in (weight.double(), weight[:5], torch.zeros(6, 159), torch.full((6,160), float('nan')), weight.tolist()):
        with pytest.raises(ValueError): GroundClassifierA(weight=invalid, metadata=metadata())
    for invalid in (features.double(), features.half(), features.to(torch.int32), features[0], features[:, :159],
                    torch.full((2,160), float('inf')), features.tolist()):
        with pytest.raises(ValueError): head.score(z_id=invalid, feature_contract=contract())
    with pytest.raises(ValueError): head.score(z_id=features, feature_contract=contract(checkpoint_sha256='b'*64))
    with pytest.raises(TypeError): GroundClassifierA(weight=weight)
    with pytest.raises(TypeError): head.score(z_id=features)


@pytest.mark.parametrize('forbidden', ['labels', 'truth', 'roles', 'quota', 'class_count', 'support_labels',
    'query_labels', 'source_features', 'prototype', 'temperature', 'corrections'])
def test_no_target_fit_truth_role_or_alternate_head_inputs(forbidden):
    weight, features = tensors(); head = GroundClassifierA(weight=weight, metadata=metadata())
    with pytest.raises(TypeError): head.score(z_id=features, feature_contract=contract(), **{forbidden: object()})
    with pytest.raises(TypeError): head.predict(z_id=features, feature_contract=contract(), **{forbidden: object()})
