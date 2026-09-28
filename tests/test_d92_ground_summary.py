"""Synthetic aggregate-only tests; no source or target observations are used."""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import numpy as np
import pytest

from cvsrffi import d92_ground_summary as reader
from cvsrffi import phase1_center_lowrank_prototype_bundle as codec


SHA = 'a' * 64
CLASSES = ('synthetic-old-0', 'synthetic-old-1')


def fixture_files(tmp_path):
    domains = ['domain-0', 'domain-1', 'domain-2', 'domain-3']
    payload = dict(schema=np.asarray(codec.SCHEMA), feature_schema=np.asarray(codec.FEATURE_SCHEMA),
        residual_rank=np.asarray(3, dtype=np.int16), center_domain_handle=np.asarray(domains[0]),
        domain_registry=np.asarray(domains), residual_domain_registry=np.asarray(domains[1:]),
        class_registry=np.asarray(CLASSES), core_q=np.ones((2, 160), dtype=np.int8),
        core_scale=np.full(2, 0.01, dtype=np.float16), residual_basis_q=np.ones((2, 3, 160), dtype=np.int8),
        residual_basis_scale=np.full((2, 3), 0.02, dtype=np.float16),
        residual_coeff_q=np.ones((3, 2, 3), dtype=np.int8),
        residual_coeff_scale=np.full((3, 2), 0.03, dtype=np.float16),
        radius_q=np.ones((4, 2), dtype=np.int8), radius_scale=np.full(2, 0.04, dtype=np.float16))
    manifest = dict(schema=codec.SCHEMA, metadata_schema='cvs.matched.ground.v2',
        checkpoint_sha256=SHA, source_prototype_artifact_sha256='b' * 64,
        provenance_status='CURRENT_FINAL_SCRATCH_EXACT_SOURCE_L', formal_phase2_eligible=True,
        feature_dim=160, residual_rank=3, radius_histogram_bins=4096,
        member_allowlist=[codec.NPZ_NAME], npz_member_allowlist=sorted(codec.ALLOWED_NPZ_MEMBERS),
        resource_audit=dict(codec._numeric_resource_audit(payload), reconstruction_rmse=0.001),
        source_role='L_s', target_access=False, component_state='CURRENT_MATCHED_SOURCE_ONLY_V2',
        historical_outer_signature_claim=False)
    save(tmp_path, payload, manifest)
    return payload, manifest


def save(root, payload, manifest):
    np.savez_compressed(root / codec.NPZ_NAME, **payload)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def load(root, **overrides):
    return reader.load_ground_summary(root, **dict(
        expected_checkpoint_sha256=SHA, expected_classes=CLASSES, **overrides))


def test_byte_audit_separates_numeric_metadata_disk_and_incremental_transfer(tmp_path):
    payload, _ = fixture_files(tmp_path)
    summary = load(tmp_path)
    audit = summary.payload_audit
    numeric = sum(payload[name].nbytes for name in reader.NUMERIC_MEMBERS)
    metadata = sum(a.nbytes for name, a in payload.items() if name not in reader.NUMERIC_MEMBERS)
    assert audit['numeric_array_bytes'] == numeric
    assert audit['registry_schema_bytes'] == metadata
    assert audit['total_array_bytes'] == numeric + metadata
    assert audit['npz_file_bytes'] == (tmp_path / codec.NPZ_NAME).stat().st_size
    assert audit['manifest_file_bytes'] == (tmp_path / 'manifest.json').stat().st_size
    total = audit['npz_file_bytes'] + audit['manifest_file_bytes']
    assert audit['total_file_bytes'] == audit['incremental_transfer_bytes'] == total
    assert load(tmp_path, already_deployed=True).payload_audit['incremental_transfer_bytes'] == 0
    assert summary.feature_schema == codec.FEATURE_SCHEMA
    assert summary.class_registry == CLASSES


@pytest.mark.parametrize('field,value', [
    ('schema', 'wrong'), ('metadata_schema', 'wrong'), ('provenance_status', 'unverified'),
    ('component_state', 'unfrozen'), ('checkpoint_sha256', 'c' * 64), ('feature_dim', 288),
    ('feature_schema', 'wrong'), ('formal_phase2_eligible', False), ('target_access', True),
    ('source_role', 'query'), ('source_samples_path', '/forbidden/source.npz'),
    ('member_allowlist', [codec.NPZ_NAME, 'samples.npz']),
])
def test_rejects_manifest_contract_or_illegal_source_member(tmp_path, field, value):
    payload, manifest = fixture_files(tmp_path)
    manifest[field] = value
    save(tmp_path, payload, manifest)
    with pytest.raises(ValueError):
        load(tmp_path)


@pytest.mark.parametrize('fault', ['schema', 'feature_schema', 'classes', 'sample_embeddings', 'dtype',
                                  'nested_source_audit', 'source_path_in_numeric_audit'])
def test_rejects_npz_or_nested_audit_violation(tmp_path, fault):
    payload, manifest = fixture_files(tmp_path)
    if fault in ('schema', 'feature_schema'):
        payload[fault] = np.asarray('wrong')
    elif fault == 'classes':
        payload['class_registry'] = np.asarray(CLASSES[::-1])
    elif fault == 'sample_embeddings':
        payload[fault] = np.zeros((10, 160), dtype=np.float32)
    elif fault == 'dtype':
        payload['core_q'] = payload['core_q'].astype(np.float32)
    elif fault == 'nested_source_audit':
        manifest['resource_audit']['source_path'] = '/forbidden/source.npy'
    else:
        manifest['resource_audit']['reconstruction_rmse'] = '/forbidden/source.npy'
    save(tmp_path, payload, manifest)
    with pytest.raises(ValueError):
        load(tmp_path)


def test_summary_component_nested_metadata_and_temporary_domain_are_immutable(tmp_path):
    fixture_files(tmp_path)
    summary = load(tmp_path)
    component = summary.component
    before = {name: array.tobytes() for name, array in vars(component).items() if isinstance(array, np.ndarray)}
    with pytest.raises(FrozenInstanceError):
        summary.component = None
    with pytest.raises(FrozenInstanceError):
        component.core_q = np.zeros((2, 160), dtype=np.int8)
    with pytest.raises(ValueError):
        component.core_q.setflags(write=True)
    with pytest.raises(TypeError):
        component.manifest['resource_audit']['reconstruction_rmse'] = 7
    with pytest.raises(TypeError):
        summary.payload_audit['per_array_bytes']['core_q'] = 7
    first = summary.reconstruct_domain('domain-1')
    assert first.shape == (2, 160) and first.dtype == np.float32
    with pytest.raises(ValueError):
        first.setflags(write=True)
    with pytest.raises(ValueError):
        summary.radius_for_domain('domain-1')[0] = 5
    second = summary.reconstruct_domain('domain-1')
    assert first is not second and not np.shares_memory(first, second)
    np.testing.assert_array_equal(first, second)
    assert before == {name: array.tobytes() for name, array in vars(component).items() if isinstance(array, np.ndarray)}
    assert not any(isinstance(value, np.ndarray) and value.dtype == np.float32 for value in vars(component).values())
    with pytest.raises(ValueError, match='unknown'):
        summary.reconstruct_domain('not-registered')


def test_reads_only_existing_two_files_without_builders_or_source_access(tmp_path, monkeypatch):
    fixture_files(tmp_path)
    allowed = {tmp_path / 'manifest.json', tmp_path / codec.NPZ_NAME}
    opened = []
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path in allowed
        mode = args[0] if args else kwargs.get('mode', 'r')
        assert mode == 'rb'
        opened.append(path)
        return original_open(path, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('An offline builder must never run in this reader')
    monkeypatch.setattr(Path, 'open', guarded_open)
    monkeypatch.setattr(codec, 'compress_v1_dense_component', forbidden)
    monkeypatch.setattr(codec, 'build_center_lowrank_component', forbidden)
    summary = load(tmp_path)
    for domain in summary.domain_registry:
        summary.reconstruct_domain(domain)
        summary.radius_for_domain(domain)
    assert opened == [tmp_path / 'manifest.json', tmp_path / codec.NPZ_NAME]
    assert summary.payload_audit['source_samples_read'] is False
    assert summary.payload_audit['checkpoint_file_read'] is False
    assert summary.payload_audit['dense_bank_persisted'] is False
    assert summary.payload_audit['dequantized_domain_cache'] is False
