"""Synthetic aggregate packets exercise the existing reader and geometry bridge."""
import json

import numpy as np
import pytest

from cvsrffi import phase1_center_lowrank_prototype_bundle as codec
from cvsrffi.d92_proto_frame_ground_geometry import load_proto_frame_ground_geometry


SHA = 'a' * 64
CLASSES = ('old-5', 'old-1', 'old-4', 'old-2', 'old-3', 'old-0')


def packet(root):
    core = np.zeros((6, 160), dtype=np.int8)
    core[np.arange(6), np.arange(6)] = 50
    payload = dict(
        schema=np.asarray(codec.SCHEMA), feature_schema=np.asarray(codec.FEATURE_SCHEMA),
        residual_rank=np.asarray(3, dtype=np.int16), center_domain_handle=np.asarray('ground-0'),
        domain_registry=np.asarray(['ground-0', 'ground-1']),
        residual_domain_registry=np.asarray(['ground-1']), class_registry=np.asarray(CLASSES),
        core_q=core, core_scale=np.full(6, .01, dtype=np.float16),
        residual_basis_q=np.zeros((6, 3, 160), dtype=np.int8),
        residual_basis_scale=np.ones((6, 3), dtype=np.float16),
        residual_coeff_q=np.zeros((1, 6, 3), dtype=np.int8),
        residual_coeff_scale=np.ones((1, 6), dtype=np.float16),
        radius_q=np.zeros((2, 6), dtype=np.int8), radius_scale=np.ones(6, dtype=np.float16),
    )
    manifest = dict(
        schema=codec.SCHEMA, metadata_schema='cvs.matched.ground.v2', checkpoint_sha256=SHA,
        source_prototype_artifact_sha256='b' * 64,
        provenance_status='CURRENT_FINAL_SCRATCH_EXACT_SOURCE_L', formal_phase2_eligible=True,
        feature_dim=160, residual_rank=3, radius_histogram_bins=4096,
        member_allowlist=[codec.NPZ_NAME], npz_member_allowlist=sorted(codec.ALLOWED_NPZ_MEMBERS),
        resource_audit=codec._numeric_resource_audit(payload), source_role='L_s',
        target_access=False, component_state='CURRENT_MATCHED_SOURCE_ONLY_V2',
        historical_outer_signature_claim=False,
    )
    write(root, payload, manifest)
    return payload, manifest


def write(root, payload, manifest):
    np.savez_compressed(root/codec.NPZ_NAME, **payload)
    (root/'manifest.json').write_text(json.dumps(manifest)+'\n', encoding='utf-8')


def load(root, **overrides):
    arguments = dict(expected_checkpoint_sha256=SHA, expected_classes=CLASSES, already_deployed=False)
    arguments.update(overrides)
    return load_proto_frame_ground_geometry(root, **arguments)


def test_existing_center_canonical_frame_and_separate_transfer_cost(tmp_path):
    packet(tmp_path)
    frame, audit = load(tmp_path)
    assert frame.classes == tuple(sorted(CLASSES))
    assert frame.Q.shape == (160, 5)
    center_rows = {name: index for index, name in enumerate(CLASSES)}
    expected = np.zeros((160, 5))
    for column, name in enumerate(sorted(CLASSES)[:-1]):
        expected[center_rows[name], column] = 1
        expected[center_rows['old-5'], column] = -1
    np.testing.assert_array_equal(frame.Q, expected)
    assert not frame.Q.flags.writeable
    assert audit['decoded_center_bytes'] == 6*160*4
    total = sum((tmp_path/name).stat().st_size for name in (codec.NPZ_NAME, 'manifest.json'))
    assert audit['ground_payload_audit']['incremental_transfer_bytes'] == total
    assert load(tmp_path, already_deployed=True)[1]['ground_payload_audit']['incremental_transfer_bytes'] == 0
    assert audit['newly_generated_ground_statistics_bytes'] == 0
    assert audit['native_wire_bytes'] is None
    assert not audit['prototype_teacher_targets'] and not audit['support_labels_read']
    with pytest.raises(TypeError):
        audit['native_wire_bytes'] = 0


def test_residuals_and_radii_do_not_define_reference_geometry(tmp_path):
    payload, manifest = packet(tmp_path)
    before, _ = load(tmp_path)
    payload['residual_basis_q'][:] = 100
    payload['residual_coeff_q'][:] = 17
    payload['radius_q'][:] = 99
    manifest['resource_audit'] = codec._numeric_resource_audit(payload)
    write(tmp_path, payload, manifest)
    after, audit = load(tmp_path)
    np.testing.assert_array_equal(before.Q, after.Q)
    assert not audit['residual_domain_reconstruction']


@pytest.mark.parametrize('fault', ['checkpoint', 'target', 'source_members', 'class_order'])
def test_existing_input_contract_still_rejects_invalid_packet(tmp_path, fault):
    payload, manifest = packet(tmp_path)
    if fault == 'checkpoint':
        manifest['checkpoint_sha256'] = 'c' * 64
    elif fault == 'target':
        manifest['target_access'] = True
    elif fault == 'source_members':
        payload['source_samples'] = np.zeros((1, 160))
    else:
        payload['class_registry'] = np.asarray(CLASSES[::-1])
    write(tmp_path, payload, manifest)
    with pytest.raises(ValueError):
        load(tmp_path)


@pytest.mark.parametrize('classes', [CLASSES[:5], CLASSES+(CLASSES[0],), 'old-0'])
def test_rejects_wrong_class_geometry_before_io(tmp_path, classes):
    with pytest.raises(ValueError):
        load(tmp_path/'not-created', expected_classes=classes)
