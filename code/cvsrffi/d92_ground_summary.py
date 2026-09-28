"""Read an existing frozen quantized ground summary without source-data access.

The only files opened are the supplied component's manifest and aggregate NPZ.
No checkpoint, source samples, source loaders, sample features, or builder are
accepted. Domain reconstructions are temporary read-only arrays, never cached.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from . import phase1_center_lowrank_prototype_bundle as codec


NUMERIC_MEMBERS = (
    'core_q', 'core_scale', 'residual_basis_q', 'residual_basis_scale',
    'residual_coeff_q', 'residual_coeff_scale', 'radius_q', 'radius_scale',
)
MATCHED_MANIFEST_FIELDS = frozenset({
    'schema', 'metadata_schema', 'checkpoint_sha256', 'source_prototype_artifact_sha256',
    'provenance_status', 'formal_phase2_eligible', 'feature_dim', 'residual_rank',
    'radius_histogram_bins', 'member_allowlist', 'npz_member_allowlist', 'resource_audit',
    'source_role', 'target_access', 'component_state', 'historical_outer_signature_claim',
})
RESOURCE_METADATA_FIELDS = frozenset({
    'center_domain_handle', 'center_domain_index_in_retained_registry', 'center_domain_policy',
    'residual_rank', 'svd_sign_canonicalization', 'rounding_rule', 'radius_provenance',
    'mean_reconstruction_cosine', 'min_reconstruction_cosine', 'mean_reconstruction_angle_deg',
    'max_reconstruction_angle_deg', 'reconstruction_rmse', 'radius_max_abs_error', 'radius_mean_abs_error',
})


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{label} must be a lowercase SHA256 digest')
    return value


@dataclass(frozen=True)
class GroundSummary:
    component: codec.CenterLowRankPrototypeComponent
    payload_audit: Mapping[str, Any]

    @property
    def domain_registry(self):
        return self.component.domain_registry

    @property
    def class_registry(self):
        return self.component.class_registry

    @property
    def feature_schema(self):
        return codec.FEATURE_SCHEMA

    def reconstruct_domain(self, domain_handle: str) -> np.ndarray:
        return self.component.reconstruct_domain(domain_handle)

    def radius_for_domain(self, domain_handle: str) -> np.ndarray:
        return self.component.radius_for_domain(domain_handle)


def load_ground_summary(
    component_dir: str | Path, *, expected_checkpoint_sha256: str,
    expected_classes: Sequence[str], already_deployed: bool = False,
) -> GroundSummary:
    """Load only the existing matched v2 ground component and audit its bytes.

    ``already_deployed`` describes this component's delivery state, not its
    creation time. Its incremental transfer is zero only when already present
    at the intended receiver. Otherwise the full two-file size is reported.
    Numeric and registry/schema bytes are ndarray storage sizes; NPZ disk bytes
    separately include compression, array headers, and archive overhead.
    """
    expected_sha = _sha(expected_checkpoint_sha256, 'expected_checkpoint_sha256')
    if isinstance(expected_classes, (str, bytes)):
        raise ValueError('expected_classes must be the ordered class registry')
    classes = tuple(expected_classes)
    if not classes or any(not isinstance(c, str) or not c for c in classes) or len(set(classes)) != len(classes):
        raise ValueError('expected_classes must contain unique nonempty class identifiers')
    if type(already_deployed) is not bool:
        raise ValueError('already_deployed must be an explicit boolean')
    root = Path(component_dir)
    manifest_bytes = (root / codec.MANIFEST_NAME).read_bytes()
    manifest = json.loads(manifest_bytes.decode('utf-8'))
    if (not isinstance(manifest, dict) or set(manifest) - {'feature_schema'} != MATCHED_MANIFEST_FIELDS
            or manifest.get('schema') != codec.SCHEMA
            or manifest.get('metadata_schema') != 'cvs.matched.ground.v2'
            or manifest.get('provenance_status') != 'CURRENT_FINAL_SCRATCH_EXACT_SOURCE_L'
            or manifest.get('component_state') != 'CURRENT_MATCHED_SOURCE_ONLY_V2'
            or manifest.get('formal_phase2_eligible') is not True
            or manifest.get('source_role') != 'L_s' or manifest.get('target_access') is not False
            or manifest.get('historical_outer_signature_claim') is not False):
        raise ValueError('Ground manifest schema, member allowlist, or source-only provenance mismatch')
    if manifest['checkpoint_sha256'] != expected_sha:
        raise ValueError('Ground/checkpoint SHA256 mismatch')
    _sha(manifest['source_prototype_artifact_sha256'], 'source_prototype_artifact_sha256')
    if (manifest.get('feature_schema', codec.FEATURE_SCHEMA) != codec.FEATURE_SCHEMA
            or manifest['feature_dim'] != codec.FEATURE_DIM or manifest['residual_rank'] != codec.RESIDUAL_RANK
            or manifest['radius_histogram_bins'] != 4096):
        raise ValueError('Ground feature schema, dimension, rank, or radius metadata mismatch')
    if (manifest['member_allowlist'] != [codec.NPZ_NAME]
            or manifest['npz_member_allowlist'] != sorted(codec.ALLOWED_NPZ_MEMBERS)):
        raise ValueError('Ground payload member allowlist mismatch')
    npz_bytes = (root / codec.NPZ_NAME).read_bytes()
    with np.load(BytesIO(npz_bytes), allow_pickle=False) as arrays:
        if set(arrays.files) != codec.ALLOWED_NPZ_MEMBERS or len(arrays.files) != len(codec.ALLOWED_NPZ_MEMBERS):
            raise ValueError('Ground NPZ contains missing, duplicate, or illegal extra members')
        payload = {name: arrays[name] for name in arrays.files}
    details = codec._validate_payload(payload)
    if details['classes'] != classes:
        raise ValueError('Ground class registry/order mismatch')
    resource = manifest['resource_audit']
    if not isinstance(resource, dict) or set(resource) - (set(details['resource_audit']) | RESOURCE_METADATA_FIELDS):
        raise ValueError('Illegal ground resource metadata; only scalar aggregate audit fields are allowed')
    fixed_resource = dict(center_domain_handle=details['center'],
        center_domain_index_in_retained_registry=details['domains'].index(details['center']),
        center_domain_policy='phase1_offline_global_maximin_fixed_before_target_access',
        residual_rank=codec.RESIDUAL_RANK, svd_sign_canonicalization=codec.SVD_SIGN_SCHEMA,
        rounding_rule=codec.ROUNDING_SCHEMA, radius_provenance='phase1_offline_aggregate_p90_cosine_distance_v1',
        **details['resource_audit'])
    for name, value in resource.items():
        if name in fixed_resource:
            if value != fixed_resource[name]:
                raise ValueError('Ground aggregate audit binding mismatch: ' + name)
        elif type(value) not in (int, float) or not np.isfinite(value):
            raise ValueError('Ground aggregate numeric audit contains invalid data: ' + name)
    immutable_manifest = _freeze(manifest)
    component = codec.CenterLowRankPrototypeComponent(
        **{name: payload[name] for name in NUMERIC_MEMBERS},
        domain_registry=details['domains'], residual_domain_registry=details['residual_domains'],
        class_registry=details['classes'], center_domain_handle=details['center'], manifest=immutable_manifest)
    numeric_bytes = sum(int(payload[name].nbytes) for name in NUMERIC_MEMBERS)
    registry_bytes = sum(int(array.nbytes) for name, array in payload.items() if name not in NUMERIC_MEMBERS)
    total_files = len(npz_bytes) + len(manifest_bytes)
    audit = _freeze(dict(
        numeric_array_bytes=numeric_bytes, registry_schema_bytes=registry_bytes,
        total_array_bytes=numeric_bytes + registry_bytes,
        per_array_bytes={name: int(array.nbytes) for name, array in payload.items()},
        npz_file_bytes=len(npz_bytes), manifest_file_bytes=len(manifest_bytes), total_file_bytes=total_files,
        already_deployed=already_deployed, incremental_transfer_bytes=0 if already_deployed else total_files,
        transfer_scope='These two component files only; excludes model, support, transport framing and other artifacts',
        registry_schema_definition='All nonnumeric-payload ndarray storage, including scalar residual_rank metadata',
        source_samples_read=False, checkpoint_file_read=False, dense_bank_persisted=False,
        dequantized_domain_cache=False))
    return GroundSummary(component=component, payload_audit=audit)


__all__ = ['GroundSummary', 'load_ground_summary']
