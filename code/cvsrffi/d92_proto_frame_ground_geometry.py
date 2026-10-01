"""Load an existing frozen ground center solely as a fixed feature dictionary.

No source observations, per-record features, checkpoint loading, prototype
builders, residual-bank reconstruction or target fitting are exposed here.
The existing ground-summary reader owns the unchanged input/schema contract.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Sequence

from .d92_ground_summary import load_ground_summary
from .d92_proto_frame_primitives import build_proto_frame_dictionary


def load_proto_frame_ground_geometry(
    component_dir: str | Path, *, expected_checkpoint_sha256: str,
    expected_classes: Sequence[str], already_deployed: bool,
):
    """Return a fixed five-direction frame and a separate payload/cost audit.

    ``already_deployed`` refers only to the two existing ground component
    files. It does not imply zero model, code or native transport wire bytes.
    Six old classes are required before opening the component. The center
    domain was fixed by Phase1; no receiver/support-based domain selection is
    performed. Only its immutable center coordinates define the frame.
    """
    if isinstance(expected_classes, (str, bytes)):
        raise ValueError('expected_classes must be the six-class registry')
    classes = tuple(expected_classes)
    if len(classes) != 6 or len(set(classes)) != 6 or any(
        not isinstance(item, str) or not item for item in classes
    ):
        raise ValueError('Exactly six unique nonempty old class identifiers required')
    if type(already_deployed) is not bool:
        raise ValueError('already_deployed must be explicit bool')
    started = perf_counter()
    ground = load_ground_summary(
        component_dir, expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_classes=classes, already_deployed=already_deployed,
    )
    load_seconds = perf_counter() - started
    center = ground.component.dequantized_center()
    frame = build_proto_frame_dictionary(prototypes=center, classes=classes)
    audit = MappingProxyType(dict(
        operation='load_existing_ground_center_fixed_proto_frame',
        feature_schema=ground.feature_schema,
        center_domain_handle=ground.component.center_domain_handle,
        center_policy='phase1_offline_global_maximin_fixed_before_target_access',
        ground_payload_audit=ground.payload_audit,
        frame_audit=frame.audit,
        component_read_seconds=load_seconds,
        decoded_center_bytes=int(center.nbytes),
        newly_generated_ground_statistics_bytes=0,
        native_wire_bytes=None,
        source_samples_read=False,
        source_per_record_features_read=False,
        checkpoint_loaded=False,
        residual_domain_reconstruction=False,
        prototype_updated=False,
        prototype_teacher_targets=False,
        support_labels_read=False,
        query_read=False,
        interpretation='Frozen center fixes feature directions; target support trains adapter/head elsewhere',
    ))
    return frame, audit


__all__ = ['load_proto_frame_ground_geometry']
