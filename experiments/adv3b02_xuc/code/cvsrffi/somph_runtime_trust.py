"""Minimal Phase2 trust anchor for signed path-free SOMP-H policy.

This module is deliberately self-contained.  It has no filesystem, dataset,
cache, builder, NumPy, or offline-controller dependency.
"""

from __future__ import annotations

import hashlib


PINNED_AUTHORITY_ISSUER = "qknnv42_stage2bc_extreme_light_route_20260716"
PINNED_AUTHORITY_KEY_ID = "somph-authority-ed25519-20260716"
PINNED_AUTHORITY_PUBLIC_KEY_HEX = (
    "ec301433b5a625f8e34f887f5aeea664e809236d1b871fcc0ffeb47cb540bdc1"
)
PINNED_AUTHORITY_PUBLIC_KEY_SHA256 = (
    "52944e59ec99d360e227cbe78e84efeca6db3ebca3d9698f5d567270c37a9444"
)

PHASE2_SINGLE_OBSERVATION_CONTRACT = {
    "phase2_physical_sample_observation_policy": (
        "single_leo_weak_observation_per_physical_sample"
    ),
    "phase2_cross_scenario_physical_sample_reuse": False,
    "phase2_additional_leo_channel_state_generation": False,
    "phase2_post_reception_equalization_augmentation_transform_allowed": True,
    "phase2_post_reception_view_from_fixed_received_iq_only": True,
    "phase2_post_reception_view_counts_as_additional_physical_sample": False,
    "phase2_physical_sample_root_id_policy": "immutable_preoverlay_lineage_token",
    "phase2_query_post_reception_view_fit_access": False,
}
PHYSICAL_SAMPLE_SCENARIO_ASSIGNMENT_POLICY = (
    "disjoint_preoverlay_tx_day_stratified_v1"
)


_ED_Q = 2**255 - 19
_ED_L = 2**252 + 27742317777372353535851937790883648493
_ED_D = (-121665 * pow(121666, _ED_Q - 2, _ED_Q)) % _ED_Q
_ED_I = pow(2, (_ED_Q - 1) // 4, _ED_Q)


def _ed_xrecover(y: int) -> int:
    xx = (y * y - 1) * pow(_ED_D * y * y + 1, _ED_Q - 2, _ED_Q) % _ED_Q
    x = pow(xx, (_ED_Q + 3) // 8, _ED_Q)
    if (x * x - xx) % _ED_Q:
        x = x * _ED_I % _ED_Q
    if (x * x - xx) % _ED_Q:
        raise ValueError("Ed25519 point is not on curve")
    if x & 1:
        x = _ED_Q - x
    return x


_ED_BY = 4 * pow(5, _ED_Q - 2, _ED_Q) % _ED_Q
_ED_BX = _ed_xrecover(_ED_BY)
_ED_B = (_ED_BX, _ED_BY)
_ED_IDENTITY = (0, 1)


def _ed_add(p: tuple[int, int], q: tuple[int, int]) -> tuple[int, int]:
    x1, y1 = p
    x2, y2 = q
    common = _ED_D * x1 * x2 * y1 * y2 % _ED_Q
    return (
        (x1 * y2 + x2 * y1) * pow(1 + common, _ED_Q - 2, _ED_Q) % _ED_Q,
        (y1 * y2 + x1 * x2) * pow(1 - common, _ED_Q - 2, _ED_Q) % _ED_Q,
    )


def _ed_scalar_mult(point: tuple[int, int], scalar: int) -> tuple[int, int]:
    result = _ED_IDENTITY
    addend = point
    value = scalar
    while value:
        if value & 1:
            result = _ed_add(result, addend)
        addend = _ed_add(addend, addend)
        value >>= 1
    return result


def _ed_encode(point: tuple[int, int]) -> bytes:
    x, y = point
    value = y | ((x & 1) << 255)
    return value.to_bytes(32, "little")


def _ed_decode(value: bytes) -> tuple[int, int]:
    if len(value) != 32:
        raise ValueError("Ed25519 point length drift")
    raw = int.from_bytes(value, "little")
    y = raw & ((1 << 255) - 1)
    if y >= _ED_Q:
        raise ValueError("Ed25519 point is non-canonical")
    x = _ed_xrecover(y)
    if (x & 1) != (raw >> 255):
        x = _ED_Q - x
    point = (x, y)
    if (-x * x + y * y - 1 - _ED_D * x * x * y * y) % _ED_Q:
        raise ValueError("Ed25519 point is not on curve")
    if _ed_encode(point) != value:
        raise ValueError("Ed25519 point encoding drift")
    return point


def verify_ed25519(public_key: bytes, message: bytes, signature: bytes) -> None:
    """Verify one canonical Ed25519 signature or raise ``ValueError``."""

    if len(public_key) != 32 or len(signature) != 64:
        raise ValueError("Ed25519 key/signature length drift")
    public_point = _ed_decode(public_key)
    r_point = _ed_decode(signature[:32])
    scalar = int.from_bytes(signature[32:], "little")
    if scalar >= _ED_L:
        raise ValueError("Ed25519 signature scalar drift")
    if public_point == _ED_IDENTITY or _ed_scalar_mult(public_point, _ED_L) != _ED_IDENTITY:
        raise ValueError("Ed25519 public key subgroup drift")
    if r_point == _ED_IDENTITY or _ed_scalar_mult(r_point, _ED_L) != _ED_IDENTITY:
        raise ValueError("Ed25519 R point subgroup drift")
    challenge = int.from_bytes(
        hashlib.sha512(signature[:32] + public_key + message).digest(), "little"
    ) % _ED_L
    left = _ed_scalar_mult(_ED_B, (8 * scalar) % _ED_L)
    right = _ed_add(
        _ed_scalar_mult(r_point, 8),
        _ed_scalar_mult(public_point, (8 * challenge) % _ED_L),
    )
    if _ed_encode(left) != _ed_encode(right):
        raise ValueError("Ed25519 authority signature invalid")
