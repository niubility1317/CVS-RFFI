from __future__ import annotations

import pytest

import cvsrffi.somph_lineage_authority as authority
from cvsrffi.somph_lineage_authority import SomphLineageAuthorityError


RFC8032_PUBLIC_KEY = bytes.fromhex(
    "d75a980182b10ab7d54bfed3c964073a"
    "0ee172f3daa62325af021a68f707511a"
)
RFC8032_SIGNATURE = bytes.fromhex(
    "e5564300c360ac729086e2cc806e828a"
    "84877f1eb8e5d974d873e06522490155"
    "5fb8821590a33bacc61e39701cf9b46b"
    "d25bf5f0595bbe24655141438e7a100b"
)


def test_ed25519_decode_rejects_non_curve_02_zero_encoding() -> None:
    invalid_point = bytes.fromhex("02" + "00" * 31)

    with pytest.raises(
        SomphLineageAuthorityError,
        match="Ed25519 point is not on curve",
    ):
        authority._ed_decode(invalid_point)

    with pytest.raises(
        SomphLineageAuthorityError,
        match="Ed25519 point is not on curve",
    ):
        authority._verify_ed25519(
            RFC8032_PUBLIC_KEY,
            b"",
            invalid_point + RFC8032_SIGNATURE[32:],
        )


def test_ed25519_rfc8032_vector_regression() -> None:
    authority._verify_ed25519(RFC8032_PUBLIC_KEY, b"", RFC8032_SIGNATURE)


def test_ed25519_rejects_small_subgroup_r_point() -> None:
    identity_encoding = bytes.fromhex("01" + "00" * 31)

    with pytest.raises(
        SomphLineageAuthorityError,
        match="Ed25519 R point subgroup drift",
    ):
        authority._verify_ed25519(
            RFC8032_PUBLIC_KEY,
            b"",
            identity_encoding + RFC8032_SIGNATURE[32:],
        )
