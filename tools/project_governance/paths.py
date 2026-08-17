"""Path normalization and stable identity helpers for governance records."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import quote

from .models import Location


_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]|^//|^\\\\")


def normalize_relative_path(
    value: str, *, location: Location | str | None = None
) -> str:
    """Return a normalized relative path that remains under its root.

    Local/unspecified inputs accept Windows separators.  N607 paths are
    POSIX paths, where a backslash is a literal filename character and must
    not collapse into a directory separator.
    """

    if not isinstance(value, str) or not value:
        raise ValueError("path must be a non-empty relative string")
    selected_location = Location(location) if location is not None else None
    normalized = unicodedata.normalize("NFC", value)
    if selected_location is not Location.N607:
        normalized = normalized.replace("\\", "/")
    if normalized.startswith("/") or _WINDOWS_ABSOLUTE.match(normalized):
        raise ValueError(f"path must be relative: {value!r}")

    parts: list[str] = []
    for part in normalized.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise ValueError(f"path escapes root: {value!r}")
            parts.pop()
            continue
        parts.append(part)
    if not parts:
        raise ValueError("path must name an entry below the root")
    return "/".join(parts)


def stable_asset_id(location: Location, root_id: str, relative_path: str) -> str:
    """Build identity from location, configured root and normalized relative path only."""

    selected_location = Location(location)
    path = normalize_relative_path(relative_path, location=selected_location)
    if selected_location is Location.LOCAL:
        path = path.casefold()
    return "asset:{location}:{root}:{path}".format(
        location=selected_location.value,
        root=quote(root_id, safe="-._~"),
        path=quote(path, safe="/-._~"),
    )


def escaped_display_name(value: str) -> str:
    """Return an ASCII, reversible rendering for unusual filesystem names."""

    if not isinstance(value, str):
        raise TypeError("display name must be a string")
    return unicodedata.normalize("NFC", value).encode("unicode_escape").decode("ascii")


__all__ = ["escaped_display_name", "normalize_relative_path", "stable_asset_id"]
