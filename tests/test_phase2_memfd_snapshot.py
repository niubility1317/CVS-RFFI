from __future__ import annotations

from cvsrffi import phase2_memfd_snapshot as snapshot


def test_create_memfd_prefers_python_native(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_native(name: str, *, flags: int) -> int:
        calls.append((name, flags))
        return 41

    monkeypatch.setattr(snapshot.os, "memfd_create", fake_native, raising=False)
    monkeypatch.setattr(
        snapshot,
        "_libc_memfd_create",
        lambda _name, _flags: (_ for _ in ()).throw(AssertionError("fallback used")),
    )

    assert snapshot._create_memfd("phase2", 3) == 41
    assert calls == [("phase2", 3)]


def test_create_memfd_uses_libc_when_python_omits_api(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_fallback(name: str, flags: int) -> int:
        calls.append((name, flags))
        return 42

    monkeypatch.setattr(snapshot.os, "memfd_create", None, raising=False)
    monkeypatch.setattr(snapshot, "_libc_memfd_create", fake_fallback)

    assert snapshot._create_memfd("phase2", 3) == 42
    assert calls == [("phase2", 3)]
