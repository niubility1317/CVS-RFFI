#!/usr/bin/env python3
"""Build the sealed same-forward pre-ReLU160 extractor from an ADV3B02 runtime.

The source runtime remains unchanged.  The derived artifact returns the output
of ``joint_proj.0`` as its first feature tuple member while retaining the
original cosine logits as the second member.  The graph transformation is
deterministic and leaves the ReLU/dropout path used to compute the logits
untouched; the caller binds the derived artifact SHA in the new method lock.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


class PR160RuntimeBuildError(ValueError):
    """Raised when the sealed runtime graph is not the expected ADV3B02 graph."""


def _regular_file(path: Path, name: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise PR160RuntimeBuildError(f"{name} must be a regular file")
    return source.resolve(strict=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patch_runtime_graph(model: torch.jit.ScriptModule) -> torch.jit.ScriptModule:
    """Return ``model`` with a first-output pre-ReLU160 tap.

    The model's original first tuple member is the post-ReLU/dropout feature.
    We replace only the tuple member returned by ``cls_head``; the original
    ``joint_proj`` call still feeds the frozen CosFace head, so the logits path
    remains a parity witness for the graph edit.
    """

    try:
        cls_head = model.model.id_backbone.cls_head
        graph = cls_head.graph
    except Exception as error:  # pragma: no cover - TorchScript backend detail
        raise PR160RuntimeBuildError("ADV3B02 cls_head graph is unavailable") from error

    nodes = list(graph.nodes())
    joint_calls = [
        node
        for node in nodes
        if node.kind() == "prim::CallMethod"
        and node.s("name") == "forward"
        and node.inputsSize() == 2
        and node.inputsAt(0).debugName().startswith("joint_proj")
    ]
    if len(joint_calls) != 1:
        raise PR160RuntimeBuildError("expected exactly one joint_proj forward call")
    joint_call = joint_calls[0]
    input8 = joint_call.inputsAt(1)

    tuples = [
        node
        for node in nodes
        if node.kind() == "prim::TupleConstruct" and node.inputsSize() == 2
    ]
    if len(tuples) != 1:
        raise PR160RuntimeBuildError("expected exactly one two-tensor cls_head output tuple")
    output_tuple = tuples[0]

    joint_attrs = [
        node
        for node in nodes
        if node.kind() == "prim::GetAttr" and node.s("name") == "joint_proj"
    ]
    if len(joint_attrs) != 1:
        raise PR160RuntimeBuildError("expected exactly one joint_proj attribute")

    try:
        linear0 = getattr(cls_head.joint_proj, "0")
        linear0_type = linear0._c._type()
    except Exception as error:  # pragma: no cover - TorchScript backend detail
        raise PR160RuntimeBuildError("joint_proj.0 linear tap is unavailable") from error

    get_linear0 = graph.create("prim::GetAttr", [joint_attrs[0].output()])
    get_linear0.s_("name", "0")
    get_linear0.output().setType(linear0_type)
    get_linear0.insertBefore(joint_call)

    pre_call = graph.create("prim::CallMethod", [get_linear0.output(), input8])
    pre_call.s_("name", "forward")
    pre_call.output().setType(output_tuple.inputsAt(0).type())
    pre_call.insertBefore(output_tuple)
    output_tuple.replaceInput(0, pre_call.output())
    return model


def build(source_path: Path, output_path: Path) -> dict[str, str | int]:
    source = _regular_file(source_path, "source runtime")
    destination = Path(output_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable extractor output already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise PR160RuntimeBuildError("extractor output parent must be a real directory")

    model = torch.jit.load(str(source), map_location="cpu")
    model.eval()
    patch_runtime_graph(model)
    torch.jit.save(model, str(destination))
    digest = _sha256(destination)
    return {
        "source_runtime_sha256": _sha256(source),
        "extractor_runtime_sha256": digest,
        "extractor_runtime_size_bytes": destination.stat().st_size,
        "extractor_runtime_path": str(destination.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
