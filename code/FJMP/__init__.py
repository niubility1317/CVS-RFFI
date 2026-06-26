"""FJMP post-stage experiment package.

Import heavy PyTorch modules explicitly from ``FJMP.frozen_joint_prototype_head``.
This keeps lightweight helpers such as ``FJMP.experiment_manifest`` usable in
environments where PyTorch is not installed.
"""

__all__ = []
