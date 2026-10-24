"""Compose the existing losses and view policy once,under one complete-L guard."""
from .hard_pl_path_objective import PathComplement
from .hard_pl_reliable_as_objective import ReliableAs
from .hard_pl_guard_objective import Guarded


class PathReliableAs(PathComplement,ReliableAs):pass


class GuardedPathReliableAs(Guarded,PathReliableAs):pass
