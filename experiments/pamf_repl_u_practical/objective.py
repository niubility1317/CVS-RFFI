"""Reuse the STAR physical-ID practical observations only at the existing U As slot."""
from pathlib import Path

import numpy as np
import torch

from experiments.cric_dura_comparison.io import read, tensor
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import GuardedReceiverExtrapolation
from experiments.practical_full_2x2.plan import SCENES

KEYS = ("eq_i", "rx_i", "day_i", "sig_i", "base_index")
U_FIELDS = set(KEYS) | {"ids", "domain"}


class SourcePractical:
    def __init__(self, c):
        root = Path(c["hard_pl_practical_u_cache"])
        built = read(root / "complete.json")
        if any(built[key] is not False for key in ("U_TX_truth_read", "validation_used", "target_read")):
            raise ValueError("Practical views must come exclusively from source training IQ")
        contract = read(c["source_contract"])
        ids = np.load(root / "U_s_ids.npy", allow_pickle=False)
        self.iq = np.load(root / "U_s_sg.npy", mmap_mode="r", allow_pickle=False)
        self.scene = np.load(root / "U_s_scene.npy", allow_pickle=False)
        with np.load(Path(c["source_cache"]) / "U_s_metadata.npz", allow_pickle=False) as m:
            if set(m.files) != U_FIELDS or m["ids"].tolist() != contract["role_ids"]["U_s"]:
                raise ValueError("Source U metadata must exclude transmitter labels")
            if not np.array_equal(ids, m["ids"]):
                raise ValueError("STAR practical cache physical IDs differ from PAMF U_s")
            positions = zip(*(m[k].tolist() for k in KEYS))
            self.index = {key: i for i, key in enumerate(positions)}
        if len(self.index) != len(ids) or self.iq.shape != (len(ids), 2, 256) or self.iq.dtype != np.float32:
            raise ValueError("Source-U practical record binding differs")
        if self.scene.shape != ids.shape or set(self.scene) != set(range(len(SCENES))):
            raise ValueError("Practical views must have the recorded source scene classes")

    def select(self, metadata, mask, device):
        visible = metadata["tx_label_visible"]
        if bool(visible.any()):
            raise ValueError("Labeled or foreign input cannot enter source U As")
        key = zip(*(metadata[name][mask].detach().cpu().tolist() for name in KEYS))
        indices = [self.index[record] for record in key]
        return tensor(self.iq[indices], device), self.scene[indices]


class CachedPracticalREPL(GuardedReceiverExtrapolation):
    def __init__(self, c):
        super().__init__(c)
        self.practical = SourcePractical(c)
        self.scene_counts = np.zeros(len(SCENES), dtype=np.int64)
        self.accepted_practical = 0

    def mixed_view(self, student, x, ordinary, physical, metadata, epoch, step, pseudo, accepted):
        view, scene = self.practical.select(metadata, physical, x.device)
        if view.shape != x[physical].shape or not bool(torch.isfinite(view).all()):
            raise ValueError("Finite, same-record practical U As required")
        mixed = ordinary.clone()
        mixed[physical] = view
        self.scene_counts += np.bincount(scene, minlength=len(SCENES))
        self.accepted_practical += int(accepted[physical].sum())
        return mixed

    def state_dict(self):
        state = super().state_dict()
        if (int(self.scene_counts.sum()) != state["seen"] // 2
                or state["physical_seen"] != state["seen"] // 2
                or state["physical_student_views"] != state["seen"]
                or state["raw_As_seen"] or not 0 <= self.accepted_practical <= state["accepted"]):
            raise ValueError("The registered U practical scene or student forward budget differs")
        state["U_practical"] = dict(scene_counts=dict(zip(SCENES, self.scene_counts.tolist())),
                                    accepted_packets=self.accepted_practical)
        return state
