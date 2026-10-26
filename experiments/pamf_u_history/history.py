"""Detached, per-physical-source-U history; no model EMA or query state."""
import torch
from torch.nn import functional as F

KEYS = ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')
METHODS = ('PL-Negative', 'PL-Temporal', 'PL-History')
RECIPE = dict(positive_threshold=.95, positive_weight=1., history_visits=2,
              negative_threshold=.05, negative_weight=.25, temporal_weight=.25,
              temporal_confidence=.5, ramp_updates=444, EMA_model=False,
              extra_student_forwards=0, extra_teacher_forwards=0, start_epoch=21)


class History:
    def __init__(self, physical, classes=6):
        rows = [tuple(int(v) for v in row) for row in physical]
        if classes != 6 or not rows or any(len(r) != 5 for r in rows) or len(set(rows)) != len(rows):
            raise ValueError('Unique source-U physical records and six classes required')
        self.lookup = {r: i for i, r in enumerate(rows)}
        self.receivers = torch.tensor([r[1] for r in rows], dtype=torch.long)
        self.by_rx = {}
        self.probabilities = torch.zeros((2, len(rows), classes), dtype=torch.float32)
        self.visits = torch.zeros(len(rows), dtype=torch.long)
        self.calls = self.seen = self.accepted = self.negative_seen = self.temporal_seen = 0
        self.negative_classes = 0
        self.negative_unique = set()
        self.temporal_unique = set()

    def indices(self, metadata, labels, *, role, training):
        if role != 'U_s' or not training or bool((labels != -1).any()) or bool(metadata['tx_label_visible'].any()):
            raise ValueError('History updates require hidden-label training U_s only')
        rows = [tuple(int(metadata[k][i]) for k in KEYS) for i in range(len(labels))]
        if len(set(rows)) != len(rows) or any(row not in self.lookup for row in rows):
            raise ValueError('Foreign or duplicate physical records in history batch')
        return torch.tensor([self.lookup[r] for r in rows], dtype=torch.long)

    def loss(self, logits, current, indices, method, *, update):
        if method not in METHODS or update <= 0:
            raise ValueError('Fixed source history method and positive update required')
        p = current.detach().to(device='cpu', dtype=torch.float32)
        ids = indices.detach().cpu()
        if (p.shape != logits.shape or p.ndim != 2 or p.shape[1] != 6 or len(ids) != len(p)
                or len(ids.unique()) != len(ids) or bool((ids < 0).any()) or bool((ids >= len(self.visits)).any())
                or not bool(torch.isfinite(p).all()) or bool(((p < 0) | (p > 1)).any())
                or not torch.allclose(p.sum(1), torch.ones(len(p)), atol=1e-5)):
            raise ValueError('Finite aligned probabilities and unique registered U indices required')
        confidence, pseudo = p.max(1)
        accepted = confidence >= RECIPE['positive_threshold']
        positive = (F.cross_entropy(logits, pseudo.to(logits.device), reduction='none') * accepted.to(logits.device)).mean()
        prior = self.probabilities[:, ids].clone()  # Read previous visits BEFORE writing current prediction.
        ready = self.visits[ids] >= RECIPE['history_visits']
        negative = (prior.max(0).values <= RECIPE['negative_threshold']) & (p <= RECIPE['negative_threshold'])
        negative &= (ready & ~accepted).unsqueeze(1)
        if method == 'PL-Temporal':
            negative.zero_()
        negative_rows = negative.any(1)
        logp = logits.log_softmax(1)
        # log(sum_{j!=c} p_j), computed without 1-softmax cancellation at extreme logits.
        log_complement = torch.stack([torch.logsumexp(logp[:, [j for j in range(6) if j != c]], 1) for c in range(6)], 1)
        nmask = negative.to(logits.device)
        per_negative = -(log_complement * nmask).sum(1) / nmask.sum(1).clamp_min(1)
        negative_loss = per_negative[negative_rows.to(logits.device)].mean() if bool(negative_rows.any()) else logits.sum() * 0.
        target = prior.mean(0)
        temporal = ready & (prior[0].argmax(1) == prior[1].argmax(1)) & (target.max(1).values >= RECIPE['temporal_confidence'])
        if method == 'PL-Negative':
            temporal.zero_()
        if bool(temporal.any()):
            temporal_loss = F.kl_div(logp[temporal.to(logits.device)], target[temporal].to(logits.device), reduction='batchmean')
        else:
            temporal_loss = logits.sum() * 0.
        ramp = min(update / RECIPE['ramp_updates'], 1.)
        total = positive + ramp * (RECIPE['negative_weight'] * negative_loss + RECIPE['temporal_weight'] * temporal_loss)
        slot = self.visits[ids].remainder(2)
        self.probabilities[slot, ids] = p
        self.visits[ids] += 1
        self.calls += 1
        self.seen += len(p)
        self.accepted += int(accepted.sum())
        self.negative_seen += int(negative_rows.sum())
        self.temporal_seen += int(temporal.sum())
        self.negative_classes += int(negative.sum())
        self.negative_unique.update(ids[negative_rows].tolist())
        self.temporal_unique.update(ids[temporal].tolist())
        for rx in self.receivers[ids].unique().tolist():
            selected = self.receivers[ids] == rx
            counts = self.by_rx.setdefault(str(rx), dict(seen=0, accepted=0, negative=0, temporal=0))
            for key, mask in (('seen', selected), ('accepted', selected & accepted),
                              ('negative', selected & negative_rows), ('temporal', selected & temporal)):
                counts[key] += int(mask.sum())
        return total, dict(UH_positive_loss=positive.detach(), UH_negative_loss=negative_loss.detach(),
                          UH_temporal_loss=temporal_loss.detach(), UH_ramp=ramp,
                          UH_negative_batch=int(negative_rows.sum()), UH_temporal_batch=int(temporal.sum()))

    def state(self):
        return dict(calls=self.calls, seen=self.seen, accepted=self.accepted,
                    negative_seen=self.negative_seen, temporal_seen=self.temporal_seen,
                    negative_classes=self.negative_classes, negative_unique=len(self.negative_unique),
                    temporal_unique=len(self.temporal_unique), unique_seen=int((self.visits > 0).sum()),
                    min_visits=int(self.visits.min()), max_visits=int(self.visits.max()),
                    by_source_RX=self.by_rx,
                    history_tensor_bytes=self.probabilities.numel()*4+self.visits.numel()*8+self.receivers.numel()*8,
                    Python_lookup_memory_not_measured=True,
                    student_U_packets=self.seen, teacher_U_packets=self.seen,
                    model_EMA_updates=0, U_truth_read=False, target_reads=0,
                    new_inference_parameters=0, default_head_joint=True)
