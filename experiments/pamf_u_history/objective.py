"""Replace only ordinary PL arithmetic; retain full SIG1 PAMF and one U forward."""
import torch
from .history import History, KEYS


class Objective:
    def __init__(self, c):
        self.c = c
        self.history = None
        self.last = {}

    def __call__(self, student, device):
        dataset = student._cd_U_loader.dataset
        if dataset.role != 'U_s' or 'y' in dataset.metadata:
            raise ValueError('Only source U with no TX truth can build history')
        if self.history is None:
            physical = zip(*(dataset.metadata[k] for k in KEYS))
            self.history = History(physical)
        try:
            batch = next(student._cd_U_iter)
        except StopIteration:
            student._cd_U_iter = iter(student._cd_U_loader)
            batch = next(student._cd_U_iter)
        ids = self.history.indices(batch[3], batch[1], role=dataset.role, training=student.training)
        x = batch[0].to(device)
        student.eval()
        try:
            with torch.no_grad():
                probabilities = student(x).softmax(1)
        finally:
            student.train()
        logits = student.forward_identity_only(x, domain_labels=batch[2].to(device))['tx_logits']
        step = student._feature_l_steps + self.c.get('step_offset', 0) - 4440
        total, self.last = self.history.loss(logits, probabilities, ids, self.c['paper_method'], update=step)
        student._cd_U_seen += len(x)
        student._cd_U_accepted += int((probabilities.detach().max(1).values >= .95).sum())
        self.last.update(UH_seen=self.history.seen, UH_accepted=self.history.accepted,
                         UH_negative_seen=self.history.negative_seen, UH_temporal_seen=self.history.temporal_seen)
        return total
