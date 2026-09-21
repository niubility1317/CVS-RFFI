"""Source-only matched-fit predictors and strict independent donor task plans."""
from __future__ import annotations

import copy
import itertools
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn

from .replay_audit import capture_rng_state, restore_rng_state


@dataclass(frozen=True)
class DonorReuseTask:
    kind: str
    query_tx: tuple
    query_rx: tuple
    donor_tx: tuple  # (branch A TX IDs, branch B TX IDs)
    donor_rx: tuple
    query_records: tuple
    donor_records: tuple  # (branch A records, branch B records)
    day_id: int
    condition_id: object
    k: int

    def validate(self):
        qids = {r.physical_sample_id for r in self.query_records}
        aids, bids = ({r.physical_sample_id for r in rows} for rows in self.donor_records)
        if len(qids) != len(self.query_records) or len(aids) != len(self.donor_records[0]) or len(bids) != len(self.donor_records[1]):
            raise ValueError("duplicate physical recording")
        if qids & (aids | bids) or aids & bids:
            raise ValueError("donor branches and query must be physically disjoint")
        query_events = {r.event_id for r in self.query_records if r.event_id is not None}
        donor_events = {r.event_id for rows in self.donor_records for r in rows if r.event_id is not None}
        if query_events & donor_events:
            raise ValueError("query/donor physical event overlap")
        if len(aids) != len(bids):
            raise ValueError("donor counts must match")
        if self.kind == "tx_reuse" and set(self.donor_tx[0]) & set(self.donor_tx[1]):
            raise ValueError("strict TX donor identities overlap")
        if self.kind == "rx_reuse" and set(self.donor_rx[0]) & set(self.donor_rx[1]):
            raise ValueError("strict RX donor identities overlap")
        for branch, rows in enumerate(self.donor_records):
            if set(self.query_tx) & set(self.donor_tx[branch]) or set(self.query_rx) & set(self.donor_rx[branch]):
                raise ValueError("donor/query identities overlap")
            cells = defaultdict(int)
            for r in rows:
                if r.day_id != self.day_id or r.condition_id != self.condition_id:
                    raise ValueError("mixed source conditions")
                cells[r.tx_id, r.rx_id] += 1
            expected = set(itertools.product(self.query_tx, self.donor_rx[branch])) | set(itertools.product(self.donor_tx[branch], self.query_rx))
            if set(cells) != expected or any(n != self.k for n in cells.values()):
                raise ValueError("donor cells must have identical K")


def build_donor_reuse_tasks(records, *, k=2, max_tasks=8):
    """Enumerate feasible source cells; never pad or borrow target receivers.

    Unchanged-axis donor cells use different physical records in A and B,
    requiring 2K records there. This is stricter than identity-set disjointness
    alone and is reported as unavailable if the data cannot support it.
    """
    if k < 1 or max_tasks < 1:
        raise ValueError("positive K and task budget required")
    cells = defaultdict(list)
    by_condition = defaultdict(list)
    for r in records:
        cells[r.day_id, r.condition_id, r.tx_id, r.rx_id].append(r)
        by_condition[r.day_id, r.condition_id].append(r)
    for rows in cells.values():
        rows.sort(key=lambda r: repr(r.physical_sample_id))
    outputs = {"tx_reuse": [], "rx_reuse": []}
    for (day, condition), rows in sorted(by_condition.items(), key=lambda v: repr(v[0])):
        txs, rxs = sorted({r.tx_id for r in rows}), sorted({r.rx_id for r in rows})
        for kind in outputs:
            if len(outputs[kind]) >= max_tasks:
                continue
            tx_count, rx_count = (6, 4) if kind == "tx_reuse" else (4, 5)
            for tx in itertools.combinations(txs, tx_count):
                for rx in itertools.combinations(rxs, rx_count):
                    qt = tx[:2]
                    qr = rx[:2] if kind == "tx_reuse" else rx[:1]
                    dt = (tx[2:4], tx[4:6]) if kind == "tx_reuse" else (tx[2:4], tx[2:4])
                    dr = (rx[2:4], rx[2:4]) if kind == "tx_reuse" else (rx[1:3], rx[3:5])
                    used = set()
                    def pick(positions):
                        chosen = []
                        for t, r in positions:
                            available = [v for v in cells[day, condition, t, r] if v.physical_sample_id not in used]
                            if len(available) < k:
                                return None
                            chosen.extend(available[:k])
                            used.update(v.physical_sample_id for v in available[:k])
                        return tuple(chosen)
                    query = pick(itertools.product(qt, qr))
                    a = pick(itertools.chain(itertools.product(qt, dr[0]), itertools.product(dt[0], qr)))
                    b = pick(itertools.chain(itertools.product(qt, dr[1]), itertools.product(dt[1], qr)))
                    if query is None or a is None or b is None:
                        continue
                    task = DonorReuseTask(kind, qt, qr, dt, dr, query, (a, b), day, condition, k)
                    try:
                        task.validate()
                    except ValueError:
                        continue
                    outputs[kind].append(task)
                    if len(outputs[kind]) >= max_tasks:
                        break
                if len(outputs[kind]) >= max_tasks:
                    break
    return {kind: {"status": "AVAILABLE" if tasks else "N/A", "tasks": tasks,
                   "reason": None if tasks else "insufficient_independent_source_cells_or_records",
                   "query_shape": [2, 2] if kind == "tx_reuse" else [2, 1],
                   "independent_query_interaction": kind == "tx_reuse"}
            for kind, tasks in outputs.items()}


class RestrictedResponseHead(nn.Module):
    """Actually fitted restricted additive predictor; no zero-input shortcut."""
    def __init__(self, descriptor_dim, target_dim, mode):
        super().__init__()
        if mode not in ("tx_only", "rx_only"):
            raise ValueError("restricted axis required")
        self.mode = mode
        self.linear = nn.Linear(descriptor_dim, target_dim)

    def forward(self, tx, rx):
        if self.mode == "tx_only":
            return self.linear(tx)[:, None].expand(-1, len(rx), -1)
        return self.linear(rx)[None].expand(len(tx), -1, -1)


def prediction_errors(prediction, truth):
    error = (prediction-truth).detach().double()
    grand = error.mean((0, 1), keepdim=True)
    tx = error.mean(1, keepdim=True)-grand
    rx = error.mean(0, keepdim=True)-grand
    interaction = error-grand-tx-rx
    return {"mse": float(error.square().mean()), "grand_mse": float(grand.square().mean()),
            "tx_mse": float(tx.square().mean()), "rx_mse": float(rx.square().mean()),
            "interaction_mse": float(interaction.square().mean()) if min(error.shape[:2]) > 1 else None}


def fit_source_baselines(predictor, fit_tasks, eval_tasks, *, steps, lr, seed,
                         fit_physical_ids, eval_physical_ids, fit_role, eval_role):
    """Tasks carry detached tx/rx/target tensors and common exact fit exposure.

    Full is the same predictor architecture, independently refit from the
    supplied initial predictor. Restricted heads copy corresponding weights.
    Thus full and restricted start from the same additive branch parameters.
    Head-only is the full refit on frozen descriptors; it is an explicit alias,
    not a separate independent experiment. Backbone training comparisons must
    be supplied by the real training caller.
    """
    if fit_role != "source_train" or eval_role != "source_validation":
        raise ValueError("only labeled source fit and source validation accepted")
    if not fit_tasks or not eval_tasks or steps < 1 or lr <= 0:
        raise ValueError("nonempty tasks and positive frozen fit budget required")
    if not fit_physical_ids or not eval_physical_ids or set(fit_physical_ids) & set(eval_physical_ids):
        raise ValueError("source fit/eval physical recordings must be disjoint")
    if not hasattr(predictor, "tx") or not hasattr(predictor, "rx"):
        raise TypeError("SharedResponsePredictor-compatible architecture required")
    caller_rng = capture_rng_state()
    try:
        torch.manual_seed(seed)
        full = copy.deepcopy(predictor)
        for p in full.parameters():
            p.requires_grad_(True)
        dim, target_dim = full.tx.in_features, full.tx.out_features
        heads = {"full_matched_fit": full}
        for name, branch in (("tx_only", full.tx), ("rx_only", full.rx)):
            head = RestrictedResponseHead(dim, target_dim, name).to(branch.weight.device)
            with torch.no_grad():
                head.linear.weight.copy_(branch.weight)
                head.linear.bias.copy_(full.bias)
            heads[name] = head
        results = {}
        for name, head in heads.items():
            opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.)
            head.train()
            for step in range(steps):
                task = fit_tasks[step % len(fit_tasks)]
                tx, rx, truth = (task[key].detach() for key in ("tx", "rx", "target"))
                opt.zero_grad(set_to_none=True)
                loss = (head(tx, rx)-truth).square().mean()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite source fit loss")
                loss.backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                errors = [prediction_errors(head(t["tx"].detach(), t["rx"].detach()), t["target"]) for t in eval_tasks]
            results[name] = {"errors": errors, "parameter_count": sum(p.numel() for p in head.parameters()),
                             "fit_steps": steps, "fit_task_indices": [i % len(fit_tasks) for i in range(steps)],
                             "fit_query_cells": sum(fit_tasks[i % len(fit_tasks)]["target"].shape[0]*fit_tasks[i % len(fit_tasks)]["target"].shape[1] for i in range(steps)),
                             "lr": lr, "optimizer": "AdamW", "weight_decay": 0.,
                             "optimizer_betas": [0.9, 0.999], "backbone_frozen": True,
                             "readouts_frozen": True, "architecture": type(head).__name__,
                             "descriptor_dim": dim, "target_dim": target_dim,
                             "predictor_mode": getattr(head, 'mode', None),
                             "initialization": 'copy_supplied_source_predictor_corresponding_branches'}
        results["head_only"] = dict(results["full_matched_fit"],
                                    alias_of="full_matched_fit", independent_evidence=False)
        results["metadata"] = {"fit_unique_physical_ids": len(set(fit_physical_ids)),
                               "eval_unique_physical_ids": len(set(eval_physical_ids)),
                               "necessity_gate_passed": None, "seed": seed,
                               "scope": "source_frozen_representation_matched_head_fit"}
        return results
    finally:
        restore_rng_state(caller_rng)
