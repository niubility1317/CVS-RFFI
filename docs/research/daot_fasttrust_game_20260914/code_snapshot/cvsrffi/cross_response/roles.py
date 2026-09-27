"""Disjoint, rotating donor/query roles and forward-path mixing permissions."""
from .schema import RolePlan
from itertools import combinations, product


def make_roles(tx_ids, rx_ids, rotation=0, *, policy="legacy_halves_v1"):
    tx, rx = tuple(tx_ids), tuple(rx_ids)
    if len(tx) < 2 or len(rx) < 2 or len(set(tx)) != len(tx) or len(set(rx)) != len(rx):
        raise ValueError("at least two distinct physical TX and RX required")
    if policy == "balanced_partitions_v2":
        if len(tx) != 4 or len(rx) != 4:
            raise ValueError("balanced_partitions_v2 requires P4 x Q4")
        plans = tuple(product(combinations(tx, 2), combinations(rx, 2)))
        plan_id = int(rotation) % len(plans)
        qt, qr = plans[plan_id]
        return RolePlan(tuple(t for t in tx if t not in qt), qt,
                        tuple(r for r in rx if r not in qr), qr,
                        plan_id, policy, plan_id)
    if policy != "legacy_halves_v1":
        raise ValueError("unknown role policy")
    # Four complementary templates. 3x3 has a 2x2 query in template zero;
    # templates with only one query axis report independent rectangle N/A.
    ta, tb = tx[:len(tx)//2], tx[len(tx)//2:]
    ra, rb = rx[:len(rx)//2], rx[len(rx)//2:]
    turn = int(rotation) % 4
    if turn & 1:
        ta, tb = tb, ta
    if turn & 2:
        ra, rb = rb, ra
    return RolePlan(ta, tb, ra, rb, turn, policy, turn)


def mixstyle_allowed_mask(plan, views=1, device=None):
    """Conservative role compartments; rows are recipients, columns sources.

    Query cannot affect either donor family. Neither can one donor family
    become an indirect route into the other. All view slots preserve this
    restriction; same-record views cannot supply each other. Ordinary
    remainder samples only mix within their own remainder compartment.
    """
    import torch
    n = len(plan.indices)
    groups = [("ordinary",)] * n
    for block in plan.blocks:
        role = block.roles
        for pos, rec in zip(block.batch_positions, block.records):
            groups[pos] = (block.block_id, rec.tx_id in role.query_tx, rec.rx_id in role.query_rx)
    groups = [(view, group) for view in range(int(views)) for group in groups]
    size = n * int(views)
    return torch.tensor([[groups[i] == groups[j] and i % n != j % n for j in range(size)]
                         for i in range(size)], dtype=torch.bool, device=device)
