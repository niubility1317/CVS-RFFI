import copy
import sys
from pathlib import Path
from itertools import combinations
from dataclasses import replace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from cvsrffi.cross_response.roles import make_roles
from cvsrffi.cross_response.roles import mixstyle_allowed_mask
from cvsrffi.cross_response.schema import SampleRecord
from cvsrffi.cross_response.sampler import CrossBlockSampler, plan_to_json, extract_batch_plan
from cvsrffi.cross_response.coverage import CoverageTracker


def records(tx=4, rx=4, days=1):
    return [SampleRecord(i, t, r, d, 'clean', (t,r,d,k))
            for i,(t,r,d,k) in enumerate((t,r,d,k) for t in range(tx)
            for r in range(rx) for d in range(days) for k in range(2))]


def sampler(**kwargs):
    return CrossBlockSampler(records(), batch_size=32, seed=77,
                             role_policy='balanced_partitions_v2', **kwargs)


def test_directed_cycle_and_legacy():
    roles = [make_roles(range(4), range(4), i, policy='balanced_partitions_v2') for i in range(36)]
    assert len({(r.query_tx,r.query_rx) for r in roles}) == 36
    assert {r.plan_id for r in roles} == set(range(36))
    for r in roles:
        assert not set(r.query_tx) & set(r.donor_tx)
        assert not set(r.query_rx) & set(r.donor_rx)
    assert make_roles(range(4),range(4)).query_tx == (2,3)


def test_sampler_cycle_resume_ticket_and_index_order():
    s = sampler()
    plans = [s.next_batch() for _ in range(13)]
    resumed = sampler()
    resumed.load_state_dict(s.state_dict())
    for _ in range(23):
        plan = s.next_batch()
        assert plan == resumed.next_batch()
        plans.append(plan)
        assert extract_batch_plan({'cr_plan_json':plan_to_json(plan)}) == plan
    assert len({p.blocks[0].roles.plan_id for p in plans}) == 36
    assert all([(r.tx_id,r.rx_id) for r in p.blocks[0].records] ==
               [(t,r) for t in range(4) for r in range(4) for _ in range(2)] for p in plans)
    old = CrossBlockSampler(records(),batch_size=32,seed=77).state_dict()
    with pytest.raises(ValueError):
        sampler().load_state_dict(old)


def test_coverage_rectangles_counts_and_transfer_denominator():
    block = sampler().next_batch().blocks[0]
    tracker = CoverageTracker()
    for step in range(80):
        tracker.commit(block,step,valid_count=32)
    assert len(tracker.observed_rectangles) == 36
    assert len(tracker.response_query_rectangles) == 1
    for table in (tracker.observed_rectangles,tracker.response_query_rectangles):
        assert all(row['record_exposures']==640 and row['unique_physical_records']==8
                   and row['valid_count']==640 for row in table.values())
    assert all(row['record_exposures']==1280 and row['unique_physical_records']==16
               for row in tracker.directed_response_transfers.values())
    restored = CoverageTracker()
    restored.load_state_dict(tracker.state_dict())
    assert restored.state_dict() == tracker.state_dict()


def test_all_feasible_query_reachability():
    queries = set()
    for tx in combinations(range(6),4):
        for rx in combinations(range(5),4):
            for d in range(3):
                for i in range(36):
                    r = make_roles(tx,rx,i,policy='balanced_partitions_v2')
                    queries.add((r.query_tx,r.query_rx,d))
    assert len(queries) == 450


def test_candidate_progress_and_independent_role_rng():
    s = CrossBlockSampler(records(6,5), batch_size=32, seed=77,
                          max_candidates=128, role_policy='balanced_partitions_v2')
    before = copy.deepcopy(s.rng.getstate())
    preview = s.next_role_plans()
    assert preview == s.next_role_plans()
    assert before == s.rng.getstate()
    counts = dict(s.role_counts)
    chosen = s.next_batch().blocks[0]
    assert chosen.roles == preview[chosen.candidate.key]
    assert all(s.role_counts[k] == n + (k == chosen.candidate.key) for k,n in counts.items())
    reached = CoverageTracker().reachable_coverage(s.candidates, 'balanced_partitions_v2')
    assert len(reached['response_query_rectangles']) == 150


def test_failed_feedback_only_exposes_physical_records():
    c = CoverageTracker()
    c.commit(sampler().next_batch().blocks[0], 0, success=False)
    assert len(c.exposed_physical) == 32
    assert not c.observed_rectangles and not c.response_query_rectangles and not c.directed


def test_device_renumbering_preserves_all_query_relationships():
    tx, rx = (83, 2, 61, 9), (32, 13, 4, 81)
    roles = [make_roles(tx,rx,i,policy='balanced_partitions_v2') for i in range(36)]
    assert {frozenset(r.query_tx) for r in roles} == {frozenset(p) for p in combinations(tx,2)}
    assert {frozenset(r.query_rx) for r in roles} == {frozenset(p) for p in combinations(rx,2)}


def test_all_roles_query_perturbation_cannot_enter_donor_mix():
    import torch
    base = sampler().next_batch()
    block = base.blocks[0]
    for turn in range(36):
        role = make_roles(range(4), range(4), turn, policy='balanced_partitions_v2')
        plan = replace(base, blocks=(replace(block, roles=role),))
        mask = mixstyle_allowed_mask(plan, views=2).float()
        query = torch.tensor([r.tx_id in role.query_tx and r.rx_id in role.query_rx
                              for r in block.records] * 2)
        donor = torch.tensor([(r.tx_id in role.query_tx) != (r.rx_id in role.query_rx)
                              for r in block.records] * 2)
        perturbed = torch.zeros(64)
        perturbed[query] = 123
        assert torch.count_nonzero((mask @ perturbed)[donor]) == 0
        assert mask.shape == (64,64)
