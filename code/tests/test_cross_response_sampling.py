import copy
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from torch.utils.data import DataLoader
from cvsrffi.cross_response.schema import SampleRecord, records_from_dataset
from cvsrffi.cross_response.sampler import CrossBlockSampler, CrossResponseDataset, extract_batch_plan
from cvsrffi.cross_response.roles import make_roles, mixstyle_allowed_mask
from cvsrffi.cross_response.coverage import CoverageTracker
from cvsrffi.cross_response.scheduler import FeedbackScheduler


def records(p=4,q=4,k=3,day=0):
    return [SampleRecord(i,t,r,day,0,(t,r,day,0,s)) for i,(t,r,s) in
            enumerate((t,r,s) for t in range(p) for r in range(q) for s in range(k))]


def test_complete_and_missing_cells_unique_k():
    rows = records()
    sampler = CrossBlockSampler(rows,batch_size=64)
    plan = sampler.next_batch()
    assert len(plan.indices)==64 and plan.blocks
    for b in plan.blocks:
        b.validate()
        assert len({r.physical_sample_id for r in b.records}) == 32
    missing = [r for r in rows if (r.tx_id,r.rx_id)!=(3,3)]
    plan = CrossBlockSampler(missing,batch_size=64).next_batch()
    assert len(plan.indices)==64 and not plan.blocks and plan.skip_reason
    duplicate = [replace(r,index=r.index+len(rows)) for r in rows if r.physical_sample_id[-1]==0]
    one = [r for r in rows if r.physical_sample_id[-1]==0]+duplicate
    assert not CrossBlockSampler(one,batch_size=32).next_batch().blocks
    with pytest.raises(ValueError,match="duplicate physical"):
        b = sampler.next_batch().blocks[0]
        replace(b,records=(b.records[0],)*32).validate()


def test_day_cannot_be_second_rx_and_conditions_separate():
    rows = records(q=1,day=0)+[replace(r,index=r.index+12) for r in records(q=1,day=1)]
    assert not CrossBlockSampler(rows,P=2,Q=2,k_menu=(1,),batch_size=8).next_batch().blocks
    rows = [replace(r,condition_id=r.rx_id) for r in records()]
    assert not CrossBlockSampler(rows).next_batch().blocks


def test_dataset_adapter_rejects_hidden_and_keeps_rx():
    ds = SimpleNamespace(split_source="ssdg_labeled_tx_visible",index=[
        SimpleNamespace(tx_i=1,rx_i=2,day_i=3,eq_i=0,sig_i=4)])
    row = records_from_dataset(ds)[0]
    assert row.rx_id==2 and row.day_id==3 and row.event_id is None
    ds.split_source="ssdg_unlabeled_tx_hidden"
    with pytest.raises(ValueError,match="source-labeled"):
        records_from_dataset(ds)


def test_rotation_directed_coverage_and_failed_step():
    sampler = CrossBlockSampler(records(),batch_size=32)
    blocks = [sampler.next_batch().blocks[0] for _ in range(4)]
    assert len({b.roles for b in blocks})==4
    coverage = CoverageTracker()
    for step,b in enumerate(blocks):
        coverage.commit(b,step)
    assert len(coverage.joint)==36
    assert len(coverage.directed)==8
    old = copy.deepcopy(coverage.joint)
    coverage.commit(blocks[0],100,success=False)
    assert coverage.joint==old
    coverage.commit(blocks[0],101)
    assert all(row["successes"]==5 for row in coverage.joint.values())
    assert all(len(row["physical"])<=12 for row in coverage.joint.values())
    assert not make_roles((0,1),(0,1)).independent_query_rectangle
    assert make_roles((0,1,2),(0,1,2)).independent_query_rectangle


def test_resume_determinism_including_pending_feedback():
    a = CrossBlockSampler(records(k=8),batch_size=64,k_menu=(1,2,4),
                          scheduler=FeedbackScheduler(mode="guided"))
    for step in range(3):
        for block in a.next_batch().blocks:
            a.scheduler.commit(block,step,response=1.2,decision=.4,reliability=.8)
    state=a.state_dict()
    expected=[a.next_batch() for _ in range(5)]
    b = CrossBlockSampler(records(k=8),batch_size=64,k_menu=(1,2,4),
                          scheduler=FeedbackScheduler(mode="guided"))
    b.load_state_dict(state)
    assert [b.next_batch() for _ in range(5)]==expected
    assert b.scheduler.pending==state["scheduler"]["pending"]


def test_noisy_high_error_cannot_monopolize_and_budget():
    s=FeedbackScheduler(mode="guided",update_interval=1)
    sampler=CrossBlockSampler(records(k=4),k_menu=(1,2,4),batch_size=64,scheduler=s)
    candidates=list(sampler.candidates)
    bad=candidates[0]
    s.history[bad.key]=dict(response=1e12,decision=1e12,reliability=.05,noise=100.)
    probs=s.probabilities(candidates,0,64)
    assert abs(sum(probs)-1)<1e-12
    assert probs[0]<1/len(probs)
    assert min(probs)>=s.exploration/len(probs)
    with pytest.raises(ValueError,match="budget"):
        s.probabilities(candidates,0,1)


class ToyDataset:
    def __len__(self):
        return 48
    def __getitem__(self,index):
        return torch.ones(2,16)*index,index//12,index%4,{"base_index":index}


def test_worker_ticket_roundtrip_mask_and_no_query_to_donor():
    sampler=CrossBlockSampler(records(),batch_size=40,steps_per_epoch=2)
    loader=DataLoader(CrossResponseDataset(ToyDataset()),batch_sampler=sampler,num_workers=0)
    batches=list(loader)
    first=extract_batch_plan(batches[0][3])
    assert first == extract_batch_plan((batches[0][2], batches[0][3]))
    assert first == extract_batch_plan([batches[0][2], batches[0][3]])
    second=extract_batch_plan(batches[1][3])
    assert first.blocks[0].roles != second.blocks[0].roles
    assert tuple(batches[0][3]["base_index"].tolist())==first.indices
    mask=mixstyle_allowed_mask(first,views=2)
    assert mask.shape==(80,80)
    assert not mask[:40,40:].any()
    assert not mask[40:,:40].any()
    b=first.blocks[0]
    query=[p for p,r in zip(b.batch_positions,b.records) if r.tx_id in b.roles.query_tx and r.rx_id in b.roles.query_rx]
    donors=[p for p,r in zip(b.batch_positions,b.records) if (r.tx_id in b.roles.query_tx) != (r.rx_id in b.roles.query_rx)]
    assert not mask[donors][:,query].any()
    assert not mask.diag().any()


def test_bounded_candidate_search_is_reported():
    s=CrossBlockSampler(records(p=8,q=8,k=2),max_candidates=3,candidate_attempts=5)
    assert len(s.candidates)<=3 and not s.candidate_search_complete


def test_fixed_roles_and_resume_contract():
    fixed=CrossBlockSampler(records(),batch_size=32,role_rotation=False)
    roles=[fixed.next_batch().blocks[0].roles for _ in range(5)]
    assert len(set(roles))==1
    state=fixed.state_dict()
    other=CrossBlockSampler(records(),batch_size=32,role_rotation=True)
    with pytest.raises(ValueError,match="configuration mismatch"):
        other.load_state_dict(state)


def test_real_events_are_disjoint_or_fallback_without_fake_events():
    rows=[replace(r,event_id="one-packet") for r in records()]
    sampler=CrossBlockSampler(rows,batch_size=32)
    plan=sampler.next_batch()
    assert not plan.blocks and len(plan.indices)==32
    assert sampler.skip_counts["event_disjoint_block_unavailable"]==1
    rows=[replace(r,event_id=r.physical_sample_id) for r in records()]
    block=CrossBlockSampler(rows,batch_size=32).next_batch().blocks[0]
    block.validate()
    with pytest.raises(ValueError,match="event overlap"):
        replace(block,records=tuple(replace(r,event_id="same") for r in block.records)).validate()


def test_k2_k3_execute_with_fixed_record_and_view_budget():
    sampler=CrossBlockSampler(records(k=12),k_menu=(2,3),batch_size=96,
                              scheduler=FeedbackScheduler(mode="guided",views=2,update_interval=2))
    seen=set()
    for step in range(20):
        plan=sampler.next_batch()
        assert len(plan.indices)==96
        assert sum(b.candidate.size*2 for b in plan.blocks)<=192
        physical=[]
        for b in plan.blocks:
            seen.add(b.candidate.k)
            b.validate()
            physical.extend(r.physical_sample_id for r in b.records)
            sampler.scheduler.commit(b,step,response=.8,reliability=.9,noise=.4)
        assert len(physical)==len(set(physical))
    assert seen=={2,3}
    assert sampler.scheduler.history
