"""Metadata-only full-grid reachability and configured 128-candidate coverage."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cvsrffi.cross_response.schema import SampleRecord
from cvsrffi.cross_response.sampler import CrossBlockSampler
from cvsrffi.cross_response.coverage import CoverageTracker


def audit():
    rows = [SampleRecord(i,t,r,d,"clean",(t,r,d,k)) for i,(t,r,d,k) in enumerate(
        (t,r,d,k) for t in range(6) for r in range(5) for d in range(3) for k in range(70))]
    full = CrossBlockSampler(rows,batch_size=128,seed=392005,max_candidates=1000,
                            role_policy="balanced_partitions_v2")
    sampler = CrossBlockSampler(rows,batch_size=128,seed=392005,max_candidates=128,
                               role_policy="balanced_partitions_v2")
    tracker = CoverageTracker()
    full_reachable = tracker.reachable_coverage(full.candidates,"balanced_partitions_v2")
    pool_reachable = tracker.reachable_coverage(sampler.candidates,"balanced_partitions_v2")
    blocks = 0
    for step in range(49):
        plan = sampler.next_batch()
        for block in plan.blocks:
            tracker.commit(block,step,success=True)
            blocks += 1
    assert len(full_reachable["response_query_rectangles"]) == 450
    assert len(sampler.candidates) == 128
    assert set(tracker.response_query_rectangles) <= pool_reachable["response_query_rectangles"]
    return dict(status="VERIFIED",scope="synthetic complete source metadata grid; not a formal run or target evaluation",
        tx=6,rx=5,days=3,K=2,physical_records=len(rows),role_combinations_per_candidate=36,
        full_feasible_candidates=len(full.candidates),actual_candidate_pool=len(sampler.candidates),
        all_feasible_reachable={k:len(v) for k,v in full_reachable.items()},
        actual_pool_reachable={k:len(v) for k,v in pool_reachable.items()},
        sampled_steps=49,sampled_blocks=blocks,
        actually_observed=dict(observed_rectangles=len(tracker.observed_rectangles),
            response_query_rectangles=len(tracker.response_query_rectangles),
            directed_response_transfers=len(tracker.directed_response_transfers)),
        interpretation="reachable sets are structural upper bounds; observed/query coverage counts are separate")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    report=audit()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
