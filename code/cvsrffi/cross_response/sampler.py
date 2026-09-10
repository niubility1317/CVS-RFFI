"""Bounded complete-block sampling with immutable DataLoader tickets."""
import copy
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations, product
from .schema import SampleRecord, BlockCandidate, CrossBlock, BatchPlan, RolePlan
from .roles import make_roles
from .scheduler import FeedbackScheduler


def _tuples(value):
    return tuple(_tuples(x) for x in value) if isinstance(value, list) else value


def plan_to_json(plan):
    return json.dumps(asdict(plan), separators=(",", ":"), allow_nan=False)


def extract_batch_plan(extra):
    """Reconstruct the immutable worker-delivered plan from default-collated meta."""
    if isinstance(extra, (tuple, list)):
        mappings = [item for item in extra if isinstance(item, dict) and "cr_plan_json" in item]
        if len(mappings) != 1:
            raise ValueError("batch extras require exactly one cross-response metadata mapping")
        extra = mappings[0]
    encoded = extra["cr_plan_json"]
    if isinstance(encoded, (list, tuple)):
        if not encoded or any(x != encoded[0] for x in encoded):
            raise ValueError("mixed batch tickets")
        encoded = encoded[0]
    raw = json.loads(encoded)
    blocks = []
    for b in raw["blocks"]:
        c = b["candidate"]
        c["tx_ids"], c["rx_ids"] = tuple(c["tx_ids"]), tuple(c["rx_ids"])
        c["condition_id"] = _tuples(c["condition_id"])
        records = []
        for record in b["records"]:
            for key in ("condition_id", "physical_sample_id", "event_id"):
                record[key] = _tuples(record[key])
            records.append(SampleRecord(**record))
        roles = {k: _tuples(v) for k, v in b["roles"].items()}
        block = CrossBlock(BlockCandidate(**c), tuple(records), tuple(b["batch_positions"]),
                           RolePlan(**roles), b["block_id"], b["selection_probability"])
        block.validate()
        blocks.append(block)
    return BatchPlan(tuple(raw["indices"]), tuple(blocks), raw["skip_reason"])


@dataclass(frozen=True)
class SampleTicket:
    index: int
    plan_json: str
    position: int


class CrossResponseDataset:
    """Preserve WiSig's (x,y,d,meta), carrying plans safely across worker prefetch.

    Use DataLoader(CrossResponseDataset(ds), batch_sampler=sampler). A plan is
    serialized once per batch, and its string is repeated by default_collate.
    Checkpoint exactness requires num_workers=0, or draining prefetched tickets.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ticket):
        if not isinstance(ticket, SampleTicket):
            raise TypeError("CrossResponseDataset expects sampler tickets")
        x, y, d, meta = self.dataset[ticket.index]
        meta = dict(meta)
        meta.update(cr_plan_json=ticket.plan_json, cr_batch_position=ticket.position)
        return x, y, d, meta


class CrossBlockSampler:
    def __init__(self, records, P=4, Q=4, k_menu=(2,), batch_size=64, seed=1337,
                 max_candidates=512, candidate_attempts=10000, scheduler=None,
                 steps_per_epoch=None, role_rotation=True):
        self.records = tuple(records)
        self.P, self.Q, self.k_menu = int(P), int(Q), tuple(sorted(set(map(int,k_menu))))
        self.batch_size = int(batch_size)
        if min(self.P,self.Q) < 2 or not self.k_menu or min(self.k_menu) < 1 or self.batch_size < 1:
            raise ValueError("invalid block/batch dimensions")
        if not self.records or len({r.index for r in self.records}) != len(self.records):
            raise ValueError("empty records or duplicate dataset indices")
        if min(max_candidates,candidate_attempts) < 1:
            raise ValueError("candidate limits must be positive")
        self.rng = random.Random(seed)
        self.scheduler = scheduler or FeedbackScheduler()
        self.step = 0
        self.role_rotation = bool(role_rotation)
        self.block_counter = 0
        self.steps_per_epoch = int(steps_per_epoch or math.ceil(len(records)/batch_size))
        self.cells = defaultdict(dict)
        physical_cells = {}
        for r in self.records:
            key = (r.day_id,r.condition_id,r.tx_id,r.rx_id)
            if r.physical_sample_id in physical_cells and physical_cells[r.physical_sample_id] != key:
                raise ValueError("physical ID occurs in conflicting TX/RX/day/condition cells")
            physical_cells[r.physical_sample_id] = key
            self.cells[key].setdefault(r.physical_sample_id,r)
        self.cells = {k: tuple(v.values()) for k,v in self.cells.items()}
        conditions = sorted({(r.day_id,r.condition_id) for r in self.records}, key=repr)
        possible = []
        for day,cond in conditions:
            tx = sorted({r.tx_id for r in self.records if (r.day_id,r.condition_id)==(day,cond)})
            rx = sorted({r.rx_id for r in self.records if (r.day_id,r.condition_id)==(day,cond)})
            if len(tx)>=self.P and len(rx)>=self.Q:
                possible.append((day,cond,tx,rx))
        candidates = {}
        def consider(day,cond,tx,rx):
            minimum = min(len(self.cells.get((day,cond,t,r),())) for t in tx for r in rx)
            for k in self.k_menu:
                c = BlockCandidate(tuple(tx),tuple(rx),day,cond,k)
                if minimum >= k and c.size <= self.batch_size:
                    candidates[c.key] = c
        total = sum(math.comb(len(t),self.P)*math.comb(len(r),self.Q) for _,_,t,r in possible)
        if total <= candidate_attempts:
            for day,cond,tx,rx in possible:
                for t,r in product(combinations(tx,self.P),combinations(rx,self.Q)):
                    consider(day,cond,t,r)
        elif possible:
            for _ in range(candidate_attempts):
                day,cond,tx,rx = self.rng.choice(possible)
                consider(day,cond,sorted(self.rng.sample(tx,self.P)),sorted(self.rng.sample(rx,self.Q)))
        candidates = list(candidates.values())
        self.rng.shuffle(candidates)
        self.candidates = tuple(candidates[:max_candidates])
        self.candidate_search_complete = total <= candidate_attempts and len(candidates)<=max_candidates
        self.skip_counts = defaultdict(int)

    def next_batch(self):
        indices, blocks, used = [], [], set()
        rejected = set()
        rotation = self.step % 4 if self.role_rotation else 0
        while True:
            remaining = self.batch_size-len(indices)
            feasible = [c for c in self.candidates if c.key not in rejected and c.size <= remaining and
                        all(sum(r.physical_sample_id not in used for r in self.cells[(c.day_id,c.condition_id,t,rx)]) >= c.k
                            for t in c.tx_ids for rx in c.rx_ids)]
            if not feasible:
                break
            probs = self.scheduler.probabilities(feasible,rotation,remaining)
            choice = self.rng.choices(range(len(feasible)),weights=probs,k=1)[0]
            c = feasible[choice]
            role = make_roles(c.tx_ids,c.rx_ids,rotation)
            block = None
            # Actual event metadata, when provided, cannot cross donor/query
            # roles. Bounded retries may conservatively skip a feasible block;
            # they never invent synchronized packets from local signal indices.
            for _ in range(8):
                picked = []
                query_events, donor_events = set(), set()
                for t in c.tx_ids:
                    for rx in c.rx_ids:
                        query = t in role.query_tx and rx in role.query_rx
                        donor = (t in role.query_tx) != (rx in role.query_rx)
                        forbidden = donor_events if query else query_events if donor else set()
                        pool = [r for r in self.cells[(c.day_id,c.condition_id,t,rx)]
                                if r.physical_sample_id not in used and (r.event_id is None or r.event_id not in forbidden)]
                        if len(pool) < c.k:
                            continue
                        selected = self.rng.sample(pool,c.k)
                        picked.extend(selected)
                        if query:
                            query_events.update(r.event_id for r in selected if r.event_id is not None)
                        if donor:
                            donor_events.update(r.event_id for r in selected if r.event_id is not None)
                if len(picked) == c.size:
                    block = CrossBlock(c,tuple(picked),tuple(range(len(indices),len(indices)+c.size)),
                                       role,self.block_counter,probs[choice])
                    block.validate()
                    break
            if block is None:
                rejected.add(c.key)
                self.skip_counts["event_disjoint_block_unavailable"] += 1
                continue
            blocks.append(block)
            indices.extend(r.index for r in picked)
            used.update(r.physical_sample_id for r in picked)
            self.block_counter += 1
        reason = "" if blocks else "no_feasible_complete_block"
        if reason:
            self.skip_counts[reason] += 1
        # Preserve the exact baseline record budget. Remainder is explicitly
        # ordinary supervision, never assigned a fake auxiliary block.
        remaining = self.batch_size-len(indices)
        pool = [r.index for r in self.records if r.physical_sample_id not in used]
        if remaining:
            picked = self.rng.sample(pool,min(remaining,len(pool)))
            indices.extend(picked)
            indices.extend(self.rng.choice(self.records).index for _ in range(remaining-len(picked)))
        self.step += 1
        return BatchPlan(tuple(indices),tuple(blocks),reason)

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            plan = self.next_batch()
            encoded = plan_to_json(plan)
            yield [SampleTicket(index,encoded,i) for i,index in enumerate(plan.indices)]

    def __len__(self):
        return self.steps_per_epoch

    def state_dict(self):
        return copy.deepcopy(dict(rng=self.rng.getstate(),step=self.step,block_counter=self.block_counter,
             scheduler=self.scheduler.state_dict(),skip_counts=dict(self.skip_counts),
             contract=(self.records,self.P,self.Q,self.k_menu,self.batch_size,self.steps_per_epoch,self.role_rotation,self.candidates)))

    def load_state_dict(self,state):
        if state["contract"] != self.state_dict()["contract"]:
            raise ValueError("sampler resume data/configuration mismatch")
        self.rng.setstate(state["rng"])
        self.step,self.block_counter = int(state["step"]),int(state["block_counter"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.skip_counts = defaultdict(int,state["skip_counts"])
