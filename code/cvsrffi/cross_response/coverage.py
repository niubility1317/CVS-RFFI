"""Sparse joint rectangles and directed transfer coverage, committed on success."""
import copy
from itertools import combinations


class CoverageTracker:
    def __init__(self):
        self.joint = {}
        self.directed = {}
        self.exposed_physical = set()

    @staticmethod
    def keys(candidate, roles):
        c = candidate
        joint = [(t, r, c.day_id, c.condition_id) for t in combinations(c.tx_ids, 2)
                 for r in combinations(c.rx_ids, 2)]
        directed = [("rx_transfer", roles.donor_rx, roles.query_rx, roles.query_tx, c.day_id, c.condition_id),
                    ("tx_transfer", roles.donor_tx, roles.query_tx, roles.query_rx, c.day_id, c.condition_id)]
        return joint, directed

    def gain(self, candidate, roles):
        joint, directed = self.keys(candidate, roles)
        values = [1.0 / (1 + table.get(key, {}).get("successes", 0))
                  for table, keys in ((self.joint, joint), (self.directed, directed)) for key in keys]
        return sum(values) / max(1, len(values))

    def expose(self, block):
        self.exposed_physical.update(r.physical_sample_id for r in block.records)

    def commit(self, block, step, reliability=1.0, valid_count=None, success=True):
        self.expose(block)
        if not success:
            return
        joint, directed = self.keys(block.candidate, block.roles)
        for table, keys in ((self.joint, joint), (self.directed, directed)):
            for key in keys:
                row = table.setdefault(key, {"successes": 0, "physical": set(), "last_step": -1,
                                             "reliability_sum": 0.0, "valid_count": 0})
                row["successes"] += 1
                # A rectangle only counts receptions in its own four corners.
                records = block.records if table is self.directed else [r for r in block.records if r.tx_id in key[0] and r.rx_id in key[1]]
                row["physical"].update(r.physical_sample_id for r in records)
                row["last_step"] = int(step)
                row["reliability_sum"] += max(0.0, min(1.0, float(reliability)))
                row["valid_count"] += int(len(records) if valid_count is None else valid_count)

    def state_dict(self):
        return copy.deepcopy(dict(joint=self.joint, directed=self.directed, exposed_physical=self.exposed_physical))

    def load_state_dict(self, state):
        for key in ("joint", "directed", "exposed_physical"):
            setattr(self, key, copy.deepcopy(state[key]))
