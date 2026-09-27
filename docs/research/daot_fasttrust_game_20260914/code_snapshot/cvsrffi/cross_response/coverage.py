"""Sparse joint rectangles and directed transfer coverage, committed on success."""
import copy
from itertools import combinations


class CoverageTracker:
    def __init__(self):
        self.joint = {}
        self.directed = {}
        self.response_query_rectangles = {}
        self.exposed_physical = set()

    @property
    def observed_rectangles(self):
        return self.joint

    @property
    def directed_response_transfers(self):
        return self.directed

    @staticmethod
    def query_keys(candidate, roles):
        return [(t, r, candidate.day_id, candidate.condition_id)
                for t in combinations(sorted(roles.query_tx), 2)
                for r in combinations(sorted(roles.query_rx), 2)]

    def reachable_coverage(self, candidates, role_policy="legacy_halves_v1"):
        """Reachable sets for the actual candidate pool, separate from observed."""
        from .roles import make_roles
        observed, query, directed = set(), set(), set()
        for c in candidates:
            for turn in range(36 if role_policy == "balanced_partitions_v2" else 4):
                role = make_roles(c.tx_ids, c.rx_ids, turn, policy=role_policy)
                o, d = self.keys(c, role)
                observed.update(o)
                query.update(self.query_keys(c, role))
                directed.update(d)
        return dict(observed_rectangles=observed, response_query_rectangles=query,
                    directed_response_transfers=directed)

    @staticmethod
    def keys(candidate, roles):
        c = candidate
        joint = [(t, r, c.day_id, c.condition_id) for t in combinations(sorted(c.tx_ids), 2)
                 for r in combinations(sorted(c.rx_ids), 2)]
        directed = [("rx_transfer", roles.donor_rx, roles.query_rx, roles.query_tx, c.day_id, c.condition_id),
                    ("tx_transfer", roles.donor_tx, roles.query_tx, roles.query_rx, c.day_id, c.condition_id)]
        if roles.role_policy_version != "legacy_halves_v1":
            directed = [key + (roles.role_policy_version,) for key in directed]
        return joint, directed

    def gain(self, candidate, roles):
        joint, directed = self.keys(candidate, roles)
        values = [1.0 / (1 + table.get(key, {}).get("successes", 0))
                  for table, keys in ((self.joint, joint), (self.directed, directed)) for key in keys]
        return sum(values) / max(1, len(values))

    def expose(self, block):
        self.exposed_physical.update(r.physical_sample_id for r in block.records)

    def commit(self, block, step, reliability=1.0, valid_count=None, success=True):
        # valid_count is accepted for API compatibility only. Full-block counts
        # cannot be broadcast into four-corner or directed-task support counts.
        self.expose(block)
        if not success:
            return
        joint, directed = self.keys(block.candidate, block.roles)
        query = self.query_keys(block.candidate, block.roles)
        for table, keys in ((self.joint, joint), (self.directed, directed),
                            (self.response_query_rectangles, query)):
            for key in keys:
                row = table.setdefault(key, {"successes": 0, "physical": set(), "last_step": -1,
                                             "reliability_sum": 0.0, "valid_count": 0,
                                             "record_exposures": 0, "unique_physical_records": 0,
                                             "valid_tasks": 0, "last_success_step": -1})
                row["successes"] += 1
                row["valid_tasks"] += 1
                # A rectangle only counts receptions in its own four corners.
                if table is self.directed:
                    # A transfer includes its query rectangle and relevant donor
                    # strip, excluding the unused donor-TX x donor-RX quadrant.
                    records = [r for r in block.records if
                               (r.tx_id in block.roles.query_tx if key[0] == "rx_transfer"
                                else r.rx_id in block.roles.query_rx)]
                else:
                    records = [r for r in block.records if r.tx_id in key[0] and r.rx_id in key[1]]
                row["physical"].update(r.physical_sample_id for r in records)
                row["unique_physical_records"] = len(row["physical"])
                row["last_step"] = row["last_success_step"] = max(row["last_step"], int(step))
                row["reliability_sum"] += max(0.0, min(1.0, float(reliability)))
                row["record_exposures"] += len(records)
                row["valid_count"] = row["record_exposures"]
                row["reliability"] = row["reliability_sum"] / row["valid_tasks"]

    def state_dict(self):
        return copy.deepcopy(dict(state_version=2,joint=self.joint, directed=self.directed,
            observed_rectangles=self.joint,response_query_rectangles=self.response_query_rectangles,
            directed_response_transfers=self.directed,exposed_physical=self.exposed_physical))

    def load_state_dict(self, state):
        for key in ("joint", "directed", "exposed_physical"):
            setattr(self, key, copy.deepcopy(state[key]))
        self.response_query_rectangles = copy.deepcopy(state.get("response_query_rectangles", {}))
        for table in (self.joint, self.directed, self.response_query_rectangles):
            for row in table.values():
                if "record_exposures" not in row:
                    # Historical overcounts cannot be reconstructed; retain the
                    # original under an explicit legacy field, start V2 at zero.
                    row["legacy_valid_count"] = row.get("valid_count", 0)
                    row["record_exposures"] = row["valid_count"] = 0
                    row["counting_since_state_version"] = 2
                row.setdefault("valid_tasks",row.get("successes",0))
                row.setdefault("unique_physical_records",len(row["physical"]))
                row.setdefault("last_success_step",row.get("last_step",-1))
