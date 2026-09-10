"""Metadata-only scheduling; feedback is historical, bounded and reliability weighted."""
import copy
import math
from .coverage import CoverageTracker
from .roles import make_roles


class FeedbackScheduler:
    def __init__(self, mode="uniform", exploration=0.2, alpha=1.0, beta=1.0,
                 error_clip=2.0, update_interval=10, views=2):
        if mode not in ("uniform", "guided") or not 0 < exploration <= 1:
            raise ValueError("invalid scheduler mode/exploration")
        if min(error_clip, update_interval, views) <= 0 or min(alpha, beta) < 0:
            raise ValueError("invalid scheduler controls")
        self.mode, self.exploration = mode, float(exploration)
        self.alpha, self.beta, self.error_clip = float(alpha), float(beta), float(error_clip)
        self.update_interval, self.views = int(update_interval), int(views)
        self.coverage = CoverageTracker()
        self.history = {}
        self.pending = {}

    def probabilities(self, candidates, rotation, remaining_records):
        if any(c.size > remaining_records for c in candidates):
            raise ValueError("candidate exceeds remaining record/view budget")
        if not candidates:
            return []
        if self.mode == "uniform":
            return [1 / len(candidates)] * len(candidates)
        scores = []
        for c in candidates:
            h = self.history.get(c.key, {})
            noise = max(0.0, h.get("noise", 0.0))
            w = h.get("reliability", 1.0) / (1.0 + noise)
            gain = self.coverage.gain(c, make_roles(c.tx_ids, c.rx_ids, rotation))
            dec = min(self.error_clip, max(0.0, h.get("decision", 0.0)))
            resp = min(self.error_clip, max(0.0, h.get("response", 0.0)))
            # Greater K can improve noisy centers, but cost still charges all views.
            variance_gain = noise / (1.0 + noise) * (1.0 - 1.0 / c.k)
            scores.append((w * gain + self.alpha*w*dec + self.beta*w*resp + w*variance_gain)
                          / (c.size*self.views + 1e-12))
        total = sum(scores)
        base = 1 / len(scores)
        return [(1-self.exploration)*(s/total if total > 0 else base) + self.exploration*base for s in scores]

    def commit(self, block, step, *, response=0.0, decision=0.0, reliability=1.0,
               noise=0.0, valid_count=None, success=True):
        values = (response, decision, reliability, noise)
        success = bool(success and all(math.isfinite(float(x)) for x in values))
        self.coverage.commit(block, step, reliability if success else 0, valid_count, success)
        if not success:
            return
        row = self.pending.setdefault(block.candidate.key, [])
        row.append(dict(response=max(0.,float(response)), decision=max(0.,float(decision)),
                        reliability=max(0.,min(1.,float(reliability))), noise=max(0.,float(noise))))
        if (int(step)+1) % self.update_interval == 0:
            for key, rows in self.pending.items():
                means = {name: sum(r[name] for r in rows)/len(rows) for name in rows[0]}
                old = self.history.get(key, means)
                self.history[key] = {name: .8*old[name]+.2*means[name] for name in means}
            self.pending.clear()

    def state_dict(self):
        return copy.deepcopy({"config": (self.mode,self.exploration,self.alpha,self.beta,self.error_clip,self.update_interval,self.views),
                              "coverage": self.coverage.state_dict(), "history": self.history, "pending": self.pending})

    def load_state_dict(self, state):
        if state["config"] != self.state_dict()["config"]:
            raise ValueError("scheduler resume configuration mismatch")
        self.coverage.load_state_dict(state["coverage"])
        self.history, self.pending = copy.deepcopy(state["history"]), copy.deepcopy(state["pending"])
