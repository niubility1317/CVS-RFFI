"""Metadata-only scheduling; feedback is historical, bounded and reliability weighted."""
import copy
import math
from .coverage import CoverageTracker
from .roles import make_roles


class FeedbackScheduler:
    def __init__(self, mode="uniform", exploration=0.2, alpha=1.0, beta=1.0,
                 error_clip=2.0, update_interval=10, views=2,
                 feedback_version="legacy_v1", role_policy_version="legacy_halves_v1",
                 direction_shrinkage=0.0, gain_strategy="legacy_gain", evidence_config=None):
        if mode not in ("uniform", "guided") or not 0 < exploration <= 1:
            raise ValueError("invalid scheduler mode/exploration")
        if min(error_clip, update_interval, views) <= 0 or min(alpha, beta) < 0:
            raise ValueError("invalid scheduler controls")
        self.mode, self.exploration = mode, float(exploration)
        self.alpha, self.beta, self.error_clip = float(alpha), float(beta), float(error_clip)
        self.update_interval, self.views = int(update_interval), int(views)
        if feedback_version not in ("legacy_v1", "transaction_v2"):
            raise ValueError("unknown feedback version")
        if not math.isfinite(float(direction_shrinkage)) or direction_shrinkage < 0:
            raise ValueError("invalid direction shrinkage")
        self.feedback_version = feedback_version
        self.role_policy_version = role_policy_version
        self.direction_shrinkage = float(direction_shrinkage)
        if gain_strategy not in ("legacy_gain", "reliable_evidence"):
            raise ValueError("unknown feedback gain strategy")
        self.gain_strategy = gain_strategy
        self.evidence_config = copy.deepcopy(evidence_config)
        if gain_strategy == "reliable_evidence":
            if mode != "guided" or feedback_version != "transaction_v2":
                raise ValueError("reliable_evidence requires guided transaction_v2")
            self._validate_evidence_config()
            self._frozen_evidence_config = copy.deepcopy(self.evidence_config)
        elif evidence_config is not None:
            raise ValueError("evidence_config is only meaningful for reliable_evidence")
        self.coverage = CoverageTracker()
        self.history = {}
        self.pending = {}
        self.direction_history = {}
        self.direction_pending = {}
        self._step_id = None
        self._step_rows = []
        self.last_finished_step = -1
        self.flush_count = 0
        self.last_step_event = None
        self.last_probability_audit = []

    def _validate_evidence_config(self):
        """Validate an explicit source freeze, never infer scientific qualification.

        The caller supplies source-derived evidence. Schema acceptance itself is
        not evidence that a real model passed these scientific conditions.
        """
        cfg = self.evidence_config
        if not isinstance(cfg, dict) or cfg.get("frozen") is not True:
            raise ValueError("reliable_evidence requires explicitly frozen source parameters")
        if hasattr(self, "_frozen_evidence_config") and cfg != self._frozen_evidence_config:
            raise ValueError("frozen evidence configuration changed; construct an explicit new source-frozen branch")
        source = cfg.get("source_evidence", {})
        if (not isinstance(source, dict) or source.get("source_only") is not True or source.get("target_used") is not False
                or source.get("joint_objective_validated") is not True
                or not isinstance(source.get("evidence_id"), str) or not source["evidence_id"].strip()):
            raise ValueError("reliable_evidence requires valid source-only joint-objective evidence")
        gate = source.get("gate_result", {})
        if not isinstance(gate, dict) or gate.get("source_role") != "source_validation":
            raise ValueError("reliable_evidence requires source_validation gate evidence")
        for name in ("capability", "necessity", "update_value"):
            row = gate.get(name, {})
            if not isinstance(row, dict):
                raise ValueError("invalid gate evidence: " + name)
            count = row.get("sample_count")
            metrics = row.get("metrics")
            if (row.get("passed") is not True or isinstance(count, bool)
                    or not isinstance(count, int) or count <= 0 or not isinstance(metrics, dict)
                    or not metrics or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                          or not math.isfinite(v) for v in metrics.values())):
                raise ValueError("reliable_evidence requires passed gate with metrics/sample_count: " + name)
        names = ("coverage", "query_gap", "new_physical", "staleness", "decision", "response")
        for group, required, positive in (("weights", names, False),
                                           ("scales", names + ("noise", "cost"), True)):
            values = cfg.get(group, {})
            if not isinstance(values, dict) or set(values) != set(required):
                raise ValueError("all evidence " + group + " must be explicit")
            if any(not isinstance(v, (int, float)) or isinstance(v, bool)
                   or not math.isfinite(v) or (v <= 0 if positive else v < 0)
                   for v in values.values()):
                raise ValueError("invalid evidence " + group)
        if not any(cfg["weights"].values()):
            raise ValueError("at least one evidence weight must be positive")
        cap, reliability = cfg.get("new_physical_cap"), cfg.get("unknown_reliability")
        if (not isinstance(cap, (int, float)) or isinstance(cap, bool)
                or not math.isfinite(cap) or cap <= 0):
            raise ValueError("new_physical_cap must be explicit and positive")
        if (not isinstance(reliability, (int, float)) or isinstance(reliability, bool)
                or not math.isfinite(reliability) or not 0 <= reliability <= 1):
            raise ValueError("unknown_reliability must be explicit and bounded")

    def probabilities(self, candidates, rotation, remaining_records, *, role_plans=None,
                      candidate_evidence=None, step_id=None):
        if any(c.size > remaining_records for c in candidates):
            raise ValueError("candidate exceeds remaining record/view budget")
        if not candidates:
            return []
        if self.mode == "uniform":
            return [1 / len(candidates)] * len(candidates)
        if self.gain_strategy == "reliable_evidence":
            return self._evidence_probabilities(candidates, role_plans, candidate_evidence, step_id)
        scores = []
        for c in candidates:
            h = self.history.get(c.key, {})
            noise = max(0.0, h.get("noise", 0.0))
            w = h.get("reliability", 1.0) / (1.0 + noise)
            role = (role_plans[c.key] if role_plans is not None
                    else make_roles(c.tx_ids, c.rx_ids, rotation))
            gain = self.coverage.gain(c, role)
            dec = min(self.error_clip, max(0.0, h.get("decision", 0.0)))
            resp = min(self.error_clip, max(0.0, h.get("response", 0.0)))
            # Greater K can improve noisy centers, but cost still charges all views.
            variance_gain = noise / (1.0 + noise) * (1.0 - 1.0 / c.k)
            scores.append((w * gain + self.alpha*w*dec + self.beta*w*resp + w*variance_gain)
                          / (c.size*self.views + 1e-12))
        total = sum(scores)
        base = 1 / len(scores)
        return [(1-self.exploration)*(s/total if total > 0 else base) + self.exploration*base for s in scores]

    def _evidence_probabilities(self, candidates, role_plans, candidate_evidence, step_id):
        # Freeze is checked again in case caller mutated public config after init.
        self._validate_evidence_config()
        if role_plans is None or candidate_evidence is None:
            raise ValueError("reliable_evidence requires prospective roles and physical metadata")
        step = self.last_finished_step + 1 if step_id is None else step_id
        if isinstance(step, bool) or int(step) != step or step < 0:
            raise ValueError("evidence step_id must be a nonnegative attempt ID")
        cfg, audit, scores = self.evidence_config, [], []
        def bounded(value, scale):
            return value / (scale + value)
        for candidate in candidates:
            role = role_plans[candidate.key]
            if role.role_policy_version != self.role_policy_version:
                raise ValueError("evidence role policy mismatch")
            meta = candidate_evidence[candidate.key]
            novelty = meta.get("new_physical_estimate")
            if (not isinstance(novelty, (int, float)) or isinstance(novelty, bool)
                    or not math.isfinite(novelty) or novelty < 0
                    or meta.get("record_budget") != candidate.size):
                raise ValueError("invalid candidate physical evidence or record budget")
            query_keys = self.coverage.query_keys(candidate, role)
            if not query_keys:
                raise ValueError("reliable_evidence requires actual response-query rectangles")
            query_rows = [self.coverage.response_query_rectangles.get(key, {}) for key in query_keys]
            direction = self.directional_estimate(candidate, role)
            h = direction["estimate"] or {}
            # Missing histories stay explicit in audit, with no invented errors.
            raw = dict(coverage=self.coverage.gain(candidate, role),
                       query_gap=sum(not row.get("valid_tasks", 0) for row in query_rows) / len(query_rows),
                       new_physical=min(float(novelty), float(cfg["new_physical_cap"]), candidate.size),
                       staleness=math.fsum(max(0, step - row.get("last_success_step", -1))
                                          for row in query_rows) / len(query_rows),
                       decision=max(0., h.get("decision", 0.)), response=max(0., h.get("response", 0.)),
                       noise=max(0., h.get("noise", 0.)), cost=candidate.size * self.views)
            normalized = {name: bounded(value, cfg["scales"][name]) for name, value in raw.items()}
            weighted = {name: cfg["weights"][name] * normalized[name] for name in cfg["weights"]}
            weighted_total = math.fsum(weighted.values())
            reliability = max(0., min(1., h.get("reliability", cfg["unknown_reliability"])))
            noise_discount = 1. / (1. + normalized["noise"])
            cost_discount = 1. - normalized["cost"]
            score = weighted_total * reliability * noise_discount * cost_discount
            scores.append(score)
            audit.append(dict(candidate_key=candidate.key, direction_key=self.direction_key(candidate, role),
                              role_plan_id=role.plan_id, step_id=int(step), gain_strategy=self.gain_strategy,
                              source_evidence_id=cfg["source_evidence"]["evidence_id"],
                              raw=raw, normalized=normalized, weighted=weighted,
                              component_fraction={name: value / weighted_total if weighted_total else 0.
                                                  for name, value in weighted.items()},
                              novelty_estimate_before_cap=float(novelty), novelty_cap=cfg["new_physical_cap"],
                              direction=direction, history_available=bool(h), reliability=reliability,
                              noise_discount=noise_discount, cost_discount=cost_discount, score=score))
        total, base = math.fsum(scores), 1. / len(scores)
        probabilities = [(1-self.exploration) * (score/total if total else base) + self.exploration*base
                         for score in scores]
        for row, probability in zip(audit, probabilities):
            row.update(score_total=total, exploration=self.exploration,
                       exploration_probability=self.exploration*base, probability=probability,
                       zero_score_fallback=not bool(total))
        self.last_probability_audit = audit
        return probabilities

    def commit(self, block, step, *, response=0.0, decision=0.0, reliability=1.0,
               noise=0.0, valid_count=None, success=True):
        if self.feedback_version != "legacy_v1":
            raise RuntimeError("V2 requires collect_block_feedback and one finish per attempted step")
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

    def _check_step(self, step):
        if isinstance(step, bool) or int(step) != step or int(step) < 0:
            raise ValueError("step must be a nonnegative optimizer attempt ID")
        step = int(step)
        if step <= self.last_finished_step:
            raise ValueError("step already finished or nonmonotonic")
        if self._step_id is not None and self._step_id != step:
            raise ValueError("finish the active step before collecting another")
        return step

    @staticmethod
    def direction_key(candidate, roles):
        return (candidate.key, tuple(roles.query_tx), tuple(roles.query_rx),
                tuple(roles.donor_tx), tuple(roles.donor_rx), candidate.condition_id,
                getattr(roles, "role_policy_version", "legacy_halves_v1"))

    def collect_block_feedback(self, block, step, *, response=0.0, decision=0.0,
                               reliability=1.0, noise=0.0, valid_count=None, success=True):
        """Stage metadata only; no history or coverage changes before optimizer outcome."""
        if self.feedback_version != "transaction_v2":
            raise RuntimeError("transaction feedback requires transaction_v2")
        step = self._check_step(step)
        block.validate()
        if getattr(block.roles, "role_policy_version", "legacy_halves_v1") != self.role_policy_version:
            raise ValueError("feedback role policy mismatch")
        values = dict(response=float(response), decision=float(decision),
                      reliability=float(reliability), noise=float(noise))
        valid = bool(success and all(math.isfinite(v) for v in values.values()))
        if valid:
            values = {name: max(0., value) for name, value in values.items()}
            values["reliability"] = min(1., values["reliability"])
        else:
            values = dict(response=0., decision=0., reliability=0., noise=0.)
        self._step_id = step
        self._step_rows.append((block, values, valid_count, valid))

    @staticmethod
    def _flush_table(pending, history):
        for key in sorted(pending, key=repr):
            rows = pending[key]
            means = {name: math.fsum(sorted(row[name] for row in rows)) / len(rows)
                     for name in ("response", "decision", "reliability", "noise")}
            old = history.get(key, {})
            history[key] = {name: .8 * old.get(name, 0.) + .2 * means[name]
                            for name in means}
            history[key]["sample_count"] = old.get("sample_count", 0) + len(rows)
        pending.clear()

    def finish_step_and_maybe_flush(self, step, *, success=True):
        """Finish an attempt exactly once, including empty/failed attempts.

        A failed attempt contributes read exposure only. Earlier successful pending
        feedback still flushes on the attempt-clock boundary; tails stay pending.
        """
        if self.feedback_version != "transaction_v2":
            raise RuntimeError("transaction feedback requires transaction_v2")
        step = self._check_step(step)
        committed = 0
        novelty_rows = []
        rows = sorted(self._step_rows, key=lambda row: repr((
            self.direction_key(row[0].candidate, row[0].roles),
            tuple(r.physical_sample_id for r in row[0].records),
            row[0].block_id, row[1], row[2], row[3])))
        for block, values, valid_count, valid in rows:
            accepted = bool(success and valid)
            actual_new = len({record.physical_sample_id for record in block.records}
                             - self.coverage.exposed_physical)
            novelty_rows.append(dict(block_id=block.block_id, candidate_key=block.candidate.key,
                                     direction_key=self.direction_key(block.candidate, block.roles),
                                     actual_new_physical=actual_new, success=accepted))
            self.coverage.commit(block, step, values["reliability"] if accepted else 0.,
                                 valid_count, accepted)
            if accepted:
                self.pending.setdefault(block.candidate.key, []).append(values)
                self.direction_pending.setdefault(self.direction_key(block.candidate, block.roles), []).append(values)
                committed += 1
        # Canonicalize checkpoint payloads too, not only the eventual means.
        for pending in (self.pending, self.direction_pending):
            for values in pending.values():
                values.sort(key=lambda row: tuple(row[name] for name in sorted(row)))
        flushed = (step + 1) % self.update_interval == 0
        if flushed:
            self._flush_table(self.pending, self.history)
            self._flush_table(self.direction_pending, self.direction_history)
            self.flush_count += 1
        self._step_rows = []
        self._step_id = None
        self.last_finished_step = step
        self.last_step_event = dict(step_id=step, success=bool(success),
                                    collected_blocks=len(rows), accepted_blocks=committed,
                                    flushed=flushed, flush_count=self.flush_count,
                                    new_physical_records=sum(row["actual_new_physical"] for row in novelty_rows),
                                    block_physical_evidence=novelty_rows)
        return copy.deepcopy(self.last_step_event)

    def directional_estimate(self, candidate, roles):
        """Inspectable sparse-direction prior; P0 probabilities retain legacy gain.

        Missing directional measurements remain None. The separately labelled
        estimate may use candidate history; it is never reported as measured data.
        """
        raw = self.direction_history.get(self.direction_key(candidate, roles))
        prior = self.history.get(candidate.key)
        n = 0 if raw is None else raw["sample_count"]
        weight = n / (n + self.direction_shrinkage) if n else 0.
        if raw is None:
            estimate = None if prior is None else {k: prior[k] for k in ("response", "decision", "reliability", "noise")}
            reason = "unobserved_direction" if prior is not None else "no_candidate_history"
        else:
            estimate = {k: weight * raw[k] + (1 - weight) * (prior or raw)[k]
                        for k in ("response", "decision", "reliability", "noise")}
            reason = "sparse_direction_shrinkage" if weight < 1 and prior else "none"
        return dict(raw=copy.deepcopy(raw), estimate=estimate, raw_sample_count=n,
                    direction_weight=weight, shrinkage_strength=self.direction_shrinkage,
                    fallback_reason=reason)

    def state_dict(self):
        state = {"config": (self.mode,self.exploration,self.alpha,self.beta,self.error_clip,self.update_interval,self.views),
                 "coverage": self.coverage.state_dict(), "history": self.history, "pending": self.pending}
        if self.feedback_version == "transaction_v2":
            state.update(state_version=2, feedback_config=(self.feedback_version,
                         self.role_policy_version, self.direction_shrinkage),
                         direction_history=self.direction_history, direction_pending=self.direction_pending,
                         active_step_id=self._step_id, active_step_rows=self._step_rows,
                         last_finished_step=self.last_finished_step, flush_count=self.flush_count,
                         last_step_event=self.last_step_event)
            if self.gain_strategy == "reliable_evidence":
                state.update(state_version=3, gain_config=(self.gain_strategy, self.evidence_config),
                             last_probability_audit=self.last_probability_audit)
        return copy.deepcopy(state)

    def load_state_dict(self, state):
        expected_version = (3 if self.gain_strategy == "reliable_evidence" else
                            2 if self.feedback_version == "transaction_v2" else 1)
        if state.get("state_version", 1) != expected_version:
            raise ValueError("scheduler resume state version mismatch")
        if state["config"] != self.state_dict()["config"]:
            raise ValueError("scheduler resume configuration mismatch")
        if expected_version >= 2:
            if state.get("feedback_config") != self.state_dict()["feedback_config"]:
                raise ValueError("scheduler resume feedback configuration mismatch")
            required = ("direction_history", "direction_pending", "active_step_id", "active_step_rows",
                        "last_finished_step", "flush_count", "last_step_event")
            if any(key not in state for key in required):
                raise ValueError("incomplete V2 scheduler resume state")
        if expected_version == 3 and (state.get("gain_config") != self.state_dict()["gain_config"]
                                      or "last_probability_audit" not in state):
            raise ValueError("scheduler resume evidence configuration mismatch")
        self.coverage.load_state_dict(state["coverage"])
        self.history, self.pending = copy.deepcopy(state["history"]), copy.deepcopy(state["pending"])
        if expected_version >= 2:
            self.direction_history = copy.deepcopy(state["direction_history"])
            self.direction_pending = copy.deepcopy(state["direction_pending"])
            self._step_id = state["active_step_id"]
            self._step_rows = copy.deepcopy(state["active_step_rows"])
            self.last_finished_step = state["last_finished_step"]
            self.flush_count = state["flush_count"]
            self.last_step_event = copy.deepcopy(state["last_step_event"])
        if expected_version == 3:
            self.last_probability_audit = copy.deepcopy(state["last_probability_audit"])
