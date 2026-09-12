"""First-order signed-field solvers with atomic optimizer state transitions.

Closures return a scalar signed-GRL loss and must reuse a fixed batch, weights,
pseudo labels and teacher. They must not commit external loss/EMA state. GRL is
valid for these first derivatives only, never for Jacobian diagnostics.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from .parameter_roles import parameter_roles, validate_optimizer_ownership
from .state import RNGState, TrainingState, clone_buffers, restore_buffers


class NonFiniteStep(FloatingPointError):
    pass


@dataclass
class StepResult:
    loss: float
    loss_replay: float | None
    grad_norm: float
    mode: str
    accepted: bool
    field_evaluations: int
    algorithm: str
    failure_stage: str | None = None
    origin_grad_norm: float | None = None
    telemetry: dict | None = None


class GameSolver:
    MODES = {"simultaneous", "alternating", "extragradient", "heun", "optimistic", "head_lookahead"}

    def __init__(self, model, optimizer, mode="simultaneous", head_names=("adv_head.",),
                 *, nonfinite="raise", max_grad_norm=None, scaler=None, stateful=(),
                 b8_impl="reference", telemetry_interval=0):
        if mode not in self.MODES:
            raise ValueError(f"Unknown solver: {mode}")
        if type(optimizer) not in (torch.optim.SGD, torch.optim.AdamW):
            raise TypeError("Only SGD and AdamW have validated transactional semantics")
        if nonfinite not in ("raise", "skip"):
            raise ValueError("nonfinite must be raise or skip")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        validate_optimizer_ownership(model, optimizer)
        self.model, self.optimizer, self.mode = model, optimizer, mode
        self.nonfinite, self.max_grad_norm = nonfinite, max_grad_norm
        self.scaler, self.stateful = scaler, tuple(stateful)
        if b8_impl not in {"reference", "head_grad_only", "graph_reuse"}:
            raise ValueError("Unknown B8 implementation")
        self.b8_impl, self.telemetry_interval = b8_impl, int(telemetry_interval)
        self.last_trace = None
        rows = parameter_roles(model, head_names=head_names)
        self.parameters = [row.parameter for row in rows]
        self.names = [row.name for row in rows]
        self.roles = [row.role for row in rows]
        self.head_ids = {id(row.parameter) for row in rows if row.role == "adversarial_head"}
        if mode == "head_lookahead" and not self.head_ids:
            raise ValueError("head_lookahead requires an identified adversarial head")
        self.previous = None
        self.pseudo_signature = None
        self.pseudo_accumulator = None
        self.reset_reasons = []
        self.steps = 0

    def reset_history(self, reason="explicit"):
        self.previous = None
        self.reset_reasons.append({"step": self.steps, "reason": str(reason)})

    def state_dict(self):
        return copy.deepcopy({"version": 1, "mode": self.mode, "names": self.names,
                              "previous": self.previous, "steps": self.steps, "b8_impl": self.b8_impl,
                              "pseudo_signature": self.pseudo_signature,
                              "pseudo_accumulator": self.pseudo_accumulator,
                              "reset_reasons": self.reset_reasons})

    def load_state_dict(self, state):
        if state.get("b8_impl", "reference") != self.b8_impl:
            raise ValueError("Solver B8 implementation differs from checkpoint")
        if state["version"] != 1 or state["mode"] != self.mode or state["names"] != self.names:
            raise ValueError("Solver checkpoint does not match mode/parameter layout")
        previous = state["previous"]
        if previous is not None:
            if len(previous) != len(self.parameters):
                raise ValueError("Optimistic history has incorrect length")
            previous = [None if g is None else g.to(device=p.device, dtype=p.dtype).clone()
                        for p, g in zip(self.parameters, previous)]
            if any(g is not None and g.shape != p.shape for p, g in zip(self.parameters, previous)):
                raise ValueError("Optimistic history has incorrect shapes")
        self.previous = previous
        self.pseudo_signature=copy.deepcopy(state.get('pseudo_signature'))
        self.pseudo_accumulator=copy.deepcopy(state.get('pseudo_accumulator'))
        self.steps, self.reset_reasons = int(state["steps"]), copy.deepcopy(state["reset_reasons"])

    def _evaluate(self, closure, stage, *, head_only=False, retain_graph=False):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = closure()
            if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not loss.requires_grad:
                raise ValueError("Closure must return one differentiable scalar loss")
            if not torch.isfinite(loss).all():
                raise NonFiniteStep(stage + ":loss")
            active = [p for p in self.parameters if p.requires_grad and (not head_only or id(p) in self.head_ids)]
            scaled = self.scaler.scale(loss) if self.scaler is not None else loss
            if self._measure_components and hasattr(closure, 'component_gradients'):
                self._components[stage] = closure.component_gradients(self.parameters)
            gradients = torch.autograd.grad(scaled, active, allow_unused=True, retain_graph=retain_graph) if active else ()
            if self.scaler is not None and self.scaler.is_enabled():
                scale = self.scaler.get_scale()
                gradients = tuple(None if g is None else g / scale for g in gradients)
        by_id = {id(p): g for p, g in zip(active, gradients)}
        gradients = [by_id.get(id(p)) for p in self.parameters]
        if any(g is not None and not torch.isfinite(g).all() for g in gradients):
            raise NonFiniteStep(stage + ":gradient")
        return float(loss.detach()), [None if g is None else g.detach().clone() for g in gradients]

    def _install(self, gradients, *, head_only=False, exclude_head=False):
        self.optimizer.zero_grad(set_to_none=True)
        for p, g in zip(self.parameters, gradients):
            use = (not head_only or id(p) in self.head_ids) and (not exclude_head or id(p) not in self.head_ids)
            p.grad = g.clone() if g is not None and use else None
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm, error_if_nonfinite=True)

    def _finite_state(self, stage):
        def finite(value):
            if isinstance(value, torch.Tensor):
                return bool(torch.isfinite(value).all())
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            if isinstance(value, (tuple, list)):
                return all(finite(v) for v in value)
            return True
        if not finite(list(self.parameters)) or not finite(self.optimizer.state) or not finite(clone_buffers(self.model)):
            raise NonFiniteStep(stage + ":state")

    @property
    def algorithm(self):
        prefix = "adamw_isolated_predictor_raw_gradient" if isinstance(self.optimizer, torch.optim.AdamW) else "sgd"
        if prefix == "sgd" and any(g.get("momentum", 0) for g in self.optimizer.param_groups):
            prefix = "sgd_momentum_isolated_predictor"
        return prefix + "_" + self.mode + ("_" + self.b8_impl if self.mode == "head_lookahead" else "")

    def _advance_scaler(self, accepted):
        # Public state serialization avoids GradScaler's optimizer-specific
        # unscale_/step stage machine being advanced by virtual evaluations.
        if self.scaler is None or not self.scaler.is_enabled():
            return
        state = self.scaler.state_dict()
        if accepted:
            tracker = int(state["_growth_tracker"]) + 1
            if tracker >= state["growth_interval"]:
                state["scale"] *= state["growth_factor"]
                tracker = 0
            state["_growth_tracker"] = tracker
        else:
            state["scale"] *= state["backoff_factor"]
            state["_growth_tracker"] = 0
        self.scaler.load_state_dict(state)

    def step(self, closure, *, exclude_head=False, predictor_grad_scope=None):
        """Commit one coupled update. Alternating head catch-up is caller-owned.

        ``exclude_head=True`` prevents the main step from updating heads already
        updated by caller-owned alternating/catch-up. Frozen head forwards must
        still track input derivatives; do not wrap them in ``no_grad``.
        """
        if exclude_head and self.mode in {"extragradient", "heun", "head_lookahead"}:
            raise ValueError("Coupled predictor modes must include all active head parameters")
        graph = self.mode == "head_lookahead" and self.b8_impl == "graph_reuse"
        autocast_active = torch.is_autocast_enabled()
        if graph and self.parameters and self.parameters[0].device.type == 'cpu':
            try:
                autocast_active = autocast_active or torch.is_autocast_enabled('cpu')
            except TypeError:  # PyTorch 2.1 lacks the device argument.
                autocast_active = autocast_active or torch.is_autocast_cpu_enabled()
        if graph and ((self.scaler is not None and self.scaler.is_enabled()) or autocast_active):
            raise ValueError("graph_reuse is validated for FP32 only; AMP is not supported")
        if graph and not hasattr(closure, "corrector"):
            raise NotImplementedError("graph_reuse requires a validated reusable objective")
        if predictor_grad_scope not in (None, "all", "head_only"):
            raise ValueError("Invalid predictor_grad_scope")
        head_only = self.mode == "head_lookahead" and (self.b8_impl != "reference" if predictor_grad_scope is None else predictor_grad_scope == "head_only")
        measure = self.telemetry_interval > 0 and self.steps % self.telemetry_interval == 0
        self._measure_components = measure
        self._components = {}
        snapshot = TrainingState(self.model, self.optimizer, scaler=self.scaler, stateful=self.stateful)
        telemetry = {"roles": {}, "origin_scope": "head_only" if head_only else "all", "b8_impl": self.b8_impl} if measure else None
        predictor_clipped = predictor_updates = None
        evaluations, loss0, loss1, norm = 0, float("nan"), None, float("nan")
        try:
            evaluations += 1
            loss0, first = self._evaluate(closure, "origin", head_only=head_only, retain_graph=graph)
            self._finite_state("origin")
            origin_norm = sum(float(g.double().square().sum()) for g in first if g is not None) ** .5
            buffers, rng_after_first = clone_buffers(self.model), RNGState.capture()
            gradients = first
            if graph:
                loss1, gradients, predictor_updates = closure.corrector(first, self)
                if measure: self._components['corrector'] = closure.predictor_components
                if measure: predictor_clipped = closure.predictor_clipped
                if not torch.isfinite(torch.tensor(loss1)):
                    raise NonFiniteStep("corrector:loss")
                evaluations += 1
            elif self.mode in {"extragradient", "heun", "head_lookahead"}:
                self._install(first, head_only=self.mode == "head_lookahead", exclude_head=exclude_head)
                predictor_clipped = [None if p.grad is None else p.grad.detach().clone() for p in self.parameters] if measure else None
                self.optimizer.step()
                predictor_updates = [p.detach()-v for p,v,_ in snapshot.parameters] if measure else None
                self._finite_state("predictor")
                restore_buffers(self.model, snapshot.buffers)
                snapshot.rng.restore()
                evaluations += 1
                loss1, second = self._evaluate(closure, "corrector")
                if self.mode == "heun":
                    gradients = [None if a is None and b is None else
                                 ((torch.zeros_like(p) if a is None else a) +
                                  (torch.zeros_like(p) if b is None else b)) * .5
                                 for p, a, b in zip(self.parameters, first, second)]
                else:
                    gradients = second
                snapshot.restore()
            elif self.mode == "optimistic" and self.previous is not None:
                gradients = [None if a is None else 2 * a - (torch.zeros_like(a) if b is None else b)
                             for a, b in zip(first, self.previous)]
            if any(g is not None and not torch.isfinite(g).all() for g in gradients):
                raise NonFiniteStep("combined:gradient")
            norm = sum(float(g.double().square().sum()) for g in gradients if g is not None) ** .5
            self._install(gradients, exclude_head=exclude_head)
            clipped = [None if p.grad is None else p.grad.detach().clone() for p in self.parameters] if measure else None
            self.optimizer.step()
            self._finite_state("formal")
            restore_buffers(self.model, buffers)
            rng_after_first.restore()
            self._advance_scaler(True)
            self.previous = [None if g is None else g.clone() for g in first] if self.mode == "optimistic" else None
            self.steps += 1
            if measure:
                original = {id(p):v for p,v,_ in snapshot.parameters}
                for role in set(self.roles):
                    indices = [i for i,r in enumerate(self.roles) if r == role]
                    def norm_of(values):
                        return sum(float(values[i].double().square().sum()) for i in indices if values[i] is not None)**.5
                    telemetry["roles"][role] = {"raw_grad_norm": norm_of(gradients), "clipped_grad_norm": norm_of(clipped),
                        "update_norm": norm_of([p.detach()-original[id(p)] for p in self.parameters]),
                        "origin_raw_grad_norm": norm_of(first),
                        "predictor_clipped_grad_norm": norm_of(predictor_clipped) if predictor_clipped is not None else None,
                        "virtual_update_norm": norm_of(predictor_updates) if predictor_updates is not None else None}
                total_raw = sum(float(g.double().square().sum()) for g in gradients if g is not None)**.5
                total_clipped = sum(float(g.double().square().sum()) for g in clipped if g is not None)**.5
                telemetry['formal_clip_ratio'] = total_clipped/total_raw if total_raw else 1.
                self.last_trace = {'origin': first, 'predictor_clipped': predictor_clipped,
                                   'corrector': gradients, 'formal_clipped': clipped,
                                   'virtual_update': predictor_updates,
                                   'formal_update': [p.detach()-original[id(p)] for p in self.parameters]}
                telemetry.update(field_evaluations=evaluations, backward_calls=evaluations,
                                 gradient_component_diagnostics="UNAVAILABLE: requires separately scoped objective gradients")
                if self._components.get('origin') is not None and self._components.get('corrector') is not None:
                    left,right=self._components['origin'],self._components['corrector']
                    indices=[i for i,p in enumerate(self.parameters) if id(p) not in self.head_ids]
                    def squared(rows): return sum(float(rows[i].double().square().sum()) for i in indices if rows[i] is not None)
                    na,nb=squared(left['adv'])**.5,squared(right['adv'])**.5
                    dot=sum(float((left['adv'][i].double()*right['adv'][i].double()).sum()) for i in indices if left['adv'][i] is not None and right['adv'][i] is not None)
                    diff=[None if a is None and b is None else (torch.zeros_like(p) if a is None else a)-(torch.zeros_like(p) if b is None else b) for p,a,b in zip(self.parameters,left['nonadv'],right['nonadv'])]
                    telemetry.update(online_predictor_adv_cosine=dot/(na*nb) if na*nb else None,
                                     nonadv_gradient_change_norm=squared(diff)**.5,
                                     component_scope='all_nonhead_parameters_full_current_objective',
                                     gradient_component_diagnostics='AVAILABLE',
                                     telemetry_backward_calls=left['backward_calls']+right['backward_calls'])
            return StepResult(loss0, loss1, norm, self.mode, True, evaluations, self.algorithm,
                              origin_grad_norm=origin_norm, telemetry=telemetry)
        except Exception as exc:
            snapshot.restore()
            if isinstance(exc, NonFiniteStep):
                self._advance_scaler(False)
            if isinstance(exc, NonFiniteStep) and self.nonfinite == "skip":
                return StepResult(loss0, loss1, norm, self.mode, False, evaluations, self.algorithm, str(exc))
            raise
