"""Install reference-route masking in the existing PTBR source trajectory."""
from contextlib import contextmanager

from . import objective, training_scope
from .reference_routing import route_stable_physical_directions


class RouteStablePL(objective.PhysicalPL):
    def loss(self, student, device):
        self.routing_student = student
        return super().loss(student, device)

    def state_dict(self):
        state = super().state_dict()
        student = self.routing_student
        calls = getattr(student, '_reference_routing_calls', 0)
        seen = getattr(student, '_reference_routing_seen', 0)
        changed = getattr(student, '_reference_routing_changed', 0)
        if calls != 2 * self.calls or not 0 <= changed <= seen:
            raise RuntimeError('Each L/U physical probe must record reference routing')
        state['reference_routing'] = dict(calls=calls, selected_samples=seen,
                                          changed_samples=changed, target_reads=0, U_truth_reads=0)
        return state


@contextmanager
def installed(c, holder):
    old_measure, old_class = objective.physical_directions, training_scope.PhysicalPL

    def measure(student, inputs, seed):
        return route_stable_physical_directions(student, inputs, seed, old_measure)

    objective.physical_directions, training_scope.PhysicalPL = measure, RouteStablePL
    try:
        with training_scope.installed(c, holder):
            yield
    finally:
        objective.physical_directions, training_scope.PhysicalPL = old_measure, old_class
