"""One mathematical objective; only its attachment position changes."""
from experiments.cvs_urban_identity50.objective import direction,labeled_reliability,EPS
from .branches import BRANCHES


def losses(student,teacher,reliability):
    return {key:direction(student[key],teacher[key],reliability) for key in (*BRANCHES,'J')}


def mixture(values,coefficients,weights):
    if set(weights)-set(BRANCHES) or not weights or abs(sum(weights.values())-1.)>1e-9 or any(w<0 for w in weights.values()):
        raise ValueError('Registered nonnegative branch weights summing to one required')
    if any(k not in coefficients or not 1e-4<=coefficients[k]<=10. for k in weights):
        raise ValueError('Frozen bounded source branch coefficients required')
    return sum(weights[k]*coefficients[k]*values[k] for k in weights)
