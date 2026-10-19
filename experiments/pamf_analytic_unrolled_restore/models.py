from . import design as d
from experiments.pamf_unrolled_restore.models import UnrolledRestorer


class AnalyticRestorer(UnrolledRestorer):
    def __init__(self,candidate,frozen):
        if candidate not in d.ROWS:raise ValueError('Registered matched analytic row required')
        super().__init__('unroll_unshared',frozen)
        self.candidate=candidate
