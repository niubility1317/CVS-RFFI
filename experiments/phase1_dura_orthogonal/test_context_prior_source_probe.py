import pytest
import torch
from experiments.phase1_dura_orthogonal.context_prior_source_probe import mixed_outputs, summarize


class Dummy:
    def __init__(self, context, prior):
        self.context, self.prior = context, prior
    def estimate_context(self, x):
        return self.context
    def stage_outputs(self, x, context):
        return ([x + context + self.prior],)


def test_swap_changes_context_and_prior_independently():
    x = torch.zeros(2, 2, 256)
    result = mixed_outputs(Dummy(1, 10), Dummy(2, 20), x)
    assert [float(t[0, 0, 0]) for t in result] == [11., 21., 12., 22.]


def test_summary_reports_paired_difference():
    val = torch.tensor([[1., 2., .5, 1.5], [2., 1., 1., 3.]])
    result = summarize(val)
    assert result['mean_nmse'] == pytest.approx([1.5, 1.5, .75, 2.25])
    assert result['improved_vs_original'] == [0, 1, 2, 0]
