from . import design as d
from experiments.pamf_gradient_balance.gradients import output_gradient as capped_gradient


def output_gradient(estimate,paired,identity,arm):
    d.configuration(arm)
    return capped_gradient(estimate,paired,identity,'per_frame_cap')
