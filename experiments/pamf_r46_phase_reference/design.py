from experiments.pamf_r46_source_phase.design import *
from experiments.pamf_r46_source_phase import design as original

RUN='20261021-diagnostic-restoration-R46-phase-reference-s2026092701-r52'


def configuration():
    value=original.configuration()
    value.update(schema='R46_phase_reference_diagnostic_v1',run=RUN,
        metrics=['equivariance_nmse','paired_clean_nmse','unrotated_clean_nmse','invariance_nmse','fixed_clean_nmse'],
        reference_definitions=dict(transported='R^-1G(Ry) versus clean x',fixed='G(Ry) versus original clean x'),
        hypothesis_rule='Compare all four source errors;small invariance error suggests phase compensation,small equivariance error suggests coordinate transport;no training/selection threshold',
        previous_diagnostic=original.RUN,previous_metrics_preserved=True,
        claim='Source phase reference control;no inference that C4 equivariance is required for clean recovery or TX authenticity')
    return value
