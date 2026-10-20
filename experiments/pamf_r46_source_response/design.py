from experiments.pamf_r46_cuda_graph import design as frozen
from experiments.pamf_r46_complex_fidelity import design as complex_run

RUN = '20261021-diagnostic-restoration-R46-source-response-s2026092701-m4-r25'
ARMS = ('R46_350521', 'R18_factor625_stage', 'R22_complex625', 'R22_complex625_identity')
REFERENCE_ARM = ARMS[0]
PARAMETERS = dict(zip(ARMS, (350521, 278180, 278180, 278180)))
SOURCE_ROWS = dict(zip(ARMS, ('prune_refine', 'factor625_stage', 'complex625', 'complex625_identity')))
SOURCE_BUDGET = dict(zip(ARMS, ((20, 4940), (20, 19700), (20, 19700), (20, 19700))))
COMPACT_PARENT_RUN, COMPACT_PARENT_COMMIT = frozen.COMPACT_PARENT_RUN, frozen.COMPACT_PARENT_COMMIT
FACTOR_RUN, FACTOR_COMMIT = frozen.FACTOR_RUN, frozen.FACTOR_COMMIT
COMPLEX_RUN, COMPLEX_COMMIT = complex_run.RUN, 'e190cf5367b03da8aafced51276a8730ad8274fa'
ACTION_RUN, KIND = frozen.ACTION_RUN, frozen.KIND
VIEWS = ('clean', 'received01', 'received20')
PROBES = ('random', 'phase')
EPSILONS = (.005, .01)
ROLE_GROUPS = ('all', 'L_s', 'U_s')
PER_ROLE, BATCH, SEED = 128, 32, 2026102125
FLOOR_FRACTION, ABSOLUTE_FLOOR = .01, 1e-8


def configuration():
    return dict(schema='R46_source_response_v1', run=RUN, arms=list(ARMS), reference_arm=REFERENCE_ARM,
        parameters=PARAMETERS, source_rows=SOURCE_ROWS,
        source_budgets={arm:list(v) for arm,v in SOURCE_BUDGET.items()},
        parent_run=COMPACT_PARENT_RUN, parent_commit=COMPACT_PARENT_COMMIT,
        factor_run=FACTOR_RUN, factor_commit=FACTOR_COMMIT,
        complex_run=COMPLEX_RUN, complex_commit=COMPLEX_COMMIT,
        views=list(VIEWS), probes=list(PROBES), epsilons=list(EPSILONS), role_groups=list(ROLE_GROUPS),
        sampling='128 evenly spaced opaque positions per original L_s/U_s; same panel as R21',
        unique_source_positions=2*PER_ROLE, source_centers_per_model=2*PER_ROLE*len(VIEWS),
        diagnostic_IQ_inputs_per_model=2*PER_ROLE*len(VIEWS)*(1+2*len(PROBES)*len(EPSILONS)),
        smoke_IQ_inputs_per_model=2, batch=BATCH, seed=SEED,
        perturbation='Common per-batch random direction and i*x phase tangent; complex RMS delta/input=epsilon',
        response='Odd half difference of frozen G(x+delta) and G(x-delta); even half sum minus G(x)',
        response_denominator='max(reference odd power, .01 input perturbation power, 1e-8)',
        floor_fraction=FLOOR_FRACTION, absolute_floor=ABSOLUTE_FLOOR,
        scaling_denominator='max(low-step scaled response power,.01 input power,1e-8)',
        summary_statistics=['mean','median','p90','p95','maximum'],
        expected_metric_rows=(len(ARMS)*len(VIEWS)*len(ROLE_GROUPS)*2
            + len(ARMS)*len(VIEWS)*len(ROLE_GROUPS)*len(PROBES)*len(EPSILONS)*7
            + len(ARMS)*len(VIEWS)*len(ROLE_GROUPS)*len(PROBES)*2),
        device='cpu', CPU_threads=2, GPU_workers=0, diagnostic_workers=1,
        new_training_updates=0, new_network_versions=0, F_weight_reads=0, F_output_reads=0,
        TX_label_reads=0, V_reads=0, target_reads=0, head_fit_updates=0,
        model_selection=False, teacher_selection=False, target_feedback=False,
        claim='Finite-perturbation source response comparison;not exact Jacobian or TX authenticity proof',
        exposure='R18/R22 formal outcomes known;fixed original reference plus same-rank output/complex/identity contrasts',
        future_training_parent='Human original R46 source-final350521 only;no newly tested student inheritance',
        future_versions='Every newly trained restoration version still completes all seven formal scenes',
        stop_rule='Preserve technical failure;no retry,update,fallback,or other-owner process mutation')
