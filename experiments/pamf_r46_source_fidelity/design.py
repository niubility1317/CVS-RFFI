from experiments.pamf_r46_cuda_graph import design as frozen

RUN = '20261021-diagnostic-restoration-R46-source-fidelity-s2026092701-m3-r21'
ARMS = frozen.ARMS
VIEWS = ('clean', 'received01', 'received20')
PER_ROLE = 128
BATCH = 32
LAGS = (1, 2, 4, 8, 16)


def configuration():
    return dict(schema='R46_source_fidelity_v1', run=RUN, arms=list(ARMS),
        model_sources=frozen.SOURCE_ROWS, parameters=frozen.PARAMETERS,
        source_budgets={a:list(v) for a,v in frozen.SOURCE_BUDGET.items()},
        views=list(VIEWS), L_s=PER_ROLE, U_s=PER_ROLE, batch=BATCH,
        sampling='Evenly spaced opaque positions within each original source-training role; no TX labels',
        metrics='Relative paired IQ/prefix232/tail24; complex FFT power quartiles; normalized lag correlation; amplitude-conditioned residual',
        lags=list(LAGS), device='cpu', CPU_threads=2, GPU_workers=0,
        new_training_updates=0, new_network_versions=0, F_weight_reads=0,
        F_output_reads=0, TX_label_reads=0, V_reads=0, target_reads=0,
        model_selection=False, teacher_selection=False, target_feedback=False,
        claim='Source fidelity diagnostic, not TX identity proof or a new blind test',
        exposure='R18/R19/R20 formal outcomes already known; all three models fixed without target ranking',
        next_training_parent='Original human-designated R46 prune_refine350521 only',
        future_versions='Every newly trained version completes all seven formal scenes',
        stop_rule='Preserve technical failure; no retry, model update, fallback, or other-owner mutation')
