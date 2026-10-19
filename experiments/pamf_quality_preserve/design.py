RUN = '20261019-phase1-restoration-quality-preserve-source-s2026092701-r01'
MODEL_SEED = 2026092701
DATA_SEED = 2026101901
EPOCHS = 20
BATCH_SIZE = 256
UPDATES = 4940
GPU = 1


def configuration():
    return dict(schema='synthetic_quality_preservation_v1', run_id=RUN,
        model_seed=MODEL_SEED, data_seed=DATA_SEED, epochs=EPOCHS, updates=UPDATES,
        batch_size_physical=BATCH_SIZE, batch_size_quality_examples=2*BATCH_SIZE,
        source_count=63000, L_count=6300, U_count=56700,
        objective='Balanced BCE: original source IQ=0, existing synthetic damaged IQ=1',
        supervision='Synthetic augmentation indicator only; no TX labels or F signals',
        observation_mix=dict(clean=.5, residual=.375, full=.125),
        damaged_route='Independent seeded 75% residual/25% full;no channel regeneration',
        cached_view_policy='R02 quantile>=.15 slots only;otherwise first cyclic valid cached epoch for same physical sample',
        route_seed='NumPy SeedSequence([data_seed,epoch,7525]);independent of cached quantile',
        optimizer='AdamW', learning_rate=3e-4, weight_decay=1e-4,
        warmup_updates=247, final_learning_rate=3e-5, gradient_clip=1., precision='FP32',
        threshold_logit=0., threshold_choice='Fixed before training;no V fitting or sweep',
        inference='Current received IQ only determines keep/restore;frozenR03 output or exact received IQ',
        normalization='Gate-only per-packet RMS;preserved/output IQ remains original coordinates',
        frozen_restorer='R03 residual_deterministic over frozenR02,E40 cumulative',
        frozen_reader='Original CROSS_RX-s2026092701 F50 with original default head',
        checkpoint_choice='Fixed last E20;no early stopping or target/source-V selection',
        training_TX_labels_read=0, training_F_access=False, target_reads=0,
        source_rule='All clean source decisions match identity and clean IQ NMSE0;each practical accuracy/F1/NMSE no worse than R03',
        target_scope='None;this run is source-only and does not start practical target tests',
        GPU=GPU, max_this_run_GPU_workers=1, all_owner_cap_per_GPU=2,
        capacity_lock='receiver_residual_capacity4.lock', CPU_threads=2,
        stop_rule='Illegal source/parent,outputcollision,nonfinite,budget/predictionfailure:preserve,no automaticretry')
