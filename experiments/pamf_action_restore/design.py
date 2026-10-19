RUN = '20261019-phase1-restoration-learned-action-source-s2026092701-r07'
MODEL_SEED = 2026092701
DATA_SEED = 2026101901
AUGMENTATION_SEED = 2026101912
RECEIVER_SEED = 2026101911
EPOCHS = 20
BATCH_SIZE = 256


def configuration():
    return dict(schema='learned_effective_action_v1', run=RUN, candidate='learned_operator',
        model_seed=MODEL_SEED, data_seed=DATA_SEED, augmentation_seed=AUGMENTATION_SEED,
        receiver_seed=RECEIVER_SEED, epochs=EPOCHS, updates=4940, batch_size=BATCH_SIZE,
        train_count=63000, L_count=6300, U_count=56700, L_exposures=126000, U_exposures=1134000,
        training_TX_labels_read=0, training_F_access=False, target_reads=0,
        training_routes=dict(clean=.15, residual=.85, full=0.),
        observation='native residual/post_sync/noEQ/fs25MHz/fc2462MHz/reflect/referenceSNR30',
        fresh_training_namespace='learned-action/train/epochNN',
        teacher='Source-only actual native action/AGC/lock from two clean-independent probes; never deployment inputs',
        objective='Mean normalized action error on two fresh complex white probes + .1 log(real noise variance) MSE',
        paired_restoration_loss=False, F_loss=False, TX_loss=False,
        action='64 complex FIR taps, per-time main/image modulation, scalar real noise variance',
        inference='Current received+frozen coarse only;6 real-inner-product CG iterations for soft Gaussian MAP',
        prior_RMS=.1169459182403555, prior_origin='R03 once-only8192 L/U RMS,not child weights',
        noise_variance_range=[1e-8, 1.], output='One IQ;FFT/PA/all original F branches derive it',
        optimizer='AdamW', learning_rate=2e-4, weight_decay=1e-4, warmup_updates=247,
        final_learning_rate=2e-5, gradient_clip=1., precision='FP32', EMA=False,
        frozen_parent='R02 source-waveform-A tf_unet_mixed E20/4940,0 TX training',
        checkpoint_choice='Fixed last E20,not V/target selection', parent_epochs=20, cumulative_epochs=40,
        source_evaluation='Original fixed1800V x7views;coarse+identity controls;originalF50/defaulthead,after freeze',
        waveform_threshold=dict(mean_six_scene_NMSE_gain_vs_coarse_min=.05, clean_NMSE_max=.02),
        identity_diagnostic='Accuracy/Macro-F1/repair/damage by clean/suburban/urban/TX/RX/day;not waveform qualification',
        target_scope='None in this source-mechanism run;no target-feedback ranking or retry',
        GPU=5, workers=16, CPU_threads=2, max_this_run_GPU_workers=1,
        capacity_lock='receiver_residual_capacity4.lock',
        stop_rule='Illegal parent/role,collision,nonfinite,invalid teacher replay or prediction coverage:preserve outputs,abort,no automaticretry')
