RUN = '20261019-phase1-restoration-analytic-views-selfsup-s2026092701-m2-r12'
PARENT = '20261019-phase1-restoration-uncertainty-selfsup-s2026092701-m3-r10'
PARENT_COMMIT = 'b57eca9946b2a912a88870e24826fe3e2f70985d'
ACTION = '20261019-phase1-restoration-learned-action-source-s2026092701-r07'
ROWS = ('pair_control', 'analytic_views')
KIND = 'native_map_prior'
GPUS = (0, 1, 5, 7)
MODEL_SEED = 2026092701
DATA_SEED = 2026101923
CALIBRATION_SEED = 2026101924


def configuration():
    return dict(schema='analytic_restore_v1', run=RUN, rows=list(ROWS), parent_run=PARENT,
        parent_commit=PARENT_COMMIT, parent_candidate=KIND, source_contract='original7/63/30',
        model_seed=MODEL_SEED, data_seed=DATA_SEED, calibration_seed=CALIBRATION_SEED,
        train_count=63000, L_count=6300, U_count=56700, epochs=20, updates=4940, batch_size=256,
        L_exposures=126000, U_exposures=1134000, calibration_count=8192, fitted_RMS_scalars=63,
        training_TX_labels_read=0, training_F_access=False, V_updates=0, target_training=False,
        optimizer='ResetAdamW', peak_lr=3e-5, final_lr=3e-6, warmup_updates=247, weight_decay=1e-4,
        gradient_clip=1., precision='FP32', parent_updates=4940, prior_cumulative_epochs=40,
        component_cumulative_epochs=80, old_parent_component_epochs=60,
        loss='IQNMSE+.1FFTlogpowerL1;analytic row adds1.0*mean_four_sourceRMS_Huber_groups',
        analytic_groups=['Native32band4channelmirrorFFT', 'Native4delay3oddordercomplexPA',
            'Native7spectral/circularitystatistics', 'Public28coordinateNoCFOreference_response'],
        normalization='RMSperchannel on8192sourceLUclean,63scalars,floor.01;noV/F/TX/target',
        training_routes=dict(clean=.15, residual=.85, full=0.),
        cache='Repeat original R07 complete20cachedobservations,not newnoise/physicalrecords',
        checkpoint_choice='Fixedlast20,notVselected', trainable='Onlyexisting prior;coarse/action frozen',
        source_thresholds=dict(mean6_NMSE_vs_control_max_ratio=1.05, mean6_NMSE_vs_parent_max_ratio=1.05, clean_NMSE_max=.02,
            lowurban_gain_vs_control_min_pp=.5, lowurban_gain_vs_parent_min_pp=.5,
            midurban_gain_vs_control_min_pp=.5, midurban_gain_vs_parent_min_pp=.5,
            clean_drop_vs_original_max_pp=.1, lowurban_worst_TX_delta_vs_parent_min_pp=-1.,
            worst_other_scene_delta_vs_parent_min_pp=-.5),
        target='Onlyanalyticprimaryifallsourcepass;thenpair_control+analytic_views original168000x7 originalF50 truthlast',
        user_goal=dict(lowurban_accuracy_min=70., other_five_max_gap_vs_original_clean_pp=5.),
        GPU_pool=list(GPUS), max_own_GPU_workers=2, capacity_lock='receiver_residual_capacity4.lock',
        all_owner_cap=2, min_free_MB=12000, CPU_threads=2, target_feedback=False,
        stop_rule='Illegal source/parent/prediction,collision,nonfinite:stopfailedrow,holdpending,preservehealthyactiverows,noautomaticretry')
