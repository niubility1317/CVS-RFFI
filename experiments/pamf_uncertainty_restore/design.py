RUN = '20261019-phase1-restoration-uncertainty-selfsup-s2026092701-m3-r10'
COARSE_RUN = '20261019-phase1-reception-restoration-selfsup-manysig-m6-r02'
COARSE_COMMIT = '7efdba92932a86bf4b0492a565d35377564a26da'
ACTION_RUN = '20261019-phase1-restoration-learned-action-source-s2026092701-r07'
ACTION_COMMIT = 'c8ccdfdbd03ab5ad8828b1eb9210b99a664ebe7b'
RESIDUAL_RUN = '20261019-phase1-restoration-residual-refinement-selfsup-m3-r03'
NAMES = ('direct_control', 'native_map_prior', 'uncertain_map_prior')
GPUS = (0, 1, 5, 7)
EPOCHS = 20
BATCH_SIZE = 256
MODEL_SEED = 2026092701
DATA_SEED = 2026101921
CALIBRATION_SEED = 2026101920


def configuration():
    return dict(schema='uncertainty_restoration_v1', run=RUN, candidates=list(NAMES),
        model_seed=MODEL_SEED, data_seed=DATA_SEED, calibration_seed=CALIBRATION_SEED,
        parent_coarse_run=COARSE_RUN, parent_coarse_commit=COARSE_COMMIT,
        parent_action_run=ACTION_RUN, parent_action_commit=ACTION_COMMIT,
        source_reference_run=RESIDUAL_RUN,
        train_count=63000, L_count=6300, U_count=56700, epochs=EPOCHS, updates=4940, batch_size=BATCH_SIZE,
        L_exposures=126000, U_exposures=1134000, calibration_exposures=8192,
        training_TX_labels_read=0, training_F_access=False, V_persistent_updates=0,
        training_routes=dict(clean=.15, residual=.85, full=0.),
        cache='Reuse complete R07 twenty fresh source epochs;no regeneration/no newphysicalrecords',
        native_augmentation_seed=2026101912, native_receiver_seed=2026101911,
        optimizer='AdamW', lr=2e-4, weight_decay=1e-4, warmup_updates=247, final_lr=2e-5,
        precision='FP32', gradient_clip=1., EMA=False, checkpoint_choice='Fixed lastE20,notV-selected',
        reconstruction='IQNMSE+.1 FFT abs log(power+1e-3),samepriorCore andR02 source scale',
        uncertainty='Native variance + global_source_excess*exp(4*tanh(current_observation_head)) pertime',
        uncertainty_inputs='Current received,coarse,estimated action(coarse);no clean/TX/F/scene/runtimeoracle',
        variance_objective='Proper residual GaussianNLL;weight.01;reconstruction variance detached,NLL prior detached',
        calibration='8192 L/U firstepoch;max(mean(real residual squared)-mean(nativevariance),1e-6);1fittedscalar',
        prior_RMS=.1169459182403555, CG_steps=6,
        inherited_cost=dict(direct_control='coarseE20+newpriorE20=40',
            native_map_prior='coarseE20+actionE20+newpriorE20=60',
            uncertain_map_prior='coarseE20+actionE20+newpriorE20=60'),
        source_thresholds=dict(mean_NMSE_ratio_vs_direct_max=.98, mean_NMSE_ratio_vs_R03_max=1.,
            clean_NMSE_max=.02, lowurban_accuracy_gain_vs_R03_min_pp=.5,
            clean_accuracy_drop_vs_R03_max_pp=.2, lowurban_worst_TX_delta_min_pp=-1.),
        qualification='Onlyuncertainprimary;waveformA noF/TX;separateidentityB useslabeledsourceV after7predictions',
        source_views='Original1800V clean+sixpractical;allclass/RX/day groups',
        target='Only if allsource thresholdsPASS:uncertain +matched direct,originalcomplete168000IQ x7views,originalF50/defaulthead,independenttruth-last',
        output='OneIQ;allPAMF FFT/PA/time features useoriginalgraph;nohardprojection/clipping/sceneoracle',
        GPUs=list(GPUS), max_simultaneous_GPU_workers=3, max_own_worker_per_GPU=1,
        capacity_lock='receiver_residual_capacity4.lock', all_owner_cap=2, min_free_MB=12000, CPU_threads=2,
        stop_rule='Nonfinite/illegal source,parent,query,collision,missinglegalprediction:stoprow,holdpending,preservehealthyactiverows,noautomaticretry',
        target_feedback=False)
