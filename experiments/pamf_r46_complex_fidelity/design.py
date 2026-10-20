from experiments.pamf_r46_conditional_factor import design as baseline
from experiments.pamf_r44_pruned_refine import design as parent
from experiments.pamf_r46_prior_preserve import design as previous

RUN = '20261021-phase1-restoration-R46-complex-fidelity-s2026092701-m2-r22'
ARMS = ('complex625', 'complex625_identity')
PARAMETERS = {a:278180 for a in ARMS}
CONTEXT_PARAMETERS = {a:186703 for a in ARMS}
TASK = {a:False for a in ARMS}
MODE = {a:'R46_allcore_SVD625_preserve_channels' for a in ARMS}
GPUS = dict(zip(ARMS, (2, 3)))
COMPLEX_WEIGHT = .02
CLEAN_WEIGHT = dict(zip(ARMS, (0., .25)))
COMPACT_PARENT_RUN, COMPACT_PARENT_COMMIT, COMPACT_PARENT_ROW = baseline.COMPACT_PARENT_RUN, baseline.COMPACT_PARENT_COMMIT, baseline.COMPACT_PARENT_ROW
PARENT_RUN, PARENT_COMMIT, ACTION_RUN = baseline.PARENT_RUN, baseline.PARENT_COMMIT, baseline.ACTION_RUN
F_RUN, F_SEED, VIEWS, KIND = baseline.F_RUN, baseline.F_SEED, baseline.VIEWS, baseline.KIND
MODEL_SEED, DATA_SEED = baseline.MODEL_SEED, baseline.DATA_SEED
EPOCHS, UPDATES, BATCH, MICRO = 20, 19700, 64, 64
SCORING_RUN = baseline.SCORING_RUN
PREDECESSOR = previous.RUN
PREDECESSOR_BY_GPU = {g:PREDECESSOR for g in GPUS.values()}
PREDECESSOR_MODULE = {PREDECESSOR:'experiments.pamf_r46_prior_preserve.runner'}
teacher_weight, learning_rate = baseline.teacher_weight, baseline.learning_rate


def configuration():
    return dict(schema='R46_complex_fidelity_v1', run=RUN, arms=list(ARMS), parameters=PARAMETERS,
        context_parameters=CONTEXT_PARAMETERS, compact_parent_run=COMPACT_PARENT_RUN,
        compact_parent_commit=COMPACT_PARENT_COMMIT, compact_parent_row=COMPACT_PARENT_ROW,
        architecture='Same full SVD625 coarse/action/three shared priors as R18stage;all channels/GN4/skips retained;3stage/6CG',
        initialization_IQ_examples=0, paired_clean_optimized=True, all_coarse_action_prior_trainable=True,
        objective='R18stage IQ+.05 powerSTFT64/128/256+teacherstageIQ linear1to.1;add.02 balancedcomplexFFT;second arm adds.25cleanidentityIQ',
        balanced_complex_weight=COMPLEX_WEIGHT, clean_identity_weight=CLEAN_WEIGHT,
        complex_normalization='Within-sample cleanpower quartiles of FFT256;complex residual mean / max(quartilecleanpower,.01fullcleanpower,1e-4);mean four groups;clean and order detached',
        clean_identity='Additional native G(clean) forward only for second arm;paired original clean detached;IQ energy clamp1e-4;no F or teacher in this branch',
        source_basis='R21 fixed256sourceIQ diagnostic199f88078ee4576b705eff84b2a0e33e67492ab0/results854e582c46c4bd935f6b4080d9bb22e4f95f23db',
        historical_control='Reuse completed R18factor625_stage E20/19700;not teacher/init;no new contemporaneous control or old target rerun',
        epochs=EPOCHS, updates=UPDATES, batch=BATCH, microbatch=MICRO, optimizer='ResetAdamW', LR=2e-4,
        final_LR=2e-5, warmup_updates=985, weight_decay=1e-4, gradient_clip=1., FP32=True, TF32=False,
        model_seed=MODEL_SEED, data_seed=DATA_SEED, received_epochs=list(range(1,21)),
        L_count=6300, U_count=56700, L_exposures=126000, U_exposures=1134000, source_IQ_examples=1260000,
        teacher_IQ_inputs=1260000, student_context_inputs=1260000,
        extra_clean_student_inputs={a:1260000*int(CLEAN_WEIGHT[a]>0) for a in ARMS},
        training_F_access=False, TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_training=False, target_training=False,
        inherited_component_epochs=80, cumulative_component_epochs=100, cumulative_component_epoch_sum=300,
        cumulative_serial_training_passes=140, freeze_rule='FixedfinalE20/19700 perarm;all7formal without source gate;all predictions before independenttruthlast',
        views=list(VIEWS), target_count_per_view=168000, target_acceptance_allowance_pp=.8,
        acceptance_baselines=['human_R46_350521','original_R17_historical1pp','R18stage_historicalmatchedrecipe'],
        GPU=GPUS, max_own_GPU_workers=2, max_own_workers_per_card=1,
        predecessor_by_GPU={str(g):v for g,v in PREDECESSOR_BY_GPU.items()}, capacity_lock='.pamf-reception-capacity4.lock',
        source_min_free_MB=18000, all_owner_cap=2, CPU_threads=2, max_own_scorers=1, scoring_run=SCORING_RUN,
        stop_rule='Preserve affected technicalfailure;healthy peer untouched;no automaticretry/hotedit/other-owner stop',
        inference='Complete native G(IQ) only;no F/teacher/clean/extra loss branch at deployment;original F50/defaulthead frozen evaluation',
        difference_from_main='Preserve channel/GN dimensions;analytic complex-frequency fidelity and separate cleanidentity;no phase augmentation or gradient-ranked channel pruning',
        claim='TX-unlabeled simulated-paired source selfsupervision;exposed single-seed engineering benchmark;not freshblind or perdetailTX proof',
        target_feedback=False, goal_complete=False)
