from experiments.pamf_r46_delayed_prune import design as parent

RUN = '20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69'
ARMS = ('stage2', 'shared_context', 'pcg1', 'polyphase')
PARENT_RUN = parent.RUN
PARENT_COMMIT = '4fd058f9bae79687f924e06d057e6a20083ff6dc'
PARENT_ROW = 'prune_after10'
ACTION_RUN, KIND, VIEWS = parent.ACTION_RUN, parent.KIND, parent.VIEWS
F_RUN, F_SEED = parent.F_RUN, parent.F_SEED
MODEL_SEED, DATA_SEED = parent.MODEL_SEED, parent.DATA_SEED
EPOCHS, UPDATES, BATCH = 20, 4940, 256
GPUS = dict(stage2=0, shared_context=1, pcg1=0, polyphase=1)
STAGE_TARGETS = dict(stage2=(1, 2), shared_context=(0, 1, 2), pcg1=(0, 1, 2), polyphase=(0, 1, 2))
learning_rate, teacher_weight = parent.learning_rate, parent.teacher_weight
SCORING_RUN = parent.SCORING_RUN


def configuration():
    return dict(schema='R65_four_route_v1', run=RUN, arms=list(ARMS), parent_run=PARENT_RUN,
        parent_commit=PARENT_COMMIT, parent_row=PARENT_ROW, parent_epoch=20, parent_updates=4940,
        parent_parameters=273765, parent_user_designated_after_historical_exposure=True,
        methods=dict(stage2='Two retained priors(indices1,2),twoCG perstage;teacher stages2,3',
            shared_context='One temporal encoder/decoder shared by coarse/action;separate shallow heads;original spectral and3priors6CG',
            pcg1='One analytic block-preconditioned CG perstage;preconditioner built once per input;3priors3PCG',
            polyphase='Reversible even/odd channel rearrangement for all3priors;length128 hidden features;context/6CG retained'),
        stage_targets={k:list(v) for k,v in STAGE_TARGETS.items()},
        objective='Final pairedclean IQNMSE+.05 original multiscaleSTFT+linear1to.1 mean alignedstage IQNMSE to frozen originalR65',
        spectral_sizes=[64,128,256], epochs=EPOCHS, updates=UPDATES, global_batch=BATCH, microbatch=BATCH,
        optimizer='ResetAdamW',lr=2e-4,final_lr=2e-5,warmup_updates=247,weight_decay=1e-4,clip_norm=1.,
        FP32=True,TF32=False, model_seed=MODEL_SEED,data_seed=DATA_SEED,split_seed=392005,
        augmentation_seed=2026101912,evaluation_seed=2027,support_seed=None,
        L_count=6300,U_count=56700,L_exposures=126000,U_exposures=1134000,
        source_IQ_examples_per_row=1260000,teacher_IQ_examples_per_row=1260000,
        train_TX_labels_read=0,F_training_access=False,V_training=False,target_training=False,
        initialization_IQ_examples=0,new_calibration_IQ=0,
        inherited_parameter_source_epochs=100,new_source_epochs=20,
        transformed_or_new_parameters='Only20new optimizer epochs;copied weights retain explicit100-epoch ancestry;noequalcomputeclaim',
        source_panel=1800,target_count=168000,views=list(VIEWS),
        freeze_rule='All4fixedfinalE20/4940;source diagnostic only;all4auto7formal regardlesssource performance;no selection/earlystop',
        GPU=GPUS,max_own_GPU_workers=2,max_own_per_GPU=1,allowed_GPUs=[0,1],
        source_min_free_MB=18000,all_owner_cap_per_GPU=1,CPU_threads=2,max_own_scorers=1,
        capacity_lock='receiver_residual_capacity4.lock',scoring_run=SCORING_RUN,
        latency='Samecard alternating originalR65/student;G andG+F;B1/8/32/128;3rounds;median/P90/P95/throughput;preloadedFP32 deviceIQ;fullpredictionwallclock separately',
        acceptance='Per-scene exact correct delta againstR65 and originalR17;R17 atmost1680lost/view;70percent lowurban andotherfivegap5pp separately',
        stop_rule='Technical failure preserves affected outputs;healthy peerscontinue;no performance stop;noauto retry/hotedit',
        target_feedback=False,goal_complete=False)
