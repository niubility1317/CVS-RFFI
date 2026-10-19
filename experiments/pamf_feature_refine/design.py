RUN='20261019-phase1-restoration-frozenF-feature-source-s2026092701-m2-r01'
PARENT_RUN='20261019-phase1-restoration-residual-refinement-selfsup-m3-r03'
PARENT_COMMIT='b01b534f52daf9cae9e940ed6ec6f9243bccbdbe'
ARMS=('waveform_control','frozenF_feature')
GPUS={'waveform_control':0,'frozenF_feature':1}
MODEL_SEED=2026092701
DATA_SEED=2026101951
EPOCHS=20
BATCH=256
UPDATES=4940


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered matched arm required')
    return dict(schema='frozenF_feature_refine_v1',run_id=RUN,arm=arm,
        parent_run=PARENT_RUN,parent_commit=PARENT_COMMIT,model_seed=MODEL_SEED,data_seed=DATA_SEED,
        epochs=EPOCHS,updates=UPDATES,physical_batch=BATCH,IQ_examples_per_physical=2,
        source_count=63000,L_count=6300,U_count=56700,
        routes='Original R02 q<.15 clean,.15<=q<.75 residual,otherwise full;pluscleanidentity for everyphysicalsample',
        optimizer='AdamW reset',learning_rate=3e-5,weight_decay=1e-4,warmup_updates=247,
        final_learning_rate=3e-6,clip_grad_norm=1.,precision='FP32',
        waveform_loss='mean IQNMSE+.1logpower,observed->clean andclean->clean equallyweighted',
        feature_weight=1. if arm=='frozenF_feature' else 0.,
        feature_loss='1-cos(originalF50features(estimate),detachedoriginalF50features(clean));no head/logit loss',
        teacher='Original F50 CROSS_RX-s2026092701/defaulthead,fully frozen;sourceL/U features only',
        teacher_features_read_in_training=arm=='frozenF_feature',TX_labels_read=0,
        F_optimizer_updates=0,head_fit_updates=0,coarse_updates=0,target_training=False,
        trainable='R03 correction module only;parent correctionE20+newE20=E40,coarseE20 separately',
        checkpoint_selection='Fixed final E20;no V earlystop/optimizer/normalization updates',
        source_views='clean+sixpractical,1800V,allpredictions before independentCPU labels',
        source_rule='Guidedlowurbanaccuracy>=control+.5pp,cleandropvsidentity<=.1pp,meanpracticalNMSE<=1.05control',
        target_scope='Pending user clean-only vs clean+sixpractical clarification;no target access in source run',
        GPU=GPUS[arm],max_this_arm_GPU_workers=1,all_owner_cap_per_GPU=2,min_free_MB=12000,
        capacity_lock='receiver_residual_capacity4.lock',CPU_threads=2,
        stop_rule='Invalid data/parent/code,collision,nonfinite,budget/predictionfailure:abort,preserve,noautoretry')
