from experiments.pamf_unrolled_restore import design as original
from experiments.pamf_uncertainty_restore.evaluation import VIEWS

RUN='20261020-phase1-restoration-r17-IQ-joint-compact-s2026092701-m2-r38'
ARMS=('dense_three_quarter','dense_half')
WIDTHS=dict(zip(ARMS,((24,48,72,96),(16,32,48,64))))
SPECTRAL_WIDTHS=dict(zip(ARMS,((16,32,48,64),(12,24,36,48))))
GPUS=dict(zip(ARMS,(0,1)))
PARAMETERS=dict(zip(ARMS,(1709713,794529)))
PARENT_RUN=original.RUN
PARENT_COMMIT='cd97744884b5e460926e48c8b7efb7d78c6da0f4'
ACTION_RUN='20261019-phase1-restoration-learned-action-source-s2026092701-r07'
PRECEDING_RUN='20261020-evaluation-restoration-R32-recovery-all-versions-F50-s2026092701-m2-r37'
GPU_PRECEDING_RUN=PRECEDING_RUN
SCORING_RUN=RUN
KIND='native_map_prior'
EPOCHS,UPDATES,BATCH,MICRO=20,4940,256,64
MODEL_SEED,DATA_SEED=2026092701,2026102001


def teacher_weight(epoch):
    if type(epoch) is not int or not 1<=epoch<=EPOCHS:raise ValueError('Fixed epoch1..20 required')
    return 10.-9.*(epoch-1)/(EPOCHS-1)


def configuration():
    return dict(schema='IQ_joint_dense_compact_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,
        widths={a:list(w) for a,w in WIDTHS.items()},spectral_widths={a:list(w) for a,w in SPECTRAL_WIDTHS.items()},
        parent_run=PARENT_RUN,parent_commit=PARENT_COMMIT,parent_row='unroll_unshared',parent_epoch=20,parent_updates=4940,
        initialization='Group-preserving deterministic dense channel slices of original R17;share long prior weights by mean;step-specific bias/GN/pointwise retained',
        initialization_IQ_examples=0,context='Jointly train all narrowed coarse/action and priors;teacher always uses original context',
        objective='paired final IQ NMSE + linear10to1(epoch1to20) mean3stage IQ NMSE to detached original R17',
        paired_clean_optimized=True,teacher_weight_schedule=[teacher_weight(e) for e in range(1,21)],
        epochs=EPOCHS,updates=UPDATES,batch=BATCH,microbatch=MICRO,optimizer='AdamW',LR=2e-4,final_LR=2e-5,warmup_updates=247,weight_decay=1e-4,gradient_clip=1.,FP32=True,TF32=False,
        L_count=6300,U_count=56700,L_exposures=126000,U_exposures=1134000,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_training=False,target_training=False,
        views=list(VIEWS),target_count_per_view=168000,target_acceptance_allowance_pp=1.,source_performance_gate=False,
        inference='Current IQ only;one restored IQ->original frozen F50/defaulthead FFT/PA/time/statistics;3prior calls and6CG steps',
        GPU=GPUS,max_own_GPU_workers=2,max_own_workers_per_card=1,capacity_lock='receiver_residual_capacity4.lock',source_min_free_MB=18000,all_owner_cap=2,
        preceding_run=PRECEDING_RUN,CPU_threads=2,max_own_scorers=1,
        stop_rule='Wrong contract/inputs/collision/nonfinite or technical failure:preserve affected output and healthy peers;no automatic retry',
        cost='All unique coarse/action/prior parameters;buffers/model bytes/MAC/latency/memory separate;optimizer excluded from inference profile',
        claim='TX-unlabeled simulated-paired self-supervision with F-free IQ teacher;exposed benchmark,not per-detail TX authenticity proof',target_feedback=False,goal_complete=False)
