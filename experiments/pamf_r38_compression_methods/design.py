"""Training refinement of two existing compressed R38 architectures; no new network."""
from experiments.pamf_iq_joint_compact import design as parent
from experiments.pamf_r38_successor import design as existing
from experiments.pamf_fullresolution_restore.training import learning_rate

RUN='20261020-phase1-restoration-R38-compression-methods-s2026092701-m2-r44'
ARMS=('low_rank','group_pruned')
TASK={a:False for a in ARMS}
MODE=dict(low_rank='R38_weight_only_SVD_rank_half',group_pruned='R38_GN4_weight_energy_channel_pruning')
WIDTHS=dict(low_rank=(16,32,48,64),group_pruned=(12,24,36,48))
SPECTRAL_WIDTHS=dict(low_rank=(12,24,36,48),group_pruned=(8,16,24,36))
PARAMETERS=dict(low_rank=530161,group_pruned=442805)
CONTEXT_PARAMETERS=dict(low_rank=358771,group_pruned=302471)
GPUS=dict(zip(ARMS,(0,1)))
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=existing.COMPACT_PARENT_RUN,existing.COMPACT_PARENT_COMMIT,existing.COMPACT_PARENT_ROW
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=parent.PARENT_RUN,parent.PARENT_COMMIT,parent.ACTION_RUN
F_RUN,F_SEED=existing.F_RUN,existing.F_SEED
VIEWS,KIND=parent.VIEWS,parent.KIND
MODEL_SEED,DATA_SEED=parent.MODEL_SEED,parent.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO=parent.EPOCHS,parent.UPDATES,parent.BATCH,parent.MICRO
teacher_weight=parent.teacher_weight
PREDECESSOR='20261020-phase1-restoration-IQ-physics-compact-s2026092701-m2-r41'
PREDECESSOR_BY_GPU={g:PREDECESSOR for g in (0,1)}
PREDECESSOR_MODULE={PREDECESSOR:'experiments.pamf_iq_physics_compact.runner'}
SCORING_RUN=PREDECESSOR


def configuration():
    return dict(schema='R38_compression_methods_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,context_parameters=CONTEXT_PARAMETERS,
        widths={a:list(w) for a,w in WIDTHS.items()},spectral_widths={a:list(w) for a,w in SPECTRAL_WIDTHS.items()},
        architecture='Compress R38 by Conv1d low rank or GN4-aware weight-importance channel pruning; original receive-chain/unroll architecture unchanged',
        compact_parent_run=COMPACT_PARENT_RUN,compact_parent_commit=COMPACT_PARENT_COMMIT,compact_parent_row=COMPACT_PARENT_ROW,
        compact_parent_epoch=20,compact_parent_updates=4940,original_R17_run=PARENT_RUN,original_R17_commit=PARENT_COMMIT,
        initialization='User-designated R38 dense_half only: data-free half-rank SVD long Conv1d with original shared aliases, or weight-energy top channels within each GN4 group; no R13/R41 students',
        parent_choice='Human-designated 794529 exposed-benchmark engineering compression baseline; not independent blind/generalization validation',
        design_evidence='User explicitly requests different compression methods; fixed low-rank and weight-importance pruning, both trained/tested regardless source gain',
        objective='Final paired-clean IQ NMSE + linear10to1(epoch1to20) mean3stage IQ NMSE to frozen R38 dense_half',
        teacher_weight_schedule=[teacher_weight(e) for e in range(1,21)],paired_clean_optimized=True,all_coarse_action_prior_trainable=True,
        epochs=EPOCHS,updates=UPDATES,batch=BATCH,microbatch=MICRO,optimizer='Reset AdamW',LR=2e-4,final_LR=2e-5,warmup_updates=247,
        weight_decay=1e-4,gradient_clip=1.,FP32=True,TF32=False,model_seed=MODEL_SEED,data_seed=DATA_SEED,received_epochs=list(range(1,21)),
        L_count=6300,U_count=56700,L_exposures=126000,U_exposures=1134000,source_IQ_examples=1260000,
        R38_teacher_inputs=1260000,student_context_inputs=1260000,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_training=False,target_training=False,
        inherited_component_epochs=40,cumulative_component_epochs=60,cumulative_component_epoch_sum=180,cumulative_serial_training_passes=100,
        freeze_rule='Fixed final E20/4940; no early stop, source score selection or target gate; both rows always full7 formal then independent truth-last',
        views=list(VIEWS),target_count_per_view=168000,target_acceptance_allowance_pp=1.,acceptance_baselines=['original_R17','user_R38_794529'],
        GPU=GPUS,max_own_GPU_workers=2,max_own_workers_per_card=1,predecessor_by_GPU={str(g):r for g,r in PREDECESSOR_BY_GPU.items()},
        capacity_lock='receiver_residual_capacity4.lock',source_min_free_MB=18000,all_owner_cap=2,CPU_threads=2,max_own_scorers=1,scoring_run=SCORING_RUN,
        stop_rule='Technical failure preserves affected outputs and healthy peers; no automatic retry, no healthy process stop or hot edit',
        inference='Current native-scale float32 IQ2x256 only; one recovered IQ->original frozen F50/defaulthead FFT/PA/time/stats;3prior/6CG',
        claim='TX-unlabeled simulated-paired source self-supervision; F-free restoration training; exposed engineering benchmark, not clean-blind or per-detail TX authenticity proof',
        historical_R13_test_exposure=True,rank_fraction=.5,pruning_score='Sum of normalized per-output-filter mean-square weights of four encoder convolutions at each level;top channels independently within each GN4 group;stable ties',target_feedback=False,goal_complete=False)
