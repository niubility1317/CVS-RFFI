"""Human-designated R44 pruned parent:matched optimization and further compression."""
from experiments.pamf_r38_compression_methods import design as parent
from experiments.pamf_fullresolution_restore.training import learning_rate

RUN='20261020-phase1-restoration-R44-pruned-refine-s2026092701-m2-r46'
ARMS=('retain_refine','prune_refine')
TASK={a:False for a in ARMS}
MODE=dict(retain_refine='R44_442805_exact_continuation',prune_refine='R44_GN4_weight_importance_further_pruning')
WIDTHS=dict(retain_refine=(12,24,36,48),prune_refine=(12,20,32,44))
SPECTRAL_WIDTHS=dict(retain_refine=(8,16,24,36),prune_refine=(8,12,20,32))
PARAMETERS=dict(retain_refine=442805,prune_refine=350521)
CONTEXT_PARAMETERS=dict(retain_refine=302471,prune_refine=237927)
GPUS=dict(zip(ARMS,(0,1)))
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=parent.RUN,'2f5247dbd5fe50f28defa38a0252d70e692934af','group_pruned'
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=parent.PARENT_RUN,parent.PARENT_COMMIT,parent.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=parent.F_RUN,parent.F_SEED,parent.VIEWS,parent.KIND
MODEL_SEED,DATA_SEED=parent.MODEL_SEED,parent.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO=20,4940,256,64
PREDECESSOR=parent.RUN
PREDECESSOR_BY_GPU={g:PREDECESSOR for g in (0,1)}
PREDECESSOR_MODULE={PREDECESSOR:'experiments.pamf_r38_compression_methods.runner'}
SCORING_RUN=parent.SCORING_RUN


def teacher_weight(epoch):
    if type(epoch) is not int or not 1<=epoch<=20:raise ValueError('Fixed twenty-epoch schedule required')
    return 1.-.9*(epoch-1)/19


def configuration():
    return dict(schema='R44_pruned_refinement_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,context_parameters=CONTEXT_PARAMETERS,
        widths={a:list(v) for a,v in WIDTHS.items()},spectral_widths={a:list(v) for a,v in SPECTRAL_WIDTHS.items()},
        compact_parent_run=COMPACT_PARENT_RUN,compact_parent_commit=COMPACT_PARENT_COMMIT,compact_parent_row=COMPACT_PARENT_ROW,
        parent_choice='Human explicitly retains and optimizes442805 after R44 formal results;exposed engineering benchmark,not independentblind validation',
        architecture='Same receive-chain/context/3shared-prior/6CG architecture;exact-parent optimization control and GN4 weight-only further channel pruning',
        objective='Final pairedclean IQ NMSE +.05 mean log1p relative-power STFT errors at64/128/256 +linear1to.1 mean3stage IQNMSE to frozen R44group_pruned',
        spectral_sizes=[64,128,256],spectral_hop='size//2',spectral_window='periodic Hann;centerFalse;orthonormal complexFFT',spectral_weight=.05,
        spectral_scale='Detached pairedclean per-example mean STFT power,clamp1e-4;both log1p arguments use same reference scale',
        teacher_weight_schedule=[teacher_weight(e) for e in range(1,21)],paired_clean_optimized=True,all_coarse_action_prior_trainable=True,
        epochs=20,updates=4940,batch=256,microbatch=64,optimizer='ResetAdamW',LR=2e-4,final_LR=2e-5,warmup_updates=247,weight_decay=1e-4,gradient_clip=1.,FP32=True,TF32=False,
        model_seed=MODEL_SEED,data_seed=DATA_SEED,received_epochs=list(range(1,21)),L_count=6300,U_count=56700,L_exposures=126000,U_exposures=1134000,
        source_IQ_examples=1260000,teacher_IQ_inputs=1260000,student_context_inputs=1260000,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_training=False,target_training=False,
        inherited_component_epochs=60,cumulative_component_epochs=80,cumulative_component_epoch_sum=240,cumulative_serial_training_passes=120,
        freeze_rule='Both fixedfinalE20/4940;source diagnostics only,no earlystop/source selection;automatically full7formal perrow then independentCPUtruthlast',
        views=list(VIEWS),target_count_per_view=168000,target_acceptance_allowance_pp=1.,acceptance_baselines=['original_R17','human_R44_442805','R38_794529_report_only'],
        GPU=GPUS,max_own_GPU_workers=2,max_own_workers_per_card=1,predecessor_by_GPU={str(g):v for g,v in PREDECESSOR_BY_GPU.items()},capacity_lock='receiver_residual_capacity4.lock',
        source_min_free_MB=18000,all_owner_cap=2,CPU_threads=2,max_own_scorers=1,scoring_run=SCORING_RUN,
        stop_rule='Technical failure preserves affectedoutputs,healthy peers unchanged;no automaticretry/hotedit/stop',
        inference='Current nativefloat32 IQ[B,2,256] only,one restoredIQ->originalF50/defaulthead FFT/PA/time/stats;no extra head',
        claim='TX-unlabeled simulated-paired source self-supervision;noF/TX in restorationtraining;notcleanfree blind learning or per-detailTX proof',target_feedback=False,goal_complete=False)
