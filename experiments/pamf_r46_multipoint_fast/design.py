"""Human-designated R46 pruned parent:matched optimization and further compression."""
from experiments.pamf_r44_pruned_refine import design as parent
from experiments.pamf_fullresolution_restore.training import learning_rate

RUN='20261020-phase1-restoration-R46-multipoint-fast-s2026092701-m6-r49'
ARMS=('retain_refine', 'prune_near', 'prune_balanced', 'prune_small', 'context_slim', 'prior_slim')
TASK={a:False for a in ARMS}
MODE={a:("R46_350521_exact_continuation" if a=="retain_refine" else "R46_GN4_weight_importance_"+a) for a in ARMS}
WIDTHS={'retain_refine': (12, 20, 32, 44), 'prune_near': (12, 20, 32, 40), 'prune_balanced': (12, 20, 28, 40), 'prune_small': (12, 16, 24, 32), 'context_slim': (12, 20, 32, 44), 'prior_slim': (12, 16, 24, 36)}
CONTEXT_WIDTHS={'retain_refine': (12, 20, 32, 44), 'prune_near': (12, 20, 32, 40), 'prune_balanced': (12, 20, 28, 40), 'prune_small': (12, 16, 24, 32), 'context_slim': (12, 16, 24, 36), 'prior_slim': (12, 20, 32, 44)}
SPECTRAL_WIDTHS={'retain_refine': (8, 12, 20, 32), 'prune_near': (8, 12, 20, 28), 'prune_balanced': (8, 12, 16, 28), 'prune_small': (8, 12, 16, 24), 'context_slim': (8, 12, 16, 24), 'prior_slim': (8, 12, 20, 32)}
PARAMETERS={'retain_refine': 350521, 'prune_near': 329885, 'prune_balanced': 293069, 'prune_small': 215049, 'context_slim': 268257, 'prior_slim': 310905}
CONTEXT_PARAMETERS={'retain_refine': 237927, 'prune_near': 223191, 'prune_balanced': 197727, 'prune_small': 146755, 'context_slim': 155663, 'prior_slim': 237927}
GPUS=dict(zip(ARMS,range(2,8)))
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=parent.RUN,'bbb8daf817de050b958d217b70d4c33f4aefe341','prune_refine'
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=parent.PARENT_RUN,parent.PARENT_COMMIT,parent.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=parent.F_RUN,parent.F_SEED,parent.VIEWS,parent.KIND
MODEL_SEED,DATA_SEED=parent.MODEL_SEED,parent.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO=20,4940,256,256
PREDECESSOR=parent.RUN
PREDECESSOR_BY_GPU={g:PREDECESSOR for g in (0,1)}
PREDECESSOR_MODULE={PREDECESSOR:'experiments.pamf_r44_pruned_refine.runner'}
SCORING_RUN=parent.SCORING_RUN


def teacher_weight(epoch):
    if type(epoch) is not int or not 1<=epoch<=20:raise ValueError('Fixed twenty-epoch schedule required')
    return 1.-.9*(epoch-1)/19


def configuration():
    return dict(schema='R46_multipoint_fast_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,context_parameters=CONTEXT_PARAMETERS,
        widths={a:list(v) for a,v in WIDTHS.items()},context_widths={a:list(v) for a,v in CONTEXT_WIDTHS.items()},spectral_widths={a:list(v) for a,v in SPECTRAL_WIDTHS.items()},
        compact_parent_run=COMPACT_PARENT_RUN,compact_parent_commit=COMPACT_PARENT_COMMIT,compact_parent_row=COMPACT_PARENT_ROW,
        parent_choice='Human explicitly retains and optimizes350521 after R46 formal results;exposed engineering benchmark,not independentblind validation',
        architecture='Same receive-chain/context/3shared-prior/6CG architecture;exact-parent optimization control and GN4 weight-only further channel pruning',
        objective='Final pairedclean IQ NMSE +.05 mean log1p relative-power STFT errors at64/128/256 +linear1to.1 mean3stage IQNMSE to frozen R46prune_refine',
        spectral_sizes=[64,128,256],spectral_hop='size//2',spectral_window='periodic Hann;centerFalse;orthonormal complexFFT',spectral_weight=.05,
        spectral_scale='Detached pairedclean per-example mean STFT power,clamp1e-4;both log1p arguments use same reference scale',
        teacher_weight_schedule=[teacher_weight(e) for e in range(1,21)],paired_clean_optimized=True,all_coarse_action_prior_trainable=True,
        epochs=20,updates=4940,batch=256,microbatch=256,optimizer='ResetAdamW',LR=2e-4,final_LR=2e-5,warmup_updates=247,weight_decay=1e-4,gradient_clip=1.,FP32=True,TF32=False,
        model_seed=MODEL_SEED,data_seed=DATA_SEED,received_epochs=list(range(1,21)),L_count=6300,U_count=56700,L_exposures=126000,U_exposures=1134000,
        source_IQ_examples=1260000,teacher_IQ_inputs=1260000,student_context_inputs=1260000,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_training=False,target_training=False,
        inherited_component_epochs=80,cumulative_component_epochs=100,cumulative_component_epoch_sum=300,cumulative_serial_training_passes=140,
        freeze_rule='Both fixedfinalE20/4940;source diagnostics only,no earlystop/source selection;automatically full7formal perrow then independentCPUtruthlast',
        views=list(VIEWS),target_count_per_view=168000,target_acceptance_allowance_pp=1.,acceptance_baselines=['original_R17','human_R46_350521','R44_442805_report_only'],
        GPU=GPUS,max_own_GPU_workers=6,max_own_across_R48_R49=8,max_own_workers_per_card=1,predecessor_by_GPU={},healthy_R48_unchanged=True,capacity_lock='receiver_residual_capacity4.lock',
        source_min_free_MB=18000,all_owner_cap=2,CPU_threads=2,max_own_scorers=1,scoring_run=SCORING_RUN,
        stop_rule='Technical failure preserves affectedoutputs,healthy peers unchanged;no automaticretry/hotedit/stop',
        inference='Current nativefloat32 IQ[B,2,256] only,one restoredIQ->originalF50/defaulthead FFT/PA/time/stats;no extra head',
        claim='TX-unlabeled simulated-paired source self-supervision;noF/TX in restorationtraining;notcleanfree blind learning or per-detailTX proof',training_tensor_backend='float32 Python buffer->Torch;no NumPy C bridge or tolist;read-only arrays copied beforebuffer',
        throughput_control='Same350521 fast-control vs R48same-width;global256/247updatesperepoch unchanged;micro256 vs64 may changeFP32roundoff;not a new scientific budget',
        authorization='Latest human requests multiple compression points and higher GPU utilization;GPU2to7 previously idle,oneworkerpercard;R48GPU0/1preserved',
        cost_plan='Matched same-card source-only G and G+F;B1 warm16/64samples x3 alternating rounds;B32 warm4/16iterations x3;median,p90,p95,throughput,peak/resident memory;ConvLinearMAC excludes FFT/CG/norm',target_feedback=False,goal_complete=False)
