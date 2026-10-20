"""Pre-formal-score fixed conditional GN correction within the compressed graph."""
from experiments.pamf_r46_fold_compensate import design as base

RUN='20261021-phase1-restoration-R46-stat-transport-s2026092701-m2-r58'
ARMS=('calibrated','identity_control')
PARAMETERS={a:276453 for a in ARMS}
CONTEXT_PARAMETERS={a:186959 for a in ARMS}
GPUS={'calibrated':3,'identity_control':2}
STAT_MODULES,STAT_PARAMETERS=168,2688
RIDGE_FRACTION,SLOPE_BOUNDS=1e-3,(.25,4.)
CAL_SEED,CAL_UNIQUE,CAL_PER_ROLE,CAL_BATCH=base.CAL_SEED,128,64,8
FOLD_CAL_RUN,FOLD_CAL_COMMIT=base.RUN,'6bea50605a1c2cfb49645b5c2f7ba8306b39791d'
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=base.COMPACT_PARENT_RUN,base.COMPACT_PARENT_COMMIT,base.COMPACT_PARENT_ROW
PREDECESSOR_RUN='20261021-phase1-restoration-R46-GN-response-s2026092701-m2-r57'
PREDECESSOR_COMMIT='38a1b4da6dd2964967cebb67935ef877df11c837'
PREDECESSOR_MODULE='experiments.pamf_r46_gn_response.runner'
PREDECESSOR_ROW={3:'gn_full',2:'gn_half'}
EPOCHS,UPDATES,BATCH,MICRO=20,4940,256,256
MODEL_SEED,DATA_SEED=base.MODEL_SEED,base.DATA_SEED
VIEWS=base.VIEWS

def configuration():
    return dict(schema='R46_GN_stat_transport_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,context_parameters=CONTEXT_PARAMETERS,
        parent=COMPACT_PARENT_RUN,parent_commit=COMPACT_PARENT_COMMIT,parent_row=COMPACT_PARENT_ROW,
        fold_calibration_run=FOLD_CAL_RUN,fold_calibration_commit=FOLD_CAL_COMMIT,trained_R55_R57_students_inherited=False,
        hypothesis='Pruning changes per-sample group statistics;small learned mean/logvariance correction may compensate this while preserving the existing compressed U-Net and3prior6CG',
        normalization='For each sample/group use own mean and biased variance over channels/time;transport mean and log(var+eps),then original per-channel gamma/beta;positive slopesexp(clampedrawlog(.25)to log4)',
        statistic_parameters=2688,GN_modules=168,original_weight_parameters=273765,new_runtime_buffers=0,
        comparison='Calibrated versus identity initialization on exactly the same276453param graph/trainingbudget;R55fullfold descriptive extra control',
        calibration=dict(source_unique_IQ=128,L_s=64,U_s=64,received_views=['01','20'],sampling_seed=CAL_SEED,parent_G_inputs=256,identity_child_G_inputs=256,
            total_new_G_inputs=512,reused_R55_parent_G_inputs=256,ridge_fraction=RIDGE_FRACTION,slope_bounds=list(SLOPE_BOUNDS),
            targets='Frozen parent pre-GN full-group mean and log(biasedvariance+eps);one pair per source sample/group;no spatial subsampling',
            model='Identity transport child with original full-fold inputweights;one shared CPU2 pass;fit fourgroup mean/logvariance coefficients perGN;no optimizer',
            calibrated='Fit initializes existing source-derived new layer parameters;all learn jointly afterward',identity_control='Identical graph with rawlog slopes0 and offsets0;all learn jointly afterward',
            TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,clean_IQ_reads=0,optimizer_updates=0),
        training=dict(epochs=20,updates=4940,batch=256,microbatch=256,pairedclean=True,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,
            source_counts=dict(L_s=126000,U_s=1134000),loss='Unchanged pairedIQNMSE+.05relativepowerMRSTFT+linear1to.1frozenhumanparent3stageIQteacher',
            original_parameters_inherited_epochs=80,original_parameters_cumulative_epochs=100,new_stat_parameters_epochs=20,
            original_component_epoch_sum=300,serial_training_passes=140,all_parameters_jointly_trainable=True),
        evaluation=dict(source_count=1800,target_count=168000,views=list(VIEWS),all_fixed_rows_full7formal=True,
            default_frozen_F50_head=True,source_performance_gate=False,predictions_before_independent_truth=True),
        GPU=GPUS,max_own_GPU_workers=2,max_own_workers_per_card=1,predecessor=PREDECESSOR_RUN,predecessor_commit=PREDECESSOR_COMMIT,
        wait_rule='ActualR57samecardownedworkerscompleteandexited beforeGPUtraining;nohealthyjobchanges',
        source_rule='Matrix/normalizationrulefixedbeforeR57formal scores read;no targetfeedback,parent/width/lossselection or selective reruns',
        stop_rule='Wrongroot/ancestry/IDs,outputcollision,nonfiniteor incompleteinputs/calibration/training/prediction:preserveoutputs,noautomaticretry;healthyjobsuntouched',
        claim='Local compression optimization hypothesis,not verified identification benefit or inputTXdetail proof',goal_complete=False)
