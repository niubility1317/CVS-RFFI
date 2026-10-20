RUN='20261020-diagnostic-restoration-independent-analytic-TX-s2026092701-r34'
PARENT='20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17'
PARENT_COMMIT='cd97744884b5e460926e48c8b7efb7d78c6da0f4'
CANDIDATE='unroll_unshared'
ACTION='20261019-phase1-restoration-learned-action-source-s2026092701-r07'
ACTION_COMMIT='c8ccdfdbd03ab5ad8828b1eb9210b99a664ebe7b'
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
GROUPS=dict(mirror=64,spectral=4,reference=24,combined=92)
CALIBRATION_COUNT=8192
PERMUTATIONS=199
SEED=2026102034


def configuration(commit):
    return dict(schema='F_independent_analytic_identity_v1',run=RUN,commit=commit,parent=PARENT,parent_commit=PARENT_COMMIT,
        candidate=CANDIDATE,views=list(VIEWS),groups=GROUPS,count=1800,calibration_count=CALIBRATION_COUNT,
        calibration='Original first8192 LU clean;FP64 population mean/std floor.05;no labels/F/V/target;not a classifier fit',
        coordinates='R22 fixed public maths:mirror logratio/asym64,spectral4,reference complexresidual24;drop response quality4',
        combined='Concatenate centered standardized groups divided by sqrt(3*group_dimension)',
        reference='Cap2 lowest opaque physicalIDs per TX/RX/day;reference RX and day both different;complete6TX nonzero cells;equal cell and other5TX means',
        negative_control=dict(seed=SEED,permutations=PERMUTATIONS,rule='Shuffle query labels within RX/day;hold reference library fixed;descriptive conditional control,not pvalue'),
        new_neural_training_updates=0,head_fit_updates=0,extra_restorer_predictions=0,F_weights_read=0,F_embeddings_logits_read=0,
        target_reads=0,source_V_fit_updates=0,CPU_threads=2,GPU_workers=0,source_selection_changed=False,
        claim_scope='F-independent analytic source association;descriptors not TX-exclusive;no decodedpayload independence,newTX,inorbit or per-detail proof',
        stop_rule='Wrong source parent/roles,missing or nonfinite input/features,outputcollision:preserve outputs,noautomaticretry;healthy peers unchanged',goal_complete=False)
