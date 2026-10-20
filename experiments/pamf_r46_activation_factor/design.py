from experiments.pamf_r46_complex_fidelity import design as baseline
from experiments.pamf_r44_pruned_refine import design as parent

RUN = '20261021-phase1-restoration-R46-activation-factor-s2026092701-m2-r23'
ARMS = ('activation625_identity', 'activation500_identity')
FRACTIONS = dict(zip(ARMS,(.625,.5)))
PARAMETERS = dict(zip(ARMS,(278180,236355)))
CONTEXT_PARAMETERS = dict(zip(ARMS,(186703,157143)))
MODE = {a:'R46_allcore_source_activation_SVD'+str(int(1000*FRACTIONS[a])) for a in ARMS}
TASK = {a:False for a in ARMS}
GPUS = dict(zip(ARMS,(2,3)))
CLEAN_WEIGHT = {a:.25 for a in ARMS}
COMPLEX_WEIGHT = .02
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW = baseline.COMPACT_PARENT_RUN,baseline.COMPACT_PARENT_COMMIT,baseline.COMPACT_PARENT_ROW
PARENT_RUN,PARENT_COMMIT,ACTION_RUN = baseline.PARENT_RUN,baseline.PARENT_COMMIT,baseline.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND = baseline.F_RUN,baseline.F_SEED,baseline.VIEWS,baseline.KIND
MODEL_SEED,DATA_SEED = baseline.MODEL_SEED,baseline.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO = 20,19700,64,64
SCORING_RUN = baseline.SCORING_RUN
PREDECESSOR = baseline.RUN
PREDECESSOR_BY_GPU = {g:PREDECESSOR for g in GPUS.values()}
PREDECESSOR_MODULE = {PREDECESSOR:'experiments.pamf_r46_complex_fidelity.runner'}
CONTROL_RUN,CONTROL_ROW = baseline.RUN,'complex625_identity'
teacher_weight,learning_rate = baseline.teacher_weight,baseline.learning_rate
CAL_PER_ROLE,CAL_UNIQUE,CALIBRATION_IQ = 128,256,768
CAL_VIEWS = ('clean','received01','received20')
CAL_POSITIONS,RIDGE_FRACTION = 8,1e-3


def calibration_configuration():
    return dict(parent_run=COMPACT_PARENT_RUN,parent_commit=COMPACT_PARENT_COMMIT,parent_row=COMPACT_PARENT_ROW,
        action_cache_run=ACTION_RUN,role_positions=CAL_PER_ROLE,unique_source_positions=CAL_UNIQUE,
        views=list(CAL_VIEWS),teacher_IQ_examples=CALIBRATION_IQ,L_exposures=CAL_PER_ROLE*len(CAL_VIEWS),
        U_exposures=CAL_PER_ROLE*len(CAL_VIEWS),batch=32,positions_per_conv=CAL_POSITIONS,
        ridge_fraction=RIDGE_FRACTION,ridge_floor=1e-8,minimum_fraction=.5,
        method='Uncentered teacher input second moments;pool all3sharedprior uses;Cholesky weighted SVD',
        source_roles=['L_s','U_s'],device='cpu',CPU_threads=2,GPU_workers=0,
        F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,optimizer_updates=0)


def configuration():
    cfg=baseline.configuration()
    cfg.update(schema='R46_activation_factor_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,
        context_parameters=CONTEXT_PARAMETERS,rank_fraction=FRACTIONS,initialization_IQ_examples=CALIBRATION_IQ,
        calibration=calibration_configuration(),clean_identity_weight=CLEAN_WEIGHT,
        clean_identity='Additional native G(clean) forward in both arms;paired original clean detached;IQ energy clamp1e-4;no F or teacher in this branch',
        acceptance_baselines=['human_R46_350521','original_R17_historical1pp'],
        result_comparison='R22complex625_identity prospectively fixed sameobjective;compare when complete,never read for training or selection',
        extra_clean_student_inputs={a:1260000 for a in ARMS},
        architecture='Source-activation weighted SVD625/500 across fullcoarse/action/sharedprior;GN4/channels/skips/bias retained;3stage6CG',
        objective='Same pre-fixed R22identity pairedIQ+.05powerSTFT+teacherstage1to.1+.02balancedcomplexFFT+.25cleanidentity',
        historical_control='Prospectively fixed R22complex625_identity sameobjective;results only,never initialization/teacher;missing control not a formal gate',
        source_basis='R21sourceIQ/R22source-only training diagnostics and known weightSVD fitting gap;reuse R03 math only,not old model or F-selection gate',
        GPU=GPUS,predecessor_by_GPU={str(g):v for g,v in PREDECESSOR_BY_GPU.items()},
        difference_from_main='Preserve all GN/channel widths and factor full kernels;main R55 folds dropped-channel linear reconstructions into pruned weights',
        inference='Own fullG/nativeIQ only;no calibration moments/sourceIQ/teacher/F/clean or training loss branch',
        claim='F-free TX-unlabeled paired-source selfsupervision;activation calibration uses sourceIQ and is not data-free;exposed benchmark,not blind or perdetailTX proof')
    return cfg
