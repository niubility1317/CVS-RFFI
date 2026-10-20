from experiments.pamf_r46_complex_fidelity import design as source
from experiments.pamf_r46_response_distill import design as previous

RUN='20261021-phase1-restoration-R46-prior-collapse-s2026092701-m2-r27'
ARMS=('R22_complex625_priorcollapsed','R22_complex625_identity_priorcollapsed')
PARAMETERS={a:299297 for a in ARMS}
SOURCE_ROWS=dict(zip(ARMS,source.ARMS))
GPUS=dict(zip(ARMS,(2,3)))
COMPLEX_RUN,COMPLEX_COMMIT=source.RUN,'e190cf5367b03da8aafced51276a8730ad8274fa'
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=source.COMPACT_PARENT_RUN,source.COMPACT_PARENT_COMMIT,source.COMPACT_PARENT_ROW
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=source.PARENT_RUN,source.PARENT_COMMIT,source.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=source.F_RUN,source.F_SEED,source.VIEWS,source.KIND
SCORING_RUN=source.SCORING_RUN
PREDECESSOR=previous.RUN
PREDECESSOR_BY_GPU={g:PREDECESSOR for g in GPUS.values()}
PREDECESSOR_MODULE={PREDECESSOR:'experiments.pamf_r46_response_distill.runner'}
BATCH,EPOCHS,UPDATES=128,20,19700


def configuration():
    return dict(schema='R46_prior_collapse_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,
        source_rows=SOURCE_ROWS,source_run=COMPLEX_RUN,source_commit=COMPLEX_COMMIT,
        source_budget=[20,19700],source_parameters=278180,source_selection=False,
        original_parent_run=COMPACT_PARENT_RUN,original_parent_commit=COMPACT_PARENT_COMMIT,
        mechanism='Frozen prior-only linear factor composition;preserve shared weight identities and independent stagebias/GN/skip;nativeFP32',
        before_parameters=278180,after_parameters=299297,original_parent_parameters=350521,
        context_parameters=186703,prior_parameters=112594,converted_factor_modules=84,unique_expanded_kernels=28,
        convolution_calls_removed_per_G=84,training_updates=0,training_IQ_examples=0,
        training_F_access=False,training_TX_labels_read=0,F_parameter_updates=0,head_fit_updates=0,
        source_choice='BOTH fixedR22 source-final models;not a new training initializer/teacher;futuretraining original350521 only',
        source_count=1800,formal_count=168000,views=list(VIEWS),batch=BATCH,
        freeze_rule='Frozen source checkpoints;eachrow source7/profile/formal7 and independenttruthlast without sourcegain gate',
        source_equivalence='Actual2Ls clean+received noquery smoke;full outputs before/after reported;not assumed bitwise identical',
        prediction_comparison='Allnew7predictions fixed,then compare IQ/features/logits and argmax with completed R22 native predictions;no native targetrerun',
        timing='Samecard originalR17,parent350521,sourceR22lowrank278180,expanded299297;nativeG/G+F;B1/B32;3alternatingrounds;64sourceV',
        timing_scope='All physics included;preloaded device FP32;excludes transfer/preprocess/loader;MAC Conv1d+Linear only,notFFT/CG/norm/F',
        memory_scope='Four frozenG and oneF resident suite;incrementalpeaks comparable,not standalone deploymentresident',
        GPU=GPUS,max_own_GPU_workers=2,max_own_workers_per_card=1,all_owner_cap=2,CPU_threads=2,max_own_scorers=1,
        scoring_run=SCORING_RUN,source_min_free_MB=18000,capacity_lock='.pamf-reception-capacity4.lock',
        predecessor_by_GPU={str(g):v for g,v in PREDECESSOR_BY_GPU.items()},
        stop_rule='Preserve affectedtechnicalfailure;healthyotherrowscontinue;no retry/hotedit/nativefallback/other-ownerstop',
        target_acceptance_allowance_pp=.8,historical_R17_allowance_pp=1.,
        performance_targets='Measured latency and complete deployed parameters;lowurban70/other5within5pp rawclean/cleanpreservation/TXfidelity remain separate',
        difference_from_main='Collapse linear prior factor pairs with increased but stillcompressed deployed coefficient count;main caches normalization slopes',
        claim='Algebraic frozen reparameterization with changed FP32 operation order;all7required;not exactfloatingidentity,new blind test,or TX authenticity proof',
        target_feedback=False,goal_complete=False)
