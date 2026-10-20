from experiments.pamf_r46_source_moments import design as base

RUN='20261021-diagnostic-restoration-R46-gradient-geometry-s2026092701-m4-r31'
ARMS,REFERENCE_ARM,PARAMETERS,SOURCE_ROWS,SOURCE_BUDGET=base.ARMS,base.REFERENCE_ARM,base.PARAMETERS,base.SOURCE_ROWS,base.SOURCE_BUDGET
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT=base.COMPACT_PARENT_RUN,base.COMPACT_PARENT_COMMIT
FACTOR_RUN,FACTOR_COMMIT,COMPLEX_RUN,COMPLEX_COMMIT=base.FACTOR_RUN,base.FACTOR_COMMIT,base.COMPLEX_RUN,base.COMPLEX_COMMIT
ACTION_RUN,KIND,VIEWS,PER_ROLE=base.ACTION_RUN,base.KIND,base.VIEWS,base.PER_ROLE
BATCH=16
GROUPS=('whole','coarse','action','prior')
OBJECTIVES=('base','moment_clean','moment_teacher')
PAIRS=(('base','moment_clean'),('base','moment_teacher'),('moment_clean','moment_teacher'))
NORM_FLOOR=1e-12


def configuration():
    return dict(schema='R46_source_gradient_geometry_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,
        source_rows=SOURCE_ROWS,source_budgets={a:list(v) for a,v in SOURCE_BUDGET.items()},
        parent_run=COMPACT_PARENT_RUN,parent_commit=COMPACT_PARENT_COMMIT,factor_run=FACTOR_RUN,factor_commit=FACTOR_COMMIT,
        complex_run=COMPLEX_RUN,complex_commit=COMPLEX_COMMIT,source_matrix='Exactly fixed R29/R25 four source-final models;no R23/target-ranked model',
        views=list(VIEWS),unique_source_positions=256,source_L=128,source_U=128,batch=16,batches_per_view=16,
        batching='Each batch takes8Ls positions then8Us;all256positions once perview;noTX labels or sample-quality selection',
        objective_names=list(OBJECTIVES),parameter_groups=list(GROUPS),norm_floor=NORM_FLOOR,
        base_objective='Counterfactual common R22identity finalepoch20 base:pairedIQ+.05powerSTFT+.1teacherstageIQ+.02balancedcomplex+.25cleanidentity',
        moment_objectives='R30 exact mean11descriptorMSE with weight1 against detached pairedclean or original350521 finalreceivedoutput',
        gradient_scope='Gradient of each 16-example batch mean w.r.t.unique model parameters;autograd.grad only,no .grad accumulation,optimizer,clip,projection or updates',
        geometry='Float64 dot/norm summaries;cosine valid only if bothnorms>1e-12;undefined cosines excluded,never treated as zero/conflict',
        teacher_cache_IQ_inputs=770,student_received_per_model=768,student_clean_identity_per_model=768,smoke_received_per_model=2,smoke_clean_per_model=2,
        student_IQ_inputs_per_model=1540,total_actual_G_IQ_inputs=6930,batch_records=192,expected_summary_rows=324,
        source_L_student_per_model=768,source_U_student_per_model=768,smoke_L_student_per_model=4,
        CPU_threads=2,GPU_workers=0,diagnostic_workers=1,new_network_versions=0,optimizer_updates=0,head_fit_updates=0,
        F_weight_reads=0,F_output_reads=0,TX_label_reads=0,V_reads=0,target_reads=0,state_updates=0,
        model_selection=False,teacher_selection=False,target_feedback=False,existing_R30_matrix_mutations=0,
        claim='Local source batch gradient geometry for a common diagnostic objective,not historical training gradients,TX identity,learning outcome or formal recognition proof',
        future_training_parent='Original human350521 only;diagnostic students never become new initialization/teacher',
        future_versions='Every new network still full seven-scene formal test;this diagnostic creates none',
        stop_rule='Preserve technical failure;no automaticretry/hotedit/fallback or other-owner mutation')
