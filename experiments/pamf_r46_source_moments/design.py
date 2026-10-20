from experiments.pamf_r46_source_response import design as base

RUN='20261021-diagnostic-restoration-R46-source-moments-s2026092701-m4-r29'
ARMS,REFERENCE_ARM,PARAMETERS,SOURCE_ROWS,SOURCE_BUDGET=base.ARMS,base.REFERENCE_ARM,base.PARAMETERS,base.SOURCE_ROWS,base.SOURCE_BUDGET
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT=base.COMPACT_PARENT_RUN,base.COMPACT_PARENT_COMMIT
FACTOR_RUN,FACTOR_COMMIT,COMPLEX_RUN,COMPLEX_COMMIT=base.FACTOR_RUN,base.FACTOR_COMMIT,base.COMPLEX_RUN,base.COMPLEX_COMMIT
ACTION_RUN,KIND,VIEWS,PER_ROLE,BATCH=base.ACTION_RUN,base.KIND,base.VIEWS,base.PER_ROLE,base.BATCH
LAGS=(1,2,4,8)
FEATURES=('abs_c40','c42','normalized_m42')+tuple('lag'+str(lag)+'_'+part for lag in LAGS for part in ('real','imag'))
POWER_FLOOR=1e-4


def configuration():
    return dict(schema='R46_source_moments_v1',run=RUN,arms=list(ARMS),reference_arm=REFERENCE_ARM,parameters=PARAMETERS,
        source_rows=SOURCE_ROWS,source_budgets={a:list(v) for a,v in SOURCE_BUDGET.items()},
        parent_run=COMPACT_PARENT_RUN,parent_commit=COMPACT_PARENT_COMMIT,factor_run=FACTOR_RUN,factor_commit=FACTOR_COMMIT,
        complex_run=COMPLEX_RUN,complex_commit=COMPLEX_COMMIT,source_matrix='Exactly the four frozen R25 models;no target-ranked or R23-selected model',
        views=list(VIEWS),PER_ROLE=PER_ROLE,unique_source_positions=256,source_roles=['L_s','U_s'],source_views=['clean','received01','received20'],
        sampling='Original R21/R25 fixed 128 evenly spaced opaque positions per L_s/U_s role',
        features=list(FEATURES),lags=list(LAGS),power_floor=POWER_FLOOR,
        moments='Centered complex moments:absC40,C42,normalizedM42;lagcum(a,a,a*,b*)=E[a*a*a*.b*]-2E[aa*]E[ab*]-E[aa]E[a*b*]',
        normalization='Marginal P^2;lag sqrt(Pa^3*Pb),powers each clamped1e-4;no nearzero-cumulant division',
        metrics='Pairedclean/parent IQNMSE,momentMSE,11 squaredfeatureerrors each;clean/parent/restored powerfloor rates',
        metric_names_per_cell=29,expected_metric_rows=1044,role_groups=['all','L_s','U_s'],summary_statistics=['mean','median','p90','p95','maximum'],
        center_inputs_per_model=768,smoke_inputs_per_model=2,total_actual_inputs=3080,diagnostic_L_per_model=384,diagnostic_U_per_model=384,
        CPU_threads=2,GPU_workers=0,diagnostic_workers=1,new_network_versions=0,new_training_updates=0,
        F_weight_reads=0,F_output_reads=0,TX_label_reads=0,V_reads=0,target_reads=0,head_fit_updates=0,
        state_updates=0,model_selection=False,teacher_selection=False,target_feedback=False,
        claim='Descriptive finite-window source fourth-order errors;not TX identity or recoverability proof,not unbiased population estimates',
        future_training_parent='Only original human R46 source-final350521;no tested student initialization/teacher',
        future_versions='Every new restoration version retains full seven-scene formal evaluation',
        stop_rule='Preserve technical failure;no retry/update/fallback/other-owner mutation')
