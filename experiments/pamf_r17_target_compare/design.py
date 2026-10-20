from experiments.pamf_r17_task import design as source

RUN='20261020-phase1-restoration-r17-compact-target-comparison-s2026092701-m2-r05'
SOURCE_RUN=source.RUN
SOURCE_COMMIT='a310b3b5ff30e23584e7bc950e8a0ff96ac685fb'
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=source.PARENT_RUN,source.PARENT_COMMIT,source.ACTION_RUN
GPU_PRECEDING_RUN=SOURCE_RUN
ARMS,VIEWS,GPUS,PARAMETERS,KIND=source.ARMS,source.VIEWS,source.GPUS,source.PARAMETERS,source.KIND


def configuration():
    return dict(schema='R17_fixed_target_comparison_v1',run=RUN,source_run=SOURCE_RUN,source_commit=SOURCE_COMMIT,
        rows=list(ARMS),views=list(VIEWS),count_per_view=168000,parameters=PARAMETERS,GPU=GPUS,
        checkpoint='BothsourcefinalE20/4940,frozenbeforeanyR04target;no retrain/selectionbytarget',
        authorization='Latestuser:performancejudgedontargetdomain,maxdrop1percentagepointineachviewvsoriginalR17;80to79explicitlyconfirmed',
        source_result='Bothsourceaccuracywithin1pp;originalR04waveformguardfailed;originalsourcequalificationfalse preserved,notforged',
        evaluation_basis='Fixedtwo-rowtargetcomparison;waveformNMSEreported,butnotanidentityperformancegate',
        target_acceptance_allowance_pp=1.,original_R17_run=source.PARENT_RUN,original_R17_commit=source.PARENT_COMMIT,
        F_run=source.F_RUN,F_commit=source.F_COMMIT,F_seed=source.F_SEED,original_default_head=True,
        new_training_updates=0,new_head_fit_updates=0,prediction_state_updates=0,
        inference='currentIQonly;no TX/RX/scene/clean/Fteacher/runtimecacheorupdates inrestorer',
        GPU_policy='physical4/5;oneownworker/card;free>=12GB/allowner<2;preserveothers',
        CPU_policy='2threads/max1ownscorer/ownscore.lock;previousR04sourceandsavedanalysis exited',
        stop_rule='Technicalfailurepreserved;healthyrowscontinue;no automaticretry;performancefailureisnottechnicalfailure',
        target_feedback=False,claim='Fixedexposedbenchmarkcomparison,notnewblindtest;reportbothrows,no targetfeedbackintotraining/hyperparameters/retries')
