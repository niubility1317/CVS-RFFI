from experiments.pamf_r17_calibrated import design as previous
RUN='20261020-phase1-restoration-r17-task-source-s2026092701-m2-r04'
PARENT_RUN,PARENT_COMMIT=previous.PARENT_RUN,previous.PARENT_COMMIT
ACTION_RUN,KIND=previous.ACTION_RUN,previous.KIND
ARMS=('step_norm_task','activation_task')
GPUS=dict(zip(ARMS,(4,5)));PARAMETERS=dict(zip(ARMS,(3118001,3775105)))
GPU_PRECEDING_RUN=CPU_PRECEDING_RUN=previous.RUN
VIEWS=previous.VIEWS
MODEL_SEED,DATA_SEED=previous.MODEL_SEED,previous.DATA_SEED
EPOCHS,UPDATES,BATCH,DISTILL_WEIGHT=20,4940,256,10.
F_RUN='20261016-phase1-F-supcon-e50-clean-manysig-m9-r02'
F_COMMIT='4df7b94d0fd8ce29056bd8236364e45afa92e07f';F_SEED=2026092701
SOURCE_ALLOWANCE_PP=1.


def configuration():
    value=previous.configuration()
    for key in ('fractions','calibration','source_rationale','matched_comparison'):value.pop(key,None)
    value.update(schema='R17_frozen_defaulthead_task_v4',run=RUN,arms=list(ARMS),GPU=GPUS,parameters=PARAMETERS,
        initialization='OnlyoriginalR17cd977448;step_norm_task=R02step-specificstate/sharedlongweights;activation_task=R03half/source512calibration;no failedstudents',
        loss='10mean3stageIQNMSE_to_originalR17 + KL(T1 originalF50posterior teacherrestored->studentrestored) + halfsquaredunitF160featuredistance;allcoefficientsfixed1;clean detachedenergy/diagnostic only',
        F_knowledge=dict(run=F_RUN,commit=F_COMMIT,seed=F_SEED,original_default_head=True,weights_frozen=True,head_fit_updates=0,
            teacher_reference='same receivedIQ through originalR17,notcleanFtarget',true_TX_labels_read=0,claim='source-supervised F knowledge assisted restoration distillation,notF-freewaveformselfsupervision'),
        calibration_per_arm=dict(step_norm_task=0,activation_task=previous.configuration()['calibration']),
        source_rationale='R02savedsourcepredictions:49/69lowurbanflips/lost20/31 despitewaveguards;48/64flips atteacherconfidence<.9;inputFfidelityhypothesis source-only,not targetfeedback. R03unchanged.',
        source_selection='Healthonly:each7Vaccuracy>=R17-1pp;mean6NMSE<=1.05xR17/cleanNMSE<=.02;qualifiedminfullparameters thenname;onlyoneautomaticfulltarget;sourcehealthisnotperformanceacceptance;targeteach7correctcount>=R17/0pp;no fallback',
        source_screening_allowance_pp=SOURCE_ALLOWANCE_PP,target_acceptance_allowance_pp=0.,
        screening_rationale='V1800 is a small health/mechanism panel;do not equate tiny source count fluctuations with final target performance;newrule only fornewFknowledgecandidate,notretroactiveR01-R03 qualification',
        GPU_queue='WaitR03rootterminal AND actualownedGPUhandlesEXITED;oneown/card4,5;source18GB/target12GB/allowner<2;nohealthyownerintervention',
        CPU_queue='WaitR03rootterminal thenownscore.lock;CPU2threads/max1ownscorer',
        matched_comparison='R02step_norm/R03activation_half are existing matchedE20 source-only controls;onlynewFknowledgeobjectiveadded;no retrospectivetargettestsorstudentinit',
        inference='CurrentIQonly;innerR17compact graph/noFreader or teacher or cache in restoration checkpoint;originalF50 downstream unchanged')
    return value
