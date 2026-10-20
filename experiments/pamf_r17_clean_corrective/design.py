from experiments.pamf_r17_joint_optimize import design as previous
RUN='20261020-phase1-restoration-r17-clean-corrective-s2026092701-m2-r10'
ARMS=('clean_only_quarter','clean_only_eighth_pointwise')
FRACTIONS=dict(zip(ARMS,(.25,.125)));POINTWISE=dict(zip(ARMS,(False,True)));CLEAN_OPT=dict.fromkeys(ARMS,True)
GPUS=dict(zip(ARMS,(4,5)));PARAMETERS=dict(zip(ARMS,(1822977,1592461)));CONTEXT_PARAMETERS=previous.CONTEXT_PARAMETERS
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=previous.PARENT_RUN,previous.PARENT_COMMIT,previous.ACTION_RUN
F_RUN,F_COMMIT,F_SEED=previous.F_RUN,previous.F_COMMIT,previous.F_SEED
VIEWS,KIND=previous.VIEWS,previous.KIND;MODEL_SEED,DATA_SEED=previous.MODEL_SEED,previous.DATA_SEED
EPOCHS,UPDATES,BATCH,DISTILL_WEIGHT=20,4940,256,0.
GPU_PRECEDING_RUN='20261020-phase1-restoration-r17-context-compact-s2026092701-m2-r08'
SCORING_RUN=previous.SCORING_RUN


def configuration():
    value=previous.configuration()
    value.update(schema='R17_clean_corrective_v1',run=RUN,arms=list(ARMS),fractions=FRACTIONS,pointwise_weight_sharing=POINTWISE,
        parameters=PARAMETERS,GPU=GPUS,clean_objective=CLEAN_OPT,waveform_teacher_weight=0.,R17_training_forward_examples=0,
        source_rationale='Userrequirescompressionandactualoptimization;legalsourcepairedcleanexists;matchedR09architecturesandbudget;removeteacherwaveformimitationasindependentfixedablation;notderivedfromR08/R09targetresults',
        initialization='Originalsource-selectedR17only;identicalR08/R09CPUcontext-half/prior-quarter-or-eighthSVD;no testedstudentinit;originalR17notusedfortrainingforwards',
        loss='FinalpairedsourcecleanIQNMSE(clampdetachedmeanenergy1e-4)+T1KL(Fclean||Frestored)+0.5unitF160distance_squared;coefficients1/1/0.5;noR17outputdistillation',
        optimization='Allcompactcoarse/action/priorjointlytrained;originalF50/defaultheadfrozen;0newTX/V/targetlabels;sourcepairedcleanonly;R17CPUinitobjectdeletedbeforeoptimizer',
        teacher_context_forward_examples=0,R17_teacher_forward_examples=0,student_context_forward_examples=1260000,
        F_teacher_forward_examples=1260000,F_student_forward_examples=1260000,
        budget='Inheritedcoarse20/action20/R17prior20+onejointE20=80serialpasses;eachcomponent40/component-epochsum120;63000sourcex20/4940;R17teachertrainingforwards0;Fclean/Fstudentcostseparate',
        evaluation='Fixedtwoarchitectures;finalE20/sourceVdiagnosticonly;eachrowautomatic168000x7formalprediction/truthlast;originalR17eachview1ppfloor/actualgain/lowurban70report',
        GPU_queue='WaitR08rootterminal AND allactualownedGPU4/5handlesEXITED withoutreadingscores;oneown/card;R09healthy2/3/6/7continue;6ownGPUmax',
        inference='CurrentreceivedIQonly;sourcejointoptimizedcompactcoarse/action/prior/3stepsx2CG;noFteacher/clean/TX/RX/scene/cache/stateupdate',
        claim='Pairedsourcecleanwaveform/Fsourceknowledgeassistedcompactoptimization;notstrictlyunsupervised/F-free;notnewblindtest/globalminimum/Phase2;no targetfeedback')
    return value
