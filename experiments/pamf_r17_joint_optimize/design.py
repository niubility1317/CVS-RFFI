from experiments.pamf_r17_context_compact import design as previous
RUN='20261020-phase1-restoration-r17-joint-optimize-s2026092701-m4-r09'
ARMS=('joint_teacher_quarter','joint_clean_quarter','joint_teacher_eighth_pointwise','joint_clean_eighth_pointwise')
FRACTIONS=dict(zip(ARMS,(.25,.25,.125,.125)));POINTWISE=dict(zip(ARMS,(False,False,True,True)))
CLEAN_OPT=dict(zip(ARMS,(False,True,False,True)))
GPUS=dict(zip(ARMS,(2,3,6,7)));PARAMETERS=dict(zip(ARMS,(1822977,1822977,1592461,1592461)))
CONTEXT_PARAMETERS=previous.CONTEXT_PARAMETERS
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=previous.PARENT_RUN,previous.PARENT_COMMIT,previous.ACTION_RUN
F_RUN,F_COMMIT,F_SEED=previous.F_RUN,previous.F_COMMIT,previous.F_SEED
VIEWS,KIND=previous.VIEWS,previous.KIND
MODEL_SEED,DATA_SEED=previous.MODEL_SEED,previous.DATA_SEED
EPOCHS,UPDATES,BATCH,DISTILL_WEIGHT=20,4940,256,10.
GPU_PRECEDING_RUN='20261020-phase1-restoration-r17-parallel-compact-s2026092701-m4-r07'
SCORING_RUN=previous.SCORING_RUN


def configuration():
    value=previous.configuration()
    for key in ('frozen_coarse_epochs','frozen_action_epochs'):value.pop(key,None)
    value.update(schema='R17_joint_optimize_v1',run=RUN,arms=list(ARMS),fractions=FRACTIONS,pointwise_weight_sharing=POINTWISE,
        parameters=PARAMETERS,GPU=GPUS,context_updates=4940,context_trainable=True,clean_objective=CLEAN_OPT,
        source_rationale='Latestuserrequiresoptimizationduringcompression;sourcecodeauditfindscontextfrozenandR17-outputtasktarget;sourcepairedcleanavailable;notrank/loss/budgettuningfromtargetscores',
        initialization='OriginalR17source-finalonly;identicalR08CPUcontext-half/prior-quarter-or-eighthSVD;noR04-R08studentweights',
        loss='Allrows10mean3stageNMSE_to_originalR17;teacherrows+T1KL/halfunitF160toR17restored;cleanrows+finalpairedcleanIQNMSE+T1KL/halfunitF160tosourcepairedclean;coefficients10/1/1fixed',
        optimization='Jointallcompactcoarse/action/priorparameters;teacherandoriginalF50/defaultheadfrozen;no newTX/V/targetlabels;sourcecleanonlycorrectivearms',
        context_updates_per_row=4940,coarse_updates=4940,action_updates=4940,prior_updates=4940,
        cumulative_coarse_epochs=40,cumulative_action_epochs=40,cumulative_prior_epochs=40,cumulative_component_epoch_sum=120,
        inherited_coarse_epochs=20,inherited_action_epochs=20,
        cumulative_serial_training_passes=80,budget='Inheritedcoarse20/action20/R17prior20 plusonejointE20;80serialpasses;eachcomponent40;source63000x20/4940;extraF/teacherforwardsseparate',
        evaluation='Fixed2architecturesx2objectives;sourceonlydiagnostic;eachrowfull168000x7targetafterE20;truthlast/eachview<=1ppplusactualgainandlowurban70report',
        GPU_queue='WaitR07rootterminal AND allactualownedGPU2/3/6/7handlesEXITED withoutreadingscores;oneown/card;R08healthy4/5continues;6ownGPUmax',
        inference='CurrentIQonly;source-jointoptimizedhalf-factorcoarse/action+compactprior/3stepsx2CG;noFteacher/clean/TX/RX/scene/cache/test-stateupdates',
        claim='CompressedrestorerjointoptimizationandpairedsourcecleanFknowledge;notmerelyteacher-preservingcompression;no newhead/targetadaptation/newblindtest/globalminimumproof')
    return value
