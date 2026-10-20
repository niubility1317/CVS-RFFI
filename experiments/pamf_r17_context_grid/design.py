from experiments.pamf_r17_clean_corrective import design as previous
RUN='20261020-phase1-restoration-r17-context-grid-s2026092701-m4-r11'
ARMS=('full_context_quarter','full_context_eighth_pointwise','quarter_context_quarter','quarter_context_eighth_pointwise')
FRACTIONS=dict(zip(ARMS,(.25,.125,.25,.125)));POINTWISE=dict(zip(ARMS,(False,True,False,True)));CLEAN_OPT=dict.fromkeys(ARMS,True)
CONTEXT_FRACTION=dict(zip(ARMS,(1.,1.,.25,.25)));CONTEXT_PARAMETERS=dict(zip(ARMS,(2157547,2157547,787455,787455)))
GPUS=dict(zip(ARMS,(2,3,6,7)));PARAMETERS=dict(zip(ARMS,(2582905,2352389,1212813,982297)))
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=previous.PARENT_RUN,previous.PARENT_COMMIT,previous.ACTION_RUN
F_RUN,F_COMMIT,F_SEED=previous.F_RUN,previous.F_COMMIT,previous.F_SEED
VIEWS,KIND=previous.VIEWS,previous.KIND;MODEL_SEED,DATA_SEED=previous.MODEL_SEED,previous.DATA_SEED
EPOCHS,UPDATES,BATCH,DISTILL_WEIGHT=20,4940,256,0.
GPU_PRECEDING_RUN='20261020-phase1-restoration-r17-joint-optimize-s2026092701-m4-r09'
SCORING_RUN=previous.SCORING_RUN


def configuration():
    value=previous.configuration()
    value.update(schema='R17_context_grid_v1',run=RUN,arms=list(ARMS),fractions=FRACTIONS,pointwise_weight_sharing=POINTWISE,
        parameters=PARAMETERS,GPU=GPUS,clean_objective=CLEAN_OPT,context_fraction=CONTEXT_FRACTION,context_parameters=CONTEXT_PARAMETERS,
        source_rationale='Matchedsourcecontextcompressionstudy:fulloriginal/.5/.25costsauditedbeforeR08targetswithoutIQ;R10supplieshalfcontextcells;holdtwofixedpriors/cleanobjective/sourcebudgetconstant;no rank/loss/budgetselectionfromtargetscores',
        initialization='Originalsource-selectedR17only;unmodifiedfullcontextorCPUquarterrankSVDof84contextlongconvs;fixedquarter/eighthpointwiseprior;no target-testedstudentweights',
        optimization='Jointallcompactcoarse/action/prior;originalR17CPUinitonly/deletedbeforeoptimizer;frozenoriginalF50/defaulthead;pairedsourcecleanonly',
        context_initialization='Fullfraction1copiesoriginalcontextbitwisebeforetraining;fraction.25datafreeCPUbalancedSVD;noIQcalibration;source40percomponentafterjointE20',
        evaluation='Fixed4cells:2contextlevelsx2priorstructures;R10halfcontextisasalreadyregisteredmatchedcontrol;everyrowfinalE20/sourceonlydiagnostic/168000x7target/truthlast/exactoriginalR17eachview1ppfloor/actualgain/lowurban70',
        GPU_queue='WaitR09rootterminal AND allactualownedGPU2/3/6/7handlesEXITED withoutreadingscores;oneown/card;R10healthyGPU4/5continue;6ownGPUmax',
        inference='CurrentreceivedIQonly;sourcejointoptimizedfull-or-quartercontext/compactprior/3stepsx2CG;noFteacher/clean/TX/RX/scene/cache/stateupdate',
        claim='Exposedbenchmarkmatchedcontextcompressionablation;sourcepairedwaveform/Fsourceknowledgeassisted;notstrictlyunsupervised/F-free/newblindtest/globalminimum/Phase2;nottargetadaptation')
    return value
