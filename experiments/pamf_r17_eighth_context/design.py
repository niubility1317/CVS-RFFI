from experiments.pamf_r17_context_grid import design as previous
RUN='20261020-phase1-restoration-r17-eighth-context-s2026092701-m2-r12'
ARMS=('eighth_context_quarter','eighth_context_eighth_pointwise')
FRACTIONS=dict(zip(ARMS,(.25,.125)));POINTWISE=dict(zip(ARMS,(False,True)));CLEAN_OPT=dict.fromkeys(ARMS,True)
CONTEXT_FRACTION=dict.fromkeys(ARMS,.125);CONTEXT_PARAMETERS=dict.fromkeys(ARMS,482299)
GPUS=dict(zip(ARMS,(4,5)));PARAMETERS=dict(zip(ARMS,(907657,677141)))
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=previous.PARENT_RUN,previous.PARENT_COMMIT,previous.ACTION_RUN
F_RUN,F_COMMIT,F_SEED=previous.F_RUN,previous.F_COMMIT,previous.F_SEED
VIEWS,KIND=previous.VIEWS,previous.KIND;MODEL_SEED,DATA_SEED=previous.MODEL_SEED,previous.DATA_SEED
EPOCHS,UPDATES,BATCH,DISTILL_WEIGHT=20,4940,256,0.
GPU_PRECEDING_RUN='20261020-phase1-restoration-r17-clean-corrective-s2026092701-m2-r10'
SCORING_RUN=previous.SCORING_RUN


def configuration():
    value=previous.configuration()
    value.update(schema='R17_eighth_context_v1',run=RUN,arms=list(ARMS),fractions=FRACTIONS,pointwise_weight_sharing=POINTWISE,
        parameters=PARAMETERS,GPU=GPUS,clean_objective=CLEAN_OPT,context_fraction=CONTEXT_FRACTION,context_parameters=CONTEXT_PARAMETERS,
        source_rationale='Completepreauditedcontextcapacitylevel.125withthesametwofixedpriors/directsourcecleanobjective/sourcebudget;staticcostspreR08targets/noIQ;R10halfandR11full/quartermatchedcontrols;notargetscoreselectedrank/loss/budget',
        initialization='Originalsource-selectedR17only;CPUeighthrankSVDof84contextlongconvs;fixedquarter/eighthpointwiseprior;no target-testedstudentweights',
        optimization='Jointallcompactcoarse/action/prior;originalR17CPUinitonly/deletedbeforeoptimizer;frozenoriginalF50/defaulthead;pairedsourcecleanonly',
        context_initialization='Fraction.125datafreeCPUbalancedSVD;noIQcalibration;source40percomponentafterjointE20',
        evaluation='Fixed2cells:eighthcontextxtwofixedpriorstructures;R10half/R11full-quarterasregisteredmatchedcontrols;everyrowfinalE20/sourceonlydiagnostic/168000x7target/truthlast/exactoriginalR17eachview1ppfloor/actualgain/lowurban70',
        GPU_queue='WaitR10rootterminal AND allactualownedGPU4/5handlesEXITED withoutreadingscores;oneown/card;R11healthyGPU2/3/6/7continue;6ownGPUmax',
        stop_rule='Contract/nonfinite/OOM/technicalerrorstopsaffectedrow,preservesoutputs;healthyrowcontinues;nosilentfallback/noautomaticretry;rootFAILEDafterhealthyrowsfinishifanyfailure',
        inference='CurrentreceivedIQonly;sourcejointoptimizedeighthcontext/compactprior/3stepsx2CG;noFteacher/clean/TX/RX/scene/cache/stateupdate',
        claim='Exposedbenchmarkmatchedcontextcompressionablation;sourcepairedwaveform/Fsourceknowledgeassisted;notstrictlyunsupervised/F-free/newblindtest/globalminimum/Phase2;nottargetadaptation')
    return value
