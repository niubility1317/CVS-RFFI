from experiments.pamf_r17_parallel_compact import design as previous
RUN='20261020-phase1-restoration-r17-context-compact-s2026092701-m2-r08'
ARMS=('context_half_prior_quarter','context_half_prior_eighth_pointwise')
FRACTIONS=dict(zip(ARMS,(.25,.125)));POINTWISE=dict(zip(ARMS,(False,True)))
GPUS=dict(zip(ARMS,(4,5)));PARAMETERS=dict(zip(ARMS,(1822977,1592461)))
CONTEXT_PARAMETERS=1397619
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=previous.PARENT_RUN,previous.PARENT_COMMIT,previous.ACTION_RUN
F_RUN,F_COMMIT,F_SEED=previous.F_RUN,previous.F_COMMIT,previous.F_SEED
VIEWS,KIND=previous.VIEWS,previous.KIND
MODEL_SEED,DATA_SEED=previous.MODEL_SEED,previous.DATA_SEED
EPOCHS,UPDATES,BATCH,DISTILL_WEIGHT=20,4940,256,10.
GPU_PRECEDING_RUN=SCORING_RUN=previous.SCORING_RUN


def configuration():
    value=previous.configuration()
    value.update(schema='R17_context_compact_v1',run=RUN,arms=list(ARMS),fractions=FRACTIONS,pointwise_weight_sharing=POINTWISE,
        parameters=PARAMETERS,GPU=GPUS,context_parameters=CONTEXT_PARAMETERS,context_factor_fraction=.5,
        rank_rule='Independentcontextlongconvs1/2;priorlongmeanshared1/4or1/8;max(1,floor(fraction*min(Cout,Cin*kernel)));factoronlywhenfewerweights',
        initialization='OnlyoriginalR17source-final;priorCPUmeansharedSVD + independentCPUhalfSVDof84contextlongconvs;optionaloriginalprior1x1meansharing;noR04-R07student',
        source_rationale='Architecture-onlycapacityauditbeforeR06/R07targetscores:originalfrozencontext2157547/lowestR07full2352389;compressdominantcontext,nottargetfeedback',
        teacher_context='OriginalR17teacherusesitsownoriginalcoarse/action/nativevariance;studentusesitsowncompressedcontext;neverteacheronstudentcontext',
        context_updates=0,calibration_IQ_examples=0,teacher_context_forward_examples=1260000,student_context_forward_examples=1260000,
        evaluation='Prospectivefixedtwo-rowafterE20/4940/source7diagnostic then automatic168000x7target/truth-last;allresultsreported',
        GPU_queue='WaitR06rootterminal AND actualownedGPU4/5handlesEXITED withoutreadingscores;oneown/card4/5;source18GB/allowner<2;R07healthy2/3/6/7continues',
        CPU_queue='2threads/max1ownscoreracrossallR17runs/sharedR06score.lock',
        inference='CurrentIQonly;half-factorfrozencoarse/action/nativevarianceestimation+compactprior/3stepsx2CG;noFteacher/clean/TX/RX/scene/cache/stateupdates',
        claim='Newdeterministiccontextweightapproximationplusprior-onlydistillation;notunchangedR17context;source-supervisedFknowledge/exposedbenchmark/no globalminimumproof')
    return value
