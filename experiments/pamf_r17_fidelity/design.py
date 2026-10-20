from experiments.pamf_r17_compact import design as previous

RUN='20261020-phase1-restoration-r17-fidelity-source-s2026092701-m2-r02'
PARENT_RUN=previous.PARENT_RUN
PARENT_COMMIT=previous.PARENT_COMMIT
ACTION_RUN=previous.ACTION_RUN
KIND=previous.KIND
ARMS=('step_norm_fidelity','factor_half_fidelity')
GPUS=dict(zip(ARMS,(4,5)))
GPU_PRECEDING_RUN=previous.RUN
CPU_PRECEDING_RUN=previous.RUN
VIEWS=previous.VIEWS
MODEL_SEED=previous.MODEL_SEED
DATA_SEED=previous.DATA_SEED
EPOCHS,UPDATES,BATCH=20,4940,256
DISTILL_WEIGHT=10.
PARAMETERS={'step_norm_fidelity':3118001,'factor_half_fidelity':3775105}


def configuration():
    value=previous.configuration()
    for key in ('shared','factor'):value.pop(key,None)
    value.update(schema='R17_pure_function_fidelity_v2',run=RUN,arms=list(ARMS),GPU=GPUS,parameters=PARAMETERS,
        step_norm='ThreeCore9;only28longConv1d weights shared asmean3originalR17weights;allconvbias/GN/1x1andotherparameters originalstep-specific',
        factor='Same independentthree half-rankSVDpriors asR01factor_half;only distillationobjective changes',
        loss='10*mean3stageIQNMSE_to_frozenR17;sourceclean energyfloor1e-4 denominator detached;pairedclean IQNMSE loggeddiagnostic only,not optimized',
        source_rationale='R01bothwaveformguardsPASSbutsource lowurban shared-.8889pp/factor-.5pp;0newtarget. Testpurefunction fidelity andretainedstage-specificnormalization/bias/shortkernels,nottargetfeedback.',
        initialization='Onlyoriginalsource-selectedR17cd977448E20/4940;noneR01orfailed/refinedstudentweights',
        GPU_queue='WaitR01rootterminal AND actualownedGPUhandlesEXITED;oneown/card4,5;source18GB/target12GB/allowner<2;nohealthyownerintervention',
        CPU_queue='WaitR01rootterminal thenownscore.lock;CPU2threads/max1ownscorer',
        inference='CurrentIQonly;3stepsx2CG;internalfixedstep-specificparameters;noF/teacher/clean/role/RX/scene/otherframes/update',
        matched_comparison='WithinR02samepureteacherloss/data/order/E20;architecturediffers. R01factor_half is matchedprior/source-onlyloss reference,notadditionalnewrow.')
    return value
