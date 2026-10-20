from experiments.pamf_r17_fidelity import design as previous
RUN='20261020-phase1-restoration-r17-calibrated-source-s2026092701-m2-r03'
PARENT_RUN,PARENT_COMMIT=previous.PARENT_RUN,previous.PARENT_COMMIT
ACTION_RUN,KIND=previous.ACTION_RUN,previous.KIND
ARMS=('activation_half','activation_two_thirds')
GPUS=dict(zip(ARMS,(4,5)))
FRACTIONS=dict(zip(ARMS,(.5,2/3)))
PARAMETERS=dict(zip(ARMS,(3775105,4241809)))
GPU_PRECEDING_RUN=CPU_PRECEDING_RUN=previous.RUN
VIEWS=previous.VIEWS
MODEL_SEED,DATA_SEED=previous.MODEL_SEED,previous.DATA_SEED
EPOCHS,UPDATES,BATCH,DISTILL_WEIGHT=20,4940,256,10.
CALIBRATION_COUNT,CALIBRATION_SEED,CALIBRATION_BATCH=512,2026102002,16
CALIBRATION_POSITIONS,RIDGE_FRACTION=8,1e-3


def configuration():
    value=previous.configuration()
    for key in ('step_norm','factor','source_rationale','matched_comparison'):value.pop(key,None)
    value.update(schema='R17_activation_calibrated_v3',run=RUN,arms=list(ARMS),GPU=GPUS,parameters=PARAMETERS,
        fractions=FRACTIONS,calibration=dict(examples=CALIBRATION_COUNT,seed=CALIBRATION_SEED,batch=CALIBRATION_BATCH,
            positions=CALIBRATION_POSITIONS,ridge_fraction=RIDGE_FRACTION,source='R07cache/trainroles+epochs/01/received.npy only',
            sampling='first512ofseededpermutation63000;8evenlyspacedoutputpositionsperlayer/example;uncenteredsecondmoment',
            math='SVD(W*Cholesky(C+ridgeI));inversewhitenleftfactor;ridge=max(1e-8,1e-3*mean(diagC));bias/GN/1x1unchanged',
            budget='512additionalL/Uforwardexposuresandteacherexamplesperarm;0optimizerupdates;separatecostandactualL/Ucounts'),
        initialization='OnlyoriginalR17cd977448E20/4940;source-activationweightedfactorinit;noR01/R02students',
        source_rationale='R01source-negative;originalkernelhalferror.52557 and99percentenergyfactorssave0;weightenergyisnotactivationerror. Fixedbeforeanycompressedtargetresult;R02liveunchanged.',
        GPU_queue='WaitR02rootterminal AND actualownedGPUhandlesEXITED;oneown/card4,5;source18GB/target12GB/allowner<2;nohealthyownerintervention',
        CPU_queue='WaitR02rootterminal thenownscore.lock;CPU2threads/max1ownscorer',
        matched_comparison='Same512calibrationIQ/positions/ridge/pureteacherloss/data/order/E20;onlyrankfractiondiffers;R02kernelhalf is source-only initialization reference,notnewrow',
        inference='CurrentIQonly;3independentpriors/3stepsx2CG;no calibration/cache/teacher/F/clean/role/RX/scene/otherframes/update')
    return value
