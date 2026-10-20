from experiments.pamf_identity_priority.design import configuration as base_configuration
from experiments.pamf_identity_priority.design import PARENT_RUN,PARENT_COMMIT,TEACHER_SEEDS,MODEL_SEED,DATA_SEED,EPOCHS,UPDATES,BATCH
from experiments.pamf_restoration.data import SCENES

RUN='20261020-phase1-restoration-spectral-precision-source-s2026092701-m2-r01'
ARMS=('spectral_uniform','spectral_whitened');GPUS=dict(zip(ARMS,(6,7)))
PRECEDING_RUN='20261020-phase1-restoration-scratch-guidance-source-s2026092701-m2-r01'
GPU_PRECEDING_RUN='20261020-phase1-restoration-spatial-conditioning-source-s2026092701-m2-r01'
SPECTRAL_WEIGHT=.1;PSD_FLOOR=.01;MIN_PRECISION=.25;MAX_PRECISION=4.


def target_matrix():
    return dict(policy='FIXED_COMPARISON',arms=list(ARMS),views=['clean',*SCENES],count=168000,
        F_seeds=list(TEACHER_SEEDS),source_selection=False,target_feedback=False,within_run_control=ARMS[0])


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered complex spectral precision arm required')
    value=base_configuration('posterior03')
    value.update(schema='source_spectral_precision_v1',run_id=RUN,arm=arm,GPU=GPUS[arm],trainable_parameters=2140840,
        loss='PairedIQNMSE+.1logpower+.1complexspectralNMSE;mean3Ffinalcos+.3T2KL/per_frame_cap',
        spectral_rule='Uniform256weights1' if arm==ARMS[0] else 'clip(lambda/(normalizedsourcePSD+.01),.25,4);64stepbisectionlambda for meanweight1',
        spectral_weight=SPECTRAL_WEIGHT,PSD_floor=PSD_FLOOR,precision_bounds=[MIN_PRECISION,MAX_PRECISION],
        PSD_source='All63000sourceL/Uclean;globalmeanFFTpower,normalizedmean1;noV/TX/RX/scene/target;fixedbeforeoptimizer',
        spectral_NMSE='Mean w*|orthonormalFFT(estimate-clean)|2 / mean|complexclean|2 floor2e-4;uniformequalsIQNMSE',
        gradient_rule='wave pairedpluscomplexspectral and cappedmean3F;tasksumIQgradients dividedbyactualIQcountonce',
        source_role='FixedfinalE10/2470;sourceVdiagnostic only,noqualification/ranking/earlystop/stateupdate',
        source_rule='NONE;sourceV diagnostic only',target_scope='Bothfixedarms168000clean+sixpractical/3nativeheads',
        checkpoint_selection='fixed_final_E10',target_matrix=target_matrix(),
        control_run=RUN,control_arm=ARMS[0],control_reuse='Newmatcheduniformcomplexspectralcontrol;oldperframecap lacks extra.1term and is historyonly',
        optimizer='ResetAdamW3e-5,warm247/cos3e-6,WD1e-4,clip1,FP32/noTF32',
        inference='CurrentIQonly;noPSD/weights/F/clean/cache/trueTX/RX/scene/otherframes/vote/update',
        GPU_queue='Waitspatialrootterminal AND actualownedsource/targethandlesEXITED;scores0;oneown/card,max4ownGPU including scratch4/5',
        CPU_queue='Waitscratchrootterminal thenownscorelock;CPU2threads/oneownscorer',
        target_selection='NONE;bothfinalE10 automaticallyfulltarget',
        matched_comparison='Only two newmatchedarms differ uniform versus boundedinverseglobalcleanPSD;nofailed/refinedweightinitialization or targetfeedback')
    return value
