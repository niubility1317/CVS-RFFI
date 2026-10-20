from experiments.pamf_unrolled_restore import design as original
from experiments.pamf_restoration.data import SCENES

RUN = '20261020-phase1-restoration-r17-compact-distill-source-s2026092701-m2-r01'
PARENT_RUN = original.RUN
PARENT_COMMIT = 'cd97744884b5e460926e48c8b7efb7d78c6da0f4'
ACTION_RUN = original.ACTION
KIND = original.KIND
ARMS = ('shared_mean', 'factor_half')
GPUS = dict(zip(ARMS, (4, 5)))
GPU_PRECEDING_RUN = '20261020-phase1-restoration-scratch-guidance-source-s2026092701-m2-r01'
CPU_PRECEDING_RUN = '20261020-phase1-restoration-spectral-precision-source-s2026092701-m2-r01'
VIEWS = ('clean', *SCENES)
MODEL_SEED = 2026092701
DATA_SEED = 2026102001
EPOCHS, UPDATES, BATCH = 20, 4940, 256
DISTILL_WEIGHT = 10.
PARAMETERS = {'shared_mean': 2993389, 'factor_half': 3775105}


def configuration():
    return dict(schema='R17_compact_distill_v1', run=RUN, arms=list(ARMS), GPU=GPUS,
        parent_run=PARENT_RUN, parent_commit=PARENT_COMMIT, parent_row='unroll_unshared', parent_epoch=20,
        epochs=EPOCHS, updates=UPDATES, batch=BATCH, model_seed=MODEL_SEED, data_seed=DATA_SEED,
        shared='Elementwise arithmetic mean of three trained R17 priors,one Core9 used in all3steps',
        factor='Each trained prior separately;each Conv1d kernel>1 with beneficial factorization:rank=floor(min(out,in*kernel)/2),balanced CPU SVD;1x1/GN/activations unchanged',
        loss='FinalIQNMSE+10*mean3stageIQNMSE_to_frozenR17;all denominators pairedclean meanIQpower floor1e-4',
        optimizer='AdamW2e-4,warm247/cos2e-5,WD1e-4,clip1,FP32/noTF32',
        data='Original R07 63000 source L6300/U56700;epochs1..20 cachedreceived15%clean/85%residual;no clean concatenation',
        L_exposures=126000, U_exposures=1134000, inherited_prior_epochs=20, cumulative_prior_epochs=40,
        frozen_coarse_epochs=20, frozen_action_epochs=20, cumulative_serial_training_passes=80,
        training_TX_labels_read=0, training_F_access=False, target_training=False, teacher_parameter_updates=0,
        source_selection='Each view accuracy>=originalR17;mean6NMSE<=1.05xR17 and cleanNMSE<=.02;amongqualified smallest fullparameters then name;onlyonefulltarget,no fallback',
        target='SelectedfinalE20 vs reusedoriginalR17/F50seed2701/168000x7;eachviewcorrect_count>=R17',
        parameters=PARAMETERS, inference='CurrentIQonly;threepriors/3stepsx2CG;no teacher/F/clean/TX/RX/scene/otherframes/update',
        GPU_queue='Waitscratchrootterminal AND actualownedGPUhandlesEXITED;oneown/card4,5,max4ownwithspectral6,7;18GBfree/allowner<2',
        CPU_queue='Waitspectralrootterminal;ownscore.lock,CPU2threads/max1ownscorer',
        capacity_lock='.pamf-reception-capacity4.lock', target_feedback=False,
        stop_rule='Technicalfailure:failedrowpreserved,healthyrowscontinue,pendingtargetheld,noautomaticretry;negativeperformanceisnottechnicalfailure')
