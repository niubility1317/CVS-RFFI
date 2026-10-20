from experiments.pamf_restoration.data import SCENES

RUN='20261020-phase1-restoration-scratch-guidance-source-s2026092701-m2-r01'
ARMS=('scratch_wave','scratch_F_guided');GPUS=dict(zip(ARMS,(4,5)))
MODEL_SEED=2026092701;DATA_SEED=2026101951
TEACHER_SEEDS=(2026092701,2026092702,2026092703)
F_COMMIT='4df7b94d0fd8ce29056bd8236364e45afa92e07f'
EPOCHS=40;UPDATES_PER_EPOCH=247;UPDATES=9880;BATCH=256
TRAINABLE_PARAMETERS=1305958
GPU_PRECEDING_RUN='20261020-phase1-restoration-teacher-agreement-source-s2026092701-m2-r01'
PRECEDING_RUN='20261020-phase1-restoration-spatial-conditioning-source-s2026092701-m2-r01'


def cache_epoch(epoch):
    if not isinstance(epoch,int) or isinstance(epoch,bool) or not 1<=epoch<=EPOCHS:
        raise ValueError('Registered scratch epoch1..40 required')
    return 1+(epoch-1)%20


def target_matrix():
    return dict(policy='FIXED_COMPARISON',arms=list(ARMS),views=['clean',*SCENES],count=168000,
        F_seeds=list(TEACHER_SEEDS),source_selection=False,target_feedback=False,within_run_control=ARMS[0])


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered from-scratch arm required')
    return dict(schema='scratch_frozenF_guidance_v1',run_id=RUN,arm=arm,GPU=GPUS[arm],
        architecture='Original ComplexTFUNet;bounded widely-linear spectrum plus IQ UNet',
        initialization='scratch',restoration_parent=None,trainable_parameters=TRAINABLE_PARAMETERS,
        model_seed=MODEL_SEED,data_seed=DATA_SEED,epochs=EPOCHS,updates=UPDATES,batch_physical=BATCH,
        source_count=63000,source_roles=['L_s','U_s'],TX_labels_read=0,target_training=False,
        source_cache='Original R02 L6300/U56700;physical IDs and role arrays reused unchanged',
        cache_epochs=[cache_epoch(e) for e in range(1,EPOCHS+1)],
        route='Original quantile .15clean/.60residual/.25full;observed+samephysicalclean per batch;shufflecontinues across40epochs',
        L_physical_exposures=252000,U_physical_exposures=2268000,IQ_examples=5040000,
        optimizer='AdamW2e-4,warm247/cos2e-5,WD1e-4,clip1,FP32/noTF32',
        wave_loss='Paired IQNMSE+.1abslogFFTpower',
        guidance_enabled=arm==ARMS[1],posterior_weight=.3,
        guidance='Mean3originalF finalcos+.3T2KL(T2) fromfirststep;per_frame_cap' if arm==ARMS[1] else 'NONE;no training teacher cache/forwards',
        F_seeds=list(TEACHER_SEEDS),F_commit=F_COMMIT,F_optimizer_updates=0,head_fit_updates=0,
        source_V='1800physical diagnostic only;fixedfinalE40,noqualification/ranking/earlystop/stateupdates',
        target_matrix=target_matrix(),inference='CurrentreceivedIQ only;noF/clean/cache/metadata/otherframes/vote/update',
        capacity_lock='receiver_residual_capacity4.lock',min_free_MB=18000,target_min_free_MB=12000,
        GPU_queue='Wait agreement rootterminal AND all actual owned source/target handles EXITED;scores_read0',
        CPU_queue='Wait spatial rootterminal;oneCPUscorer via shared runscore.lock;twoCPUthreads',
        matched_comparison='Same newrandomstate/architecture/40epochs/data/order/routes/optimizer;onlyFguidance differs;oldR03/40+10notmatched',
        source_hypothesis='OriginalR03 learned waveform objective before laterFrefinement;testFpreservation during inverse-map learning,not promotion of prior target winner')
