"""Second fixed source-driven study, incorporating the two supplied reports."""
from .models import NAMES

RUN_ID='20261019-phase1-reception-restoration-selfsup-manysig-m6-r02'
MODEL_SEED=2026092701
DATA_SEED=2026101901
AUGMENTATION_SEED=2026101902
EVALUATION_SEED=2026101903
EPOCHS=20
BATCH_SIZE=256
LEARNING_RATE=2e-4
WEIGHT_DECAY=1e-4
GPUS=(0,1,5)


def configuration():
    return dict(schema='reception_restorer_v2',run_id=RUN_ID,candidates=list(NAMES),
        model_seed=MODEL_SEED,data_seed=DATA_SEED,augmentation_seed=AUGMENTATION_SEED,
        evaluation_seed=EVALUATION_SEED,epochs=EPOCHS,batch_size=BATCH_SIZE,
        optimizer='AdamW',lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY,
        warmup_updates=247,schedule='247 warmup updates then cosine to 2e-5 over remaining updates',
        train_count=63000,updates_per_epoch=247,total_updates=4940,
        training_TX_label_reads=0,F_used_in_training=False,
        clean_fraction=.15,mixed_residual_fraction=.60,mixed_full_fraction=.25,
        control_nonclean_fraction=.85,
        channel='native residual/full post_sync noEQ fs25MHz fc2462MHz boundary reflect referenceSNR30',
        channel_schedule='fresh namespace every epoch; paired initial latent for two routes; same single observation quota',
        objective='IQ NMSE + .1 mean absolute log(power+1e-3) FFT difference; no F/PL/native learned features',
        generative_objective='v/velocity MSE + .2 reconstruction objective',
        precision='FP32',gradient_clip=1.,EMA=False,
        diffusion='64 cosine noise levels, v-prediction, 16 fixed DDIM steps eta0, multiscale received condition',
        flow='received+.02*noise; straight paired path; 16 Euler steps; not I2SB',
        source_selection=dict(
            waveform='No TX/F inputs: clean NMSE<=.02 then minimum mean raw IQ NMSE on original six residual sourceV views; fixed row order tie; if none minimum clean NMSE',
            identity_assisted='Separately disclosed labeled sourceV selection: best AsF lowurban among qualified rows; no label loss in restorer training',
            thresholds=dict(lowurban_gain_pp=.5,clean_drop_pp_max=.2,clean_nmse_max=.02,raw_mean_ratio_max=1.)),
        prediction='One fixed ID-seeded draw; first128 source/target IDs have two extra stochastic draws, never selected/averaged',
        condition_shuffle='Source-only fixed derangement; separate complete sampling with same noise, no selection or state update',
        source_full_diagnostic='Same initial latent as original source residual views; correlated cross-implementation control',
        target='At most two unique preregistered source selections; reuse original fixed1200 panel received bytes, clean plus six residual views; independently score only after all row predictions complete',
        GPUs=list(GPUS),max_simultaneous_GPU_workers=3,max_this_run_worker_per_GPU=1,
        cpu_threads_per_worker=2,data_generation_workers=16,
        stop_rule='Nonfinite loss/gradient/output, mismatched source/latent pair, missing input, output collision, checkpoint mismatch or invalid prediction coverage: stop row/prevent unstarted rows; preserve healthy active rows; no automatic retry',
        expansion='No extra epochs/seeds/full target matrix unless source-qualified; no target-driven parameter or selection changes',
        deferred=['physics-unrolled effective-operator validation','coarse+generative residual with deterministic correction control',
                  'labeled identity fine-tuning','joint F fine-tuning with matched As control','full/pre_sync','independent physical channel family'])
