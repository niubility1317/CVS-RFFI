from .models import NAMES

RUN_ID='20261019-phase1-restoration-residual-refinement-selfsup-m3-r03'
PARENT_RUN='20261019-phase1-reception-restoration-selfsup-manysig-m6-r02'
PARENT_COMMIT='7efdba92932a86bf4b0492a565d35377564a26da'
COARSE_NAME='tf_unet_mixed'
MODEL_SEED=2026092701
DATA_SEED=2026101901
AUGMENTATION_SEED=2026101902
EVALUATION_SEED=2026101903
EPOCHS=20
BATCH_SIZE=256
LEARNING_RATE=2e-4
WEIGHT_DECAY=1e-4
GPUS=(0,1,5)
SCALE_SAMPLES=8192
NAMESPACE='residual-refinement/train'


def configuration():
    return dict(schema='residual_refinement_v3',run_id=RUN_ID,parent_run=PARENT_RUN,
        parent_commit=PARENT_COMMIT,coarse_candidate=COARSE_NAME,
        parent_selection='Frozen source waveform-only R02 rule A, not target ranking',
        candidates=list(NAMES),model_seed=MODEL_SEED,data_seed=DATA_SEED,
        augmentation_seed=AUGMENTATION_SEED,evaluation_seed=EVALUATION_SEED,
        epochs=EPOCHS,batch_size=BATCH_SIZE,total_updates=4940,updates_per_epoch=247,
        train_count=63000,L_count=6300,U_count=56700,training_TX_labels_read=0,
        L_exposures=126000,U_exposures=1134000,F_training_access=False,
        optimizer='AdamW',lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY,
        warmup_updates=247,final_lr=2e-5,gradient_clip=1.,precision='FP32',EMA=False,
        coarse_updates=0,coarse_evaluation_mode=True,
        fixed_scale=dict(source_only=True,samples=SCALE_SAMPLES,
            selected_seed=DATA_SEED+33,definition='RMS of x-base(r) over first-epoch sampled training observations',
            floor=1e-4,target_or_V_access=False),
        training_routes=dict(clean=.15,residual=.60,full=.25,mode='post_sync',EQ=False),
        training_namespace=NAMESPACE,one_observation_per_record_per_epoch=True,
        objective='IQ NMSE+.1 abs log(power+1e-3); generative velocity MSE+.2 same reconstruction',
        diffusion='Normalized residual v-prediction,64 cosine levels,16 DDIM eta0; condition(r,b) encoded once',
        flow='Normalized residual start=.02*noise; straight paired path,16 Euler; not I2SB',
        output='One IQ=b+fixed_source_scale*estimated_residual; all PAMF views derive this IQ',
        selection_A='Coarse+three correctors: clean NMSE<=.02 then minimum six-residual mean IQ NMSE; fixed order ties; no F/TX access',
        selection_B='Source V TX-label-assisted incremental qualification vs frozen coarse; no labels in training',
        thresholds=dict(lowurban_gain_over_coarse_pp=.5,clean_drop_over_coarse_pp_max=.2,
                        clean_nmse_max=.02,mean_nmse_ratio_vs_coarse_max=1.,stochastic_switch_percent_max=5.),
        prediction='At most2 unique new A/B choices; if coarse selected reuse prior completed predictions; seven-view completion before independent scoring',
        stochastic='One fixed main draw;128 fixed source/target records with2 extra draws, no best-of-N/averaging',
        source_diagnostics='Original residual+same-latent full; encoded-condition shuffle; synchronized batch1',
        target_scope='Original R01 fixed paired1200 input bytes,clean+six practical; not complete formal received IQ',
        GPUs=list(GPUS),max_simultaneous_GPU_workers=3,max_this_run_worker_per_GPU=1,
        data_generation_workers=16,cpu_threads_per_worker=2,
        stop_rule='Nonfinite loss/gradient/output, illegal parent/source/data, collision, invalid predictions or missing inputs: preserve outputs, prevent unstarted rows; healthy active rows finish; no automatic retry',
        expansion='Only source-qualified refinements may expand; no target-feedback tuning or selective retries')
