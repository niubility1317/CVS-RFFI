"""Fixed first falsification matrix. No target-driven candidate changes."""
from .models import NAMES
from .data import SCENES

RUN_ID = '20261019-phase1-pamf-restoration-selfsup-manysig-m7-r01'
MODEL_SEED = 2026092701
DATA_SEED = 2026101901
AUGMENTATION_SEED = 2026101902
EVALUATION_SEED = 2026101903
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
TRAIN_L_PER_SESSION = 420
TRAIN_U_PER_SESSION = 3780
VALIDATION_PER_SESSION = 120
TARGET_PANEL_SIZE = 1200
TARGET_PANEL_SEED = 2026101904


def configuration():
    return dict(schema='pamf_restoration_screen_v1', run_id=RUN_ID, candidates=list(NAMES),
                model_seed=MODEL_SEED, data_seed=DATA_SEED, augmentation_seed=AUGMENTATION_SEED,
                evaluation_seed=EVALUATION_SEED, epochs=EPOCHS, batch_size=BATCH_SIZE,
                optimizer='AdamW', lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                schedule='cosine to zero over all 20 epochs, no early stopping; final checkpoint',
                train_l_per_session=TRAIN_L_PER_SESSION, train_u_per_session=TRAIN_U_PER_SESSION,
                validation_per_session=VALIDATION_PER_SESSION, scenes=list(SCENES),
                clean_identity_probability=.2, trained_TX_labels=0,
                objective='raw IQ NMSE + .2 log FFT MSE '
                          '+ .05 each native frequency/PA/default Sinc/highfreq/reference NMSE',
                diffusion=dict(training_steps=64, cosine_schedule=True, parameterization='v',
                               sampling='16-step DDIM eta=0; one fixed ID-seeded noise draw',
                               loss='velocity MSE + .2 reconstruction objective'),
                flow=dict(path='z0=received+.02*noise; zt=(1-t)*z0+t*clean',
                          objective='velocity MSE + .2 reconstruction objective', sampling='16 fixed Euler steps'),
                stochastic_diagnostics='Diffusion/flow: main draw plus two fixed extra draws on first128 V/target IDs; '
                                       'never selected or averaged; costs reported separately',
                coordinate_policy='Native causal residual with known 24-sample numerical latency; '
                                  'no truth-derived alignment at inference; full-window main NMSE and prefix232 diagnostic',
                source_thresholds=dict(low_urban_accuracy_gain_pp_min=.5, clean_accuracy_drop_pp_max=.2,
                                       raw_iq_mean_ratio_max=1.0, clean_iq_nmse_max=.02),
                selection='Eligible source-V model with highest frozen existing AsF low-urban accuracy; '
                          'if none qualifies, source-best once for explicitly negative confirmation; '
                          'exact tie: fixed candidate order. Never target scores; no expansion after source failure.',
                target_confirmation=dict(selected_only=True, panel_size=TARGET_PANEL_SIZE,
                                         panel_seed=TARGET_PANEL_SEED, views=['clean', *SCENES],
                                         waveform_truth_last=True, TX_truth_reads=0),
                scope='Standalone ground-trained self-supervised reconstruction; '
                      'not a Phase2 deployment method or identity classifier improvement claim',
                restorer_initialization='scratch; frozen recognizers used for evaluation only',
                training_host='N607', GPUs=[0, 1, 5], max_simultaneous_GPU_workers=3,
                max_this_run_worker_per_GPU=1, cpu_threads_per_worker=2, data_generation_workers=8,
                stop_rule='A nonfinite loss/gradient/output, channel/shape/permission error, output conflict '
                          'or checkpoint mismatch stops that row and prevents unstarted rows; '
                          'healthy active rows finish; preserve all outputs; no automatic retry.')
