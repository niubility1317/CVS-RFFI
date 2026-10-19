from experiments.pamf_iterative_restore import design as previous

RUN='20261020-phase1-restoration-alternating-tf-selfsup-s2026092701-m2-r19'
UNROLLED=previous.UNROLLED
UNROLLED_COMMIT=previous.UNROLLED_COMMIT
REFERENCE_CANDIDATE=previous.REFERENCE_CANDIDATE
SINGLE=previous.SINGLE
SINGLE_COMMIT=previous.SINGLE_COMMIT
FFT=previous.FFT
FFT_COMMIT=previous.FFT_COMMIT
ACTION=previous.ACTION
REFERENCE=previous.REFERENCE
KIND=previous.KIND
MODEL_SEED=previous.MODEL_SEED
DATA_SEED=previous.DATA_SEED
GPUS=previous.GPUS
AUX_WIDTHS=(24,48,72,96)
TRAINABLE_PARAMETERS=3926220
ROWS=('time_time','time_frequency')
ALL_ROWS=ROWS


def configuration():
    value=previous.configuration()
    value.update(schema='alternating_TF_v1',run=RUN,rows=list(ROWS),CG_per_step=2,
        candidate_designs={name:dict(auxiliary_representation='frequency' if name=='time_frequency' else 'time',
            auxiliary_incoming=17,auxiliary_widths=list(AUX_WIDTHS),CG_schedule=[2,2,2]) for name in ROWS},
        prior_anchor='coarse+SCALE*temporal9,then+SCALE*auxiliary17(inverseFFT forfrequency)',
        prior_architecture='Three unshared stages,eachoriginalCore9 plusCore17 widths24/48/72/96',
        prior_Core_calls=6,shared_initialization='TemporalweightsmatchR17unsharedE0;newauxclonedperstep;fullnewstate matchedbetweenrows;allinitialoutputs matchR17E0',
        trainable_parameters_per_row=TRAINABLE_PARAMETERS,
        parameter_control='3926220 trainable parameters each,sixCorecalls,sixCGiterations;FFT/mirror transform costs differ',
        representation='Aux:fourcomplexsignals(received,coarse,u,adjointinnovationatu),theirconjugates ornegativefrequencyconjugatemirrors,andlognativevariance',
        FFT_normalization='ortho',mirror_index='(-k)%256',new_auxiliary_initialization='zerooutputprojection,otherweightsnormalnativeCoreinit')
    return value
