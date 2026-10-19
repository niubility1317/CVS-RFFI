from experiments.pamf_unrolled_restore import design as unrolled
from experiments.pamf_joint_coarse_restore import design as control

RUN='20261020-phase1-restoration-fullresolution-priors-selfsup-s2026092701-m2-r24'
ROWS=('tcn_prior','attention_prior')
ALL_ROWS=ROWS
KIND=unrolled.KIND
ACTION=unrolled.ACTION
REFERENCE=unrolled.REFERENCE
MODEL_SEED=unrolled.MODEL_SEED
DATA_SEED=unrolled.DATA_SEED
GPUS=unrolled.GPUS
SINGLE=unrolled.SINGLE
SINGLE_COMMIT=unrolled.SINGLE_COMMIT
FFT=unrolled.FFT
FFT_COMMIT=unrolled.FFT_COMMIT
UNROLLED=control.UNROLLED
UNROLLED_COMMIT=control.UNROLLED_COMMIT
REFERENCE_CANDIDATE=control.REFERENCE_CANDIDATE
ALTERNATING=control.ALTERNATING
ALTERNATING_COMMIT=control.ALTERNATING_COMMIT
ALTERNATING_CANDIDATE=control.ALTERNATING_CANDIDATE
CONTROL_RUN=control.RUN
CONTROL_COMMIT='d43dbab326d0b1e2cb6fb7faa23dc9a3ec25a025'
CONTROL_CANDIDATE='prior_only'
PRECEDING=CONTROL_RUN
PRECEDING_COMMIT=CONTROL_COMMIT


def configuration():
    value=unrolled.configuration()
    value.update(schema='fullresolution_priors_v1',run=RUN,rows=list(ROWS),
        prior_architecture='Three unshared full256 TCN or attention priors with zero output heads',
        prior_initialization='Scratch;only original source R02/R07 context inherited;R23 is evaluation control only',
        shared_initialization='Equal complete initial output;different hidden architectures and parameter counts',
        tcn=dict(incoming=9,width=128,stem_kernel=5,blocks=6,dilations=[1,2,4,8,16,32],full_length=256,
            nominal_convolutional_receptive_field=269,normalization='GroupNorm4 within current packet'),
        attention=dict(incoming=9,width=128,heads=4,blocks=6,tokens=256,MLP_expansion=2,local_kernel=5,
            position='Fixed0.1sinusoidal',residual_scale=.5,dropout=0,activation_checkpoint='nonreentrant per block;training only'),
        source_selection='All native/R15/R16/R17/R19 and reused R23control guards;rank lowmidmean,lowestTX,lowestRX,lexicalname',
        source_control_run=CONTROL_RUN,source_control_commit=CONTROL_COMMIT,source_control_candidate=CONTROL_CANDIDATE,
        unrolled_source_run=UNROLLED,unrolled_source_commit=UNROLLED_COMMIT,
        alternating_source_run=ALTERNATING,alternating_source_commit=ALTERNATING_COMMIT,
        predecessor_run=PRECEDING,predecessor_commit=PRECEDING_COMMIT,
        target='One source-selected candidate only;original168000x7 F50/defaulthead truthlast;noneeligible0target',
        scientific_scope='TX-unlabeled simulated-paired self-supervision;source control reused;no per-detail TX attribution guarantee',
        calibration_count=0,training_Core_calls=3,inference_Core_calls=3,training_CG_iterations=6,inference_CG_iterations=6)
    return value
