from experiments.pamf_unrolled_restore import design as unrolled
from experiments.pamf_endpoint_restore import design as refs

RUN='20261020-phase1-restoration-analytic-unrolled-selfsup-s2026092701-m2-r22'
ROWS=('iq_control','analytic_views')
ALL_ROWS=ROWS
KIND=unrolled.KIND
ACTION=unrolled.ACTION
REFERENCE=unrolled.REFERENCE
MODEL_SEED=unrolled.MODEL_SEED
DATA_SEED=unrolled.DATA_SEED
GPUS=unrolled.GPUS
SINGLE=refs.SINGLE
SINGLE_COMMIT=refs.SINGLE_COMMIT
FFT=refs.FFT
FFT_COMMIT=refs.FFT_COMMIT
UNROLLED=refs.UNROLLED
UNROLLED_COMMIT=refs.UNROLLED_COMMIT
REFERENCE_CANDIDATE=refs.REFERENCE_CANDIDATE
ALTERNATING=refs.ALTERNATING
ALTERNATING_COMMIT=refs.ALTERNATING_COMMIT
ALTERNATING_CANDIDATE=refs.ALTERNATING_CANDIDATE
PRECEDING=refs.RUN
PRECEDING_COMMIT='130e9f594fc9da0244b3ad22fb2ca2021485a6bb'


def configuration():
    value=unrolled.configuration()
    value.update(schema='analytic_unrolled_v1',run=RUN,rows=list(ROWS),
        loss='IQNMSE;analytic row adds .01 mean3 coordinate-std-scaled SmoothL1;no FFT auxiliary loss',
        analytic_groups=dict(mirror=128,spectral=4,reference=28),analytic_group_weights=[1/3]*3,analytic_weight=.01,
        calibration_count=8192,calibration_order='original_first8192',calibration_floor=.05,calibration_coordinates=160,
        calibration='Population std percoordinate on original sourceLUclean only;FP64sum;no TX/F/V/target,0optimizer',
        candidate_designs={name:dict(architecture='R17 threeunshared Core9/2CG each',analytic_loss=name=='analytic_views') for name in ROWS},
        prior_initialization='Scratch R17unshared after R07 context;no R10/R12/R17/R19/R21 prior inheritance',
        training_Core_calls=3,inference_Core_calls=3,training_CG_iterations=6,inference_CG_iterations=6,
        parameter_control='2507526trainable/4665073total;identical deployment;analytic objective costs extra training transforms',
        source_selection='All native/R15/R16/R17/R19 sourceguards;rank lowmidmean,lowestTX,lowestRX,lexicalname',
        unrolled_source_run=UNROLLED,unrolled_source_commit=UNROLLED_COMMIT,
        alternating_source_run=ALTERNATING,alternating_source_commit=ALTERNATING_COMMIT,
        predecessor_run=PRECEDING,predecessor_commit=PRECEDING_COMMIT,
        target='One source-selected candidate only;original168000x7 F50/defaulthead truthlast;noneeligible0target',
        scientific_scope='R12 follow-up:paired analytic observation fidelity;not new first analytic trial or identifiedTX hardware',
        explicit_F_training=False)
    return value
