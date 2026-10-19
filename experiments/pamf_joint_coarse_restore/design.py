from experiments.pamf_analytic_unrolled_restore import design as previous

RUN='20261020-phase1-restoration-joint-coarse-selfsup-s2026092701-m2-r23'
ROWS=('prior_only','joint_coarse')
ALL_ROWS=ROWS
KIND=previous.KIND
ACTION=previous.ACTION
REFERENCE=previous.REFERENCE
MODEL_SEED=previous.MODEL_SEED
DATA_SEED=previous.DATA_SEED
GPUS=previous.GPUS
SINGLE=previous.SINGLE
SINGLE_COMMIT=previous.SINGLE_COMMIT
FFT=previous.FFT
FFT_COMMIT=previous.FFT_COMMIT
UNROLLED=previous.UNROLLED
UNROLLED_COMMIT=previous.UNROLLED_COMMIT
REFERENCE_CANDIDATE=previous.REFERENCE_CANDIDATE
ALTERNATING=previous.ALTERNATING
ALTERNATING_COMMIT=previous.ALTERNATING_COMMIT
ALTERNATING_CANDIDATE=previous.ALTERNATING_CANDIDATE
PRECEDING=previous.RUN
PRECEDING_COMMIT='ee41478dce40cfdf7778d46f8c44bb58eaf82367'
COARSE_LR_MULTIPLIER=.1


def configuration():
    value=previous.unrolled.configuration()
    value.update(schema='joint_coarse_v1',run=RUN,rows=list(ROWS),
        loss='Paired detached-clean IQNMSE only;no analytic/FFT/F/TX loss',
        candidate_designs={name:dict(coarse_trainable=name=='joint_coarse',estimator_trainable=False,
            prior_trainable=True,coarse_LR_multiplier=.1 if name=='joint_coarse' else None) for name in ROWS},
        prior_initialization='Scratch R17unshared;only original source R02/R07 context inherited',
        estimator='Frozen weights;forward and input derivatives depend on current coarse;action outputs are not fixed',
        coarse='R02 pretrained20;0 or20 additional epochs',
        gradient_clipping='Prior and coarse groups independently norm1;reported separately',
        training_Core_calls=3,inference_Core_calls=3,training_CG_iterations=6,inference_CG_iterations=6,
        calibration_count=0,coarse_LR_multiplier=.1,parameter_control='Identical full model/state/deployment;more trainable parameters and FLOPs in joint arm',
        source_selection='All native/R15/R16/R17/R19 sourceguards;rank lowmidmean,lowestTX,lowestRX,lexicalname',
        unrolled_source_run=UNROLLED,unrolled_source_commit=UNROLLED_COMMIT,
        alternating_source_run=ALTERNATING,alternating_source_commit=ALTERNATING_COMMIT,
        predecessor_run=PRECEDING,predecessor_commit=PRECEDING_COMMIT,
        target='One source-selected candidate only;original168000x7 F50/defaulthead truthlast;noneeligible0target',
        scientific_scope='Unfreeze source-trained coarse with no trained F/TX loss;not identifiable physical channel parameters',
        explicit_F_training=False)
    return value
