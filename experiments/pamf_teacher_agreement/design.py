from experiments.pamf_identity_priority.design import configuration as base_configuration
from experiments.pamf_identity_priority.design import PARENT_RUN,PARENT_COMMIT,TEACHER_SEEDS,MODEL_SEED,DATA_SEED,EPOCHS,UPDATES,BATCH
from experiments.pamf_restoration.data import SCENES

RUN='20261020-phase1-restoration-teacher-agreement-source-s2026092701-m2-r01'
ARMS=('frame_agreement','time_agreement');ARM=ARMS[0];GPUS=dict(zip(ARMS,(4,5)))
CONTROL_RUN='20261020-phase1-restoration-gradient-balance-source-s2026092701-m2-r01';CONTROL_ARM='per_frame_cap'
PRECEDING_RUN='20261020-phase1-restoration-tangent-guidance-source-s2026092701-m2-r01'
TRAINABLE_PARAMETERS=2140840;COSINE_DENOMINATOR_MIN=1e-20


def target_matrix():
    return dict(policy='FIXED_COMPARISON',arms=list(ARMS),views=['clean',*SCENES],count=168000,
        F_seeds=list(TEACHER_SEEDS),source_selection=False,target_feedback=False,
        external_control=dict(run=CONTROL_RUN,arm=CONTROL_ARM,reuse=True,new_training=False))


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered whole-frame/complex-time teacher agreement required')
    value=base_configuration('posterior03')
    value.update(schema='frozen_teacher_gradient_agreement_v1',run_id=RUN,arm=arm,GPU=GPUS[arm],trainable_parameters=TRAINABLE_PARAMETERS,
        trainable='Original R03 coarse/correction only;three original F/defaultheads frozen',gradient_epochs=EPOCHS,gradient_updates=UPDATES,
        source_role='Fixed finalE10/2470;sourceV diagnostic only,no accuracy/NMSE candidate gate',
        source_rule='NONE;source V diagnosis only,no candidate qualification or ranking',
        source_readout='Fixed F1/source V diagnostic;three original F guide training and independently read target',
        target_scope='Both fixed arms;168000 clean+six practical/three original nativeheads;external per_frame_cap reused without retraining',
        control_run=CONTROL_RUN,control_arm=CONTROL_ARM,control_reuse='Operator-structure fixed mean3F/per-frame cap control;no performance selection or initialization',
        agreement_space='whole restoredIQ2x256 frame' if arm==ARMS[0] else 'each complex time sample,realI/Q vector together',
        agreement_rule='Keep unweighted mean3 teacher gradients only where all three pairwise dots>=0;otherwise set identity gradient to0',
        agreement_threshold=0,agreement_zero='Zero pair dots accepted;zero-norm pair cosine omitted only from diagnostic averages',
        agreement_symmetry='Teacher-permutation and orthogonal I/Q-coordinate invariant;no per-axis sign mask,class/RX/scene-specific rule',
        agreement_limit='Raw identity first-order dot property only;wave/AdamW/clip/finite steps may change any identity loss;no causal or target guarantee',
        gradient_space='RestoredIQ2x256 real coordinates during source training only',
        gradient_rule='wave plus agreed-identity after existing per_frame_cap;no teacher normalization,optimization or conflict projection',
        gradient_normalization='Each teacher and wave task sum-loss output gradients;mean3,agreement mask,per-frame cap,then divide actual observed+clean batch count once',
        inference='Original R03 current-frame coarse/corrector only;no runtime F/gradient/iteration/RX/scene/clean/otherframe/update',
        loss='PairedIQNMSE+.1logpower and logged mean3F(finalcos+.3T2KL) values unchanged;backward consensus differs',
        optimizer='ResetAdamW3e-5,warm247/cos3e-6,WD1e-4,clip1,FP32/noTF32',target_matrix=target_matrix(),
        target_selection='NONE;both fixed arms automatically complete all target views',
        matched_comparison='Same originalR03state/params/E10/data/order/routes/optimizer/3Floss/per-frame cap;only agreement granularity differs;extra teacher pullbacks disclosed')
    return value
