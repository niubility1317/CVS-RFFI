from experiments.pamf_identity_priority.design import configuration as base_configuration
from experiments.pamf_identity_priority.design import PARENT_RUN,PARENT_COMMIT,TEACHER_SEEDS,MODEL_SEED,DATA_SEED,EPOCHS,UPDATES,BATCH
from experiments.pamf_restoration.data import SCENES

RUN='20261020-phase1-restoration-gradient-balance-source-s2026092701-m2-r01'
ARMS=('global_cap','per_frame_cap');ARM=ARMS[0];GPUS=dict(zip(ARMS,(4,5)))
CONTROL_RUN='20261020-phase1-restoration-gradient-projection-source-s2026092701-m2-r01';CONTROL_ARM='scalar_sum'
PRECEDING_RUN=CONTROL_RUN;TRAINABLE_PARAMETERS=2140840;NORM_SQUARED_MIN=1e-20


def target_matrix():
    return dict(policy='FIXED_COMPARISON',arms=list(ARMS),views=['clean',*SCENES],count=168000,
                F_seeds=list(TEACHER_SEEDS),source_selection=False,target_feedback=False,
                external_control=dict(run=CONTROL_RUN,arm=CONTROL_ARM,reuse=True,new_training=False))


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered global/per-frame norm-cap arm required')
    value=base_configuration('posterior03')
    value.update(schema='output_gradient_balance_v1',run_id=RUN,arm=arm,GPU=GPUS[arm],trainable_parameters=TRAINABLE_PARAMETERS,
        trainable='Original R03 coarse/correction only;three original F/defaultheads frozen',gradient_epochs=EPOCHS,gradient_updates=UPDATES,
        gradient_space='RestoredIQ2x256 real coordinates during source training only',gradient_rule='gw+s*gi;0<=s<=1;no conflict projection',
        gradient_scope='whole observed+clean batch' if arm==ARMS[0] else 'each IQ frame',
        gradient_scale='min(1,sqrt(sum norm2(gw)/sum norm2(gi)))' if arm==ARMS[0] else 'min(1,sqrt(norm2(gw)/norm2(gi)))',
        gradient_zero_rule='Identity norm2<=1e-20 leaves scale1;nonzero identity with zero waveform scales to0',
        gradient_normalization='Task sum-loss output gradients merged then divided by actual observed+clean batch count',
        gradient_limit='Detached source output-gradient norm cap;not parameter clipping/GradNorm/PCGrad;no parameter-step/final waveform or identity guarantee',
        source_role='Fixed finalE10/2470;sourceV diagnostic only,no accuracy/NMSE candidate gate',
        source_rule='NONE;source V diagnosis only,no candidate qualification or ranking',
        source_readout='Fixed F1/source V diagnostic;three original F guide training and independently read target',
        target_scope='Both new fixed arms;168000 clean+six practical/three original nativeheads;external scalar row reused without retraining',
        control_run=CONTROL_RUN,control_arm=CONTROL_ARM,control_reuse='Existing independently preregistered scalar_sum E10;no target-based selection;no initialization from it',
        inference='Original R03 current-frame coarse/corrector only;no runtime gradient/F,iteration/RX/scene/clean/otherframe/update',
        loss='Paired IQNMSE+.1logpower and mean3F(finalcos+.3T2KL) values unchanged;only backward scale differs',
        optimizer='ResetAdamW3e-5,warm247/cos to3e-6,WD1e-4,clip1,FP32/noTF32',target_matrix=target_matrix(),
        target_selection='NONE;both fixed arms automatically complete all target views',
        matched_comparison='Same original R03 state/params/E10/data/order/routes/optimizer/3Floss/separate gradient work;new global vs per-frame scope only;reuse same-budget scalar control')
    return value
