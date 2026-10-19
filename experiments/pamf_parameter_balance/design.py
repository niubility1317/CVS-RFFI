from experiments.pamf_identity_priority.design import configuration as base_configuration
from experiments.pamf_identity_priority.design import PARENT_RUN,PARENT_COMMIT,TEACHER_SEEDS,MODEL_SEED,DATA_SEED,EPOCHS,UPDATES,BATCH
from experiments.pamf_restoration.data import SCENES

RUN='20261020-phase1-restoration-parameter-balance-source-s2026092701-m2-r01'
ARMS=('parameter_global','parameter_parts');ARM=ARMS[0];GPUS=dict(zip(ARMS,(4,5)))
PRECEDING_RUN='20261020-phase1-restoration-residual-flow-source-s2026092701-m2-r01'
CONTROL_RUN='20261020-phase1-restoration-gradient-balance-source-s2026092701-m2-r01';CONTROL_ARM='per_frame_cap'
TRAINABLE_PARAMETERS=2140840;NORM_SQUARED_MIN=1e-20


def target_matrix():
    return dict(policy='FIXED_COMPARISON',arms=list(ARMS),views=['clean',*SCENES],count=168000,F_seeds=list(TEACHER_SEEDS),
        source_selection=False,target_feedback=False,external_control=dict(run=CONTROL_RUN,arm=CONTROL_ARM,reuse=True,new_training=False))


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered whole-model/per-component parameter cap required')
    value=base_configuration('posterior03')
    value.update(schema='parameter_gradient_balance_v1',run_id=RUN,arm=arm,GPU=GPUS[arm],trainable_parameters=TRAINABLE_PARAMETERS,
        trainable='Original R03 coarse/correction only;three originalF/defaultheads frozen',gradient_epochs=EPOCHS,gradient_updates=UPDATES,
        gradient_space='All trainable restorer parameter coordinates;task mean losses',
        gradient_scope='Whole restorer vector' if arm==ARMS[0] else 'Separate coarse and correction vectors',
        gradient_rule='gw+s*gi;s=min(1,sqrt(norm2(gw)/norm2(gi)));identity norm2<=1e-20 leaves1',
        gradient_normalization='Compute two task mean-loss parameter gradients;detach cap;assign merged gradient;clip1 then AdamW',
        gradient_limit='Raw Euclidean direction only;AdamW momentum/preconditioning/decay prevent parameter-step or final waveform/identity guarantee',
        loss='PairedIQNMSE+.1logpower and mean3F(finalcos+.3T2KL),T2,unchanged',
        optimizer='ResetAdamW3e-5,warm247/cos3e-6,WD1e-4,clip1,FP32/noTF32',
        control_run=CONTROL_RUN,control_arm=CONTROL_ARM,control_reuse='Closest output-space per-frame cap by operator definition;not chosen by target score;no initialization/retraining/prediction',
        source_role='Fixed finalE10/2470;sourceV diagnosis only,no qualification/ranking',source_rule='NONE;both rows complete target',
        target_scope='Both new rows;168000clean+sixpractical/three originalheads;reuse fixed output-cap control',target_matrix=target_matrix(),
        inference='Original R03 current-frame coarse/correction only;no runtimeF/gradient/cap/clean/truechannel/TX/RX/scene/otherframe or update')
    return value
