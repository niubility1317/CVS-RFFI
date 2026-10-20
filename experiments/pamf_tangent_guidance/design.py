from experiments.pamf_identity_priority.design import configuration as base_configuration
from experiments.pamf_identity_priority.design import PARENT_RUN,PARENT_COMMIT,TEACHER_SEEDS,MODEL_SEED,DATA_SEED,EPOCHS,UPDATES,BATCH
from experiments.pamf_restoration.data import SCENES

RUN='20261020-phase1-restoration-tangent-guidance-source-s2026092701-m2-r01'
ARMS=('feature_tangent','posterior_tangent');ARM=ARMS[0];GPUS=dict(zip(ARMS,(6,7)))
CONTROL_RUN='20261020-phase1-restoration-gradient-balance-source-s2026092701-m2-r01';CONTROL_ARM='per_frame_cap'
PRECEDING_RUN='20261020-phase1-restoration-parameter-balance-source-s2026092701-m2-r01'
TRAINABLE_PARAMETERS=2140840;PROBES=6;PROJECTION_SEED=2026102081;TEMPERATURE=2.;POSTERIOR_FLOOR=1e-12


def target_matrix():
    return dict(policy='FIXED_COMPARISON',arms=list(ARMS),views=['clean',*SCENES],count=168000,
        F_seeds=list(TEACHER_SEEDS),source_selection=False,target_feedback=False,
        external_control=dict(run=CONTROL_RUN,arm=CONTROL_ARM,reuse=True,new_training=False))


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered source-clean tangent arm required')
    value=base_configuration('posterior03')
    value.update(schema='source_clean_tangent_v1',run_id=RUN,arm=arm,GPU=GPUS[arm],trainable_parameters=TRAINABLE_PARAMETERS,
        trainable='Original R03 coarse/correction only;three original F/defaultheads frozen',gradient_epochs=EPOCHS,gradient_updates=UPDATES,
        source_role='Fixed finalE10/2470;sourceV diagnostic only,no accuracy/NMSE candidate gate',
        source_rule='NONE;source V diagnosis only,no candidate qualification or ranking',
        source_readout='Fixed F1/source V diagnostic;three original F guide training and independently read target',
        target_scope='Both fixed arms;168000 clean+six practical/three original nativeheads;external per_frame_cap row reused without retraining',
        control_run=CONTROL_RUN,control_arm=CONTROL_ARM,control_reuse='Structurally fixed per-frame cap operator control;no target-based selection,no initialization from control',
        tangent_origin='Clean IQ of same source physical sample;all63000 L_s/U_s;no TX/V/target used in tangent computation',
        tangent_coordinates='normalized160feature times fixed six Rademacher probes divided by sqrt2' if arm==ARMS[0] else 'sqrt2*T*sqrt(native six-class softmax(logits/T),floor1e-12)',
        tangent_feature_projection=dict(probes=PROBES,seed=PROJECTION_SEED,per_teacher_seed='PROJECTION_SEED+F_seed',distribution='independent Rademacher/sqrt6',fitted=False),
        tangent_metric='mean3F(sum6((J_clean@(restored-clean))**2))',tangent_dtype='float32',tangent_temperature=TEMPERATURE,
        tangent_cache_shape=[63000,3,PROBES,2,256],tangent_cache_bytes=63000*3*PROBES*2*256*4,
        tangent_limit='Fixed local first-order metric;feature uses six random probes,not full160 Jacobian;posterior is stabilized local KL quadratic;no finite-error or target guarantee',
        gradient_space='RestoredIQ2x256 real coordinates;existing per_frame_cap operator',
        gradient_rule='gw+min(1,sqrt(norm2(gw)/norm2(gi)))*gi;identity norm2<=1e-20 gives scale1;no amplification/projection',
        gradient_normalization='Task sum-loss output gradients merged then divided by actual observed+clean batch count',
        loss='IQNMSE+.1logpower and mean3F(localfeature+.3exactT2KL)' if arm==ARMS[0] else 'IQNMSE+.1logpower and mean3F(exactfinalcos+.3localposterior)',
        optimizer='ResetAdamW3e-5,warm247/cos to3e-6,WD1e-4,clip1,FP32/noTF32',target_matrix=target_matrix(),
        target_selection='NONE;both fixed arms automatically complete all target views',
        inference='Original R03 current-frame coarse/corrector only;no J/cache/F/gradient/clean/RX/scene/otherframe/update',
        matched_comparison='Same original R03 state/params/E10/data/order/routes/optimizer/3F/per-frame cap;replace exactly one identity component;extra source-J compute/storage disclosed')
    return value
