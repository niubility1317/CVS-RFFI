from experiments.pamf_identity_priority.design import configuration as base_configuration
from experiments.pamf_identity_priority.design import PARENT_RUN,PARENT_COMMIT,TEACHER_SEEDS,MODEL_SEED,DATA_SEED,EPOCHS,UPDATES,BATCH
from experiments.pamf_restoration.data import SCENES

RUN='20261020-phase1-restoration-gradient-projection-source-s2026092701-m2-r01'
ARMS=('scalar_sum','waveform_projection');ARM=ARMS[0];GPUS=dict(zip(ARMS,(6,7)))
PRECEDING_RUN='20261020-phase1-restoration-basis-filter-source-s2026092701-m2-r01'
TRAINABLE_PARAMETERS=2140840;WAVE_NORM_SQUARED_MIN=1e-20


def target_matrix():
    return dict(policy='FIXED_COMPARISON',arms=list(ARMS),views=['clean',*SCENES],count=168000,
                F_seeds=list(TEACHER_SEEDS),source_selection=False,target_feedback=False)


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered scalar/projection arm required')
    value=base_configuration('posterior03')
    value.update(schema='output_gradient_projection_v1',run_id=RUN,arm=arm,GPU=GPUS[arm],trainable_parameters=TRAINABLE_PARAMETERS,
        trainable='Original R03 coarse/correction only;three original F/defaultheads frozen',
        gradient_epochs=EPOCHS,gradient_updates=UPDATES,gradient_space='Each current restored IQ frame,2x256 real coordinates',
        gradient_rule='gw+gi' if arm==ARMS[0] else 'gw+gi-dot(gw,gi)/norm2(gw)*gw only when dot<0 and norm2(gw)>1e-20',
        gradient_normalization='Per-frame sum-loss gradients merged then divided by actual observed+clean batch count',
        gradient_wave_norm_squared_min=WAVE_NORM_SQUARED_MIN,
        gradient_limit='One-sided output-space projection;not full parameter-space PCGrad;shared Jacobian/AdamW means no parameter-step or final waveform guarantee',
        source_role='Fixed finalE10/2470;sourceV diagnostic only,no accuracy/NMSE candidate gate',
        source_rule='NONE;source V diagnosis only,no candidate qualification or ranking',
        source_readout='Fixed F1/source V diagnostic;three original F guide training and independently read target',
        target_scope='Both fixed arms;168000 clean + six practical;three original native heads independently',
        control_run=RUN,control_arm=ARMS[0],control_reuse='New matched scalar output-gradient sum trained once in fixed matrix',
        inference='One current IQ frame,one original coarse/corrector forward;no gradient/F inside restorer,iteration/RX/pseudo/bank/clean/otherframe/update',
        loss='Paired IQNMSE+.1logpower and mean3F(finalcos+.3T2KL) have same values;only backward merge differs',
        optimizer='ResetAdamW3e-5,warm247/cos to3e-6,WD1e-4,clip1,FP32/noTF32',
        target_matrix=target_matrix(),target_selection='NONE;both fixed arms automatically complete all target views',
        matched_comparison='Same original R03 state/params/E10/data/order/routes/optimizer/3Floss;both compute separate output gradients/conflict stats;only merge differs')
    return value
