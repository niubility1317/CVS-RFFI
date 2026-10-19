from experiments.pamf_identity_priority.design import configuration as base_configuration
from experiments.pamf_identity_priority.design import PARENT_RUN,PARENT_COMMIT,TEACHER_SEEDS,MODEL_SEED,DATA_SEED,EPOCHS,UPDATES,BATCH
from experiments.pamf_restoration.data import SCENES

RUN='20261020-phase1-restoration-temporal-context-source-s2026092701-m2-r01'
ARMS=('window_attention','global_attention');ARM=ARMS[0];GPUS=dict(zip(ARMS,(4,5)))
PRECEDING_RUN='20261020-phase1-restoration-style-generalization-source-s2026092701-m2-r02'
WIDTH=128;POSITIONS=32;HEADS=4;RADIUS=2;ATTENTION_PARAMETERS=66304;TRAINABLE_PARAMETERS=2207144


def target_matrix():
    return dict(policy='FIXED_COMPARISON',arms=list(ARMS),views=['clean',*SCENES],count=168000,
                F_seeds=list(TEACHER_SEEDS),source_selection=False,target_feedback=False)


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered narrow/full-window attention arm required')
    value=base_configuration('posterior03')
    value.update(schema='temporal_context_refine_v1',run_id=RUN,arm=arm,GPU=GPUS[arm],
        trainable_parameters=TRAINABLE_PARAMETERS,attention_parameters=ATTENTION_PARAMETERS,
        trainable='Original R03 coarse/correction plus new attention;all three original F/defaultheads frozen',
        attention_epochs=EPOCHS,attention_updates=UPDATES,
        attention_location='Corrector encoder bottleneck,128channels x32time positions',attention_heads=HEADS,
        attention_window_radius=RADIUS if arm==ARMS[0] else None,
        attention='PreLayerNorm,current-frame fixed sinusoidal positions,manual QKV/softmax,zero output projection,residual,no dropout/FFN',
        context_scope='Existing convolutions/GroupNorm already have wide/global context;only explicit attention window varies',
        new_learning_rate=2e-4,new_final_learning_rate=2e-5,parameter_groups='Original3e-5/newattention2e-4;warm247/cos to0.1,WD1e-4,clip1',
        source_role='Fixed finalE10/2470;sourceV diagnostic only,no accuracy/NMSE candidate gate',
        source_rule='NONE;source V diagnosis only,no candidate qualification or ranking',
        source_readout='Fixed F1/source V diagnostic;three original F guide training and independently read target',
        target_scope='Both fixed arms;168000 clean + six practical;three original native heads independently',
        control_run=RUN,control_arm=ARMS[0],control_reuse='New matched narrow-window attention trained once in fixed matrix',
        inference='One current IQ frame,one coarse/corrector forward;no runtime F inside restorer,no iteration/RX/pseudo/bank/clean/otherframe/update',
        loss='Original posterior03 paired IQNMSE+.1logpower+mean3F(finalcos+.3T2KL);no CE/prototype/focus/style/FiLM',
        target_matrix=target_matrix(),target_selection='NONE;both fixed arms automatically complete all target views',
        matched_comparison='Same R03 plus matching zero-output attention initialization,parameters/E10/data/order/routes/optimizer/threeFloss;only attention mask differs;both compute dense32x32')
    return value
