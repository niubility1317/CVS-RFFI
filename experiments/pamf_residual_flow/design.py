from experiments.pamf_identity_priority.design import (PARENT_RUN,PARENT_COMMIT,DATA_SEED,MODEL_SEED,BATCH,TEACHER_SEEDS)
from experiments.pamf_identity_priority.design import configuration as base_configuration
from experiments.pamf_restoration.data import SCENES

RUN='20261020-phase1-restoration-residual-flow-source-s2026092701-m2-r01'
ARMS=('direct_residual','rectified_flow4');GPUS=dict(zip(ARMS,(6,7)))
EPOCHS=10;UPDATES=2470;TIME_SEED=2026102071
PRECEDING_RUN='20261020-phase1-restoration-gradient-balance-source-s2026092701-m2-r01'
WIDTHS=(16,32,48,64);TRAINABLE_PARAMETERS=210882;TOTAL_PARAMETERS=2351722


def target_matrix():
    return dict(policy='FIXED_COMPARISON',arms=list(ARMS),views=['clean',*SCENES],count=168000,
                F_seeds=list(TEACHER_SEEDS),source_selection=False,target_feedback=False)


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered direct/flow IQ arm required')
    value=base_configuration('posterior03')
    value.update(schema='frozen_R03_residual_flow_v1',run_id=RUN,arm=arm,GPU=GPUS[arm],
        trainable='New Core7->2 widths16/32/48/64 only;original R03/three F/defaultheads frozen',
        trainable_parameters=TRAINABLE_PARAMETERS,total_parameters=TOTAL_PARAMETERS,
        new_epochs=EPOCHS,new_updates=UPDATES,parent_coarse_epochs=20,parent_correction_epochs=20,
        serial_inherited_passes=50,component_epochs=50,
        field_input='Current receivedIQ/R03anchor/(currentstate-anchor),all divided by anchorRMS;one time channel',
        field_target='clean-anchor;per-IQ clean-power-normalized velocity error',
        training_time='0 always' if arm==ARMS[0] else 'Half0/halfUniform[0,1),same t for observed/clean pair',
        training_point='anchor+t*(clean-anchor);paired clean ONLY in source training',
        training_guidance='anchor+predicted_velocity:original .1logpower + mean3F(finalcos+.3T2KL),T2',
        inference_steps=1 if arm==ARMS[0] else 4,inference_rule='Euler currentframe only,dt=1/K,t=i/K,initialstate=R03anchor;no noise',
        inference='Only current receivedIQ;frozen R03 once,new field1/4times;no F,clean,TX/RX/scene/oracle/otherframe or update',
        initial_output='Zero new output projection;both complete initial IQ exactly originalR03',
        optimizer='NewfieldAdamW2e-4,warm247/cos2e-5,WD1e-4,clip1,FP32/noTF32',
        source_role='Fixed finalE10/2470;sourceV diagnosis only,no qualification/ranking',
        source_rule='NONE;both fixed rows automatically complete target',target_matrix=target_matrix(),
        target_scope='168000 physical clean+six practical,three independently read original defaultheads',
        limitation='Inspired by rectified flow;not PMRF replication/optimality claim;no guaranteed identity or physical inverse;no added Gaussian noise')
    return value
