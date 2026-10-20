from experiments.pamf_identity_priority.design import configuration as base_configuration
from experiments.pamf_identity_priority.design import PARENT_RUN,PARENT_COMMIT,TEACHER_SEEDS,MODEL_SEED,DATA_SEED,EPOCHS,UPDATES,BATCH
from experiments.pamf_restoration.data import SCENES

RUN='20261020-phase1-restoration-spatial-conditioning-source-s2026092701-m2-r01'
ARMS=('pooled_features','spatial_features');ARM=ARMS[0];GPUS=dict(zip(ARMS,(6,7)))
PRECEDING_RUN='20261020-phase1-restoration-teacher-agreement-source-s2026092701-m2-r01'
GPU_PRECEDING_RUN='20261020-phase1-restoration-tangent-guidance-source-s2026092701-m2-r01'
RUNTIME_F_SEED=2026092701;RUNTIME_F_COMMIT='4df7b94d0fd8ce29056bd8236364e45afa92e07f'
WIDTHS=(32,64,96,128);MAP_CHANNELS=(72,96)
TRAINABLE_PARAMETERS=925378;RUNTIME_F_PARAMETERS=232993;TOTAL_PARAMETERS=3299211


def target_matrix():
    return dict(policy='FIXED_COMPARISON',arms=list(ARMS),views=['clean',*SCENES],count=168000,
        F_seeds=list(TEACHER_SEEDS),source_selection=False,target_feedback=False,within_run_control=ARMS[0])


def configuration(arm):
    if arm not in ARMS:raise ValueError('Registered pooled/spatial frozen-F condition required')
    value=base_configuration('posterior03')
    value.update(schema='frozen_F_spatial_condition_v1',run_id=RUN,arm=arm,GPU=GPUS[arm],trainable_parameters=TRAINABLE_PARAMETERS,
        total_parameters=TOTAL_PARAMETERS,runtime_F_parameters=RUNTIME_F_PARAMETERS,runtime_F_seed=RUNTIME_F_SEED,runtime_F_commit=RUNTIME_F_COMMIT,
        trainable='New zero-output IQ residual field only;originalR03 and originalF1/defaulthead frozen',
        gradient_epochs=EPOCHS,gradient_updates=UPDATES,field_epochs=EPOCHS,
        condition='OriginalF1 native t1/t2 maps of current R03 anchor;per-map per-frame RMS normalization with norm2 floor1e-4',
        condition_layout='t1/t1/t2/t2 to field widths32/64/96/128 via learned1x1 projections and fusion',
        condition_temporal='Globaltime mean then broadcast/interpolate' if arm==ARMS[0] else 'Native time maps linearly interpolated;align_corners=False',
        condition_gradient='RuntimeF and anchor no_grad;projections/fusions trained;no class logits or fitted identity head',
        output='anchor+anchorRMS*field(concat(received,anchor)/anchorRMS,condition);anchorRMS norm2 floor1e-4;zero field head starts exactly R03',
        source_role='Fixed finalE10/2470;sourceV diagnostic only,no accuracy/NMSE candidate gate',
        source_rule='NONE;source V diagnosis only,no candidate qualification or ranking',
        source_readout='Fixed F1/sourceV;three original F guide training and independently read target',
        target_scope='Both fixed arms;168000 clean+six practical/three original nativeheads',
        control_run=RUN,control_arm=ARMS[0],control_reuse='New matched pooled-features control;same runtimeF,parent,field/parameter count/data/seed/budget/loss/optimizer',
        gradient_space='RestoredIQ2x256 real coordinates;existing per_frame_cap operator',
        gradient_rule='wave plus capped mean3F finalcos+.3T2KL;sum-loss output gradients divided by actual observed+clean count once',
        loss='PairedIQNMSE+.1logpower and mean3F(finalcos+.3T2KL)',
        optimizer='ResetAdamW2e-4,warm247/cos2e-5,WD1e-4,clip1,FP32/noTF32',target_matrix=target_matrix(),
        GPU_queue='After tangent root terminal AND its owned source/target handles exited;no result score inspected;then oneown/card and source18GB/target12GB/allowner<2',
        target_selection='NONE;both fixed arms automatically complete all target views',
        inference='Current receivedIQ,originalR03anchor and frozenF1 native feature maps only;no clean/cache/trueTX/RX/scene/otherframe/vote/update',
        matched_comparison='Two new arms differ only time pooling of normalized native maps;old FiLM/global160 conditioning and middle-loss rows not reused/promoted;historical target best context only')
    return value
