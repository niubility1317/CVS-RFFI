"""Fixed source-only hypothesis; no new inference architecture or parameters."""
from experiments.pamf_r46_anchor_refine import design as base

RUN='20261021-phase1-restoration-R46-feature-hint-s2026092701-m2-r60'
ARMS=('context_hint','all_hint')
PARAMETERS={a:273765 for a in ARMS}
CONTEXT_PARAMETERS={a:185615 for a in ARMS}
GPUS={'context_hint':3,'all_hint':2}
HINT_WEIGHT=.02
ENERGY_FLOOR=1e-4
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=base.COMPACT_PARENT_RUN,base.COMPACT_PARENT_COMMIT,base.COMPACT_PARENT_ROW
REUSED_CONTROL=dict(run=base.RUN,row='prune_decay',method_commit='df36c3d3da4e026b747776b65b58d4921dc1a71e',parameters=273765,
    role='Same weight-only pruned graph/full E20 and IQteacher schedule,without feature loss',training_repeated=False)
PREDECESSOR_RUN='20261021-phase1-restoration-R46-stat-transport-s2026092701-m2-r58'
PREDECESSOR_COMMIT='3e1c14a733b2278c659c653c75a98657663cd8d4'
PREDECESSOR_ROW={3:'calibrated',2:'identity_control'}
EPOCHS,UPDATES,BATCH,MICRO=20,4940,256,256
VIEWS=base.VIEWS
MODEL_SEED,DATA_SEED=base.MODEL_SEED,base.DATA_SEED

def teacher_weight(epoch,arm):
    if arm not in ARMS:raise ValueError('Registered feature-hint scope required')
    return base.teacher_weight(epoch,'prune_decay')

def configuration():
    return dict(schema='R46_feature_hint_prototype_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,context_parameters=CONTEXT_PARAMETERS,
        compact_parent=COMPACT_PARENT_RUN,compact_parent_commit=COMPACT_PARENT_COMMIT,compact_parent_row=COMPACT_PARENT_ROW,
        architecture='Existing weight-only pruned U-Net/GN4/3prior6CG;no normalization replacement or new inference parameter',
        initialization='Direct human350521;same deterministic weight-only GN4 quota pruning as R53prune_decay;no R53/R55/R57/R58 trained student',
        hypothesis='IQ-output teacher alone may leave retained internal responses unconstrained;align retained source-derived teacher channels during compression training',
        hint=dict(weight=HINT_WEIGHT,energy_floor=ENERGY_FLOOR,locations='post-affine GN output/pre-SiLU;all retained channels/full time,no spatial sampling',
            context_modules=84,prior_modules=84,context_hint='context_mean only',all_hint='context_mean+prior_mean',
            normalization='Per IQ and per module mean_squared_student_minus_detached_teacher divided by retained_teacher_energy floor1e-4;equal module mean inside context/prior',
            teacher='Frozen human350521 on same current source received IQ;retained outputs from fixed weight-only channel maps',
            extra_teacher_or_student_G_forwards=0,extra_calibration_IQ=0,new_parameters=0,teacher_updates=0),
        training=dict(epochs=20,updates=4940,global_batch=256,microbatch=256,L_exposures=126000,U_exposures=1134000,
            loss='pairedIQNMSE+.05relativepowerMRSTFT+linear1to.1threeStageIQteacher+.02(contextfeatureNMSE+optionalpriorfeatureNMSE)',
            same_context_weight_both_rows=True,source_clean_pairs=True,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,
            original_parameters_inherited_epochs=80,original_parameters_new_epochs=20,original_parameters_cumulative_epochs=100,
            new_trainable_parameters=0,reset_optimizer=True,all_coarse_action_prior_jointly_train=True),
        evaluation=dict(source1800x7=True,target168000x7=True,all_rows_regardless_source=True,frozen_F50_default_head=True,independent_truth_last=True),
        prototype_only=True,full_pipeline_implemented=False,GPU=GPUS,predecessor=PREDECESSOR_RUN,max_own_GPU_workers=2,
        no_target_feedback='Hypothesis/scopes/coefficient fixed before reading R58source/formal scores;all rows fixed,no score-based parent/width/loss choice',
        claim='Data-free module verification only;no new calibration,training or performance evidence;not per-detail TX proof',goal_complete=False)
