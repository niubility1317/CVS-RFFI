from experiments.pamf_r38_prior_compact import design as r14
from experiments.pamf_r38_factor_prior import design as r15
from experiments.pamf_r38_successor import design as base
parent=base.parent
RUN='20261020-phase1-restoration-r38-fidelity-only-s2026092701-m3-r16'
ARMS=('factor625_fidelity','factor500_fidelity','dense750_fidelity')
LEVEL=dict(zip(ARMS,('five_eighth','half','three_quarter')))
FRACTIONS=dict(zip(ARMS,(.625,.5,None)))
WIDTHS={a:(12,24,36,48) if a=='dense750_fidelity' else (16,32,48,64) for a in ARMS}
INITIALIZATION_MODE={a:'R38_context_exact_prior_GN4_slice' if a=='dense750_fidelity' else 'R38_context_exact_prior_SVD_preserve_channels' for a in ARMS}
PARAMETERS=dict(zip(ARMS,(750035,720185,689129)))
CONTEXT_PARAMETERS={a:548795 for a in ARMS}
PRIOR_PARAMETERS={a:PARAMETERS[a]-548795 for a in ARMS}
TASK={a:False for a in ARMS};GPUS=dict(zip(ARMS,(2,3,6)))
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=base.COMPACT_PARENT_RUN,base.COMPACT_PARENT_COMMIT,base.COMPACT_PARENT_ROW
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=base.PARENT_RUN,base.PARENT_COMMIT,base.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=base.F_RUN,base.F_SEED,base.VIEWS,base.KIND
MODEL_SEED,DATA_SEED=base.MODEL_SEED,base.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO=10,2470,256,64
PREDECESSOR_BY_GPU={2:r15.RUN,3:r15.RUN,6:r14.RUN}
PREDECESSOR_MODULE={r15.RUN:'experiments.pamf_r38_factor_prior.runner',r14.RUN:'experiments.pamf_r38_prior_compact.runner'}
SCORING_RUN=base.SCORING_RUN
learning_rate=base.learning_rate
def configuration():
 return dict(schema='R38_fidelity_only_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,prior_parameters=PRIOR_PARAMETERS,
  context_parameters=CONTEXT_PARAMETERS,prior_widths={a:list(w) for a,w in WIDTHS.items()},factor_rank_fraction=FRACTIONS,
  compact_parent_run=COMPACT_PARENT_RUN,compact_parent_commit=COMPACT_PARENT_COMMIT,compact_parent_row=COMPACT_PARENT_ROW,
  parent_choice='Original user-designated R38 dense_half source-final E20;not R13 target-scored descendant',
  initialization='Fixed original R38 only;context exact frozen;prior either .625/.5 saving SVD or GN4 .75 channel slice;no IQ calibration',
  objective='Mean three-stage IQ NMSE to frozen original R38 only;paired-clean NMSE detached diagnostic;no paired-clean optimization',
  context_trainable=False,prior_trainable=True,training_F_access=False,TX_labels_read=0,V_training=False,target_training=False,
  shared_context_forward=True,teacher_context_inputs=630000,student_context_inputs=0,teacher_prior_inputs=630000,student_prior_inputs=630000,
  epochs=10,updates=2470,batch=256,microbatch=64,optimizer='Reset AdamW1e-4/warm247/cos1e-5/WD1e-4/clip1',
  model_seed=MODEL_SEED,data_seed=DATA_SEED,received_epochs=list(range(1,11)),L_exposures=63000,U_exposures=567000,
  cumulative_coarse_epochs=40,cumulative_action_epochs=40,cumulative_prior_epochs=50,cumulative_component_epoch_sum=130,
  cumulative_serial_training_passes=90,FP32=True,TF32=False,
  freeze_rule='Fixed final E10/2470;all three rows automatically complete all seven target scenes;no score gate',
  views=list(VIEWS),target_count_per_view=168000,target_acceptance_allowance_pp=.8,
  GPU=GPUS,max_own_GPU_workers=3,max_own_workers_per_card=1,capacity_lock='.pamf-reception-capacity4.lock',
  source_min_free_MB=18000,all_owner_cap=2,CPU_threads=2,max_own_scorers=1,scoring_run=SCORING_RUN,
  stop_rule='Technical failure preserves affected row;healthy rows continue;no automatic retry/fallback',
  inference='Current native-scale float32[B,2,256] IQ;full own coarse/action/prior and 3x2CG;no teacher/clean/TX/RX/scene/cache/state update',
  claim='F-free TX-unlabeled source IQ compression;single seed/exposed benchmark;not new blind test/global minimum/Phase2',
  target_feedback=False,goal_complete=False)
