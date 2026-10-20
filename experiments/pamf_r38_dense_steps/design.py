import math
from experiments.pamf_r38_fidelity_only import design as previous
from experiments.pamf_r38_successor import design as base
parent=base.parent
RUN='20261020-phase1-restoration-r38-dense-steps-s2026092701-m3-r17'
ARMS=('factor625_b64_lr1','factor625_b64_lr3','dense750_b64_lr3')
STRUCTURE=dict(zip(ARMS,('factor625_fidelity','factor625_fidelity','dense750_fidelity')))
LEVEL=dict(zip(ARMS,('five_eighth','five_eighth','three_quarter')))
FRACTIONS=dict(zip(ARMS,(.625,.625,None)))
WIDTHS={a:(12,24,36,48) if a=='dense750_b64_lr3' else (16,32,48,64) for a in ARMS}
INITIALIZATION_MODE={a:'R38_context_exact_prior_GN4_slice' if a=='dense750_b64_lr3' else 'R38_context_exact_prior_SVD_preserve_channels' for a in ARMS}
PARAMETERS=dict(zip(ARMS,(750035,750035,689129)))
CONTEXT_PARAMETERS={a:548795 for a in ARMS}
PRIOR_PARAMETERS={a:PARAMETERS[a]-548795 for a in ARMS}
TASK={a:False for a in ARMS};GPUS=dict(zip(ARMS,(2,3,6)))
LEARNING_RATES=dict(zip(ARMS,(1e-4,3e-4,3e-4)))
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=base.COMPACT_PARENT_RUN,base.COMPACT_PARENT_COMMIT,base.COMPACT_PARENT_ROW
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=base.PARENT_RUN,base.PARENT_COMMIT,base.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=base.F_RUN,base.F_SEED,base.VIEWS,base.KIND
MODEL_SEED,DATA_SEED=base.MODEL_SEED,base.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO=10,9850,64,64
PREDECESSOR_BY_GPU={g:previous.RUN for g in GPUS.values()}
PREDECESSOR_MODULE={previous.RUN:'experiments.pamf_r38_fidelity_only.runner'}
SCORING_RUN=base.SCORING_RUN
def learning_rate(step,arm):
 if arm not in ARMS or not 0<=step<9850:raise ValueError('Registered dense-step schedule required')
 peak=LEARNING_RATES[arm]
 if step<985:return peak*(step+1)/985
 return 1e-5+(peak-1e-5)*.5*(1+math.cos(math.pi*(step-985)/(9850-985-1)))
def configuration():
 return dict(schema='R38_dense_steps_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,prior_parameters=PRIOR_PARAMETERS,
  context_parameters=CONTEXT_PARAMETERS,prior_widths={a:list(w) for a,w in WIDTHS.items()},factor_rank_fraction=FRACTIONS,
  compact_parent_run=COMPACT_PARENT_RUN,compact_parent_commit=COMPACT_PARENT_COMMIT,compact_parent_row=COMPACT_PARENT_ROW,
  parent_choice='Original user-designated R38 dense_half source-final E20;not R13 target-scored descendant',
  initialization='Fixed original R38 only;context exact frozen;prior either .625 saving SVD or GN4 .75 channel slice;no IQ calibration',
  objective='Mean three-stage IQ NMSE to frozen original R38 only;paired-clean NMSE detached diagnostic;no paired-clean optimization',
  context_trainable=False,prior_trainable=True,training_F_access=False,TX_labels_read=0,V_training=False,target_training=False,
  shared_context_forward=True,teacher_context_inputs=630000,student_context_inputs=0,teacher_prior_inputs=630000,student_prior_inputs=630000,
  epochs=10,updates=9850,batch=64,microbatch=64,optimizer='Reset prior-only AdamW;B64 same source exposure;9850updates/warm985/cos1e-5/WD1e-4/clip1',learning_rates=LEARNING_RATES,
  model_seed=MODEL_SEED,data_seed=DATA_SEED,received_epochs=list(range(1,11)),L_exposures=63000,U_exposures=567000,
  cumulative_coarse_epochs=40,cumulative_action_epochs=40,cumulative_prior_epochs=50,cumulative_component_epoch_sum=130,
  cumulative_serial_training_passes=90,FP32=True,TF32=False,
  freeze_rule='Fixed final E10/9850;all three rows automatically complete all seven target scenes;no score gate',
  views=list(VIEWS),target_count_per_view=168000,target_acceptance_allowance_pp=.8,
  GPU=GPUS,max_own_GPU_workers=3,max_own_workers_per_card=1,capacity_lock='.pamf-reception-capacity4.lock',
  source_min_free_MB=18000,all_owner_cap=2,CPU_threads=2,max_own_scorers=1,scoring_run=SCORING_RUN,
  stop_rule='Technical failure preserves affected row;healthy rows continue;no automatic retry/fallback',
  inference='Current native-scale float32[B,2,256] IQ;full own coarse/action/prior and 3x2CG;no teacher/clean/TX/RX/scene/cache/state update',
  claim='F-free TX-unlabeled source IQ compression;single seed/exposed benchmark;not new blind test/global minimum/Phase2',
  target_feedback=False,goal_complete=False)
