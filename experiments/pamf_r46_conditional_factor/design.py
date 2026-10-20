"""R46 parent: GN-preserving low-rank receive-chain compression, with conditional prior matching."""
import math
from experiments.pamf_r44_pruned_refine import design as parent
from experiments.pamf_r38_dense_steps import design as predecessor
RUN='20261020-phase1-restoration-R46-conditional-factor-s2026092701-m2-r18'
ARMS=('factor625_stage','factor625_cond')
TASK={a:False for a in ARMS};MODE={a:'R46_allcore_SVD625_preserve_channels' for a in ARMS}
FRACTIONS={a:.625 for a in ARMS};CONDITIONAL_WEIGHT=dict(zip(ARMS,(0.,.05)))
PARAMETERS={a:278180 for a in ARMS};CONTEXT_PARAMETERS={a:186703 for a in ARMS}
GPUS=dict(zip(ARMS,(2,3)))
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=parent.RUN,'bbb8daf817de050b958d217b70d4c33f4aefe341','prune_refine'
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=parent.PARENT_RUN,parent.PARENT_COMMIT,parent.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=parent.F_RUN,parent.F_SEED,parent.VIEWS,parent.KIND
MODEL_SEED,DATA_SEED=parent.MODEL_SEED,parent.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO=20,19700,64,64
PREDECESSOR=predecessor.RUN
PREDECESSOR_BY_GPU={g:predecessor.RUN for g in GPUS.values()}
PREDECESSOR_MODULE={predecessor.RUN:'experiments.pamf_r38_dense_steps.runner'}
SCORING_RUN=predecessor.SCORING_RUN

def teacher_weight(epoch):
    if type(epoch) is not int or not 1<=epoch<=EPOCHS:raise ValueError('Fixed20epoch schedule required')
    return 1.-.9*(epoch-1)/19

def learning_rate(step):
    if type(step) is not int or not 0<=step<UPDATES:raise ValueError('Fixed19700update schedule required')
    if step<985:return 2e-4*(step+1)/985
    return 2e-5+(2e-4-2e-5)*.5*(1+math.cos(math.pi*(step-985)/(UPDATES-985-1)))

def configuration():
    return dict(schema='R46_conditional_factor_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,context_parameters=CONTEXT_PARAMETERS,
        compact_parent_run=COMPACT_PARENT_RUN,compact_parent_commit=COMPACT_PARENT_COMMIT,compact_parent_row=COMPACT_PARENT_ROW,
        parent_choice='Human-designated source-final350521 after disclosed formal results;exposed engineering benchmark,not cleanblind generalization',
        architecture='SVD625 of every saving kernel>1 convolution in coarse/action/3shared-prior;unchanged channels,GN4,skips,bias;3stages/6CG',
        factor_rank_fraction=FRACTIONS,conditional_weight=CONDITIONAL_WEIGHT,initialization_IQ_examples=0,
        objective='Pairedclean IQNMSE +.05 relativepower STFT64/128/256 +linear1to.1 teacherstageIQ;conditionalarm adds.05 teacher-prior relativeNMSE on detached studentstageinputs',
        conditional_normalization='Per-example teacher prior-output mean square clamped1e-4;three stages mean;teacher input and output detached;no conditional compute on control',
        paired_clean_optimized=True,all_coarse_action_prior_trainable=True,training_F_access=False,
        epochs=20,updates=19700,batch=64,microbatch=64,optimizer='ResetAdamW',LR=2e-4,final_LR=2e-5,warmup_updates=985,weight_decay=1e-4,gradient_clip=1.,FP32=True,TF32=False,
        model_seed=MODEL_SEED,data_seed=DATA_SEED,received_epochs=list(range(1,21)),L_count=6300,U_count=56700,L_exposures=126000,U_exposures=1134000,
        source_IQ_examples=1260000,teacher_IQ_inputs=1260000,student_context_inputs=1260000,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_training=False,target_training=False,
        conditional_teacher_prior_calls_per_source_example={a:3*int(CONDITIONAL_WEIGHT[a]>0) for a in ARMS},
        inherited_component_epochs=80,cumulative_component_epochs=100,cumulative_component_epoch_sum=300,cumulative_serial_training_passes=140,
        freeze_rule='Both fixedfinalE20/19700;all7formal perrow without source gate;independenttruthlast after full prediction',
        views=list(VIEWS),target_count_per_view=168000,target_acceptance_allowance_pp=.8,acceptance_baselines=['human_R46_350521','original_R17_historical1pp'],
        GPU=GPUS,max_own_GPU_workers=2,max_own_workers_per_card=1,predecessor_by_GPU={str(g):v for g,v in PREDECESSOR_BY_GPU.items()},capacity_lock='.pamf-reception-capacity4.lock',
        source_min_free_MB=18000,all_owner_cap=2,CPU_threads=2,max_own_scorers=1,scoring_run=SCORING_RUN,
        stop_rule='Affectedtechnicalfailure preserved;healthy peer unchanged;no autoretry/hotedit/unrelatedstop',
        inference='Own full G nativefloat32[B,2,256] IQ only;originalF50/defaulthead evaluation only;no teacher/clean/conditional branch in deployment',
        difference_from_other_thread='R48 prunes GN channels jointly;this route preserves all channel/GN dimensions, factors spatial kernels and isolates conditionalfunction matching',
        cost_plan='Samecard G and G+F B1/B32 3 alternating rounds;64sourceV only;median/p90/p95/throughput/resident/peak;ConvLinearMAC excludes FFT/CG/norm',
        claim='TX-unlabeled simulated-paired source selfsupervision;0F/TX/V/target training;single seed exposed benchmark;no perdetailTX proof',target_feedback=False,goal_complete=False)
