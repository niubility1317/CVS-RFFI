"""Performance-first R46 compression with exact frozen physical context."""
import math
from experiments.pamf_r44_pruned_refine import design as parent
from experiments.pamf_r46_conditional_factor import design as predecessor
RUN='20261020-phase1-restoration-R46-prior-preserve-s2026092701-m2-r19'
ARMS=('prior625_cond','prior750_cond')
TASK={a:False for a in ARMS};MODE={a:'R46_context_exact_prior_SVD_preserve_channels' for a in ARMS}
FRACTIONS=dict(zip(ARMS,(.625,.75)));CONDITIONAL_WEIGHT={a:.05 for a in ARMS}
PARAMETERS={'prior625_cond': 329404, 'prior750_cond': 343974};CONTEXT_PARAMETERS={a:237927 for a in ARMS};PRIOR_PARAMETERS={'prior625_cond': 91477, 'prior750_cond': 106047}
GPUS=dict(zip(ARMS,(2,3)))
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=parent.RUN,'bbb8daf817de050b958d217b70d4c33f4aefe341','prune_refine'
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=parent.PARENT_RUN,parent.PARENT_COMMIT,parent.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=parent.F_RUN,parent.F_SEED,parent.VIEWS,parent.KIND
MODEL_SEED,DATA_SEED=parent.MODEL_SEED,parent.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO=20,4940,256,256
PREDECESSOR=predecessor.RUN;PREDECESSOR_BY_GPU={g:PREDECESSOR for g in GPUS.values()};PREDECESSOR_MODULE={PREDECESSOR:'experiments.pamf_r46_conditional_factor.runner'}
SCORING_RUN=predecessor.SCORING_RUN
def teacher_weight(epoch):
    if type(epoch) is not int or not 1<=epoch<=20:raise ValueError('Fixed20epochs required')
    return 1.
def learning_rate(step):
    if type(step) is not int or not 0<=step<4940:raise ValueError('Fixed4940steps required')
    if step<247:return 2e-4*(step+1)/247
    return 2e-5+(2e-4-2e-5)*.5*(1+math.cos(math.pi*(step-247)/(4940-247-1)))
def configuration():
    return dict(schema='R46_prior_preserve_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,context_parameters=CONTEXT_PARAMETERS,prior_parameters=PRIOR_PARAMETERS,
        compact_parent_run=COMPACT_PARENT_RUN,compact_parent_commit=COMPACT_PARENT_COMMIT,compact_parent_row=COMPACT_PARENT_ROW,
        parent_choice='Human-designated source-final350521 after prior exposure;only direct parent and teacher;not any new student',
        architecture='Exact original237927coarse/action frozen;onlysharedprior savinglongConv SVD625/750;alloriginalchannels/GN/bias and3stages6CG retained',
        factor_rank_fraction=FRACTIONS,conditional_weight=CONDITIONAL_WEIGHT,initialization_IQ_examples=0,context_exact=True,context_trainable=False,shared_context_forward=True,
        objective='Pairedclean IQNMSE +.05 relativepower STFT64/128/256 +constant1 three-stage IQNMSE to original350521 +.05 conditionalprior relativeNMSE on detached actualstudent inputs',
        conditional_normalization='Teacherprior output mean square clamped1e-4 perexample;three stages mean;target/input detached',
        paired_clean_optimized=True,training_F_access=False,epochs=20,updates=4940,batch=256,microbatch=256,optimizer='Reset prior-onlyAdamW',LR=2e-4,final_LR=2e-5,warmup_updates=247,weight_decay=1e-4,gradient_clip=1.,FP32=True,TF32=False,
        model_seed=MODEL_SEED,data_seed=DATA_SEED,received_epochs=list(range(1,21)),L_count=6300,U_count=56700,L_exposures=126000,U_exposures=1134000,source_IQ_examples=1260000,
        teacher_IQ_inputs=1260000,shared_context_inputs=1260000,student_context_inputs=0,conditional_teacher_prior_calls_per_source_example=3,
        TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_training=False,target_training=False,
        inherited_component_epochs=80,cumulative_coarse_epochs=80,cumulative_action_epochs=80,cumulative_prior_epochs=100,cumulative_component_epoch_sum=260,cumulative_serial_training_passes=140,
        freeze_rule='Both fixedfinalE20/4940;everyrow source7 and formal168000x7 without sourcegate;independenttruthlast after complete row prediction',
        views=list(VIEWS),target_count_per_view=168000,target_acceptance_allowance_pp=.8,acceptance_baselines=['human_R46_350521','original_R17_historical1pp'],
        GPU=GPUS,max_own_GPU_workers=2,max_own_workers_per_card=1,predecessor_by_GPU={str(g):r for g,r in PREDECESSOR_BY_GPU.items()},capacity_lock='.pamf-reception-capacity4.lock',source_min_free_MB=18000,all_owner_cap=2,CPU_threads=2,max_own_scorers=1,scoring_run=SCORING_RUN,
        stop_rule='Affected technicalfailure preserved;healthy peer/otherowners unchanged;no automaticretry/hotedit/stop',
        inference='Full own G,nativefloat32[B,2,256]IQ only;originalF50/defaulthead evaluation only;no teacher/clean/conditional branch at deployment',
        difference_from_main='Main prunes GNchannels;this route keeps all channels/GN,exact physical context and factors only prior kernels;performance before compression',
        source_basis='BeforeR18 target access,its complete sourceepochs1..13/12 show nonzero teacherstage IQgap and initial large coarse/action sensitivities;test preserving physical context as independent hypothesis',
        cost_plan='Samecard64sourceV,G andG+F,B1/B32 three alternating rounds;latency/throughput/resident/peak/fullparams/bytes/ConvLinearMAC excluding FFT/CG/norm',
        claim='F-free TX-unlabeled simulatedpairedsource learning;single seed exposed engineeringbenchmark,not blind validation or perdetailTX proof',target_feedback=False,goal_complete=False)
