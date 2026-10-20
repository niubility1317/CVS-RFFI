import math
from experiments.pamf_iq_joint_compact import design as parent

RUN='20261020-phase1-restoration-r38-successor-s2026092701-m6-r13'
LEVELS=('keep','three_quarter','half')
ARMS=tuple(level+'_'+objective for objective in ('iq','task') for level in LEVELS)
TASK={arm:arm.endswith('_task') for arm in ARMS}
LEVEL={arm:next(level for level in LEVELS if arm==level+('_task' if TASK[arm] else '_iq')) for arm in ARMS}
_WIDTHS=dict(keep=(16,32,48,64),three_quarter=(12,24,36,48),half=(8,16,24,32))
_SPECTRAL=dict(keep=(12,24,36,48),three_quarter=(8,16,24,36),half=(8,12,16,24))
WIDTHS={a:_WIDTHS[LEVEL[a]] for a in ARMS};SPECTRAL_WIDTHS={a:_SPECTRAL[LEVEL[a]] for a in ARMS}
PARAMETERS={a:dict(keep=794529,three_quarter=442805,half=204561)[LEVEL[a]] for a in ARMS}
CONTEXT_PARAMETERS={a:dict(keep=548795,three_quarter=302471,half=140315)[LEVEL[a]] for a in ARMS}
GPUS=dict(zip(ARMS,(2,3,6,7,4,5)))
COMPACT_PARENT_RUN=parent.RUN;COMPACT_PARENT_COMMIT='e0868f0a0c3cf056a1753a5b4e5ce4f0bff84381';COMPACT_PARENT_ROW='dense_half'
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=parent.PARENT_RUN,parent.PARENT_COMMIT,parent.ACTION_RUN
F_RUN='20261016-phase1-F-supcon-e50-clean-manysig-m9-r02';F_SEED=2026092701
VIEWS,KIND=parent.VIEWS,parent.KIND
MODEL_SEED,DATA_SEED=parent.MODEL_SEED,parent.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO=10,2470,256,64
R11='20261020-phase1-restoration-r17-context-grid-s2026092701-m4-r11'
R12='20261020-phase1-restoration-r17-eighth-context-s2026092701-m2-r12'
PREDECESSOR_BY_GPU={gpu:(R12 if gpu in (4,5) else R11) for gpu in range(2,8)}
PREDECESSOR_MODULE={R11:'experiments.pamf_r17_context_grid.runner',R12:'experiments.pamf_r17_eighth_context.runner'}
SCORING_RUN='20261020-phase1-restoration-r17-shared-factor-s2026092701-m2-r06'


def learning_rate(update):
    if type(update) is not int or not 0<=update<UPDATES:raise ValueError('Fixed successor update budget required')
    return 1e-4*(update+1)/247 if update<247 else 1e-4*(.1+.9*(1+math.cos(math.pi*(update-247)/(UPDATES-1-247)))/2)


def configuration():
    return dict(schema='R38_successor_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,context_parameters=CONTEXT_PARAMETERS,
        widths={a:list(w) for a,w in WIDTHS.items()},spectral_widths={a:list(w) for a,w in SPECTRAL_WIDTHS.items()},task_fidelity=TASK,
        compact_parent_run=COMPACT_PARENT_RUN,compact_parent_commit=COMPACT_PARENT_COMMIT,compact_parent_row=COMPACT_PARENT_ROW,
        compact_parent_epoch=20,compact_parent_updates=4940,original_R17_run=PARENT_RUN,original_R17_commit=PARENT_COMMIT,
        initialization='User-designated R38 dense_half fixed source-final E20;full width exact state copy or GN4-preserving channel slices;no IQ calibration',
        parent_choice='User explicitly designates 794529 benchmark-exposed model;its training/final-E20 selection source-only;not new independent blind test',
        objective='All:final paired-clean IQNMSE+1xmean3stage IQNMSE to frozen R38;task arms additionally T1KL(Fclean||Frestored)+0.5unitF160squared distance',
        teacher_weight=1.,paired_clean_optimized=True,all_coarse_action_prior_trainable=True,
        epochs=EPOCHS,updates=UPDATES,batch=BATCH,microbatch=MICRO,optimizer='Reset AdamW;fixed LR1e-4 warm247 cosine1e-5',
        weight_decay=1e-4,gradient_clip=1.,FP32=True,TF32=False,model_seed=MODEL_SEED,data_seed=DATA_SEED,
        received_epochs=list(range(1,11)),L_count=6300,U_count=56700,L_exposures=63000,U_exposures=567000,source_IQ_examples=630000,
        R38_teacher_inputs=630000,student_context_inputs=630000,F_teacher_inputs={a:630000*int(TASK[a]) for a in ARMS},
        F_student_inputs={a:630000*int(TASK[a]) for a in ARMS},TX_labels_read=0,V_training=False,target_training=False,
        inherited_component_epochs=40,cumulative_component_epochs=50,cumulative_component_epoch_sum=150,cumulative_serial_training_passes=90,
        original_F50_training_epochs=50,F_parameter_updates=0,head_fit_updates=0,initialization_IQ_examples=0,
        freeze_rule='Fixed final E10/2470 each row;no early stop or V/source-score selection;all six rows automatically predict all7target then independent truth-last',
        views=list(VIEWS),target_count_per_view=168000,target_acceptance_allowance_pp=1.,acceptance_baselines=['user_R38_794529','original_R17'],
        GPU=GPUS,max_own_GPU_workers=6,max_own_workers_per_card=1,predecessor_by_GPU={str(g):r for g,r in PREDECESSOR_BY_GPU.items()},
        capacity_lock='.pamf-reception-capacity4.lock',source_min_free_MB=18000,all_owner_cap=2,CPU_threads=2,max_own_scorers=1,scoring_run=SCORING_RUN,
        stop_rule='Technical failure preserves affected outputs;healthy rows continue;no automatic retry/fallback;failed root after healthy rows complete',
        inference='Current native-scale IQ only;one restored IQ through original F50/defaulthead;3prior/6CG;no teacher/clean/TX/RX/scene/cache/state updates',
        claim='User-designated exposed-benchmark compression;IQ controls F-free during training;task variants use original supervised F knowledge;not strictly unsupervised/global minimum/Phase2',
        target_feedback=False,goal_complete=False)
