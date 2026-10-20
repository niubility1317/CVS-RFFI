"""Fixed-source teacher-anchor optimization of the existing two compression widths."""
from experiments.pamf_r46_multipoint_fast import design as previous
from experiments.pamf_r46_compact_refine import design as widths
parent=previous.parent
learning_rate=previous.learning_rate
RUN='20261021-phase1-restoration-R46-anchor-refine-s2026092701-m3-r53'
ARMS=('prune_decay','retain_anchor','prune_anchor')
IS_RETAIN={a:a=='retain_anchor' for a in ARMS}
TASK={a:False for a in ARMS}
MODE={a:('R46_350521_exact_continuation' if IS_RETAIN[a] else 'R46_GN4_weight_importance_further_pruning') for a in ARMS}
WIDTHS={a:widths.WIDTHS['retain_refine' if IS_RETAIN[a] else 'prune_refine'] for a in ARMS}
CONTEXT_WIDTHS=WIDTHS.copy()
SPECTRAL_WIDTHS={a:widths.SPECTRAL_WIDTHS['retain_refine' if IS_RETAIN[a] else 'prune_refine'] for a in ARMS}
PARAMETERS={a:350521 if IS_RETAIN[a] else 273765 for a in ARMS}
CONTEXT_PARAMETERS={a:237927 if IS_RETAIN[a] else 185615 for a in ARMS}
GPUS={'prune_decay':2,'retain_anchor':3,'prune_anchor':2}
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=previous.COMPACT_PARENT_RUN,previous.COMPACT_PARENT_COMMIT,previous.COMPACT_PARENT_ROW
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=previous.PARENT_RUN,previous.PARENT_COMMIT,previous.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=previous.F_RUN,previous.F_SEED,previous.VIEWS,previous.KIND
MODEL_SEED,DATA_SEED=previous.MODEL_SEED,previous.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO=20,4940,256,256
SCORING_RUN=previous.SCORING_RUN
REUSED_RETAIN_CONTROL=dict(run=previous.RUN,row='retain_refine',method_commit='e96397bc3c7036f085317f9cc0e59074fb6ee24f',parameters=350521,epochs=20,updates=4940,micro=256)

def teacher_weight(epoch,arm):
    if arm not in ARMS or type(epoch) is not int or not 1<=epoch<=20:raise ValueError('Registered arm and twenty-epoch schedule required')
    return previous.teacher_weight(epoch) if arm=='prune_decay' else 1.

def configuration():
    value=previous.configuration()
    value.update(schema='R46_teacher_anchor_refine_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,context_parameters=CONTEXT_PARAMETERS,
        widths={a:list(v) for a,v in WIDTHS.items()},context_widths={a:list(v) for a,v in CONTEXT_WIDTHS.items()},spectral_widths={a:list(v) for a,v in SPECTRAL_WIDTHS.items()},
        objective='Final pairedclean IQNMSE +.05 relative-power multiscale STFT + mean3stage IQNMSE to frozen humanR46 parent;weight constant1 in anchor rows versus linear1to.1 in decay control',
        teacher_weight_schedule={a:[teacher_weight(e,a) for e in range(1,21)] for a in ARMS},
        hypothesis='Source-only complete objective audit showed reconstruction and frozen-reader accuracy can diverge;test sustained per-input teacher fidelity without F/TX supervision;not motivated by target reranking or C4 equivariance assumption',
        reused_retain_decay_control=REUSED_RETAIN_CONTROL,control_policy='Reuse exact R49retain micro256 decay control;new prune_decay matches anchor pruning/micro256/seed/budget;R48prune micro64 retained as historical throughput comparator,not exact matchedcontrol',
        freeze_rule='All three new rows fixedfinalE20/4940;no earlystop/source selection;each automatically full7formal after source/profile then independentCPUtruthlast;no target-feedback adjustments',
        GPU=GPUS,max_own_GPU_workers=2,max_own_workers_per_card=1,predecessor_by_GPU={},all_owner_cap=2,
        authorization='Standing human compression/performance goal;latestperformancefirst;physicalGPU2/3only,twoactiveworkers,onepercard;thirdrowqueuesonGPU2;otherownersunchanged',
        throughput_control='Unchanged directfloat32buffer/micro256/global256;no extra inference branch or parameter;teacher only during training',
        row_order=list(ARMS),target_feedback=False,goal_complete=False)
    value.pop('max_own_across_R48_R49',None);value.pop('healthy_R48_unchanged',None)
    return value
