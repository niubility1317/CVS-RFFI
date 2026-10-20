"""Fix the two compression schedules before any R64 performance is consumed."""
import copy
from experiments.pamf_r46_multipoint_fast import design as baseline
from experiments.pamf_r46_anchor_refine import design as static

RUN='20261021-phase1-restoration-R46-delayed-prune-s2026092701-m2-r65'
ARMS=('prune_after5','prune_after10')
WARMUP_EPOCHS=dict(prune_after5=5,prune_after10=10)
GPUS=dict(prune_after5=3,prune_after10=2)
EPOCHS,UPDATES,BATCH,MICRO=20,4940,256,256
INITIAL_PARAMETERS=350521
PARAMETERS={arm:273765 for arm in ARMS}
CONTEXT_PARAMETERS={arm:185615 for arm in ARMS}
WIDTHS={arm:(12,16,28,40) for arm in ARMS}
CONTEXT_WIDTHS=WIDTHS.copy()
SPECTRAL_WIDTHS={arm:(8,12,16,28) for arm in ARMS}
COMPACT_PARENT_RUN=baseline.COMPACT_PARENT_RUN
COMPACT_PARENT_COMMIT=baseline.COMPACT_PARENT_COMMIT
COMPACT_PARENT_ROW=baseline.COMPACT_PARENT_ROW
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=baseline.PARENT_RUN,baseline.PARENT_COMMIT,baseline.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=baseline.F_RUN,baseline.F_SEED,baseline.VIEWS,baseline.KIND
MODEL_SEED,DATA_SEED=baseline.MODEL_SEED,baseline.DATA_SEED
SCORING_RUN=baseline.SCORING_RUN
learning_rate,teacher_weight=baseline.learning_rate,baseline.teacher_weight
STATIC_CONTROL=dict(run=static.RUN,commit='df36c3d3da4e026b747776b65b58d4921dc1a71e',
    row='prune_decay',parameters=273765,epochs=20,updates=4940,used_as_teacher=False)
RETAIN_CONTROL=dict(run=baseline.RUN,commit='e96397bc3c7036f085317f9cc0e59074fb6ee24f',
    row='retain_refine',parameters=350521,epochs=20,updates=4940,used_as_teacher=False)


def phase(epoch,arm):
    if arm not in ARMS or type(epoch) is not int or not 1<=epoch<=EPOCHS:
        raise ValueError('Registered arm and one of the fixed twenty epochs required')
    return 'dense' if epoch<=WARMUP_EPOCHS[arm] else 'compact'


def configuration():
    value=copy.deepcopy(baseline.configuration())
    for key in ('max_own_across_R48_R49','healthy_R48_unchanged','throughput_control'):
        value.pop(key,None)
    value.update(schema='R46_delayed_fixed_map_pruning_v1',run=RUN,arms=list(ARMS),
        initial_parameters=INITIAL_PARAMETERS,parameters=PARAMETERS,context_parameters=CONTEXT_PARAMETERS,
        widths={a:list(v) for a,v in WIDTHS.items()},context_widths={a:list(v) for a,v in CONTEXT_WIDTHS.items()},
        spectral_widths={a:list(v) for a,v in SPECTRAL_WIDTHS.items()},warmup_epochs=WARMUP_EPOCHS,
        compact_epochs={a:EPOCHS-WARMUP_EPOCHS[a] for a in ARMS},
        schedule={a:[phase(e,a) for e in range(1,21)] for a in ARMS},
        architecture='Existing GN4 context and three tied-long-weight priors with six CG steps;static273765 final graph,zero new inference parameters',
        hypothesis='Source objective jointly adapts every original component;test retaining dense capacity in early source updates before irreversible channel removal,within the same global twenty-epoch budget',
        direct_initialization='Exactly the human-designated M35 source-final checkpoint;no trained child or R64 checkpoint',
        teacher='The same frozen original M35 for all twenty epochs;never replace with the within-row warmup student',
        map_policy='Freeze original M35 GN4-normalized encoder Conv weight-importance maps at initialization;both rows and static R53 layout use the same maps;never recompute on warmup weights or scores',
        transition='After epoch5 or10,CPU-slice only the same row source-trained student using the already fixed maps;reset AdamW once because parameter shapes change;continue global update/LR/teacher schedules without restart',
        optimizer_state_policy='One recorded reset at transition;no reset of data order,global updates,LR schedule or teacher weight',
        objective='Unchanged pairedclean IQNMSE +.05 multiscale relative-power STFT +linear1to.1 frozen M35 mean-three-stage IQNMSE',
        static_control=STATIC_CONTROL,uncompressed_incremental_control=RETAIN_CONTROL,
        comparison_limit='New rows have different dense/compact compute and one optimizer reset;historical static control has no reset,so a difference is not attributed to timing alone',
        inherited_component_epochs=80,new_total_epochs=20,cumulative_component_epochs=100,
        discarded_dense_parameters_have_new_epochs=WARMUP_EPOCHS,
        source_IQ_examples_per_row=1260000,teacher_IQ_examples_per_row=1260000,
        new_calibration_IQ=0,new_trainable_parameters=0,TX_labels_read=0,F_training_access=False,V_training=False,target_training=False,
        GPU=GPUS,max_own_GPU_workers=2,max_own_workers_per_card=1,
        predecessor_by_GPU={3:'fold_full',2:'fold_half'},predecessor_run='20261021-phase1-restoration-R46-whole-group-fold-s2026092701-m2-r64',
        freeze_rule='Both fixedfinalE20/4940;automatic source1800x7/profile/formal168000x7 for each row regardless source scores;allseven predictions fixed before independent truthlast;no earlystop or source candidate gate',
        target_feedback=False,goal_complete=False)
    return value
