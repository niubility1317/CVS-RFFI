"""Fixed before R62 source/target accuracy access; no new restorer architecture."""
from experiments.pamf_r46_group_drop import design as base
from experiments.pamf_r46_fold_compensate import design as moments

RUN='20261021-phase1-restoration-R46-whole-group-fold-s2026092701-m2-r64'
ARMS=('fold_full','fold_half')
FOLD_STRENGTH=dict(fold_full=1.,fold_half=.5)
PARAMETERS={arm:248828 for arm in ARMS}
CONTEXT_PARAMETERS={arm:136234 for arm in ARMS}
PRIOR_PARAMETERS=112594
CONTEXT_GROUPS={arm:3 for arm in ARMS}
GPUS=dict(fold_full=3,fold_half=2)
MATCHED_RUN,MATCHED_COMMIT,MATCHED_ROW=base.RUN,'7a8fa7bd836946c3493635cbe0966591ecf14e1e','group_preserve'
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=base.COMPACT_PARENT_RUN,base.COMPACT_PARENT_COMMIT,base.COMPACT_PARENT_ROW
RIDGE_FRACTION=moments.RIDGE_FRACTION
CAL_POSITIONS=moments.CAL_POSITIONS
CAL_SEED,CAL_PER_ROLE,CAL_BATCH=2026102064,64,16
CAL_MODULE_COUNT=103
parent=base.parent
TASK={a:False for a in ARMS}
MODE={a:'whole_context_group_source_moment_fold' for a in ARMS}
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=base.PARENT_RUN,base.PARENT_COMMIT,base.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=base.F_RUN,base.F_SEED,base.VIEWS,base.KIND
MODEL_SEED,DATA_SEED=base.MODEL_SEED,base.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO=20,4940,256,256
SCORING_RUN=base.SCORING_RUN
teacher_weight,learning_rate=base.teacher_weight,base.learning_rate
PREDECESSORS={3:(base.RUN,'group_preserve'),2:(base.RUN,'regroup_control')}


def configuration():
    value=base.configuration()
    value.pop('matched_uncompressed_control',None)
    value.update(run=RUN,arms=list(ARMS),parameters=PARAMETERS,context_parameters=CONTEXT_PARAMETERS,context_GN_groups=CONTEXT_GROUPS,GPU=GPUS,
        widths={a:list(base.WIDTHS['group_preserve']) for a in ARMS},context_widths={a:list(base.CONTEXT_WIDTHS['group_preserve']) for a in ARMS},
        spectral_widths={a:list(base.SPECTRAL_WIDTHS['group_preserve']) for a in ARMS},
        schema='R46_whole_context_group_source_fold_v1',predecessor_by_GPU={str(g):dict(run=v[0],row=v[1],commit=MATCHED_COMMIT) for g,v in PREDECESSORS.items()},fold_strength=FOLD_STRENGTH,ridge_fraction=RIDGE_FRACTION,
        architecture='Unchanged R62 GN3 context/full112594prior/3sharedprior6CG;two source inputweight folding initializations',
        initialization='M35 direct weight-only complete-group GN3 map plus inputweight correction from a complete shared source-only teacher input second-moment calibration',
        shared_calibration=dict(parent_G_inputs=256,unique_source_IQ=128,L_exposures=128,U_exposures=128,
            source_views=['01','20'],TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0,clean_IQ_reads=0),
        matched_control=dict(run=MATCHED_RUN,commit=MATCHED_COMMIT,row=MATCHED_ROW,parameters=248828,training_repeated=False),
        comparison='Same GN3/fullprior/wholegroups/sourcebudget/IQteacher as R62group_preserve;only source-derived inputweight initialization differs;full versushalf correction',
        prior_copied_exactly=True,new_calibration_IQ=256,new_G_forwards=256,calibration_shared_once=True,new_trainable_parameters=0,new_analytic_projection_fits=True,
        authorization='Existing compressed U-Net optimization,performance first;GPU3/2 each1own/max2acrossR62R64;newGPUallocation only after samecardR62rowcomplete+actualPIDexit;healthy jobs untouched',
        goal_complete=False,target_feedback=False)
    return value
