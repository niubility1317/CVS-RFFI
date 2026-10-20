"""Source-only complete-GN-group pruning of the human designated350521 parent."""
import copy
from experiments.pamf_r44_pruned_refine import design as parent
from experiments.pamf_r46_multipoint_fast import design as baseline
from experiments.pamf_fullresolution_restore.training import learning_rate

RUN='20261021-phase1-restoration-R46-whole-group-s2026092701-m2-r62'
ARMS=('group_preserve','regroup_control')
TASK={arm:False for arm in ARMS}
MODE={arm:'whole_context_groups_weight_only' for arm in ARMS}
CONTEXT_GROUPS=dict(group_preserve=3,regroup_control=1)
PARAMETERS={arm:248828 for arm in ARMS}
CONTEXT_PARAMETERS={arm:136234 for arm in ARMS}
PRIOR_PARAMETERS=112594
CONTEXT_WIDTHS={arm:(9,15,24,33) for arm in ARMS}
SPECTRAL_WIDTHS={arm:(6,9,15,24) for arm in ARMS}
WIDTHS={arm:(12,20,32,44) for arm in ARMS}
GPUS=dict(group_preserve=3,regroup_control=2)
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=parent.RUN,'bbb8daf817de050b958d217b70d4c33f4aefe341','prune_refine'
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=parent.PARENT_RUN,parent.PARENT_COMMIT,parent.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=parent.F_RUN,parent.F_SEED,parent.VIEWS,parent.KIND
MODEL_SEED,DATA_SEED=parent.MODEL_SEED,parent.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO=20,4940,256,256
SCORING_RUN=parent.SCORING_RUN
MATCHED_RUN=baseline.RUN
MATCHED_COMMIT='e96397bc3c7036f085317f9cc0e59074fb6ee24f'
MATCHED_ROW='retain_refine'
teacher_weight=baseline.teacher_weight

def configuration():
    value=copy.deepcopy(baseline.configuration())
    for key in ('max_own_across_R48_R49','healthy_R48_unchanged','throughput_control'):value.pop(key,None)
    value.update(schema='R46_whole_context_group_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,context_parameters=CONTEXT_PARAMETERS,
        widths={a:list(v) for a,v in WIDTHS.items()},context_widths={a:list(v) for a,v in CONTEXT_WIDTHS.items()},
        spectral_widths={a:list(v) for a,v in SPECTRAL_WIDTHS.items()},context_GN_groups=CONTEXT_GROUPS,prior_GN_groups=4,
        original_context_GN_groups=4,retained_original_context_groups=3,prior_parameters=PRIOR_PARAMETERS,
        architecture='Same context/three-shared-long-prior/sixCG;drop complete original context GN groups;prior whole350521parent widths preserved;GN3 vsGN1 same weights/dims/parameters/budget',
        initialization='Data-free parent encoder normalized mean-squared Conv outputweight scores;sumwithin eachoriginalgroup,top3of4,stableties;same maps bothrows;skipoffset usesold widths;fullprior copied',
        layerwise_norm_equivalence='GN3 uses unchanged retained group widths;mathematical layerwise equivalence,not fullrestorer or bitwiseGPU equivalence;GN1 pools retained groups',
        matched_uncompressed_control=dict(run=MATCHED_RUN,commit=MATCHED_COMMIT,row=MATCHED_ROW,parameters=350521,training_repeated=False),
        comparison='PrimaryGN3vsGN1 samegraphweights/dims/parametercount/budget;grouping/reduction/actualcost differ;M35differentbudget,R49retain sameincrementalbudget butdifferentwidth',
        GPU=GPUS,max_own_GPU_workers=2,max_own_workers_per_card=1,predecessor_by_GPU={},
        authorization='Human requests further compression with performancefirst andmultipleGPU validation;physicalGPU2/3only,max2own;allfixedR60GPUhandles alreadyEXITED before release',
        new_calibration_IQ=0,new_trainable_parameters=0,source_score_selection=False,
        freeze_rule='BothfixedfinalE20/4940;allsourceandfull7formal automatically evenifsourceworse;no earlystop/candidatefilter or targetfeedback',
        goal_complete=False,target_feedback=False)
    return value
