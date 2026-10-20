from experiments.pamf_r44_pruned_refine import design as parent
from experiments.pamf_r46_compact_refine import design as r48
from experiments.pamf_r46_multipoint_fast import design as r49
from experiments.pamf_analytic_identity_probe import design as source

RUN='20261020-diagnostic-restoration-multipoint-formal-analytic-TX-s2026092701-m9-r50'
PARENT,PARENT_COMMIT=r49.RUN,'e96397bc3c7036f085317f9cc0e59074fb6ee24f'
PARENTS={r48.RUN:dict(commit='86b89ba6bc09b0b3f10fcaff8fc7cd1ecf6d0508',rows=list(r48.ARMS)),r49.RUN:dict(commit=PARENT_COMMIT,rows=list(r49.ARMS))}
BASELINE,BASELINE_COMMIT,BASELINE_ROW=parent.RUN,'bbb8daf817de050b958d217b70d4c33f4aefe341','prune_refine'
ROWS={**{'r48_'+a:dict(run=r48.RUN,row=a,parameters=r48.PARAMETERS[a]) for a in r48.ARMS},**{'r49_'+a:dict(run=r49.RUN,row=a,parameters=r49.PARAMETERS[a]) for a in r49.ARMS}}
NEW_ROWS=tuple(ROWS)
REFERENCE,REFERENCE_COMMIT=source.RUN,'60f0d924aa0a43f4213c683df17d7ae9e2ea54ee'
FORMAL='20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05'
TRUTH_RUN='20260927-phase1-baselines-final-clean-satellite-m5-r01'
VIEWS,GROUPS=source.VIEWS,source.GROUPS
COUNT=168000
CACHED_RUN='20261020-diagnostic-restoration-R46-formal-analytic-TX-s2026092701-m2-r47'
CACHED_COMMIT='a29dc38f7dd527edbf250c4d5b0e88f40f8ed032'
CACHED_READOUTS={'raw':'raw','r46_350521':'prune_refine'}
CANDIDATES=('r46_350521',*NEW_ROWS)
READOUTS=('raw',*CANDIDATES)
COMPARISONS=tuple([(a,'r46_350521') for a in NEW_ROWS]+[('r48_prune_refine','r48_retain_refine')]+[('r49_'+a,'r49_retain_refine') for a in r49.ARMS if a!='retain_refine']+[('r49_retain_refine','r48_retain_refine')])
SCORING_RUN=parent.SCORING_RUN

def configuration(commit):
    return dict(schema='multipoint_formal_F_independent_identity_v1',run=RUN,commit=commit,parent=PARENT,parent_commit=PARENT_COMMIT,parents=PARENTS,rows=ROWS,
        baseline=BASELINE,baseline_commit=BASELINE_COMMIT,reference=REFERENCE,reference_commit=REFERENCE_COMMIT,count=COUNT,views=list(VIEWS),groups=GROUPS,
        candidates=list(CANDIDATES),readouts=list(READOUTS),cached_run=CACHED_RUN,cached_commit=CACHED_COMMIT,cached_feature_cells=56,new_feature_cells=224,
        new_analytic_transform_samples=9408000,calibration_count=8192,new_calibration_samples=0,
        coordinates='Exactly R34/R39/R47 mirror64/spectral4/reference24/combined92;unchanged LU moments/std floor.05',
        reference_rule='Post-score evaluation clean reference only;cap2 lowest opaque physicalIDs per TX/RX/day;RX and day both different;complete6TX cells equal weight',
        label_use='Evaluation statistics after all280features fixed and all8newrows full7formal scores complete;no labels in transform or parent training',
        controls=dict(permutations=199,seed=2026102034,query_labels='within RX/day',reference_labels='fixed',is_p_value=False),comparisons=[list(v) for v in COMPARISONS],
        CPU_threads=2,GPU_workers=0,max_own_scorers=1,scoring_lock=SCORING_RUN+'/score.lock',F_weights_read=0,F_embeddings_logits_read=0,model_checkpoint_loads=0,
        extra_model_predictions=0,neural_training_updates=0,head_fit_updates=0,parent_training_or_decisions_changed=False,
        wait_rule='R48both andR49all6 fullformal scores complete before target IQ/features;never gates or delays parent testing',
        claim='Descriptive same-TX association on existing exposed formal benchmark;not classifier,new blind validation or per-detail/TX-exclusive authenticity proof',
        stop_rule='Parent failure/wrong paths,IDs,calibration,missing/nonfinite inputs/features or outputcollision:preserve outputs,no retry;healthy parents untouched',goal_complete=False)
