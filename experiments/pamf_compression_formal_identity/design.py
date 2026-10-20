from experiments.pamf_r44_pruned_refine import design as parent
from experiments.pamf_r46_anchor_refine import design as r53
from experiments.pamf_r46_source_prune import design as r54
from experiments.pamf_r46_fold_compensate import design as r55
from experiments.pamf_multipoint_formal_identity import design as cached

RUN='20261021-diagnostic-restoration-compression-formal-analytic-TX-s2026092701-m7-r56'
PARENT,PARENT_COMMIT=r55.RUN,'6bea50605a1c2cfb49645b5c2f7ba8306b39791d'
PARENTS={r53.RUN:dict(commit='df36c3d3da4e026b747776b65b58d4921dc1a71e',rows=list(r53.ARMS)),
    r54.RUN:dict(commit='554c52baf75ab8da2618a5480851b74271018103',rows=list(r54.ARMS)),
    r55.RUN:dict(commit=PARENT_COMMIT,rows=list(r55.ARMS))}
ROWS={**{'r53_'+a:dict(run=r53.RUN,row=a,parameters=r53.PARAMETERS[a]) for a in r53.ARMS},
    **{'r54_'+a:dict(run=r54.RUN,row=a,parameters=r54.PARAMETERS[a]) for a in r54.ARMS},
    **{'r55_'+a:dict(run=r55.RUN,row=a,parameters=r55.PARAMETERS[a]) for a in r55.ARMS}}
NEW_ROWS=tuple(ROWS)
BASELINE,BASELINE_COMMIT,BASELINE_ROW=cached.BASELINE,cached.BASELINE_COMMIT,cached.BASELINE_ROW
REFERENCE,REFERENCE_COMMIT=cached.REFERENCE,cached.REFERENCE_COMMIT
FORMAL,TRUTH_RUN,VIEWS,GROUPS,COUNT=cached.FORMAL,cached.TRUTH_RUN,cached.VIEWS,cached.GROUPS,cached.COUNT
CACHED_RUN,CACHED_COMMIT=cached.RUN,'3c03b2e8a8ad41e23c1785de801c295e54b2eec9'
CACHED_FEATURE_READOUTS=list(cached.READOUTS)
CACHED_READOUTS=dict(raw='raw',r46_350521='r46_350521',r49_retain_control='r49_retain_refine')
CANDIDATES=('r46_350521','r49_retain_control',*NEW_ROWS)
READOUTS=('raw',*CANDIDATES)
COMPARISONS=tuple([(a,'r46_350521') for a in NEW_ROWS]+[('r53_retain_anchor','r49_retain_control'),('r53_prune_anchor','r53_prune_decay')]
    +[('r54_'+a,'r53_prune_decay') for a in r54.ARMS]+[('r55_'+a,'r53_prune_decay') for a in r55.ARMS]+[('r55_fold_full','r55_fold_half')])
SCORING_RUN=parent.SCORING_RUN
CACHED_CELLS=len(CACHED_READOUTS)*len(VIEWS)*len(GROUPS)
NEW_CELLS=len(NEW_ROWS)*len(VIEWS)*len(GROUPS)
TOTAL_CELLS=CACHED_CELLS+NEW_CELLS
EXPECTED_RECORDS=len(CANDIDATES)*len(VIEWS)*len(GROUPS)*(1+6+7+4)
EXPECTED_CONTROLS=len(CANDIDATES)*len(VIEWS)*len(GROUPS)*3
EXPECTED_COMPARISONS=len(COMPARISONS)*len(VIEWS)*len(GROUPS)

def configuration(commit):
    return dict(schema='compression_formal_F_independent_identity_v1',run=RUN,commit=commit,parent=PARENT,parent_commit=PARENT_COMMIT,parents=PARENTS,rows=ROWS,
        baseline=BASELINE,baseline_commit=BASELINE_COMMIT,reference=REFERENCE,reference_commit=REFERENCE_COMMIT,count=COUNT,views=list(VIEWS),groups=GROUPS,
        candidates=list(CANDIDATES),readouts=list(READOUTS),cached_run=CACHED_RUN,cached_commit=CACHED_COMMIT,cached_feature_cells=CACHED_CELLS,new_feature_cells=NEW_CELLS,
        total_feature_cells=TOTAL_CELLS,new_analytic_transform_samples=COUNT*len(VIEWS)*len(NEW_ROWS),calibration_count=8192,new_calibration_samples=0,
        coordinates='Exactly fixed R34/R39/R47/R50 mirror64/spectral4/reference24/combined92;unchanged sourceLU moments/std floor.05;no new descriptors or thresholds',
        reference_rule='Post-score evaluation clean reference only;cap2 lowest opaque physicalIDs per TX/RX/day;RX and day both different;complete6TX cells equal weight',
        label_use='Evaluation statistics after all280features fixed and all7newrows full7formal scoring complete;no labels in analytic transform or parent training',
        controls=dict(permutations=199,seed=2026102034,query_labels='within RX/day',reference_labels='fixed',is_p_value=False),comparisons=[list(v) for v in COMPARISONS],
        expected_records=EXPECTED_RECORDS,expected_controls=EXPECTED_CONTROLS,expected_comparisons=EXPECTED_COMPARISONS,
        CPU_threads=2,GPU_workers=0,max_own_scorers=1,scoring_lock=SCORING_RUN+'/score.lock',F_weights_read=0,F_embeddings_logits_read=0,model_checkpoint_loads=0,
        extra_model_predictions=0,neural_training_updates=0,head_fit_updates=0,parent_training_or_decisions_changed=False,
        wait_rule='All R53/R54/R55 fullformal scoring completed before target IQ/features;never gates or delays parent training/testing',
        claim='Descriptive same-TX association only;clean descriptor may fail TX2;negative restored margin alone not proof of wrongTX;not per-detail or TX-exclusive authenticity,not a new blind validation',
        stop_rule='Parent technical failure/wrong paths,IDs,calibration,missing/nonfinite inputs/features or outputcollision:preserve outputs,no retry;healthy parents untouched',goal_complete=False)
