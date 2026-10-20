from experiments.pamf_r44_pruned_refine import design as parent
from experiments.pamf_r46_conditional_factor import design as r18
from experiments.pamf_r46_prior_preserve import design as r19
from experiments.pamf_r46_complex_fidelity import design as r22
from experiments.pamf_r46_activation_factor import design as r23
from experiments.pamf_multipoint_formal_identity import design as cached

RUN='20261021-diagnostic-restoration-factor-formal-analytic-TX-s2026092701-m8-r24'
PARENT,PARENT_COMMIT=r23.RUN,'816b6fceac537d644be3425b8186159ecdd6300b'
PARENTS={r18.RUN:dict(commit='aca99499d3f0170df71380f6429ac9a01ef412f8',rows=list(r18.ARMS)),
    r19.RUN:dict(commit='3ac37ca1138f266f5fd91f74e7feca0eb11911cc',rows=list(r19.ARMS)),
    r22.RUN:dict(commit='e190cf5367b03da8aafced51276a8730ad8274fa',rows=list(r22.ARMS)),
    r23.RUN:dict(commit=PARENT_COMMIT,rows=list(r23.ARMS))}
ROWS={**{'r18_'+a:dict(run=r18.RUN,row=a,parameters=r18.PARAMETERS[a],updates=r18.UPDATES) for a in r18.ARMS},
    **{'r19_'+a:dict(run=r19.RUN,row=a,parameters=r19.PARAMETERS[a],updates=r19.UPDATES) for a in r19.ARMS},
    **{'r22_'+a:dict(run=r22.RUN,row=a,parameters=r22.PARAMETERS[a],updates=r22.UPDATES) for a in r22.ARMS},
    **{'r23_'+a:dict(run=r23.RUN,row=a,parameters=r23.PARAMETERS[a],updates=r23.UPDATES) for a in r23.ARMS}}
NEW_ROWS=tuple(ROWS)
BASELINE,BASELINE_COMMIT,BASELINE_ROW=cached.BASELINE,cached.BASELINE_COMMIT,cached.BASELINE_ROW
REFERENCE,REFERENCE_COMMIT=cached.REFERENCE,cached.REFERENCE_COMMIT
FORMAL,TRUTH_RUN,VIEWS,GROUPS,COUNT=cached.FORMAL,cached.TRUTH_RUN,cached.VIEWS,cached.GROUPS,cached.COUNT
CACHED_RUN,CACHED_COMMIT=cached.RUN,'3c03b2e8a8ad41e23c1785de801c295e54b2eec9'
CACHED_FEATURE_READOUTS=list(cached.READOUTS)
CACHED_READOUTS=dict(raw='raw',r46_350521='r46_350521')
CANDIDATES=('r46_350521',*NEW_ROWS)
READOUTS=('raw',*CANDIDATES)
COMPARISONS=tuple([(a,'r46_350521') for a in NEW_ROWS]+[
    ('r18_factor625_cond','r18_factor625_stage'),('r19_prior750_cond','r19_prior625_cond'),
    ('r22_complex625_identity','r22_complex625'),('r23_activation625_identity','r22_complex625_identity'),
    ('r23_activation500_identity','r23_activation625_identity')])
SCORING_RUN=parent.SCORING_RUN
CACHED_CELLS=len(CACHED_READOUTS)*len(VIEWS)*len(GROUPS)
NEW_CELLS=len(NEW_ROWS)*len(VIEWS)*len(GROUPS)
TOTAL_CELLS=CACHED_CELLS+NEW_CELLS
EXPECTED_RECORDS=len(CANDIDATES)*len(VIEWS)*len(GROUPS)*(1+6+7+4)
EXPECTED_CONTROLS=len(CANDIDATES)*len(VIEWS)*len(GROUPS)*3
EXPECTED_COMPARISONS=len(COMPARISONS)*len(VIEWS)*len(GROUPS)


def configuration(commit):
    return dict(schema='factor_formal_F_independent_identity_v1',run=RUN,commit=commit,parent=PARENT,parent_commit=PARENT_COMMIT,parents=PARENTS,rows=ROWS,
        baseline=BASELINE,baseline_commit=BASELINE_COMMIT,reference=REFERENCE,reference_commit=REFERENCE_COMMIT,count=COUNT,views=list(VIEWS),groups=GROUPS,
        candidates=list(CANDIDATES),readouts=list(READOUTS),cached_run=CACHED_RUN,cached_commit=CACHED_COMMIT,cached_feature_cells=CACHED_CELLS,new_feature_cells=NEW_CELLS,
        total_feature_cells=TOTAL_CELLS,new_analytic_transform_samples=COUNT*len(VIEWS)*len(NEW_ROWS),calibration_count=8192,new_calibration_samples=0,
        coordinates='Unchanged R34/R47/R50/R56 mirror64/spectral4/reference24/combined92;reuse sourceLU moments/std floor.05;no descriptor or threshold changes',
        reference_rule='Post-score evaluation clean reference only;cap2 lowest opaque physicalIDs per TX/RX/day;RX and day both different;complete6TX cells equal weight',
        label_use='Evaluation statistics only after all280features fixed and all8newrows full7formal scoring complete;no labels in transform or parent training',
        controls=dict(permutations=199,seed=2026102034,query_labels='within RX/day',reference_labels='fixed',is_p_value=False),comparisons=[list(v) for v in COMPARISONS],
        expected_records=EXPECTED_RECORDS,expected_controls=EXPECTED_CONTROLS,expected_comparisons=EXPECTED_COMPARISONS,
        CPU_threads=2,GPU_workers=0,max_own_scorers=1,scoring_lock=SCORING_RUN+'/score.lock',F_weights_read=0,F_embeddings_logits_read=0,model_checkpoint_loads=0,
        extra_model_predictions=0,neural_training_updates=0,head_fit_updates=0,parent_training_or_decisions_changed=False,
        wait_rule='All R18/R19/R22/R23 fullformal scores complete before savedtargetIQ/feature access;no gating or delaying parent training/testing',
        inclusion='All8 newlytrained versions in this factor branch,including R19negative results;R20 sameweights/equalIQ reuse existing R18 functions without duplicate neural predictions',
        claim='Descriptive same-TX association only;clean descriptor fails to separate someTX2;negative margin alone not wrongTX or perdetail authenticity proof;exposed engineering benchmark',
        stop_rule='Parent technicalfailure/wrongpath or IDs/calibration/missing or nonfinite inputs/features/outputcollision:preserve,no automaticretry;healthy parents untouched',goal_complete=False)
