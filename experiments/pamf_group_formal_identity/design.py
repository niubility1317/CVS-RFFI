"""Fixed R62 post-score diagnostic; unchanged source coordinates and math."""
from experiments.pamf_r44_pruned_refine import design as parent
from experiments.pamf_r46_group_drop import design as r62
from experiments.pamf_compression_formal_identity import design as cached

RUN='20261021-diagnostic-restoration-whole-group-formal-analytic-TX-s2026092701-m2-r63'
PARENT,PARENT_COMMIT=r62.RUN,'7a8fa7bd836946c3493635cbe0966591ecf14e1e'
PARENTS={PARENT:dict(commit=PARENT_COMMIT,rows=list(r62.ARMS))}
ROWS={'r62_'+a:dict(run=PARENT,row=a,parameters=r62.PARAMETERS[a]) for a in r62.ARMS}
CONTROL_RUN='20261020-phase1-restoration-R46-multipoint-fast-s2026092701-m6-r49'
CONTROL_COMMIT='e96397bc3c7036f085317f9cc0e59074fb6ee24f'
CONTROL_ROW='retain_refine'
BASELINE,BASELINE_COMMIT,BASELINE_ROW=cached.BASELINE,cached.BASELINE_COMMIT,cached.BASELINE_ROW
REFERENCE,REFERENCE_COMMIT=cached.REFERENCE,cached.REFERENCE_COMMIT
FORMAL,TRUTH_RUN,VIEWS,GROUPS,COUNT=cached.FORMAL,cached.TRUTH_RUN,cached.VIEWS,cached.GROUPS,cached.COUNT
CACHED_RUN,CACHED_COMMIT=cached.RUN,'c7c1641d3e4555eb69c80fd4a601ed5e66751166'
CACHED_FEATURE_READOUTS=list(cached.READOUTS)
CACHED_READOUTS=dict(raw='raw',r46_350521='r46_350521')
ROWS['r49_retain_control']=dict(run=CONTROL_RUN,row=CONTROL_ROW,parameters=350521)
NEW_ROWS=tuple(ROWS)
CANDIDATES=('r46_350521','r49_retain_control',*('r62_'+a for a in r62.ARMS))
READOUTS=('raw',*CANDIDATES)
COMPARISONS=tuple([(a,'r46_350521') for a in NEW_ROWS if a!='r49_retain_control']+[(a,'r49_retain_control') for a in NEW_ROWS if a!='r49_retain_control']+[('r62_group_preserve','r62_regroup_control')])
SCORING_RUN=parent.SCORING_RUN
CACHED_CELLS=len(CACHED_READOUTS)*len(VIEWS)*len(GROUPS)
NEW_CELLS=len(NEW_ROWS)*len(VIEWS)*len(GROUPS)
TOTAL_CELLS=CACHED_CELLS+NEW_CELLS
EXPECTED_RECORDS=len(CANDIDATES)*len(VIEWS)*len(GROUPS)*(1+6+7+4)
EXPECTED_CONTROLS=len(CANDIDATES)*len(VIEWS)*len(GROUPS)*3
EXPECTED_COMPARISONS=len(COMPARISONS)*len(VIEWS)*len(GROUPS)

def configuration(commit):
    return dict(schema='whole_group_formal_F_independent_identity_v1',run=RUN,commit=commit,parent=PARENT,parent_commit=PARENT_COMMIT,parents=PARENTS,rows=ROWS,
        matched_control=dict(run=CONTROL_RUN,commit=CONTROL_COMMIT,row=CONTROL_ROW,parameters=350521),
        baseline=BASELINE,baseline_commit=BASELINE_COMMIT,reference=REFERENCE,reference_commit=REFERENCE_COMMIT,count=COUNT,views=list(VIEWS),groups=GROUPS,
        candidates=list(CANDIDATES),readouts=list(READOUTS),cached_run=CACHED_RUN,cached_commit=CACHED_COMMIT,cached_feature_cells=CACHED_CELLS,new_feature_cells=NEW_CELLS,
        total_feature_cells=TOTAL_CELLS,new_analytic_transform_samples=COUNT*len(VIEWS)*len(NEW_ROWS),calibration_count=8192,new_calibration_samples=0,
        coordinates='Unchanged R34/R39/R47/R50/R56 mirror64/spectral4/reference24/combined92;sourceLU moments/std floor.05;no new descriptors/thresholds',
        reference_rule='Postscore clean reference only;cap2 lowest opaque physicalIDs per TX/RX/day;RX anddaybothdifferent;complete6TX cells equal weight',
        label_use='All140cells fixed andbothR62rows full7formal scoring completed before evaluationtruth;0labels in transform/training;R62 source/target accuracy not consumed in this design',
        controls=dict(permutations=199,seed=2026102034,query_labels='within RX/day',reference_labels='fixed',is_p_value=False),comparisons=[list(v) for v in COMPARISONS],
        expected_records=EXPECTED_RECORDS,expected_controls=EXPECTED_CONTROLS,expected_comparisons=EXPECTED_COMPARISONS,
        CPU_threads=2,GPU_workers=0,max_own_scorers=1,scoring_lock=SCORING_RUN+'/score.lock',F_weights_read=0,F_embeddings_logits_read=0,model_checkpoint_loads=0,
        extra_model_predictions=0,neural_training_updates=0,head_fit_updates=0,parent_training_or_decisions_changed=False,
        wait_rule='BothR62fixedrows full7formal scored before target IQ/features;doesnot gate/delay parent testing/delivery',
        matched_controls='R62 GN3/GN1 identical initial weights/maps/dims/budget;R49retain sameparent/E20/4940/globalmicro256/IQteacher withdifferentwidth/norm;humanM35 differentextra-budget',
        claim='Descriptive sameTXassociation only;weak/negative cleanTX2 retained;not perdetail or TXexclusive authenticity/not newblind validation;no parent/method selection',
        stop_rule='Parent failure/wrongpaths/IDs/calibration/missingnonfiniteinputs/features/outputcollision:preserveoutputs,no retry;healthyparentsuntouched',goal_complete=False)
