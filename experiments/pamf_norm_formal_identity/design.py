"""Fixed R57/R58 post-score diagnostic; unchanged R56 mathematical evaluation."""
from experiments.pamf_r44_pruned_refine import design as parent
from experiments.pamf_r46_gn_response import design as r57
from experiments.pamf_r46_stat_transport import design as r58
from experiments.pamf_compression_formal_identity import design as cached

RUN='20261021-diagnostic-restoration-normalization-formal-analytic-TX-s2026092701-m4-r59'
PARENT,PARENT_COMMIT=r58.RUN,'3e1c14a733b2278c659c653c75a98657663cd8d4'
PARENTS={r57.RUN:dict(commit='38a1b4da6dd2964967cebb67935ef877df11c837',rows=list(r57.ARMS)),
    r58.RUN:dict(commit=PARENT_COMMIT,rows=list(r58.ARMS))}
ROWS={**{'r57_'+a:dict(run=r57.RUN,row=a,parameters=r57.PARAMETERS[a]) for a in r57.ARMS},
    **{'r58_'+a:dict(run=r58.RUN,row=a,parameters=r58.PARAMETERS[a]) for a in r58.ARMS}}
NEW_ROWS=tuple(ROWS)
BASELINE,BASELINE_COMMIT,BASELINE_ROW=cached.BASELINE,cached.BASELINE_COMMIT,cached.BASELINE_ROW
REFERENCE,REFERENCE_COMMIT=cached.REFERENCE,cached.REFERENCE_COMMIT
FORMAL,TRUTH_RUN,VIEWS,GROUPS,COUNT=cached.FORMAL,cached.TRUTH_RUN,cached.VIEWS,cached.GROUPS,cached.COUNT
CACHED_RUN,CACHED_COMMIT=cached.RUN,'c7c1641d3e4555eb69c80fd4a601ed5e66751166'
CACHED_FEATURE_READOUTS=list(cached.READOUTS)
CACHED_READOUTS=dict(raw='raw',r46_350521='r46_350521',r55_fold_full_reference='r55_fold_full')
CANDIDATES=('r46_350521','r55_fold_full_reference',*NEW_ROWS)
READOUTS=('raw',*CANDIDATES)
COMPARISONS=tuple([(a,'r46_350521') for a in NEW_ROWS]+[(a,'r55_fold_full_reference') for a in NEW_ROWS]
    +[('r57_gn_full','r57_gn_half'),('r58_calibrated','r58_identity_control')])
SCORING_RUN=parent.SCORING_RUN
CACHED_CELLS=len(CACHED_READOUTS)*len(VIEWS)*len(GROUPS)
NEW_CELLS=len(NEW_ROWS)*len(VIEWS)*len(GROUPS)
TOTAL_CELLS=CACHED_CELLS+NEW_CELLS
EXPECTED_RECORDS=len(CANDIDATES)*len(VIEWS)*len(GROUPS)*(1+6+7+4)
EXPECTED_CONTROLS=len(CANDIDATES)*len(VIEWS)*len(GROUPS)*3
EXPECTED_COMPARISONS=len(COMPARISONS)*len(VIEWS)*len(GROUPS)

def configuration(commit):
    return dict(schema='normalization_formal_F_independent_identity_v1',run=RUN,commit=commit,parent=PARENT,parent_commit=PARENT_COMMIT,parents=PARENTS,rows=ROWS,
        baseline=BASELINE,baseline_commit=BASELINE_COMMIT,reference=REFERENCE,reference_commit=REFERENCE_COMMIT,count=COUNT,views=list(VIEWS),groups=GROUPS,
        candidates=list(CANDIDATES),readouts=list(READOUTS),cached_run=CACHED_RUN,cached_commit=CACHED_COMMIT,cached_feature_cells=CACHED_CELLS,new_feature_cells=NEW_CELLS,
        total_feature_cells=TOTAL_CELLS,new_analytic_transform_samples=COUNT*len(VIEWS)*len(NEW_ROWS),calibration_count=8192,new_calibration_samples=0,
        coordinates='Unchanged R34/R39/R47/R50/R56 mirror64/spectral4/reference24/combined92;sourceLU moments/std floor.05;no new descriptors/thresholds',
        reference_rule='Postscore clean reference only;cap2 lowest opaque physicalIDs per TX/RX/day;RX anddaybothdifferent;complete6TX cells equal weight',
        label_use='All196cells fixed andall4rows full7formal scoring completed before evaluationtruth;0labels in transform/training;R58 scores not consumed in this design',
        controls=dict(permutations=199,seed=2026102034,query_labels='within RX/day',reference_labels='fixed',is_p_value=False),comparisons=[list(v) for v in COMPARISONS],
        expected_records=EXPECTED_RECORDS,expected_controls=EXPECTED_CONTROLS,expected_comparisons=EXPECTED_COMPARISONS,
        CPU_threads=2,GPU_workers=0,max_own_scorers=1,scoring_lock=SCORING_RUN+'/score.lock',F_weights_read=0,F_embeddings_logits_read=0,model_checkpoint_loads=0,
        extra_model_predictions=0,neural_training_updates=0,head_fit_updates=0,parent_training_or_decisions_changed=False,
        wait_rule='AllR57andR58 fixed4rows full7formal scored before target IQ/features;doesnot gate/delay parent testing/delivery',
        matched_controls='R57 same273765full/half strength;R58 same276453calibrated/identity graph;R55oldnormalization reference notcompleteparam/compute matched toR58',
        claim='Descriptive sameTXassociation only;weak/negative cleanTX2 retained;not perdetail or TXexclusive authenticity/not newblind validation;no parent/method selection',
        stop_rule='Parent failure/wrongpaths/IDs/calibration/missingnonfiniteinputs/features/outputcollision:preserveoutputs,no retry;healthyparentsuntouched',goal_complete=False)
