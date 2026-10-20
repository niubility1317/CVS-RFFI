from experiments.pamf_r38_compression_methods import design as parent
from experiments.pamf_analytic_identity_probe import design as source

RUN='20261020-diagnostic-restoration-compression-formal-analytic-TX-s2026092701-m2-r45'
PARENT,PARENT_COMMIT=parent.RUN,'2f5247dbd5fe50f28defa38a0252d70e692934af'
BASELINE,BASELINE_COMMIT=parent.COMPACT_PARENT_RUN,parent.COMPACT_PARENT_COMMIT
REFERENCE,REFERENCE_COMMIT=source.RUN,'60f0d924aa0a43f4213c683df17d7ae9e2ea54ee'
FORMAL='20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05'
TRUTH_RUN='20260927-phase1-baselines-final-clean-satellite-m5-r01'
VIEWS,GROUPS=source.VIEWS,source.GROUPS
COUNT=168000
NEW_ROWS=parent.ARMS
CANDIDATES=('r38_794529',*NEW_ROWS)
READOUTS=('raw',*CANDIDATES)
ALIASES=dict(zip(READOUTS,('raw','r17_reference','dense_three_quarter','dense_half')))
SCORING_RUN=parent.SCORING_RUN


def configuration(commit):
    return dict(schema='formal_F_independent_compression_identity_v1',run=RUN,commit=commit,
        parent=PARENT,parent_commit=PARENT_COMMIT,baseline=BASELINE,baseline_commit=BASELINE_COMMIT,
        reference=REFERENCE,reference_commit=REFERENCE_COMMIT,count=COUNT,views=list(VIEWS),groups=GROUPS,
        candidates=list(CANDIDATES),readouts=list(READOUTS),calibration_count=8192,new_calibration_samples=0,
        coordinates='Exactly R34/R39/R42 mirror64/spectral4/reference24/combined92;unchanged LU moments/std floor.05',
        reference_rule='Post-score evaluation clean reference only;cap2 lowest opaque physicalIDs per TX/RX/day;RX and day both different;complete6TX cells equal weight',
        label_use='Evaluation statistics after all112features fixed and both R44 full7formal scores complete;no labels in transform or parent training',
        controls=dict(permutations=199,seed=2026102034,query_labels='within RX/day',reference_labels='fixed',is_p_value=False),
        comparisons=[['group_pruned','low_rank'],['low_rank','r38_794529'],['group_pruned','r38_794529']],
        CPU_threads=2,GPU_workers=0,max_own_scorers=1,scoring_lock=SCORING_RUN+'/score.lock',
        F_weights_read=0,F_embeddings_logits_read=0,model_checkpoint_loads=0,extra_model_predictions=0,
        neural_training_updates=0,head_fit_updates=0,parent_training_or_decisions_changed=False,
        wait_rule='Both R44 rows full formal scores complete before target IQ/features;this diagnostic never gates or delays parent testing',
        claim='Descriptive same-TX association on existing exposed formal benchmark;not a classifier,not new blind validation,not TX-exclusive/per-detail authenticity proof',
        stop_rule='Parent failure/wrong paths,IDs,calibration,missing/nonfinite input/features or outputcollision:preserve outputs,no retry;healthy parents untouched',goal_complete=False)
