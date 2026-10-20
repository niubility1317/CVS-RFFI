"""Source-timing-fixed implementation comparison, without new training."""
from experiments.pamf_r46_conditional_factor import design as factor
from experiments.pamf_r46_prior_preserve import design as previous

RUN = '20261021-phase1-restoration-R46-cudagraph-s2026092701-m3-r20'
ARMS = ('R46_350521_graph', 'R18_factor625_stage_graph', 'R18_factor625_cond_graph')
PARAMETERS = dict(zip(ARMS, (350521, 278180, 278180)))
GPUS = dict(zip(ARMS, (2, 2, 3)))
SOURCE_ROWS = dict(zip(ARMS, ('prune_refine', 'factor625_stage', 'factor625_cond')))
SOURCE_BUDGET = dict(zip(ARMS, ((20, 4940), (20, 19700), (20, 19700))))
FACTOR_RUN, FACTOR_COMMIT = factor.RUN, 'aca99499d3f0170df71380f6429ac9a01ef412f8'
COMPACT_PARENT_RUN, COMPACT_PARENT_COMMIT = factor.COMPACT_PARENT_RUN, factor.COMPACT_PARENT_COMMIT
PARENT_RUN, PARENT_COMMIT, ACTION_RUN = factor.PARENT_RUN, factor.PARENT_COMMIT, factor.ACTION_RUN
F_RUN, F_SEED, VIEWS, KIND = factor.F_RUN, factor.F_SEED, factor.VIEWS, factor.KIND
MODEL_SEED, DATA_SEED, SCORING_RUN = factor.MODEL_SEED, factor.DATA_SEED, factor.SCORING_RUN
PREDECESSOR = previous.RUN
PREDECESSOR_MODULE = 'experiments.pamf_r46_prior_preserve.runner'
BATCH = 256


def configuration():
    return dict(schema='R46_cuda_graph_v1', run=RUN, arms=list(ARMS), parameters=PARAMETERS,
        source_rows=SOURCE_ROWS, source_budgets={a:list(v) for a,v in SOURCE_BUDGET.items()},
        parent_run=COMPACT_PARENT_RUN, parent_commit=COMPACT_PARENT_COMMIT,
        factor_run=FACTOR_RUN, factor_commit=FACTOR_COMMIT,
        source_choice='Human original R46 plus BOTH fixed R18 arms; no new target-score selection',
        mechanism='Same frozen FP32 mathematical function and original F50/defaulthead; static CUDA graph replay; owned output clones',
        training_updates=0, training_IQ_examples=0, training_TX_labels_read=0, training_F_access=False,
        new_parameters=0, optimizer_updates=0, head_fit_updates=0, TF32=False,
        source_count=1800, formal_count=168000, views=list(VIEWS), batch=BATCH,
        source_selection=False, freeze_rule='Existing source-final checkpoint; all3rows source7 and formal7 without source gain gate',
        source_equivalence='Two actual Ls smoke; all source/formal stored outputs compared with original completed native predictions; no target-native rerun',
        prediction_comparison='Report native-vs-graph IQ/features/logits differences and argmax agreement; batches differ128vs256, no presumption of bitwise identity',
        timing='Matched samecard native/graph parent and candidate; B1/B32;3 alternating rounds;64sourceV; completeG andG+F',
        timing_includes='New input copy and output ownership clones in graph cases; synchronization; all physics; G+F original defaulthead',
        timing_excludes='Host transfer,loader,preprocess; capture/warmup reported separately; ConvLinearMAC excludes FFT/CG/norm/activation/memory/F',
        GPU=GPUS, max_own_GPU_workers=2, max_own_workers_per_card=1, all_owner_cap=2,
        min_free_MB=18000, capacity_lock='.pamf-reception-capacity4.lock', predecessor=PREDECESSOR,
        CPU_threads=2, max_own_scorers=1, scoring_run=SCORING_RUN,
        stop_rule='Affected technicalfailure preserved; no native fallback for formal prediction; no automaticretry/hotedit/processstop',
        target_acceptance_allowance_pp=.8, historical_R17_allowance_pp=1.,
        performance_targets='Accuracy improvements and compression; lowurban70/other5within5pp rawclean reported separately; B1 speedup requires measurement',
        claim='Same already-exposed single-seed engineering benchmark; no fresh blind validation or per-detail TX proof',
        target_feedback=False, goal_complete=False)
