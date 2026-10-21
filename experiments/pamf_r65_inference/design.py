from experiments.pamf_r46_delayed_prune import design as parent

RUN = '20261021-phase1-restoration-R65-inference-opt-s2026092701-m2-r01'
ARMS = ('static_graph', 'residual_graph')
PARENT_RUN, PARENT_COMMIT = parent.RUN, '4fd058f9bae79687f924e06d057e6a20083ff6dc'
PARENT_ROW = 'prune_after10'
GPUS = dict(static_graph=2, residual_graph=3)
PARAMETERS = 273765
BATCH = 128
F_RUN, F_SEED, VIEWS, KIND = parent.F_RUN, parent.F_SEED, parent.VIEWS, parent.KIND
ACTION_RUN, SCORING_RUN = parent.ACTION_RUN, parent.SCORING_RUN
R17_RUN = parent.PARENT_RUN


def configuration():
    return dict(schema='R65_frozen_inference_v1', run=RUN, arms=list(ARMS), parameters=PARAMETERS,
        parent_run=PARENT_RUN, parent_commit=PARENT_COMMIT, parent_row=PARENT_ROW,
        checkpoint_selection='User-designated original R65 prune_after10 E20/4940; no new selection',
        mechanisms=dict(static_graph='Original FP32 operation order with device validity accumulation and CUDA graph',
                        residual_graph='Same graph execution plus per-window-constant-variance residual reuse; six CG updates retained'),
        FFT_calls=dict(static_graph=63,residual_graph=39), prior_calls=3, CG_updates=6,
        training_updates=0, optimizer_updates=0, training_TX_labels_read=0, training_F_access=False,
        F_parameter_updates=0, head_fit_updates=0, TF32=False,
        formal_count=168000, source_count=1800, views=list(VIEWS), batch=BATCH,
        source_selection=False, all_versions_formal=True, target_feedback=False,
        GPU=GPUS, max_own_GPU_workers=2, max_own_workers_per_card=1, all_owner_cap=2,
        min_free_MB=18000, CPU_threads=2, max_own_scorers=1,
        timing='Matched source-only native/static/residual, G and G+F, B1/8/32/128, 3 alternating rounds; graph copy/owned outputs/one validity check included',
        prediction='Identical original received bytes, original F50/default head, B128; all7 predictions before independent CPU truthlast',
        no_native_fallback=True, stop_rule='Affected technical failure preserved; no automatic retry or unrelated process intervention',
        claim='Already-exposed single-seed engineering benchmark; zero new training; no independent blind-test claim')
