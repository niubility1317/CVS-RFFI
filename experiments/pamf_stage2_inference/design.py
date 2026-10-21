from experiments.pamf_r65_four_route import design as parent

RUN='20261021-phase1-restoration-stage2-graph-s2026092701-r01'
ARMS=('stage2_residual_graph',)
PARENT_RUN,PARENT_COMMIT=parent.RUN,'b2a2ce937a64654e17e84c2aa817b9da4257a48f'
PARENT_ROW='stage2'
GPUS={ARMS[0]:4}
PARAMETERS=266951
BATCH=128
F_RUN,F_SEED,VIEWS,KIND=parent.F_RUN,parent.F_SEED,parent.VIEWS,parent.KIND
ACTION_RUN,SCORING_RUN=parent.ACTION_RUN,parent.SCORING_RUN
R17_RUN='20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17'


def configuration():
    return dict(schema='stage2_frozen_graph_residual_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,
        parent_run=PARENT_RUN,parent_commit=PARENT_COMMIT,parent_row=PARENT_ROW,parent_epoch=20,parent_updates=4940,
        checkpoint_selection='User-designated existing frozen two-stage distillation weights; no new selection/training',
        mechanism='CUDA graph, device validity accumulation, inference_mode, per-window-constant-variance residual reuse; unchanged two priors/four CG',
        FFT_calls={ARMS[0]:27},native_FFT_calls=43,prior_calls=2,CG_updates=4,
        training_updates=0,optimizer_updates=0,training_IQ_examples=0,training_TX_labels_read=0,training_F_access=False,
        F_parameter_updates=0,head_fit_updates=0,TF32=False,
        formal_count=168000,source_count=1800,views=list(VIEWS),batch=BATCH,
        source_selection=False,all_versions_formal=True,target_feedback=False,
        GPU=GPUS,max_own_GPU_workers=1,max_own_workers_per_card=1,all_owner_cap=1,min_free_MB=18000,CPU_threads=2,max_own_scorers=1,
        timing='Same-card original two-stage native vs combined engine;G and G+F;B1/8/32/128;3 alternating rounds;copy/owned outputs/validity included;setup separate',
        prediction='Original received IQ bytes;original F50/defaulthead;B128;all seven predictions before independent CPU truthlast;reuse original two-stage predictions for agreement',
        no_native_fallback=True,stop_rule='Preserve technical failures; no automatic retry, output overwrite, or unrelated process intervention',
        claim='User-designated already exposed single-seed engineering benchmark; no new blind validation or per-detail TX claim')
