from experiments.pamf_r46_compact_refine import design as parent
from experiments.pamf_uncertainty_restore import design as ancestor

RUN='20261021-diagnostic-restoration-R46-source-phase-s2026092701-r51'
PARENT=parent.COMPACT_PARENT_RUN
PARENT_COMMIT=parent.COMPACT_PARENT_COMMIT
PARENT_ROW='prune_refine'
ACTION=parent.ACTION_RUN
ACTION_COMMIT=ancestor.ACTION_COMMIT
KIND=parent.KIND
SEED=2026102051
PER_ROLE=64
TURNS=(0,1,2,3)
VIEWS=('clean','received01','received20')
THRESHOLD=.001


def configuration():
    return dict(schema='R46_source_C4_diagnostic_v1',run=RUN,parent=PARENT,parent_commit=PARENT_COMMIT,
        parent_row=PARENT_ROW,parent_parameters=350521,parent_source_epoch=20,parent_source_updates=4940,
        action_cache=ACTION,action_commit=ACTION_COMMIT,source_data_contract='original7/63/30;READY/EXACT_MATCH reuse',
        source_L_count=6300,source_U_count=56700,source_L_selected=PER_ROLE,source_U_selected=PER_ROLE,
        sampler_seed=SEED,sampler='role-balanced uniform without replacement,sorted source row positions',
        source_views=list(VIEWS),quarter_turns=list(TURNS),stages=3,batch=128,CPU_threads=2,GPU_workers=0,
        metric='Per-record IQ error after inverse C4 rotation,normalized by pairedclean energy clamped1e-4',
        hypothesis_threshold=THRESHOLD,hypothesis_rule='Mean final-stage equivariance NMSE across two received views and three nonzero turns >0.001;descriptive source evidence only',
        source_IQ_unique=128,full_probe_model_IQ_inputs=1536,smoke_additional_L_inputs=2,
        training_updates=0,TX_labels_read=0,F_access=0,V_reads=0,target_reads=0,truth_reads=0,
        new_network_versions=0,new_classifier_heads=0,model_selection=False,target_feedback=False,
        runtime_change=False,claim='Source coordinate-equivariance probe,not TX identity/per-detail authenticity or target accuracy proof',
        stop_rule='On technical failure preserve own output and error;no retries,other-process stops,or model changes')
