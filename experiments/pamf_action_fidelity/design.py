RUN='20261019-diagnostic-restoration-action-fidelity-F50-s2026092701-m5-r13'
PARENT='20261019-phase1-restoration-uncertainty-selfsup-s2026092701-m3-r10'
PARENT_COMMIT='b57eca9946b2a912a88870e24826fe3e2f70985d'
ACTION='20261019-phase1-restoration-learned-action-source-s2026092701-r07'
ROWS=('prior_only','known_action_estvar','est_action_simvar','known_action_simvar','coarse_known_action_simvar')
REFERENCES=('identity','coarse','r07_coarse_estimated','native_estimated')
GPUS=(0,1,5,7)


def configuration():
    return dict(schema='action_fidelity_source_v1',run=RUN,parent=PARENT,parent_commit=PARENT_COMMIT,action_cache=ACTION,
        source_count=1800,rows=list(ROWS),references=list(REFERENCES),epochs=0,optimizer_updates=0,
        TX_training_labels=0,F_training_reads=0,source_V_state_updates=0,target_reads=0,target_promotion=False,
        cg_steps=6,prior_scale='Original sourceLU SCALE=.1169459182403555,not refit',
        known_action='Actual conditional residual action including AGC/lock;oracle only,not deployment input',
        simulator_variance='Cached native white-input equivalent real noise variance proxy,not full true covariance',
        prior='Unchanged learned R10native prior trained with estimatedaction;swappingaction isnot theoretical upperbound',
        source_rule=dict(mean6_NMSE_ratio_max=.95,lowurban_gain_min_pp=1.,midurban_gain_min_pp=.5,
            clean_drop_vs_original_max_pp=.1,worst_lowurban_TX_delta_min_pp=-1.),
        target_feedback=False,goal_complete=False,GPU_pool=list(GPUS),max_own_GPU_workers=1,
        all_owner_cap=2,min_free_MB=12000,capacity_lock='receiver_residual_capacity4.lock',CPU_threads=2,
        stop_rule='Wrongsource/parent,collision,nonfinite,incompleteprediction:preserveoutputs,abortownrun,noautomaticretry;healthyjobsunchanged')
