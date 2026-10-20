RUN='20261020-diagnostic-restoration-unrolled-action-F50-s2026092701-m5-r28'
PARENT='20261020-phase1-restoration-unrolled-selfsup-s2026092701-m2-r17'
PARENT_COMMIT='cd97744884b5e460926e48c8b7efb7d78c6da0f4'
CANDIDATE='unroll_unshared'
REFERENCE='20261019-phase1-restoration-uncertainty-selfsup-s2026092701-m3-r10'
REFERENCE_COMMIT='b57eca9946b2a912a88870e24826fe3e2f70985d'
ACTION='20261019-phase1-restoration-learned-action-source-s2026092701-r07'
ROWS=('native_estimated','known_action_estvar','est_action_simvar','known_action_simvar','prior_only')
REFERENCES=('identity','coarse')
GPUS=(0,1,5,7)


def configuration():
    return dict(schema='unrolled_action_fidelity_source_v1',run=RUN,parent=PARENT,parent_commit=PARENT_COMMIT,
        parent_candidate=CANDIDATE,reference=REFERENCE,reference_commit=REFERENCE_COMMIT,action_cache=ACTION,
        source_count=1800,rows=list(ROWS),references=list(REFERENCES),epochs=0,optimizer_updates=0,
        TX_training_labels=0,F_training_reads=0,source_V_state_updates=0,target_reads=0,target_promotion=False,
        cg_schedule=[2,2,2],prior_scale='Original sourceLU SCALE;not refit',
        known_action='Cached actual conditional AGC/lock/FIR/main/image;source oracle only',
        simulator_variance='Cached native white-input equivalent real noise variance proxy;not full true covariance',
        prior='Original frozen R17 unshared three Core9 priors;substitution changes innovation/logvar and all three MAP steps',
        prior_only='Last prior before final MAP on unchanged estimated-action/native-variance trajectory;first two MAP steps retained',
        source_rule=dict(mean6_NMSE_ratio_max=.95,lowurban_gain_min_pp=1.,midurban_gain_min_pp=.5,
            clean_drop_vs_original_max_pp=.1,worst_lowurban_TX_delta_min_pp=-1.),
        question_origin='R17 pre-existing A17-08 source action counterfactual deferred;R14 only tested one-step R10 prior',
        target_feedback=False,goal_complete=False,GPU_pool=list(GPUS),max_own_GPU_workers=1,
        all_owner_cap=2,min_free_MB=12000,capacity_lock='receiver_residual_capacity4.lock',CPU_threads=2,
        new_random_draws=0,stop_rule='Wrongsource/parent,collision,nonfinite,incompleteprediction:preserveoutputs,abortownrun,noautomaticretry;healthyjobsunchanged')
