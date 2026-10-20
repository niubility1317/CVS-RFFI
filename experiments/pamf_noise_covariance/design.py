from experiments.pamf_unrolled_action_fidelity.design import PARENT,PARENT_COMMIT,CANDIDATE,REFERENCE,REFERENCE_COMMIT,ACTION,GPUS
RUN='20261020-diagnostic-restoration-IQ-noise-covariance-F50-s2026092701-m2-r30'
COV_REFERENCE='20261020-diagnostic-restoration-unrolled-action-F50-s2026092701-m5-r28'
COV_REFERENCE_COMMIT='2af222f06595e8f7d355d032c0144d49802a6174'
ROWS=('estimated_covariance','known_covariance')
ISOTROPIC_CONTROLS=('native_estimated','est_action_simvar','known_action_simvar')
REFERENCES=('identity','coarse',*ISOTROPIC_CONTROLS)
CONTROL_FOR=dict(estimated_covariance='est_action_simvar',known_covariance='known_action_simvar')


def configuration():
    return dict(schema='IQ_noise_covariance_source_v1',run=RUN,parent=PARENT,parent_commit=PARENT_COMMIT,parent_candidate=CANDIDATE,
        reference=REFERENCE,reference_commit=REFERENCE_COMMIT,action_cache=ACTION,
        covariance_reference=COV_REFERENCE,covariance_reference_commit=COV_REFERENCE_COMMIT,
        source_count=1800,rows=list(ROWS),references=list(REFERENCES),epochs=0,optimizer_updates=0,new_random_draws=0,
        TX_training_labels=0,F_training_reads=0,source_V_state_updates=0,target_reads=0,target_promotion=False,
        covariance='Fixed cached native AGC/rotation;noEQ g=1;per-time real2x2 covariance,n*(I+[[Re(q),Im(q)],[Im(q),-Re(q)]]),q=2ab/(|a|²+|b|²)',
        scope='Frozen transformation noise proxy;not conditional-on-measured-AGC posterior,ground-recording noise,operator-error covariance or deployable oracle',
        temporal_covariance='Pre-AGC injected white draws with fixed phase/AGC and noEQ;no added temporal filtering',
        cg_schedule=[2,2,2],prior_scale='Original SCALE²',innovation='Original normalized unweighted action adjoint;unchanged except original known/estimated action choice',
        logvar='Original scalar simulator variance at all three priors;unchanged versus corresponding R28 control',
        source_rule=dict(mean6_NMSE_ratio_max=.95,lowurban_gain_min_pp=1.,midurban_gain_min_pp=.5,clean_drop_vs_original_max_pp=.1,worst_lowurban_TX_delta_min_pp=-1.),
        primary='known_covariance vs known_action_simvar;estimated path secondary;no automatic target promotion',
        question_origin='R10/R28 source limitation:scalar simulator proxy is not full noise covariance;source code noEQ shows only IQ block correction remains',
        target_feedback=False,goal_complete=False,GPU_pool=list(GPUS),max_own_GPU_workers=1,
        all_owner_cap=2,min_free_MB=12000,capacity_lock='receiver_residual_capacity4.lock',CPU_threads=2,
        stop_rule='Illegal source/parent,collision,nonfinite/incomplete outputs:preserve,abort ownrun,noautomaticretry;healthy peers unchanged')
