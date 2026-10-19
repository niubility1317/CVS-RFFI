from experiments.pamf_flow_start_restore import design as previous

RUN='20261020-phase1-restoration-endpoint-selfsup-s2026092701-m2-r21'
UNROLLED=previous.UNROLLED
UNROLLED_COMMIT=previous.UNROLLED_COMMIT
REFERENCE_CANDIDATE=previous.REFERENCE_CANDIDATE
SINGLE=previous.SINGLE
SINGLE_COMMIT=previous.SINGLE_COMMIT
FFT=previous.FFT
FFT_COMMIT=previous.FFT_COMMIT
ACTION=previous.ACTION
REFERENCE=previous.REFERENCE
KIND=previous.KIND
MODEL_SEED=previous.MODEL_SEED
DATA_SEED=previous.DATA_SEED
GPUS=previous.GPUS
PRECEDING=previous.RUN
PRECEDING_COMMIT='98c1822f88baa546db267dd97efe00f10870afd3'
ALTERNATING=previous.PRECEDING
ALTERNATING_COMMIT=previous.PRECEDING_COMMIT
ALTERNATING_CANDIDATE='time_time'
ROWS=('path_four','endpoint_four')
ALL_ROWS=ROWS
SIGMA={name:0. for name in ROWS}
TRAINING_CORE_CALLS=dict(path_four=1,endpoint_four=4)
SAMPLING_STEPS=4
TRAINABLE_PARAMETERS=previous.TRAINABLE_PARAMETERS
TRAINING_NOISE_SEED=202610202100
PRIMARY_SEED=202610202101
STABILITY_SEEDS=(202610202102,202610202103,202610202104)


def configuration():
    value=previous.configuration()
    value.update(schema='endpoint_training_v1',run=RUN,rows=list(ROWS),steps=4,
        candidate_designs={name:dict(sigma0=0.,sampling_steps=4,training_objective='velocity_path' if name=='path_four' else 'full_deployed_IQ_endpoint') for name in ROWS},
        training_objective='candidate-specific;path_velocity or full4Euler+6CG output IQNMSE;no extra loss',
        training_Core_calls=dict(TRAINING_CORE_CALLS),training_CG_iterations=dict(path_four=0,endpoint_four=6),inference_Core_calls=4,
        training_noise_seed=TRAINING_NOISE_SEED,primary_seed=PRIMARY_SEED,stability_seeds=list(STABILITY_SEEDS),
        sampling='Deterministiczero-start;externalnoise/time consumed identically;3draw0isconstruction notidentityproof',
        inference='4Euler fromzero,currentreceivedonly,contextcachedonce;then6CG warmstartcoarse',
        shared_initialization='Complete newinitialweights AND completeinitialoutputs match betweenrows;scratchafterR07only',
        parameter_control='1389250trainable/3546797total;same4Core6CG deployment;training1Core vs4Core+6CG not equalFLOPs',
        source_selection='All native/R15/R16/R17/R19 sourceguards;rank lowmidmean,lowestTX,lowestRX,lexicalname',
        alternating_source_run=ALTERNATING,alternating_source_commit=ALTERNATING_COMMIT,
        predecessor_run=PRECEDING,predecessor_commit=PRECEDING_COMMIT,
        predecessor_stop_rule='All originalR20 processes must exit before R21GPUwork;preservehealthyothers')
    return value
