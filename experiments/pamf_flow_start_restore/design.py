from experiments.pamf_iterative_restore import design as previous

RUN='20261020-phase1-restoration-flow-start-selfsup-s2026092701-m2-r20'
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
PRECEDING='20261020-phase1-restoration-alternating-tf-selfsup-s2026092701-m2-r19'
PRECEDING_COMMIT='5bbc0ccdc32de19e75cb6a01e2cda0352e39f36a'
SIGMA={'zero_start':0.,'noise_start':.02}
ROWS=tuple(SIGMA)
ALL_ROWS=ROWS
SAMPLING_STEPS=16
TRAINABLE_PARAMETERS=1389250
TRAINING_NOISE_SEED=202610202000
PRIMARY_SEED=202610202001
STABILITY_SEEDS=(202610202002,202610202003,202610202004)


def configuration():
    value=previous.configuration()
    value.update(schema='flow_start_v1',run=RUN,rows=list(ROWS),steps=SAMPLING_STEPS,CG_per_step=None,CG_total=6,
        candidate_designs={k:dict(sigma0=v,sampling_steps=SAMPLING_STEPS) for k,v in SIGMA.items()},
        prior_anchor='coarse+SCALE*samplednormalizedresidual',prior_architecture='ConditionalCore2/Encoder5/time64;multiscalecontextcachedonceperpacket',
        training_objective='SCALE² mean velocity error / currentclean mean power;uniformt,straightresidualpath',
        training_noise_seed=TRAINING_NOISE_SEED,primary_seed=PRIMARY_SEED,stability_seeds=list(STABILITY_SEEDS),
        stability_count=128,stability_max_percent=5.,stability_view='practical_low_urban',
        trainable_parameters=TRAINABLE_PARAMETERS,training_Core_calls=1,inference_Core_calls=SAMPLING_STEPS,
        inference='16Euler from0orsigma*independentnoise,currentreceivedonly;then6CG warmstartcoarse',
        sampling='Fixedprimarytorchgenerator/batch128;3fixedsourcediagnosticseeds;no drawselection;conditionalpacketindependence givenepsilon',
        shared_initialization='Complete newinitialweights match,not completeinitialoutputs;both draw same trainingnoise/time streams',
        parameter_control='Same conditionalvelocityfield,same20/4940 exposure;onlysigma0 differs;common16inferenceCorecalls',
        predecessor_run=PRECEDING,predecessor_commit=PRECEDING_COMMIT,
        predecessor_stop_rule='Do not start anyR20GPUworker before originalR19controllerandallworkers exit;no stoppedhealthyjobs')
    return value
