from experiments.pamf_iq_joint_compact import design as base

RUN='20261020-phase1-restoration-IQ-physics-compact-s2026092701-m2-r41'
ARMS,WIDTHS,SPECTRAL_WIDTHS,GPUS,PARAMETERS=base.ARMS,base.WIDTHS,base.SPECTRAL_WIDTHS,base.GPUS,base.PARAMETERS
PARENT_RUN,PARENT_COMMIT,ACTION_RUN,KIND=base.PARENT_RUN,base.PARENT_COMMIT,base.ACTION_RUN,base.KIND
ACTION_COMMIT='c8ccdfdbd03ab5ad8828b1eb9210b99a664ebe7b'
PRECEDING_RUN=base.RUN
SCORING_RUN=RUN
EPOCHS,UPDATES,BATCH,MICRO=base.EPOCHS,base.UPDATES,base.BATCH,base.MICRO
MODEL_SEED,DATA_SEED=base.MODEL_SEED,base.DATA_SEED
PROBE_SEED=2026102041
PROBE_COUNT=2
VIEWS=base.VIEWS
teacher_weight=base.teacher_weight
WEIGHTS=dict(coarse_clean_nmse=.1,action_probe_nmse=.1,noise_log_variance_mse=.01)


def configuration():
    value=base.configuration()
    value.update(schema='IQ_physics_dense_compact_v1',run=RUN,preceding_run=PRECEDING_RUN,
        objective='R38 paired final IQ NMSE + linear10to1 mean3stage originalR17 IQ NMSE + .1coarse-cleanNMSE + .1native action probeNMSE + .01log(native variance ratio)^2',
        physics_weights=WEIGHTS.copy(),training_probes_per_IQ=PROBE_COUNT,training_probe_seed=PROBE_SEED,
        physics_targets='Existing originalR07 per-epoch simulated composite actions and absolute per-real-axis native variance;no TX labels',
        physics_weight_basis='Fixed .1 weight for coarse and originalR07(action+.1logvarianceMSE) block,from sourceR40 and originalR07 objective;no target tuning',
        source_physics='All7context estimates fixed;independentNumPy clean/operator/noise score,4common probes seed2026102040;0sourcefit',
        source_physics_records_per_arm=63,physics_target_examples=1260000,training_action_probe_vectors=2520000,
        source_evidence='R40 fixed all3models sourceV receive-chain errors andR39 analytic association;no formal accuracy based parent,width,weight or row selection',
        source_model_selection='None;both original fixed widths always receive formal7 after fixedE20',
        claim='TX-unlabeled simulated-paired self-supervision plus known synthetic-channel parameter supervision;not blind unpaired or TX-detail proof',
        context='Jointly train all narrowed coarse/action/priors with physical source anchors;teacher uses original context')
    value.pop('GPU_preceding_run',None)
    return value
