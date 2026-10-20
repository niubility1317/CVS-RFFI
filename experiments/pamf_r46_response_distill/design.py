import copy
from experiments.pamf_r46_complex_fidelity import design as baseline
from experiments.pamf_r46_activation_factor import design as previous

RUN='20261021-phase1-restoration-R46-response-distill-s2026092701-m2-r26'
ARMS=('pair_aug_control','odd_response')
PARAMETERS={a:278180 for a in ARMS}
CONTEXT_PARAMETERS={a:186703 for a in ARMS}
TASK={a:False for a in ARMS}
MODE={a:'R46_allcore_SVD625_preserve_channels' for a in ARMS}
GPUS=dict(zip(ARMS,(2,3)))
COMPLEX_WEIGHT=.02
CLEAN_WEIGHT={a:.25 for a in ARMS}
PROBE_WEIGHT,PROBE_COUNT,PROBE_SEED=.01,16,2026102126
PROBE_CYCLE=(('random',.005),('phase',.005),('random',.01),('phase',.01))
PROBE_FLOOR_FRACTION,PROBE_ABSOLUTE_FLOOR=.01,1e-8
parent=baseline.parent
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=baseline.COMPACT_PARENT_RUN,baseline.COMPACT_PARENT_COMMIT,baseline.COMPACT_PARENT_ROW
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=baseline.PARENT_RUN,baseline.PARENT_COMMIT,baseline.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=baseline.F_RUN,baseline.F_SEED,baseline.VIEWS,baseline.KIND
MODEL_SEED,DATA_SEED=baseline.MODEL_SEED,baseline.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO=20,19700,64,64
SCORING_RUN=baseline.SCORING_RUN
PREDECESSOR=previous.RUN
PREDECESSOR_BY_GPU={g:PREDECESSOR for g in GPUS.values()}
PREDECESSOR_MODULE={PREDECESSOR:'experiments.pamf_r46_activation_factor.runner'}
teacher_weight,learning_rate=baseline.teacher_weight,baseline.learning_rate


def configuration():
    cfg=copy.deepcopy(baseline.configuration())
    cfg.update(schema='R46_response_distill_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,
        context_parameters=CONTEXT_PARAMETERS,clean_identity_weight=CLEAN_WEIGHT,
        objective='Common R22 complex+cleanidentity base;matched extra fullG plus/minus calls;pair_aug_control fits perturbed teacher outputs;odd_response fits antithetic odd differences',
        probe_weight=PROBE_WEIGHT,probe_count_per_update=PROBE_COUNT,probe_seed=PROBE_SEED,
        probe_cycle=[list(v) for v in PROBE_CYCLE],probe_cells_per_model=4925,
        probe_selection='First16 examples of original source-permuted batch;no labels,role-based choice,or V/target',
        probe_normalization='Complex power=sum(I2+Q2).mean256;equal input RMS epsilon;output control uses each detached parent output power clamped1e-8;odd uses max(parentoddpower,.01deltapower,1e-8)',
        probe_teacher='Only fixed original350521 E20/4940 fullG eval,no_grad;all weights unchanged;noF',
        selected_probe_examples_per_model=315200,extra_probe_inputs_per_model=630400,
        teacher_IQ_inputs=1890400,student_context_inputs=1260000,student_total_IQ_inputs=3150400,
        extra_clean_student_inputs={a:1260000 for a in ARMS},
        initialization_IQ_examples=0,source_basis='R25 fixed-source finite-response diagnostic method b7f562acbec5855044e07fa0197ed3c8f8370ee9/results3ecb3ebfc23bb1d5f6a3128722f077bd8f6d81d7',
        historical_control='R18/R22 frozen outputs descriptive only;both R26 rows newly trained with equal additional fullG calls;no tested student init or teacher',
        GPU=GPUS,predecessor_by_GPU={str(g):v for g,v in PREDECESSOR_BY_GPU.items()},
        difference_from_main='Training finite input-output response constraints with preserved channel dimensions;main tests intermediate feature hints;no extra inference branch',
        inference='Same 278180 complete native G;probe loss,paired teacher,clean branch are training-only;full7 originalF50/defaulthead evaluation',
        source_selection=False,target_feedback=False,goal_complete=False)
    return cfg
