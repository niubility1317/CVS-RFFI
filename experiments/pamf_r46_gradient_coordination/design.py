import copy
from experiments.pamf_r46_moment_fidelity import design as base

RUN='20261021-phase1-restoration-R46-gradient-coordination-s2026092701-m2-r32'
ARMS=('moment_teacher_grad_sum','moment_teacher_symproj')
PARAMETERS=dict.fromkeys(ARMS,278180)
CONTEXT_PARAMETERS=dict.fromkeys(ARMS,186703)
TASK=dict.fromkeys(ARMS,False)
MODE=dict.fromkeys(ARMS,'R46_allcore_SVD625_preserve_channels')
GPUS=dict(zip(ARMS,(2,3)))
COMPLEX_WEIGHT=.02
CLEAN_WEIGHT=dict.fromkeys(ARMS,.25)
MOMENT_WEIGHT=1.
POWER_FLOOR,LAGS=base.POWER_FLOOR,base.LAGS
MOMENT_REFERENCE=dict.fromkeys(ARMS,'original_350521_restored_IQ')
GRADIENT_POLICY=dict(zip(ARMS,('sum','symmetric_module_projection')))
GRADIENT_GROUPS=('coarse','action','prior')
NORM_FLOOR=1e-12
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=base.COMPACT_PARENT_RUN,base.COMPACT_PARENT_COMMIT,base.COMPACT_PARENT_ROW
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=base.PARENT_RUN,base.PARENT_COMMIT,base.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=base.F_RUN,base.F_SEED,base.VIEWS,base.KIND
MODEL_SEED,DATA_SEED=base.MODEL_SEED,base.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO=20,19700,64,64
SCORING_RUN=base.SCORING_RUN
PREDECESSOR=base.RUN
PREDECESSOR_BY_GPU=dict.fromkeys(GPUS.values(),PREDECESSOR)
PREDECESSOR_MODULE={PREDECESSOR:'experiments.pamf_r46_moment_fidelity.runner'}
teacher_weight,learning_rate=base.teacher_weight,base.learning_rate
parent,baseline=base.parent,base.base


def configuration():
    cfg=copy.deepcopy(base.configuration())
    cfg.update(schema='R46_gradient_coordination_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,context_parameters=CONTEXT_PARAMETERS,
        GPU=GPUS,clean_identity_weight=CLEAN_WEIGHT,moment_reference=MOMENT_REFERENCE,gradient_policy=GRADIENT_POLICY,
        extra_clean_student_inputs=dict.fromkeys(ARMS,1260000),
        moment_objective='Perexamplemean11squared_descriptor_errors against original350521 finalreceivedoutput in botharms;clean/teacher references detached;no relative nearzero denominator',
        gradient_groups=list(GRADIENT_GROUPS),gradient_norm_floor=NORM_FLOOR,gradient_evaluations=39400,
        gradient_objectives='Two autograd.grad calls/update:batchmean base and weighted1.0originalparentmoment;same completeG forward counts and source exposure',
        gradient_formula='For each unique coarse/action/prior group a=base,b=moment,c=dot(a,b);ifc<0 and bothnorms>1e-12:symmetric=(a-c/b²*b)+(b-c/a²*a);elsea+b;computeFP64/castFP32',
        gradient_control='Botharms calculate same raw/proposed vectors and two gradients;only applied vector differs;both-None gradient remainsNone soAdamW skips unchanged unused params',
        gradient_telemetry='Per-group conflict/projected counts,norm/dot means and valid applied-gradient cosine means;properties only before clip/AdamW,not guaranteed actual loss or recognition improvement',
        source_basis='R31 source-only375bf471fa5eb30a9ad36eee0411771fdace9e8a/results827cbd8c0980fe738805ca7ff1c1c874ee031ed4;R22identity teacher conflicts8/16,10/16 andmodule18/32,17/32,13/32;noTX/F/V/target',
        historical_control='R30moment_teacher same source objective;historical reference only,never initializer/teacher;newraw-gradientcontrol matches two-gradient computation',
        acceptance_baselines=['human_R46_350521','original_R17_historical1pp','R22identity_historical','R30momentteacher_historical'],
        predecessor_by_GPU={str(g):v for g,v in PREDECESSOR_BY_GPU.items()},
        difference_from_main='Fullchannel lowrank and source-only gradient coordination;no GroupNorm group/channel pruning or Ffeature guidance',
        inference='Unchanged fullnative278180 G(IQ) only;no gradient/descriptor/teacher/F/clean branch at deployment',
        target_feedback=False,goal_complete=False)
    return copy.deepcopy(cfg)
