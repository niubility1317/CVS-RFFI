"""Two fixed source-only fourth-order fidelity objectives, original350521 parent."""
import copy
from experiments.pamf_r46_complex_fidelity import design as base
from experiments.pamf_r46_prior_collapse import design as queued

RUN='20261021-phase1-restoration-R46-moment-fidelity-s2026092701-m2-r30'
ARMS=('moment_clean','moment_teacher')
PARAMETERS=dict.fromkeys(ARMS,278180)
CONTEXT_PARAMETERS=dict.fromkeys(ARMS,186703)
TASK=dict.fromkeys(ARMS,False)
MODE=dict.fromkeys(ARMS,'R46_allcore_SVD625_preserve_channels')
GPUS=dict(zip(ARMS,(2,3)))
COMPLEX_WEIGHT=.02
CLEAN_WEIGHT=dict.fromkeys(ARMS,.25)
MOMENT_WEIGHT=1.
POWER_FLOOR=1e-4
LAGS=(1,2,4,8)
MOMENT_REFERENCE=dict(moment_clean='paired_clean',moment_teacher='original_350521_restored_IQ')
COMPACT_PARENT_RUN,COMPACT_PARENT_COMMIT,COMPACT_PARENT_ROW=base.COMPACT_PARENT_RUN,base.COMPACT_PARENT_COMMIT,base.COMPACT_PARENT_ROW
PARENT_RUN,PARENT_COMMIT,ACTION_RUN=base.PARENT_RUN,base.PARENT_COMMIT,base.ACTION_RUN
F_RUN,F_SEED,VIEWS,KIND=base.F_RUN,base.F_SEED,base.VIEWS,base.KIND
MODEL_SEED,DATA_SEED=base.MODEL_SEED,base.DATA_SEED
EPOCHS,UPDATES,BATCH,MICRO=20,19700,64,64
SCORING_RUN=base.SCORING_RUN
PREDECESSOR=queued.RUN
PREDECESSOR_BY_GPU=dict.fromkeys(GPUS.values(),PREDECESSOR)
PREDECESSOR_MODULE={PREDECESSOR:'experiments.pamf_r46_prior_collapse.runner'}
teacher_weight,learning_rate=base.teacher_weight,base.learning_rate
parent,baseline=base.parent,base.baseline


def configuration():
    cfg=copy.deepcopy(base.configuration())
    cfg.update(schema='R46_moment_fidelity_v1',run=RUN,arms=list(ARMS),parameters=PARAMETERS,
        context_parameters=CONTEXT_PARAMETERS,clean_identity_weight=CLEAN_WEIGHT,
        objective='R22identity base IQ+.05powerSTFT+stageIQ1to.1+.02balancedcomplex+.25cleanidentity;add1.0mean11momentdescriptorMSE',
        clean_identity='Botharms additional G(clean) IQ identity;all reference IQ detached;no added descriptor on identity output',
        moment_reference=MOMENT_REFERENCE,moment_weight=MOMENT_WEIGHT,moment_power_floor=POWER_FLOOR,moment_lags=list(LAGS),
        moment_descriptor='R29 exact11dimensions:normalizedabsC40,C42,M42,and realimaglag1/2/4/8cum(a,a,a*,b*);per-windowcentering;powerfloor1e-4',
        moment_objective='Perexamplemean11squared_descriptor_errors;selected clean or original350521 finalreceivedoutput;both references detached;no relative nearzero cumulant denominator',
        moment_weight_basis='Fixed1.0 from completed R29 sourceIQ scale only;pairedmomentabout.0012 vsIQabout.0195;not calibrated to target or equalgradient strength',
        moment_telemetry='Log selected/weightedobjective and both clean/teacherdescriptorMSE everyepoch;botharms compute all3descriptors with same fullG forward counts',
        moment_descriptor_inputs=3780000,extra_teacher_G_inputs=0,extra_student_G_inputs=0,
        teacher_IQ_inputs=1260000,student_context_inputs=1260000,student_total_IQ_inputs=2520000,
        extra_clean_student_inputs=dict.fromkeys(ARMS,1260000),
        source_basis='R29 sourceonly diagnostic3eff8ce2a778d1f952f625b2a5a7c94a66c7e723/results2f633a51bc7c93daa143a63343ad754caa562c66;noTX/F/V/target',
        historical_control='Completed R22complex625_identity E20/19700 sameSVD625/base;result comparison only,never initialization or teacher;no oldtarget rerun',
        acceptance_baselines=['human_R46_350521','original_R17_historical1pp','R22identity_historicalmatchedbase'],
        GPU=GPUS,predecessor_by_GPU={str(g):v for g,v in PREDECESSOR_BY_GPU.items()},
        difference_from_main='Fullchannel lowrank plus fixed parameterfree IQ fourthorder fidelity;no learned Ffeaturehint or channel pruning',
        inference='Same complete native278180 G(IQ) only;no descriptor,clean,teacher,F or added lossbranch at deployment',
        claim='TX-unlabeled F-free simulatedpaired source training;finitewindow moment preservation not perdetailTXproof;exposed single-seed engineering comparison',
        target_feedback=False,goal_complete=False)
    return cfg
