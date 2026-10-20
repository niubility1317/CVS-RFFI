from experiments.pamf_response_restore import design as parent
from experiments.pamf_uncertainty_restore.evaluation import VIEWS
from experiments.pamf_fcrossrx_readout.frozen import REFERENCE_RUN

RUN='20261020-evaluation-restoration-R32-positive-control-F50-s2026092701-r33'
PARENT=parent.RUN
PARENT_COMMIT='1857f2c26eabbf68838bad1074e4fdc9b02e267f'
CANDIDATE='pair_aug_control'
KIND=parent.KIND
UNROLLED=parent.UNROLLED
UNROLLED_COMMIT=parent.UNROLLED_COMMIT
REFERENCE=parent.REFERENCE
REFERENCE_COMMIT='b57eca9946b2a912a88870e24826fe3e2f70985d'
GPUS=(0,1,5,7)
POLICY=dict(metric='Equal-weight mean accuracy over six impaired source views',strict_positive=True,
    positivity='Sum of six confusion-matrix diagonal counts strictly greater than source R17 unshared',
    clean_drop_vs_raw_max_pp=.1,target_feedback=False)


def configuration(commit):
    return dict(schema='r32_positive_control_formal_v1',run=RUN,commit=commit,parent_run=PARENT,
        parent_commit=PARENT_COMMIT,candidate=CANDIDATE,kind=KIND,source_policy=POLICY,
        source_reference=UNROLLED,source_raw_reference=REFERENCE,source_count=1800,
        parent_must_finish_before_launch=True,new_training_updates=0,head_fit_updates=0,state_updates=0,
        training_TX_F_access=False,target_feedback=False,truth_last=True,
        count=168000,views=list(VIEWS),F_run=REFERENCE_RUN,F_seed=2026092701,original_default_head=True,
        inference='One restored IQ from current IQ;original frozen F derives all its branches',
        checkpoint='Original R32 pair_aug_control E20/4940,strict complete state;no new training',
        parent_decisions_modified=False,GPUS=list(GPUS),max_own_GPU_workers=1,
        capacity_lock='receiver_residual_capacity4.lock',all_owner_cap=2,min_free_MB=12000,
        benchmark='Previously exposed formal test;conditional source-positive audit,not a new blind test',
        stop_rule='Wrong parent/permissions/inputs,collision,nonfinite,illegal or failed prediction:preserve artifacts and healthy peers;no automatic retry',
        TX_fidelity_unproven=True,goal_complete=False)
