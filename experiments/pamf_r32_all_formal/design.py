from experiments.pamf_response_restore import design as parent
from experiments.pamf_uncertainty_restore.evaluation import VIEWS
from experiments.pamf_fcrossrx_readout.frozen import REFERENCE_RUN

RUN='20261020-evaluation-restoration-R32-all-versions-F50-s2026092701-m2-r36'
PARENT=parent.RUN
PARENT_COMMIT='1857f2c26eabbf68838bad1074e4fdc9b02e267f'
ROWS=tuple(parent.ROWS)
KIND=parent.KIND
GPUS=(0,1,5,7)
POLICY='User 2026-10-20: every new network version receives formal testing regardless of source performance'


def configuration(commit):
    return dict(schema='r32_all_versions_formal_v1',run=RUN,commit=commit,parent_run=PARENT,parent_commit=PARENT_COMMIT,
        rows=list(ROWS),kind=KIND,formal_policy=POLICY,source_performance_gate=False,
        parent_must_finish_before_launch=True,new_training_updates=0,head_fit_updates=0,state_updates=0,
        training_TX_F_access=False,target_feedback=False,truth_last=True,count=168000,views=list(VIEWS),
        F_run=REFERENCE_RUN,F_seed=2026092701,original_default_head=True,
        inference='One restored current IQ;original frozen F derives FFT/PA/time/statistics',
        checkpoint='Both original R32 E20/4940 complete states;no new training or target adaptation',
        reuse='Reuse only complete same-row original R32 seven-view predictions and independent scores;never use accuracy to decide reuse',
        parent_decisions_modified=False,GPUS=list(GPUS),max_own_GPU_workers=2,max_own_workers_per_card=1,
        capacity_lock='receiver_residual_capacity4.lock',all_owner_cap=2,min_free_MB=12000,
        benchmark='Previously exposed fixed formal benchmark;not a new blind test',
        stop_rule='Wrong inputs/lineage,collision,nonfinite or technical failure:preserve artifacts and healthy peers;hold pending rows;no automatic retry',
        TX_fidelity_unproven=True,goal_complete=False)
