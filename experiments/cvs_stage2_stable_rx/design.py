"""Single source-justified S3 arm; fixed existing E40 controls, no target tuning."""
from pathlib import Path
from experiments.cvs_stage2_geometry import design as previous

ROOT = Path(__file__).resolve().parents[2]
PROJECT, SOURCE, CAPSULE, TRUTH, CLASSES = (getattr(previous,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES'))
FULL_FP32_POLICY = previous.FULL_FP32_POLICY
read, write, parent_config = previous.read, previous.write, previous.parent_config
SEEDS = previous.SEEDS
ARMS = ('S3',)
CONTROL_ARMS = ('B0','B2','B3')
RUN = '20261011-phase1-stage2-stable-rx-manysig-m3-r01'
RELEASE = 'cvs_stage2_stable_rx_20261011_r01'
OWNER = 'codex/root/stage2-stable-rx-20261011'
BASE = Path(PROJECT)/'runs'/RUN
AUDIT_RUN, AUDIT_BASE = previous.AUDIT_RUN, previous.AUDIT_BASE
RELEASE_BASE = Path(PROJECT)/'releases'/RELEASE
METHOD = 'stable_rx_cross_tx_day_shrink04'


def rows():
    return [dict(arm='S3',model_seed=s,row_id=f'S3-s{s}') for s in SEEDS]


def config(row):
    c = previous.config(dict(arm='B3',model_seed=row['model_seed'],row_id=f"B3-s{row['model_seed']}"))
    c.update(schema='stage2_stable_receiver_geometry_v1',run_id=RUN,method='cvs_stage2_stable_rx',
             **row,output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,
             receiver_payload=(RELEASE_BASE/'receiver_payloads'/f"s{row['model_seed']}.pt").as_posix(),
             receiver_method=METHOD)
    c['receiver']=dict(estimator=METHOD,shrink=.4,max_rank=4,rank_truncation=False,
        crossfit='other four TX and other two source days, nine folds',max_alpha=.25,
        vector_reliability=True,complete_feature_dim=160,finite_population_variance_gradient=True,
        block_potential_scaling=True,clip_negative_estimator=False,
        inference='identity only; no RX labels, offset bank or calibration')
    c['evaluation']=dict(primary_arm='S3',goal_three_seed_mean=.84,truth_last=True,
        feedback_to_training=False,reused_control_run=previous.RUN,reused_control_arms=list(CONTROL_ARMS))
    return c


def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Unregistered stable RX configuration')
    return c
