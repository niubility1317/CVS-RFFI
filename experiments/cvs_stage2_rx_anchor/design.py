"""Matched alpha-zero control; frozen S3 reliability and sampling are unchanged."""
from pathlib import Path
from experiments.cvs_stage2_stable_rx import design as stable
previous=stable.previous
ROOT=Path(__file__).resolve().parents[2]
PROJECT,SOURCE,CAPSULE,TRUTH,CLASSES=(getattr(stable,k) for k in ('PROJECT','SOURCE','CAPSULE','TRUTH','CLASSES'))
FULL_FP32_POLICY=stable.FULL_FP32_POLICY
read,write,parent_config=stable.read,stable.write,stable.parent_config
SEEDS=stable.SEEDS
ARMS=('Z3',)
CONTROL_ARMS=('B0','B2','B3','S3')
RUN='20261011-phase1-stage2-rx-anchor-manysig-m3-r01'
RELEASE='cvs_stage2_rx_anchor_20261011_r01'
OWNER='codex/root/stage2-rx-anchor-20261011'
BASE=Path(PROJECT)/'runs'/RUN
AUDIT_RUN,AUDIT_BASE=stable.AUDIT_RUN,stable.AUDIT_BASE
RELEASE_BASE=Path(PROJECT)/'releases'/RELEASE
METHOD=stable.METHOD+'_alpha0'

def rows():return [dict(arm='Z3',model_seed=s,row_id=f'Z3-s{s}') for s in SEEDS]

def config(row):
    seed=row['model_seed'];c=stable.config(dict(arm='S3',model_seed=seed,row_id=f'S3-s{seed}'))
    c.update(schema='stage2_receiver_anchor_v1',run_id=RUN,method='cvs_stage2_rx_anchor',**row,
        output_root=(BASE/row['row_id']/'source').as_posix(),launch_owner=OWNER,
        receiver_payload=(RELEASE_BASE/'receiver_payloads'/f's{seed}.pt').as_posix(),receiver_method=METHOD)
    c['receiver'].update(estimator=METHOD,max_alpha=0.,frozen_reliability_from='S3',
        original_payload=(stable.RELEASE_BASE/'receiver_payloads'/f's{seed}.pt').as_posix(),
        effect='Remove RX direction term; retain full160 parent group-difference anchoring and unbiased variance correction')
    c['evaluation'].update(primary_arm='Z3',reused_control_runs=[previous.RUN,stable.RUN],reused_control_arms=list(CONTROL_ARMS))
    c['evaluation'].pop('reused_control_run',None)
    return c

def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Unregistered matched RX anchor configuration')
    return c
