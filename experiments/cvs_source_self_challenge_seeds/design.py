from copy import deepcopy
from pathlib import Path
from experiments.cvs_source_self_challenge import design as first
from experiments.cvs_receiver_dro_seeds import design as parent

ROOT=Path(__file__).resolve().parents[2]
for k in ('PROJECT','SOURCE','FULL_FP32_POLICY','read','write','PARAMETERS','COHORTS','MASK_SEED','QUALIFICATION'):
    globals()[k]=getattr(first,k)
RUN='20261015-phase1-source-temporal-self-challenge-manysig-s0203-m2-r01'
RELEASE='cvs_source_self_challenge_seeds_20261015_r01'
OWNER='codex/root/source-self-challenge-seeds-20261015'
BASE=Path(PROJECT)/'runs'/RUN
SEEDS=(2026092702,2026092703)
COMPLETION=dict(status='ANALYZED',models=2,source_packets=1800,forward_packets=7200,
    gradient_batches=10,layer_records=40,saved_vectors=120,physical_unique_packets=900,
    source_only=True,query_read=False,U_truth_read=False,head_updates=0,optimizer_steps=0)


def rows():
    return [dict(row_id=f'SIG_TIME-s{s}',arm='SIG_TIME',model_seed=s) for s in SEEDS]


def source_config(row):
    if row not in rows():raise ValueError('Unregistered remaining source model')
    return parent.config(next(r for r in parent.training_rows() if r['arm']=='SIG_TIME' and r['model_seed']==row['model_seed']))


def config(row):
    if row not in rows():raise ValueError('Unregistered remaining source challenge row')
    c=deepcopy(first.config(first.rows()[0]))
    c.update(run_id=RUN,**row,source_config=source_config(row),parent_run=parent.RUN,
        parent_execution='a283291279cb01b939325cf5ccba1d0a671b28c4',
        parent_rule='Predefined own nominalseed SIG_TIME control final50 only; never target-best',
        output_root=(BASE/row['row_id']).as_posix(),launch_owner=OWNER,
        source_expansion_authority=first.RUN,source_expansion_decision_commit='737b65502bc5d9f204a605108b80f430bfe59541',
        first_seed_retrained=False,cohort_and_mask_seed_unchanged=True,
        physical_unique_packets_all_models=900)
    return c


def validate(c):
    row=next((r for r in rows() if r['row_id']==c.get('row_id')),None)
    if row is None or c!=config(row):raise ValueError('Fixed remaining source challenge differs')


def validate_qualification(q):
    if (q.get('status')!='PASS' or q.get('fixed_rules')!=QUALIFICATION
        or not q.get('source_only') or q.get('target_metrics_used') is not False
        or q.get('no_automatic_training_release') is not True):
        raise ValueError('Original fixed first-source PASS missing or changed')
    return q


def require_source():
    return validate_qualification(read(first.BASE/'source_qualification.json'))
