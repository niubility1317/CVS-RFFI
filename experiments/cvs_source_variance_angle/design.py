from pathlib import Path
from experiments.cvs_backbone_fishr50 import design as parent

ROOT = Path(__file__).resolve().parents[2]
PROJECT, SOURCE, FULL_FP32_POLICY, read, write = (
    getattr(parent, k) for k in ('PROJECT', 'SOURCE', 'FULL_FP32_POLICY', 'read', 'write'))
RUN = '20261015-phase1-source-backbone-variance-angle-manysig-s2026092701-r01'
RELEASE = 'cvs_source_variance_angle_20261015_r01'
OWNER = 'codex/root/source-variance-angle-20261015'
BASE = Path(PROJECT) / 'runs' / RUN
PARAMETERS = parent.FISHR_PARAMETERS
COHORTS = 5
EPS = 1e-15
QUALIFICATION = dict(source_only=True, sample_count=900, cohorts=5,
    all_layer_cohort_gradients_nonzero=True, nonzero_floor=1e-12,
    aggregate_angle_CE_cosine_min=0., both_t1_t2_required=True,
    no_parameter_update=True, no_automatic_training_release=True,
    on_fail='Do not expand this fixed RAW-angle formula or train it',
    on_pass='Only a separately registered remaining-seed source diagnostic may follow')


def rows():
    return [dict(row_id='SIG_TIME-s2026092701', arm='SIG_TIME', model_seed=2026092701)]


def source_config():
    return parent.config(next(r for r in parent.rows() if r['arm'] == 'SIG_TIME'))


def config(row):
    if row not in rows():
        raise ValueError('Unregistered frozen source model')
    return dict(schema='source_backbone_variance_angle_v1', run_id=RUN, **row,
        source_config=source_config(), role='L_s', source_pool_count=6300,
        source_packets=900, cohort_indices=list(range(COHORTS)), cohort_packets=180,
        cohort_definition='First five existing deterministic BF L180 cohorts; TX/day/RX2 each',
        source_only=True, query_read=False, U_IQ_read=False, U_truth_read=False,
        V_forward=False, V_gradient=False,
        V_builder='Shared VALIDATED_ONCE source builder returns V; discard without forward',
        parameters=list(PARAMETERS), classifier='Existing default cosine6x160scale30 unchanged',
        objective='RAW within-RX unbiased CE-gradient variance vectors; mean ten pairwise 1-cos, mean t1/t2',
        variance_norm_epsilon=EPS, label_smoothing=.01,
        auxiliary_W='Original W constant in per-packet CE derivative graph; never fitted',
        main_CE_gradient='Mean actual same-L180 per-packet CE derivative matrices',
        compare='Original normalized RAW variance objective, unweighted derivatives for both',
        checkpoint_epoch=50, checkpoint_parent_run=parent.RUN,
        checkpoint_parent_execution='600afeeefe84e0d9f50f07c38ae0767bf1cabf07',
        model_mode='eval; actual t1/t2 parameter derivatives only; state never updated',
        extra_training_epochs=0, optimizer_steps=0, head_fit_updates=0,
        qualification=QUALIFICATION.copy(), output_root=(BASE / row['row_id']).as_posix(),
        launch_owner=OWNER)


def validate(c):
    if c != config(rows()[0]):
        raise ValueError('Fixed single-seed source-angle diagnostic differs')
