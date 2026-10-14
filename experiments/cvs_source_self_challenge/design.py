from pathlib import Path
from experiments.cvs_receiver_dro50 import design as parent

ROOT = Path(__file__).resolve().parents[2]
PROJECT, SOURCE, FULL_FP32_POLICY, read, write = (getattr(parent, k) for k in
    ('PROJECT', 'SOURCE', 'FULL_FP32_POLICY', 'read', 'write'))
RUN = '20261015-phase1-source-temporal-self-challenge-manysig-s2026092701-r01'
RELEASE = 'cvs_source_self_challenge_20261015_r01'
OWNER = 'codex/root/source-self-challenge-20261015'
BASE = Path(PROJECT) / 'runs' / RUN
PARAMETERS = (*parent.PARAMETERS, 'id_backbone.encoder.core.id_backbone.cls_head.weight')
COHORTS = 5
SEED = 2026092701
MASK_SEED = SEED + 214701
QUALIFICATION = dict(source_only=True, physical_packets=900, cohorts=5,
    three_actual_body_delta_gradients_nonzero_each_cohort=True,
    mean_selected_RSC_CE_minus_matched_random_min_exclusive=0.,
    mean_selected_RSC_true_probability_drop_minus_matched_random_min_exclusive=0.,
    model_state_and_default_head_unchanged=True, no_automatic_training_release=True,
    on_pass='Only a separately registered unchanged02/03 source diagnostic may follow',
    on_fail='Stop this fixed source self-challenge route; no mask-rate or layer scan')


def rows():
    return [dict(row_id='SIG_TIME-s2026092701', arm='SIG_TIME', model_seed=SEED)]


def source_config():
    return parent.config(next(r for r in parent.training_rows() if r['arm'] == 'SIG_TIME'))


def config(row):
    if row not in rows():
        raise ValueError('Unregistered temporal self-challenge source row')
    return dict(schema='source_temporal_self_challenge_v1', run_id=RUN, **row,
        source_config=source_config(), role='L_s', source_only=True,
        source_pool_count=6300, source_packets=900, cohorts=COHORTS, cohort_packets=180,
        cohort_definition='First five existing deterministic L180 cohorts;2 per TX/RX/day',
        parent_run=parent.RUN, parent_execution='977d4fb609436bac23621ef8e32c2017495acef7',
        checkpoint_epoch=50, parent_rule='Predefined first nominal seed SIG_TIME control, never target-best parent',
        layer='id_backbone.encoder.core.id_backbone.t2', activation_shape=[180,96,32],
        gradient='Gradient of sum true-class default-head logits with respect to actual t2 output; signed, mean over96 channels',
        feature_drop_fraction=1/3, batch_drop_fraction=1/3,
        spatial_rule='Keep signed gradient mean strictly below linear two-thirds quantile per packet',
        packet_rule='Apply spatial mask only to probability-drop values at or above linear two-thirds batch quantile',
        random_control='Same selected physical packets and exact per-packet zero count; independently shuffle spatial mask per packet',
        mask_seed=MASK_SEED, label_smoothing=.01,
        observations=['ORIGINAL','SPATIAL_ALL','RSC_SELECTED','MATCHED_RANDOM'],
        model_forward_packets=3600, parameters=list(PARAMETERS),
        masked_parameter_gradients=['ORIGINAL','RSC_SELECTED','MATCHED_RANDOM'],
        classifier='Unchanged existing default ResidualPhysicalHead cosine6x160 scale30; never fitted',
        model_mode='eval; diagnostic derivatives only; no training/update; masks removed after each row',
        query_read=False, U_IQ_read=False, U_truth_read=False, V_forward=False,V_gradient=False,
        V_builder='Reuse existing VALIDATED_ONCE loader; V returned and discarded without forward',
        optimizer_steps=0, head_fit_updates=0, training_epochs=0,
        qualification=QUALIFICATION.copy(), output_root=(BASE/row['row_id']).as_posix(),
        launch_owner=OWNER, target_metrics_for_design=False,
        limitation='Source compatibility of temporal RSC, not a PACS reproduction or proof of new evidence, DG, SSL or84')


def validate(c):
    if c != config(rows()[0]):
        raise ValueError('Fixed temporal self-challenge source recipe differs')
