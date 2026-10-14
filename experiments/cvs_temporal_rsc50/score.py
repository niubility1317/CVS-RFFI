from . import design as d
from .predict import preflight
from .evidence import recount
from experiments.cvs_pa_cross_delay.score import score as shared_score, validate_records, row_completion


def score(row):
    return shared_score(row, design=d, preflight_fn=preflight)


def source_evidence():
    from .feature_check import run
    features = run()
    if features['status'] != 'VERIFIED' or features['models'] != 3 or len(features['paired']) != 2:
        raise ValueError('Three-arm actual feature comparison incomplete')
    d.write(d.BASE / 'source_features.json', features)
    observations = {r['arm']: recount(r) for r in d.training_rows()}
    difference = {r['arm']: r['source_features']['records'][0]['joint160_L2_mean'] for r in features['paired']}
    passed = (all(r['last5_unmasked_CE'] <= .5 and r['native_U_objective_unchanged'] for r in observations.values())
              and all(difference[a] >= 1e-6 and observations[a]['body_signal'] for a in ('MATCHED_RANDOM','TEMPORAL_RSC')))
    q = dict(status='PASS' if passed else 'FAIL', source_only=True, target_metrics_read_for_decision=False,
        actual_observations=observations, actual_L720_feature_difference=difference,
        fixed_rules=d.config(d.training_rows()[0])['source_falsification'],
        predefined_remaining_seeds=[2026092702, 2026092703], expand_all_three_arms_unchanged=True,
        head_fit_updates=0, limitation='Source health, actual challenge gradients and deployment-feature changes only; no DG or U correctness proof')
    d.write(d.BASE / 'source_qualification.json', q)
    return q


def finish():
    qualification = source_evidence(); records = []
    for row in d.rows():
        c = d.config(row); p = d.BASE / row['row_id']; value = d.read(p / 'scores/metrics.json')
        count = validate_records(c, value['records'], design=d)
        if d.read(p / 'completion.json') != row_completion(c, count, design=d) or not value['truth_last']:
            raise ValueError('Scored temporal-RSC row incomplete')
        if value['independent_recount'] != 'VERIFIED':
            raise ValueError('Independent scorer recount missing')
        records.extend(value['records'])
    from experiments.cvs_signature_stats_seeds.score import summarize
    summary, paired = summarize(records, d.SEEDS, design=d)
    result = dict(status='VERIFIED', run_id=d.RUN, models=4, newly_trained_models=3,
        baseline_reused_models=1, prediction_count=672000, metric_records=72,
        summary=summary, paired=paired, records=records, source_qualification=qualification,
        goal_84=False, first_seed_is_not_goal_completion=True, goal_requires_three_predefined_seeds=True,
        head_fit_updates=0, truth_last=True, target_feedback_to_training=False, Phase2_claim=False)
    d.write(d.BASE / 'analysis.json', result)
    return result
