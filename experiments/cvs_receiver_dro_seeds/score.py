from . import design as d
from .predict import preflight
from .feature_check import run
from experiments.cvs_receiver_dro50.evidence import recount
from experiments.cvs_pa_cross_delay.score import score as shared_score, validate_records, row_completion


def score(row):
    return shared_score(row, design=d, preflight_fn=preflight)


def finish():
    first_source = d.require_source()
    features = run()
    if features['status'] != 'VERIFIED' or features['models'] != 6 or len(features['paired']) != 4:
        raise ValueError('Remaining receiver-risk source features incomplete')
    d.write(d.BASE / 'source_features.json', features)
    observations = [recount(r, design=d) for r in d.training_rows()]
    d.write(d.BASE / 'source_observations.json', dict(status='VERIFIED',
        first_source_qualification_run=d.first.RUN, first_source_qualification=first_source,
        new_selection_or_expansion_gate=False, observations=observations))
    records = []
    for row in d.rows():
        c = d.config(row); p = d.BASE / row['row_id']; value = d.read(p / 'scores/metrics.json')
        count = validate_records(c, value['records'], design=d)
        if (d.read(p / 'completion.json') != row_completion(c, count, design=d)
                or not value['truth_last'] or value['independent_recount'] != 'VERIFIED'):
            raise ValueError('Remaining receiver-risk scored row incomplete')
        records.extend(value['records'])
    from experiments.cvs_signature_stats_seeds.score import summarize
    summary, paired = summarize(records, d.SEEDS, design=d)
    result = dict(status='VERIFIED', run_id=d.RUN, models=8, newly_trained_models=6,
        baseline_reused_models=2, prediction_count=1344000, metric_records=144,
        summary=summary, paired=paired, records=records, source_observations=observations,
        goal_84=False, first_seed_retrained=False, goal_requires_three_predefined_seeds=True,
        head_fit_updates=0, truth_last=True, target_feedback_to_training=False, Phase2_claim=False)
    d.write(d.BASE / 'analysis.json', result)
    return result
