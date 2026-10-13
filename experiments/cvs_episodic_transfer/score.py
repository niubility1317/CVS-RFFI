"""Independent truth-side scorer after all three row predictions are fixed."""
import json
import math
import statistics as st
from . import design as d


def recount_record(record):
    import numpy as np
    cm = np.asarray(record['confusion'])
    if cm.shape != (6, 6) or cm.dtype.kind not in 'iu' or (cm < 0).any():
        raise ValueError('Malformed six-class confusion matrix')
    count = int(cm.sum())
    if count <= 0 or count != record['query_count']:
        raise ValueError('Confusion/query count recount differs')
    accuracy = float(np.trace(cm) / count)
    denominator = cm.sum(0) + cm.sum(1)
    f1 = float(np.divide(2 * np.diag(cm), denominator, out=np.zeros(6), where=denominator > 0).mean())
    if (not math.isfinite(record['accuracy']) or not math.isfinite(record['macro_f1'])
            or abs(record['accuracy'] - accuracy) > 1e-12 or abs(record['macro_f1'] - f1) > 1e-12):
        raise ValueError('Independent accuracy/F1 recount differs')


def validate_records(c, records):
    from experiments.cvs_report_ssl.evaluate import HELDOUT_RX, TARGET_DAYS
    strata = dict(overall=['ALL'], receiver=list(HELDOUT_RX), transmitter=list(d.CLASSES), day=list(TARGET_DAYS))
    expected = {(readout, dimension, stratum) for readout in d.READOUTS
                for dimension, values in strata.items() for stratum in values}
    actual = set()
    for record in records:
        if (any(record.get(k) != c[k] for k in ('row_id', 'arm', 'model_seed'))
                or record.get('stage') != c['stage'] or record.get('view') != 'clean'):
            raise ValueError('Score row/readout identity differs')
        key = record.get('readout'), record.get('dimension'), record.get('stratum')
        if key not in expected or key in actual:
            raise ValueError('Duplicate or unregistered score stratum')
        actual.add(key)
        dimension = record['dimension']
        if record['query_count'] != d.QUERY_COUNT // len(strata[dimension]):
            raise ValueError('Fixed receiver/TX/day score coverage differs')
        recount_record(record)
    if actual != expected or len(records) != len(expected):
        raise ValueError('Incomplete three-readout score matrix')
    return len(expected)


def row_completion(c, record_count):
    return dict(status='TRAINED_PREDICTED_SCORED', config=c, views=list(d.VIEWS),
                readouts=list(d.READOUTS), count_per_readout=d.QUERY_COUNT,
                metric_records=record_count, truth_last=True, independent_recount='VERIFIED',
                baseline_original_reused=c['arm'] == 'LEGACY50', Phase2_claim=False)


def score(row):
    import numpy as np
    from .predict import preflight
    from experiments.cvs_report_ssl.evaluate import truth_arrays, metric_rows
    c, ids = preflight(row)  # Every readout and its source fit are fixed before truth is opened.
    output = d.BASE / row / 'scores'
    if output.exists():
        raise FileExistsError('Existing scoring must be reconciled, never overwritten')
    output.mkdir(parents=True, exist_ok=False)
    truth = d.read(d.TRUTH)
    y, rx, days = truth_arrays(truth, ids)
    records = []
    with np.load(d.BASE / row / 'prediction/predictions.npz', allow_pickle=False) as saved:
        for readout in d.READOUTS:
            main, day = metric_rows(c, saved[readout], ids, y, rx, days)
            for record in main + day:
                record['readout'] = readout
            records.extend(main + day)
    count = validate_records(c, records)
    d.write(output / 'metrics.json', dict(records=records, readouts=list(d.READOUTS),
            count_per_readout=d.QUERY_COUNT, truth_last=True, independent_recount='VERIFIED',
            phase1_source_readouts=True, Phase2_claim=False))
    d.write(d.BASE / row / 'completion.json', row_completion(c, count))
    print('SCORED ' + json.dumps(dict(row_id=row, readouts=list(d.READOUTS),
                                    metric_records=count, truth_last=True)), flush=True)


def finish():
    records = []
    for row in d.rows():
        c = d.config(row)
        done = d.read(d.BASE / row['row_id'] / 'completion.json')
        value = d.read(d.BASE / row['row_id'] / 'scores/metrics.json')
        count = validate_records(c, value.get('records', []))
        if (done != row_completion(c, count) or value.get('truth_last') is not True
                or value.get('independent_recount') != 'VERIFIED'
                or value.get('readouts') != list(d.READOUTS) or value.get('count_per_readout') != d.QUERY_COUNT
                or value.get('phase1_source_readouts') is not True or value.get('Phase2_claim') is not False):
            raise ValueError('Incomplete row scoring/provenance')
        records.extend(value['records'])
    summary, paired = [], []
    for readout in d.READOUTS:
        for arm in d.ARMS:
            group = [r for r in records if r['arm'] == arm and r['readout'] == readout
                     and r['view'] == 'clean' and r['dimension'] == 'overall']
            if len(group) != len(d.SEEDS) or {r['model_seed'] for r in group} != set(d.SEEDS):
                raise ValueError('Incomplete three-seed readout comparison')
            accuracy = [r['accuracy'] for r in group]
            worst_rx = [min(r['accuracy'] for r in records if r['row_id'] == item['row_id']
                            and r['readout'] == readout and r['dimension'] == 'receiver') for item in group]
            summary.append(dict(arm=arm, readout=readout, view='clean', seed_count=len(group),
                    accuracy_mean=st.mean(accuracy), accuracy_sd=st.stdev(accuracy),
                    seed_accuracy={str(r['model_seed']): r['accuracy'] for r in group},
                    macro_f1_mean=st.mean(r['macro_f1'] for r in group), worst_RX_mean=st.mean(worst_rx)))
        index = {(r['arm'], r['model_seed']): r['accuracy'] for r in records
                 if r['readout'] == readout and r['view'] == 'clean' and r['dimension'] == 'overall'}
        for arm, control in [('EPISODE_SAME_RX', 'LEGACY50'), ('EPISODE_CROSS_RX', 'LEGACY50'),
                             ('EPISODE_CROSS_RX', 'EPISODE_SAME_RX')]:
            differences = [index[arm, seed] - index[control, seed] for seed in d.SEEDS]
            paired.append(dict(arm=arm, control=control, readout=readout, view='clean',
                    seeds=list(d.SEEDS), seed_differences=differences, mean_difference=st.mean(differences),
                    sd_difference=st.stdev(differences), positive_seeds=sum(v > 0 for v in differences)))
    goal_config = d.config(d.rows()[0])['goal']
    primary = next(r for r in summary if r['arm'] == d.PRIMARY and r['readout'] == goal_config['readout']
                   and r['view'] == 'clean')
    value = dict(status='VERIFIED', run_id=d.RUN, models=len(d.rows()),
            newly_trained_models=len(d.training_rows()), baseline_reused_models=len(d.rows()) - len(d.training_rows()),
            readouts=list(d.READOUTS), views=list(d.VIEWS), query_count_per_row=d.QUERY_COUNT,
            prediction_count=len(d.rows()) * len(d.READOUTS) * d.QUERY_COUNT,
            baseline_original_prediction_reuse_count=(len(d.rows()) - len(d.training_rows())) * d.QUERY_COUNT,
            summary=summary, paired=paired, records=records, truth_last=True, independent_recount='VERIFIED',
            primary_arm=d.PRIMARY, primary_readout=goal_config['readout'],
            goal_84=primary['accuracy_mean'] >= goal_config['clean_three_seed_mean'],
            target_feedback_to_training=False, Phase2_claim=False,
            feature_evidence_scope='six old source TX; source-fitted frozen classifiers; no target adaptation')
    output = d.BASE / 'analysis.json'
    if output.exists():
        if d.read(output) != value:
            raise ValueError('Existing fixed analysis differs; preserve and reconcile')
        return value
    d.write(output, value)
    return value
