"""Independent truth-last scoring; never imported by the predictor."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--truth', type=Path, required=True)
    a = p.parse_args()
    state = json.loads((a.run_root/'state.json').read_text(encoding='utf-8'))
    if not state or any(row['status'] != 'PREDICTIONS_COMPLETE' for row in state.values()):
        raise ValueError('Freeze all registered predictions before opening truth')
    truth = json.loads(a.truth.read_text(encoding='utf-8'))
    results = []
    for row_id in state:
        row_dir = a.run_root/row_id
        complete = json.loads((row_dir/'predictions_complete.json').read_text(encoding='utf-8'))
        count = 0
        with (row_dir/'predictions.jsonl').open(encoding='utf-8') as f:
            for line in f:
                prediction = json.loads(line)
                ids, indices = prediction['query_ids'], prediction['predicted_indices']
                if len(ids)!=len(indices) or len(set(ids))!=len(ids):
                    raise ValueError('Invalid prediction record')
                by_class, old, new = defaultdict(list), [], []
                for sid, index in zip(ids,indices):
                    target = truth[sid]
                    if target['pool_role'] != 'query':
                        raise ValueError('Scoring non-query ID')
                    ok = int(prediction['classes'][index] == target['transmitter'])
                    by_class[target['transmitter']].append(ok)
                    (old if target['old'] else new).append(ok)
                acc = lambda xs: sum(xs)/len(xs) if xs else None
                oa, na = acc(old), acc(new)
                results.append(dict(row_id=row_id,split_id=prediction['split_id'],mode=prediction['mode'],
                    receiver=prediction['receiver'],scenario=prediction['scenario'],k=prediction['k'],support_seed=prediction['support_seed'],
                    accuracy=acc(old+new), macro_accuracy=sum(acc(v) for v in by_class.values())/len(by_class),
                    old_accuracy=oa,new_accuracy=na,harmonic_mean=(2*oa*na/(oa+na) if oa+na else 0) if na is not None else None,
                    query_count=len(ids),class_count=len(by_class)))
                count += 1
        if count != complete['predictions']:
            raise ValueError('Incomplete predictions')
    with (a.run_root/'scored_results.json').open('x',encoding='utf-8') as f:
        json.dump(dict(status='SCORED',results=results,selection_feedback_forbidden=True),f)
    print(json.dumps(dict(status='SCORED',records=len(results))))


if __name__=='__main__':
    main()
