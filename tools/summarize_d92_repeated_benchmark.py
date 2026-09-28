"""Combine preregistered disjoint receiver cohorts without selecting scores or cells."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from summarize_d92_confirmation import KEY, matrix_definition, summarize, write_summary


CLAIM = ('REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS: user authorized reuse; '
         'frozen D92-SGJoint-v1 support-only formula, no query fitting; '
         'all four receivers pooled with equal paired-cell weight; '
         'not a new independent confirmation or evidence of unexposed generalization')


def combine(specs, scored):
    if len(specs) != 2 or len(scored) != 2:
        raise ValueError('Exactly the preregistered rx3 and rx1 cohorts are required')
    infos = [matrix_definition(s) for s in specs]
    validated = [summarize(d, s) for d, s in zip(scored, specs)]
    method=infos[0][0][1]
    if method not in ('D92-SGJoint-v1','D92-MVKME-v1') or any(info[0] != ('D92',method) for info in infos):
        raise ValueError('Only matching frozen repeated-benchmark candidates may be pooled')
    claim=CLAIM.replace('D92-SGJoint-v1',method)
    if len({s['run_id'] for s in specs}) != 2:
        raise ValueError('Duplicated run')
    if any('REPEATED_BENCHMARK' not in s['permissions']['claim_scope'] for s in specs):
        raise ValueError('Both cohorts must declare repeated benchmark exposure')
    first, second = (info[1] for info in infos)
    if any(first[k] != second[k] for k in KEY if k != 'receiver') or infos[0][2] != infos[1][2]:
        raise ValueError('Cohorts must share every non-receiver matrix axis and acceptance rule')
    receivers = first['receiver'] + second['receiver']
    if len(receivers) != len(set(receivers)) or sorted(map(lambda v: len(v[1]['receiver']), infos)) != [1, 3]:
        raise ValueError('Expected disjoint three-receiver and one-receiver cohorts')
    if (len(first['model_seed']) != 4 or first['k'] != [1, 5, 10, 20]
            or first['new_count'] != [0, 2, 5, 10, 20] or len(first['scenario']) != 3
            or len(first['support_seed']) != 5):
        raise ValueError('Full preregistered matrix required')
    capsule_ids = [s['confirmation']['reuse_validated_capsule_id'] for s in specs]
    if len(set(capsule_ids)) != 2:
        raise ValueError('The cohorts must retain their two distinct capsule identities')
    # This is an analysis-only matrix, never a launch specification or data capsule.
    composite = dict(analysis_only=True, source_run_ids=[s['run_id'] for s in specs],
        permissions=dict(claim_scope=claim), rows=deepcopy(specs[0]['rows']),
        joint_benchmark=deepcopy(specs[0].get('joint_benchmark',{})),
        data=dict(target_receivers=receivers, scenarios=first['scenario'], k=first['k'],
                  new_class_counts=first['new_count'], support_seeds=first['support_seed'],
                  capsule_ids=capsule_ids),
        metrics_plan=dict(acceptance=dict(old_max_drop=0.01, require_new_improvement=True)),
        confirmation={k: specs[0]['confirmation'][k] for k in
            ('candidate_method', 'candidate_folder', 'candidate_predictor', 'candidate_mode')})
    composite['confirmation'].update(expected_split_count=1200, splits_per_model=1200,
        model_rows=4, predictions_total=9648)
    data = dict(status='SCORED', selection_feedback_forbidden=True, claim_scope=claim,
        results=[r for d in scored for r in d['results']])
    result = summarize(data, composite)
    result['source_cohorts'] = [dict(run_id=s['run_id'], receivers=v['matrix']['receiver'],
        rows=v['rows'], cohort_guard_pass=v['preregistered_guard_pass']) for s, v in zip(specs, validated)]
    result['primary_acceptance_scope'] = 'All four receivers pooled; complete per-cohort and per-receiver reporting retained'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--specs', nargs=2, type=Path, required=True)
    p.add_argument('--scores', nargs=2, type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = combine([json.loads(p.read_text(encoding='utf-8')) for p in a.specs],
                     [json.loads(p.read_text(encoding='utf-8')) for p in a.scores])
    write_summary(result, a.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'tables'}, indent=2))


if __name__ == '__main__':
    main()
