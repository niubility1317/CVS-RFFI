"""Prepare the predeclared three-seed E30 fork using the same truth-last runner."""
import argparse

from .plan import SEEDS
from .publish import prepare as scaffold_prepare, publish
from .view_anchor import RUN, PREVIOUS_RUN, RESOURCE_GPU_BY_SEED, configuration


def prepare(context, project, release, commit):
    rt, scaffold = scaffold_prepare(context, project, release, commit)
    rt.update(run_id=RUN, run_root=project + '/runs/' + RUN, methods=['VIEWANCHOR'],
              launch_owner='codex-pamf-view-anchor-20261023', max_total_workers=3,
              gpu_by_seed=RESOURCE_GPU_BY_SEED, allowed_gpus=[5, 6, 7],
              replaces_queued_runtime=project + '/releases/' + PREVIOUS_RUN + '/runtime.json')
    rows, blobs = [], {}
    for seed in SEEDS:
        old_id = 'RXPSM-s' + str(seed)
        old = next(row for row in rt['rows'] if row['row_id'] == old_id)
        c = configuration(context['_rxpsm_parents'][str(seed)], project, commit)
        c['parent_prefix_training_seconds'] = context['_parent_timing'][str(seed)]
        row_id = c['row_id']
        prefix = rt['run_root'] + '/' + row_id
        prediction = scaffold['configs/' + old_id + '-predict.json']
        prediction.update(run_id=RUN, row_id=row_id, source_run_id=RUN, source_row_id=row_id,
                          source_output=c['output_root'], output_root=prefix + '/predictions')
        scoring = scaffold['configs/' + old_id + '-score.json']
        scoring.update(run_id=RUN, row_id=row_id, prediction_output=prediction['output_root'],
                       output_root=prefix + '/scores')
        paths = {key:release + '/configs/' + row_id + '-' + key + '.json'
                 for key in ('train', 'predict', 'score', 'row')}
        row = dict(old, row_id=row_id, method='VIEWANCHOR', gpu=RESOURCE_GPU_BY_SEED[seed],
            train_config=paths['train'], prediction_config=paths['predict'],
            score_config=paths['score'], row_config=paths['row'])
        rows.append(row)
        for key, value in (('train', c), ('predict', prediction), ('score', scoring), ('row', row)):
            blobs['configs/' + row_id + '-' + key + '.json'] = value
    rt['rows'] = rows
    blobs['runtime.json'] = rt
    from .control import validate_rows
    validate_rows(rows, rt['methods'], RESOURCE_GPU_BY_SEED)
    return rt, blobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('repo', 'context', 'output', 'ssh-config'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    publish(args.repo, args.context, args.output, args.ssh_config, prepare_run=prepare, run_id=RUN)
