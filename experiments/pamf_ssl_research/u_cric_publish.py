"""Second seed-pinned slot alongside the already dispatched paired-U source run."""
import argparse
from copy import deepcopy
from .publish import publish
from .paired_u_publish import prepare as paired_prepare
from .u_cric import RUN, PAIR_RUN, RESOURCE_GPU_BY_SEED, configuration


def prepare(context, project, release, commit):
    rt, previous = paired_prepare(context, project, release, commit)
    rt.update(run_id=RUN, run_root=project+'/runs/'+RUN, methods=['UCRIC'],
        launch_owner='codex-pamf-U-CRIC-20261023', gpu_predecessor_state=project+'/runs/'+PAIR_RUN+'/state.json',
        prior_score_owner_state=project+'/runs/'+PAIR_RUN+'/state.json', max_compute_processes_per_gpu=3)
    rows, blobs = [], {}
    for old in rt['rows']:
        seed, old_id = old['model_seed'], old['row_id']
        c = configuration(context['_rxpsm_parents'][str(seed)], project, commit)
        c['parent_prefix_training_seconds'] = context['_parent_timing'][str(seed)]
        rid, prefix = c['row_id'], rt['run_root']+'/'+c['row_id']
        p = deepcopy(previous['configs/'+old_id+'-predict.json'])
        p.update(run_id=RUN, row_id=rid, source_run_id=RUN, source_row_id=rid,
                 source_output=c['output_root'], output_root=prefix+'/predictions')
        s = deepcopy(previous['configs/'+old_id+'-score.json'])
        s.update(run_id=RUN, row_id=rid, prediction_output=p['output_root'], output_root=prefix+'/scores')
        paths = {key: release+'/configs/'+rid+'-'+key+'.json' for key in ('train','predict','score','row')}
        row = dict(old, row_id=rid, method='UCRIC', train_config=paths['train'], prediction_config=paths['predict'],
                   score_config=paths['score'], row_config=paths['row'])
        rows.append(row)
        for key, value in (('train', c), ('predict', p), ('score', s), ('row', row)):
            blobs['configs/'+rid+'-'+key+'.json'] = value
    rt['rows'] = rows
    blobs['runtime.json'] = rt
    from .control import validate_rows
    validate_rows(rows, rt['methods'], RESOURCE_GPU_BY_SEED)
    return rt, blobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('repo', 'context', 'output', 'ssh-config'): parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    publish(args.repo, args.context, args.output, args.ssh_config, prepare_run=prepare, run_id=RUN)
