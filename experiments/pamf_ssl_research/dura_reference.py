"""Supplement frozen PAMF readouts with the user-designated DURA v1.0."""
from copy import deepcopy
from pathlib import Path

from experiments.cric_dura_comparison.evaluation import VIEWS, reject_truth
from experiments.cric_dura_comparison.io import read
from .plan import SEEDS
from .view_anchor import RESOURCE_GPU_BY_SEED

RUN = '20261023-phase1-PAMF-SSL-DURA-reference-m9-r01'
METHODS = ('PSM', 'RXPSM', 'VIEWANCHOR')
READOUTS = ('raw', 'DURA-LU')
DURA_RELATIVE = ('20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69/'
                 'stage2/native_map_prior/final.pt')


def validate_rows(rows):
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    if (len(rows) != len(expected) or {(r['method'], r['model_seed']) for r in rows} != expected
            or len({r['row_id'] for r in rows}) != len(expected)
            or any(r['row_id'] != r['method'] + '-s' + str(r['model_seed'])
                   or r['gpu'] != RESOURCE_GPU_BY_SEED[r['model_seed']] for r in rows)):
        raise ValueError('Exact three-method/three-seed fixed DURA reference matrix required')


def ready(c):
    """Wait for existing source/raw owner, never train or launch raw predictions here."""
    reject_truth(c)
    source_file = Path(c['source_output']) / 'complete.json'
    raw_file = Path(c['raw_predictions_from']) / 'complete.json'
    if not source_file.is_file() or not raw_file.is_file():
        return False
    source, raw = read(source_file), read(raw_file)
    identity = (c['source_run_id'], c['source_row_id'], c['model_seed'])
    actual = source.get('config', {})
    if (source.get('status') != 'SOURCE_FROZEN' or source.get('epoch') != 50
            or source.get('updates') != 11100 or source.get('target_access') is not False
            or tuple(actual.get(k) for k in ('run_id', 'row_id', 'model_seed')) != identity):
        raise ValueError('Frozen E50 source identity/budget differs')
    if (raw.get('status') != 'ALL_PREDICTIONS_COMPLETE'
            or (raw.get('run_id'), raw.get('row_id')) != identity[:2]
            or raw.get('truth_read') is not False or raw.get('query_fit') is not False
            or raw.get('views') != list(VIEWS) or raw.get('readouts') != ['raw']
            or set(raw.get('tests', {})) != {'main'}
            or raw['tests']['main']['count'] != 168000):
        raise ValueError('Existing full raw predictions belong to another F row or matrix')
    return True


def prepare(context, project, release, commit):
    root = project + '/runs/' + RUN
    original = context[str(SEEDS[0])]['F21']
    g = deepcopy(original['predict']['readouts']['DURA-LU'])
    if (g.get('kind') != 'original_F0_LU' or g.get('roles') != 'LU'
            or g.get('checkpoint') != project + '/runs/' + DURA_RELATIVE):
        raise ValueError('User-fixed original DURA v1.0 required')
    for seed in SEEDS:
        for arm in ('F20', 'F21'):
            if context[str(seed)][arm]['predict']['readouts']['DURA-LU'] != g:
                raise ValueError('Matched baselines must use the same fixed DURA')
    blobs, rows = {}, []
    for entry in context['_reference_rows']:
        method, seed = entry['method'], entry['model_seed']
        row_id = method + '-s' + str(seed)
        cp = deepcopy(entry['prediction'])
        if (cp['row_id'] != row_id or cp['model_seed'] != seed or cp['readouts'] != {'raw': {}}
                or cp['source_run_id'] != cp['run_id'] or cp['source_row_id'] != row_id
                or cp['source_contract'] != original['predict']['source_contract']):
            raise ValueError('Reference source differs from original frozen raw row')
        cp.update(run_id=RUN, row_id=row_id, raw_predictions_from=cp['output_root'],
                  readouts={'raw': {}, 'DURA-LU': deepcopy(g)}, code_commit=commit,
                  output_root=root + '/' + row_id + '/predictions')
        reject_truth(cp)
        cs = deepcopy(entry['scoring'])
        cs.update(run_id=RUN, row_id=row_id, readouts=list(READOUTS),
                  prediction_output=cp['output_root'], output_root=root + '/' + row_id + '/scores')
        paths = {key: release + '/configs/' + row_id + '-' + key + '.json'
                 for key in ('predict', 'score')}
        rows.append(dict(row_id=row_id, method=method, model_seed=seed,
                         gpu=RESOURCE_GPU_BY_SEED[seed], prediction_config=paths['predict'],
                         score_config=paths['score']))
        blobs['configs/' + row_id + '-predict.json'] = cp
        blobs['configs/' + row_id + '-score.json'] = cs
    validate_rows(rows)
    view_run = next(e['prediction']['run_id'] for e in context['_reference_rows']
                    if e['method'] == 'VIEWANCHOR')
    rt = dict(run_id=RUN, project=project, release_root=release, run_root=root,
              runtime_file=release + '/runtime.json', code_commit=commit, rows=rows,
              methods=list(METHODS), readouts=list(READOUTS), allowed_gpus=[5, 6, 7],
              max_total_workers=6, max_per_gpu=2, independent_CPU_scorers=1,
              controller_module='experiments.pamf_ssl_research.dura_reference_control',
              launch_owner='codex-pamf-ssl-dura-reference-20261023',
              python='/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python',
              prior_score_owner_state=project + '/runs/' + view_run + '/state.json',
              baseline_scores={str(seed): {arm: context[str(seed)][arm]['score']['output_root']
                               for arm in ('F20', 'F21')} for seed in SEEDS},
              stop_rule='Two identical technical failures stop new dispatch; preserve healthy jobs and outputs. No performance-based stop/reorder/rerun.')
    blobs['runtime.json'] = rt
    return rt, blobs
