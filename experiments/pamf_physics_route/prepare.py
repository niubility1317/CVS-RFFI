"""Prepare only clean and the six practical test scenes."""
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_single_view_paper.plan import TASKS, READOUTS
from experiments.pamf_physics_branch.plan import RUN as PRIOR
from .plan import RUN, REPLICATION_RUN, candidate_rows, configuration


def prepare(rt):
    ctx, root = read(rt['context']), Path(rt['run_root'])
    run_id = rt.get('run_id', RUN)
    h = ctx['handles']['F0']
    jobs, sources = [], {}
    if run_id == RUN:
        previous = Path(rt['project']) / 'runs' / PRIOR
        if read(previous / 'pipeline_state.json')['status'] != 'ANALYZED':
            raise ValueError('The equal-beta matched control is not complete')
        rid = 'ptbr-beta05-s2026092701'
        prior = read(previous / 'sources.json')[rid]
        if read(Path(prior['scores']) / 'complete.json')['status'] != 'SCORED_COMPLETE':
            raise ValueError('The equal-beta matched scores are not complete')
        sources[rid] = {k: prior[k] for k in ('source_output', 'predictions', 'scores')}

    def add(rid, kind, config, folder, status, dependencies=(), nonblocking=False):
        path = root / 'configs' / (kind + '__' + rid + '.json')
        write(path, config)
        jobs.append(dict(id=kind + '__' + rid, kind=kind, config=str(path),
                         completion=str(folder / 'complete.json'), successful_statuses=[status],
                         dependencies=list(dependencies), nonblocking=nonblocking))

    rows = candidate_rows(run_id)
    predictions = [str(root / 'readouts' / row['row_id'] / 'predictions' / 'complete.json') for row in rows]
    for row in rows:
        rid, seed = row['row_id'], row['model_seed']
        baseline = ctx['existing'][f'F0-ordinary_PL-s{seed}']
        tri = (Path(rt['project']) / 'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/readouts'
               / f'main_split-repl-shuffled_time-3view-s{seed}')
        for name, pred, score in ((f'baseline_pl-s{seed}', baseline['predictions'], baseline['scores']),
                                   (f'tri_sigtime_ssl-s{seed}', tri / 'predictions', tri / 'scores')):
            frozen = read(Path(pred) / 'complete.json')
            if frozen['status'] != 'ALL_PREDICTIONS_COMPLETE' or frozen['truth_read']:
                raise ValueError('Matched controls must have predictions frozen without truth')
            if read(Path(score) / 'complete.json')['status'] != 'SCORED_COMPLETE':
                raise ValueError('Matched clean/practical scores are required')
            sources[name] = dict(predictions=str(pred), scores=str(score), reused=True)
        c = configuration(ctx, row, rt['project'], rt['code_commit'], run_id=run_id)
        c['frozen_sources'] = rt['frozen_sources']
        source = Path(c['output_root'])
        out = root / 'readouts' / rid
        add(rid, 'train', c, source, 'SOURCE_FROZEN')
        if run_id != REPLICATION_RUN:
            diagnostic = root / 'source_diagnostics' / rid
            dc = dict(run_id=run_id, row_id=rid, paper_row=c['paper_row'], source_output=str(source),
                      source_contract=h['source_contract'], source_cache=c['source_cache'],
                      frozen_sources=rt['frozen_sources'], output_root=str(diagnostic))
            add(rid, 'source_diagnostic', dc, diagnostic, 'SOURCE_DIAGNOSTIC_COMPLETE',
                [str(source / 'complete.json')], nonblocking=True)
        p = dict(run_id=run_id, row_id=rid, paper_row=c['paper_row'], protocol_id='F0', model_seed=seed,
                 source_output=str(source), source_contract=h['source_contract'], classes=h['classes'],
                 source_days=[1, 2, 3], tests={t: dict(views_root=h['tests'][t]['views_root']) for t in TASKS},
                 restored_caches={k: v['path'] for k, v in ctx['caches']['F0'].items()},
                 views=row['views'], readouts=list(READOUTS), reuse_predictions=None,
                 output_root=str(out / 'predictions'), frozen_sources=rt['frozen_sources'])
        s = deepcopy(p)
        s.update(prediction_output=p['output_root'], output_root=str(out / 'scores'),
                 tests={t: h['tests'][t] for t in TASKS})
        for key in ('source_output', 'restored_caches', 'frozen_sources', 'reuse_predictions'):
            s.pop(key, None)
        add(rid, 'predict', p, out / 'predictions', 'ALL_PREDICTIONS_COMPLETE', [str(source / 'complete.json')])
        add(rid, 'score', s, out / 'scores', 'SCORED_COMPLETE', predictions)
        sources[rid] = dict(source_output=str(source), predictions=p['output_root'],
                            scores=s['output_root'], reused=False)
    write(root / 'jobs.json', jobs)
    write(root / 'sources.json', sources)
    return jobs
