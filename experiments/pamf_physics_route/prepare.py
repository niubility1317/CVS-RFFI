"""Prepare only clean and the six practical test scenes."""
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_single_view_paper.plan import TASKS, READOUTS
from experiments.pamf_physics_branch.plan import RUN as PRIOR
from .plan import RUN, candidate_rows, configuration


def prepare(rt):
    ctx, root = read(rt['context']), Path(rt['run_root'])
    previous = Path(rt['project']) / 'runs' / PRIOR
    state = read(previous / 'pipeline_state.json')
    if state['status'] != 'ANALYZED':
        raise ValueError('The equal-beta matched control is not complete')
    prior = read(previous / 'sources.json')['ptbr-beta05-s2026092701']
    baseline = ctx['existing']['F0-ordinary_PL-s2026092701']
    tri = (Path(rt['project']) / 'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02/readouts'
           / 'main_split-repl-shuffled_time-3view-s2026092701')
    for path in (prior['scores'], baseline['scores'], tri / 'scores'):
        if read(Path(path) / 'complete.json')['status'] != 'SCORED_COMPLETE':
            raise ValueError('Matched clean/practical scores are required')
    tri_prediction = read(tri / 'predictions' / 'complete.json')
    if tri_prediction['status'] != 'ALL_PREDICTIONS_COMPLETE' or tri_prediction['truth_read']:
        raise ValueError('Tri-SIG-TIME predictions must be frozen without truth')
    row = candidate_rows()[0]
    c = configuration(ctx, row, rt['project'], rt['code_commit'])
    c['frozen_sources'] = rt['frozen_sources']
    h, rid = ctx['handles']['F0'], row['row_id']
    source = Path(c['output_root'])
    out = root / 'readouts' / rid
    diagnostic = root / 'source_diagnostics' / rid
    jobs = []

    def add(kind, config, folder, status, dependencies=(), nonblocking=False):
        path = root / 'configs' / (kind + '__' + rid + '.json')
        write(path, config)
        jobs.append(dict(id=kind + '__' + rid, kind=kind, config=str(path),
                         completion=str(folder / 'complete.json'), successful_statuses=[status],
                         dependencies=list(dependencies), nonblocking=nonblocking))

    add('train', c, source, 'SOURCE_FROZEN')
    dc = dict(run_id=RUN, row_id=rid, paper_row=c['paper_row'], source_output=str(source),
              source_contract=h['source_contract'], source_cache=c['source_cache'],
              frozen_sources=rt['frozen_sources'], output_root=str(diagnostic))
    add('source_diagnostic', dc, diagnostic, 'SOURCE_DIAGNOSTIC_COMPLETE',
        [str(source / 'complete.json')], nonblocking=True)
    p = dict(run_id=RUN, row_id=rid, paper_row=c['paper_row'], protocol_id='F0', model_seed=row['model_seed'],
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
    add('predict', p, out / 'predictions', 'ALL_PREDICTIONS_COMPLETE', [str(source / 'complete.json')])
    add('score', s, out / 'scores', 'SCORED_COMPLETE', [str(out / 'predictions' / 'complete.json')])
    sources = {'baseline_pl-s2026092701': dict(source_output=baseline['source_output'],
                   predictions=baseline['predictions'], scores=baseline['scores'], reused=True),
               'ptbr-beta05-s2026092701': {k: prior[k] for k in ('source_output', 'predictions', 'scores')},
               'tri_sigtime_ssl-s2026092701': dict(predictions=str(tri / 'predictions'),
                                                   scores=str(tri / 'scores'), reused=True),
               rid: dict(source_output=str(source), predictions=p['output_root'], scores=s['output_root'], reused=False)}
    write(root / 'jobs.json', jobs)
    write(root / 'sources.json', sources)
    return jobs
