"""Frozen practical/weak job matrix; reuse previous matched PL readouts."""
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_single_view_paper.plan import TASKS, READOUTS
from experiments.pamf_branch_training.plan import RUN as PREVIOUS
from experiments.pamf_branch_training.prepare import score_dependencies
from .plan import RUN, candidate_rows, configuration


def prepare(rt):
    ctx = read(rt['context'])
    root, previous_root = Path(rt['run_root']), Path(rt['project']) / 'runs' / PREVIOUS
    # A second controller must not compete with the still-active predecessor.
    previous_state = read(previous_root / 'pipeline_state.json')
    if (previous_state['status'] not in ('ANALYZED', 'PARTIAL_TECHNICAL_FAILURE')
            or any(v.get('status') == 'RUNNING' for v in previous_state['jobs'].values())):
        raise ValueError('Prior pilot must finish before this run uses its GPU reservation')
    if [r['physical_row'] for r in read(Path(rt['release_root']) / 'experiment.json')['rows']] != candidate_rows():
        raise ValueError('Registered PTBR matrix differs')
    jobs, sources, h = [], {}, ctx['handles']['F0']

    def add(kind, rid, cfg, output, status, deps=()):
        path = root / 'configs' / (kind + '__' + rid + '.json')
        write(path, cfg)
        jobs.append(dict(id=kind + '__' + rid, kind=kind, config=str(path),
                         completion=str(output / 'complete.json'), successful_statuses=[status],
                         dependencies=list(deps)))

    baseline = ctx['existing']['F0-ordinary_PL-s2026092701']
    weak = previous_root / 'readouts/baseline_pl-s2026092701'
    if read(Path(baseline['scores']) / 'complete.json')['status'] != 'SCORED_COMPLETE':
        raise ValueError('Reusable matched PL seven-view scores incomplete')
    weak_done = weak / 'weak_scores/complete.json'
    weak_reusable = weak_done.is_file() and read(weak_done)['status'] == 'SCORED_WEAK_COMPLETE'
    if not weak_reusable:
        rid = 'baseline_pl-s2026092701'
        previous_prediction = weak / 'weak_predictions'
        predicted = previous_prediction / 'complete.json'
        reusable_predictions = predicted.is_file() and read(predicted)['status'] == 'ALL_WEAK_PREDICTIONS_COMPLETE'
        target = root / 'readouts' / rid
        c = configuration(ctx, candidate_rows()[0], rt['project'], rt['code_commit'])
        if reusable_predictions:
            pred_out = previous_prediction
        else:
            pred_out = target / 'weak_predictions'
            cfg = dict(run_id=RUN, row_id=rid, paper_row=c['paper_row'], source_output=baseline['source_output'],
                       source_contract=h['source_contract'], weak_root=rt['weak_root'], frozen_sources=rt['frozen_sources'],
                       output_root=str(pred_out))
            add('weak_predict', rid, cfg, pred_out, 'ALL_WEAK_PREDICTIONS_COMPLETE')
        ws = dict(run_id=RUN, row_id=rid, weak_root=rt['weak_root'], prediction_output=str(pred_out),
                  output_root=str(target / 'weak_scores'), truth=rt['weak_truth'], classes=h['classes'],
                  double_cross_views_root=h['tests']['cross_both']['views_root'])
        add('weak_score', rid, ws, target / 'weak_scores', 'SCORED_WEAK_COMPLETE', [str(pred_out / 'complete.json')])
        weak_predictions, weak_scores = str(pred_out), ws['output_root']
    else:
        weak_predictions, weak_scores = str(weak / 'weak_predictions'), str(weak / 'weak_scores')
    sources['baseline_pl-s2026092701'] = dict(source_output=baseline['source_output'],
          predictions=baseline['predictions'], scores=baseline['scores'],
          weak_predictions=weak_predictions, weak_scores=weak_scores, reused=True)
    diagnostic_root = root / 'source_diagnostics'
    all_diag = [str(diagnostic_root / (r['row_id'] + '/complete.json')) for r in candidate_rows()]
    all_diag.append(str(diagnostic_root / 'baseline_pl-s2026092701/complete.json'))
    example = configuration(ctx, candidate_rows()[0], rt['project'], rt['code_commit'])
    base_diag = dict(run_id=RUN, row_id='baseline_pl-s2026092701', paper_row=example['paper_row'],
                     source_output=baseline['source_output'], source_contract=h['source_contract'],
                     source_cache=example['source_cache'], frozen_sources=rt['frozen_sources'],
                     output_root=str(diagnostic_root / 'baseline_pl-s2026092701'))
    add('source_diagnostic', base_diag['row_id'], base_diag, Path(base_diag['output_root']), 'SOURCE_DIAGNOSTIC_COMPLETE')

    for row in candidate_rows():
        c = configuration(ctx, row, rt['project'], rt['code_commit'])
        c['frozen_sources'] = rt['frozen_sources']
        rid, source = row['row_id'], Path(c['output_root'])
        out = root / 'readouts' / rid
        add('train', rid, c, source, 'SOURCE_FROZEN')
        diagnostic = dict(run_id=RUN, row_id=rid, paper_row=c['paper_row'], source_output=str(source),
                          source_contract=h['source_contract'], source_cache=c['source_cache'],
                          frozen_sources=rt['frozen_sources'], output_root=str(diagnostic_root / rid))
        add('source_diagnostic', rid, diagnostic, Path(diagnostic['output_root']), 'SOURCE_DIAGNOSTIC_COMPLETE',
            [str(source / 'complete.json')])
        p = dict(run_id=RUN, row_id=rid, paper_row=c['paper_row'], protocol_id='F0', model_seed=row['model_seed'],
                 source_output=str(source), source_contract=h['source_contract'], classes=h['classes'], source_days=[1, 2, 3],
                 tests={t: dict(views_root=h['tests'][t]['views_root']) for t in TASKS},
                 restored_caches={k: v['path'] for k, v in ctx['caches']['F0'].items()},
                 views=row['views'], readouts=list(READOUTS), reuse_predictions=None,
                 output_root=str(out / 'predictions'), frozen_sources=rt['frozen_sources'])
        s = deepcopy(p)
        s.update(prediction_output=p['output_root'], output_root=str(out / 'scores'), tests={t: h['tests'][t] for t in TASKS})
        for key in ('source_output', 'restored_caches', 'frozen_sources', 'reuse_predictions'):
            s.pop(key, None)
        wp = dict(run_id=RUN, row_id=rid, paper_row=c['paper_row'], source_output=str(source),
                  source_contract=h['source_contract'], weak_root=rt['weak_root'], frozen_sources=rt['frozen_sources'],
                  output_root=str(out / 'weak_predictions'))
        ws = dict(run_id=RUN, row_id=rid, weak_root=rt['weak_root'], prediction_output=wp['output_root'],
                  output_root=str(out / 'weak_scores'), truth=rt['weak_truth'], classes=h['classes'],
                  double_cross_views_root=h['tests']['cross_both']['views_root'])
        add('predict', rid, p, out / 'predictions', 'ALL_PREDICTIONS_COMPLETE', all_diag)
        add('weak_predict', rid, wp, out / 'weak_predictions', 'ALL_WEAK_PREDICTIONS_COMPLETE', all_diag)
        add('score', rid, s, out / 'scores', 'SCORED_COMPLETE', score_dependencies(out, True))
        add('weak_score', rid, ws, out / 'weak_scores', 'SCORED_WEAK_COMPLETE', score_dependencies(out, True))
        sources[rid] = dict(source_output=str(source), predictions=p['output_root'], scores=s['output_root'],
                            weak_predictions=wp['output_root'], weak_scores=ws['output_root'], reused=False)
    write(root / 'jobs.json', jobs)
    write(root / 'sources.json', sources)
    return jobs
