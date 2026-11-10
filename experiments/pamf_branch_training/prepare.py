"""Create source-only training and independent fixed prediction/score jobs."""
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_single_view_paper.plan import TASKS, READOUTS
from .plan import RUN, candidate_rows, configuration


def score_dependencies(out, trained):
    weak = str(out / 'weak_predictions/complete.json')
    practical = str(out / 'predictions/complete.json')
    return [practical, weak] if trained else [weak]


def prepare(rt):
    ctx = read(rt['context'])
    root = Path(rt['run_root'])
    declared = read(Path(rt['release_root']) / 'experiment.json')
    if [r['branch_row'] for r in declared['rows']] != candidate_rows():
        raise ValueError('Registered rows differ')
    jobs, sources = [], {}

    def add(kind, rid, cfg, output, status, deps=()):
        path = root / 'configs' / (kind + '__' + rid + '.json')
        write(path, cfg)
        jobs.append(dict(id=kind + '__' + rid, kind=kind, config=str(path),
                         completion=str(output / 'complete.json'), successful_statuses=[status],
                         dependencies=list(deps)))

    h = ctx['handles']['F0']
    rows = []
    for row in candidate_rows():
        c = configuration(ctx, row, rt['project'], rt['code_commit'])
        c['frozen_sources'] = rt['frozen_sources']
        source = Path(c['output_root'])
        add('train', row['row_id'], c, source, 'SOURCE_FROZEN')
        rows.append((row['row_id'], c['paper_row'], str(source), True))
    for regime, key in (('sup', 'F0-supervised-s2026092701'), ('pl', 'F0-ordinary_PL-s2026092701')):
        row = next(r for r in candidate_rows() if r['regime'] == regime)
        c = configuration(ctx, row, rt['project'], rt['code_commit'])
        previous = ctx['existing'][key]
        done = read(Path(previous['source_output']) / 'complete.json')
        if (done['status'] != 'SOURCE_FROZEN' or done['epoch'] != 50 or done['updates'] != 11100
                or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
                or read(done['config']['source_contract']) != read(c['source_contract'])
                or done['config']['ablation'] != ('F20' if regime == 'sup' else 'F21')):
            raise ValueError('Historical comparison checkpoint not reusable')
        for name, status in (('predictions', 'ALL_PREDICTIONS_COMPLETE'), ('scores', 'SCORED_COMPLETE')):
            if read(Path(previous[name]) / 'complete.json')['status'] != status:
                raise ValueError('Historical comparison readout incomplete')
        rid = 'baseline_' + regime + '-s2026092701'
        rows.append((rid, c['paper_row'], previous['source_output'], False))
        sources[rid] = dict(source_output=previous['source_output'], predictions=previous['predictions'],
                            scores=previous['scores'], reused=True)

    for rid, paper_row, source, trained in rows:
        out = root / 'readouts' / rid
        source_dep = [str(Path(source) / 'complete.json')] if trained else []
        if trained:
            p = dict(run_id=RUN, row_id=rid, paper_row=paper_row, protocol_id='F0', model_seed=2026092701,
                     source_output=source, source_contract=h['source_contract'], classes=h['classes'], source_days=[1, 2, 3],
                     tests={t: dict(views_root=h['tests'][t]['views_root']) for t in TASKS},
                     restored_caches={k: v['path'] for k, v in ctx['caches']['F0'].items()},
                     views=list(candidate_rows()[0]['views']), readouts=list(READOUTS), reuse_predictions=None,
                     output_root=str(out / 'predictions'), frozen_sources=rt['frozen_sources'])
            s = deepcopy(p)
            s.update(prediction_output=p['output_root'], output_root=str(out / 'scores'), tests={t: h['tests'][t] for t in TASKS})
            for k in ('source_output', 'restored_caches', 'frozen_sources', 'reuse_predictions'):
                s.pop(k, None)
            add('predict', rid, p, out / 'predictions', 'ALL_PREDICTIONS_COMPLETE', source_dep)
            add('score', rid, s, out / 'scores', 'SCORED_COMPLETE',
                score_dependencies(out, trained))
            sources[rid] = dict(source_output=source, predictions=p['output_root'], scores=s['output_root'], reused=False)
        weak = dict(run_id=RUN, row_id=rid, paper_row=paper_row, source_output=source,
                    source_contract=h['source_contract'], weak_root=rt['weak_root'],
                    frozen_sources=rt['frozen_sources'], output_root=str(out / 'weak_predictions'))
        weak_score = dict(run_id=RUN, row_id=rid, weak_root=rt['weak_root'],
                          prediction_output=weak['output_root'], output_root=str(out / 'weak_scores'),
                          truth=rt['weak_truth'], classes=h['classes'],
                          double_cross_views_root=h['tests']['cross_both']['views_root'])
        add('weak_predict', rid, weak, out / 'weak_predictions', 'ALL_WEAK_PREDICTIONS_COMPLETE', source_dep)
        add('weak_score', rid, weak_score, out / 'weak_scores', 'SCORED_WEAK_COMPLETE',
            score_dependencies(out, trained))
        sources[rid].update(weak_predictions=weak['output_root'], weak_scores=weak_score['output_root'])
    write(root / 'jobs.json', jobs)
    write(root / 'sources.json', sources)
    return jobs
