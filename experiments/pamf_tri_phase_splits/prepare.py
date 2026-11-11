from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_paper_completion import plan as paper
from experiments.pamf_single_view_paper.plan import TASKS, READOUTS
from .plan import RUN, candidate_id, candidate_rows, configuration


def prepare(rt):
    if rt['run_id'] != RUN:
        raise ValueError('Wrong run')
    root, project, ctx = Path(rt['run_root']), Path(rt['project']), read(rt['context'])
    paper_root = project / 'runs' / paper.RUN
    old = read(paper_root / 'sources.json')
    jobs, sources = [], {}
    predictions = [str(root / 'readouts' / candidate_id(r) / 'predictions' / 'complete.json') for r in candidate_rows()]

    def add(rid, kind, c, folder, status, dependencies=()):
        path = root / 'configs' / (kind + '__' + rid + '.json')
        write(path, c)
        jobs.append(dict(id=kind+'__'+rid, kind=kind, config=str(path), completion=str(folder/'complete.json'),
                         successful_statuses=[status], dependencies=list(dependencies), nonblocking=False))

    for row in candidate_rows():
        rid, seed, pid = candidate_id(row), row['model_seed'], row['protocol_id']
        h = ctx['handles'][pid]
        for method, order in (('ssl', 'permuted'), ('pl', 'ordered')):
            old_id = f"{row['data_setting']}-factorial-{method}-full-{order}-s{seed}"
            entry = old[old_id]
            done = read(Path(entry['source_output']) / 'complete.json')
            cfg = done['config']
            marker = read(Path(entry['predictions']) / 'complete.json')
            score = read(Path(entry['scores']) / 'complete.json')
            if (done['status'] != 'SOURCE_FROZEN' or done['epoch'] != 50 or done['updates'] != 11100
                or done['target_access'] or done['U_truth_read'] or done['head_fit_updates']
                or cfg['model_seed'] != seed or cfg['protocol_id'] != pid
                or read(cfg['source_contract']) != read(h['source_contract'])
                or marker['status'] != 'ALL_PREDICTIONS_COMPLETE' or marker['truth_read'] or marker['query_fit']
                or marker['views'] != row['views'] or marker['readouts'] != list(READOUTS)
                or score['status'] != 'SCORED_COMPLETE'):
                raise ValueError('Historical comparison source or prediction differs')
            if method == 'ssl' and (entry['row'] != row or cfg.get('paper_row') != row):
                raise ValueError('Original same-setting Tri model required')
            if method == 'pl' and (cfg.get('paper_method') != 'ordinary_PL' or cfg['repl_row']['method'] != 'ordinary_PL'):
                raise ValueError('Original ordinary PL required')
            sources[f'{method}-{row["data_setting"]}-s{seed}'] = dict(
                source_output=entry['source_output'], predictions=entry['predictions'], scores=entry['scores'], reused=True)

        c = configuration(ctx, row, project, rt['code_commit'])
        c['frozen_sources'] = rt['frozen_sources']
        source, out = Path(c['output_root']), root / 'readouts' / rid
        add(rid, 'train', c, source, 'SOURCE_FROZEN')
        p = dict(run_id=RUN, row_id=rid, paper_row=row, protocol_id=pid, model_seed=seed,
                 source_output=str(source), source_contract=h['source_contract'], classes=h['classes'], source_days=[1,2,3],
                 tests={t:dict(views_root=h['tests'][t]['views_root']) for t in TASKS},
                 restored_caches={k:v['path'] for k,v in ctx['caches'][pid].items()}, views=row['views'],
                 readouts=list(READOUTS), reuse_predictions=None, output_root=str(out/'predictions'), frozen_sources=rt['frozen_sources'])
        s = deepcopy(p)
        s.update(prediction_output=p['output_root'], output_root=str(out/'scores'), tests={t:h['tests'][t] for t in TASKS})
        for k in ('source_output','restored_caches','frozen_sources','reuse_predictions'):
            s.pop(k, None)
        add(rid, 'predict', p, out/'predictions', 'ALL_PREDICTIONS_COMPLETE', [str(source/'complete.json')])
        add(rid, 'score', s, out/'scores', 'SCORED_COMPLETE', predictions)
        sources[rid] = dict(source_output=str(source), predictions=p['output_root'], scores=s['output_root'], reused=False)
    write(root/'jobs.json', jobs)
    write(root/'sources.json', sources)
    return jobs
