"""Seven-view evaluation of the three already-trained development models."""
from copy import deepcopy
from pathlib import Path

from experiments.pamf_paper_completion import plan as paper
from experiments.pamf_repl_view_shuffle.plan import VIEWS
from experiments.pamf_single_view_paper.plan import TASKS
from . import source_tuning_plan

RUN = '20261112-phase1-pamf-development-seven-view-m3-r01'


def rows():
    return [dict(row_id=value['row_id'], setting=value['setting'], protocol_id=value['protocol_id'],
                 model_seed=value['model_seed'], source_run=source_tuning_plan.RUN, source_row=value['row_id'])
            for value in source_tuning_plan.candidate_rows()]


def source_path(row, project):
    return Path(project) / 'runs' / row['source_run'] / 'F' / row['source_row'] / 'source'


def make_jobs(run_root, project, ctx, frozen_sources):
    root = Path(run_root)
    jobs, sources, configs = [], {}, {}
    for row in rows():
        rid, pid = row['row_id'], row['protocol_id']
        h = ctx['handles'][pid]
        paper_id = f"{row['setting']}-factorial-ssl-full-permuted-s{row['model_seed']}"
        paper_row = next(item for item in paper.make_plan()['rows'] if item['row_id'] == paper_id)
        source = source_path(row, project)
        out = root / 'readouts' / rid
        pred = dict(run_id=RUN, row_id=rid, paper_row=paper_row, protocol_id=pid, model_seed=row['model_seed'],
                    source_output=str(source), source_contract=h['source_contract'], classes=h['classes'],
                    source_days=[1, 2, 3], tests={t: dict(views_root=h['tests'][t]['views_root']) for t in TASKS},
                    restored_caches={}, views=list(VIEWS), readouts=['raw'], reuse_predictions=None,
                    output_root=str(out / 'predictions'), frozen_sources=frozen_sources)
        score = deepcopy(pred)
        score.update(prediction_output=pred['output_root'], output_root=str(out / 'scores'),
                     tests={t: h['tests'][t] for t in TASKS})
        for key in ('source_output', 'restored_caches', 'frozen_sources', 'reuse_predictions'):
            score.pop(key, None)
        for kind, cfg, completion, status, dependency in (
            ('predict', pred, out / 'predictions' / 'complete.json', 'ALL_PREDICTIONS_COMPLETE', source / 'complete.json'),
            ('score', score, out / 'scores' / 'complete.json', 'SCORED_COMPLETE', out / 'predictions' / 'complete.json')):
            jid = kind + '__' + rid
            configs[jid] = cfg
            jobs.append(dict(id=jid, kind=kind, config=str(root / 'configs' / (jid + '.json')),
                             completion=str(completion), successful_statuses=[status], dependencies=[str(dependency)]))
        sources[rid] = dict(source_output=str(source), predictions=pred['output_root'], scores=score['output_root'],
                            source_label_budget='80% of original L; 20% held for source development')
    return jobs, sources, configs
