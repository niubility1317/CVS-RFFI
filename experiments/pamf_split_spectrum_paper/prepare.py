"""Create fixed source/predict/score jobs and reuse existing frozen results."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_no_reference_completion.prepare import external_sources
from .plan import RUN, TASKS, rows, configuration


def prepare(rt):
    context = read(rt['context'])
    root = Path(rt['run_root'])
    (root/'configs').mkdir()
    jobs, sources = [], {}
    oldroot = Path(rt['project'])/'runs/20261109-phase1-pamf-branch-factorial-m36-r01'
    oldsources = read(oldroot/'sources.json')
    def add(kind, row, c, complete, status, deps=()):
        path = root/'configs'/(kind+'__'+row['row_id']+'.json')
        write(path, c)
        jobs.append(dict(id=kind+'__'+row['row_id'], kind=kind, family=row['family'],
                         learning=row['learning_regime'], config=str(path), completion=str(complete),
                         successful_statuses=[status], dependencies=list(deps)))
    for row in rows():
        if not row['new_training']:
            old = oldsources[row['reuse_key']]
            source, pred, scores = (Path(old[k]) for k in ('source_output','predictions','scores'))
            d, pd, sd = (read(source/'complete.json'), read(pred/'complete.json'), read(scores/'complete.json'))
            c = d['config']
            if (d['status'] != 'SOURCE_FROZEN' or d['epoch'] != 50 or d['updates'] != 11100 or
                    d['target_access'] or d['U_truth_read'] or d['head_fit_updates'] or
                    c['model_seed'] != row['model_seed'] or c['ablation'] != 'F15' or c['U_use'] or
                    c['satellite_CE_weight'] != .68 or read(c['source_contract']) != read(context['handles']['F0']['source_contract']) or
                    c.get('branch_factorial_arm') != row['reuse_key'].rsplit('-s',1)[0] or
                    not (source/'final_ssdg.pth').is_file() or pd['status'] != 'ALL_PREDICTIONS_COMPLETE' or
                    pd['truth_read'] or sd['status'] != 'SCORED_COMPLETE'):
                raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: branch reuse')
            sources[row['row_id']] = dict(row=row, source_output=str(source), predictions=str(pred), scores=str(scores), reused=True)
            continue
        c = configuration(context, row, rt['project'], rt['code_commit'])
        c['frozen_sources'] = rt['frozen_sources']
        src = Path(c['output_root'])
        add('train', row, c, src/'complete.json', 'SOURCE_FROZEN')
        out = root/'readouts'/row['row_id']
        h = context['handles'][row['protocol_id']]
        if row['family'] == 'architecture':
            p = dict(run_id=RUN,row_id=row['row_id'],paper_row=row,variant=c['variant'],model_seed=row['model_seed'],
                     source_output=str(src),source_contract=c['source_contract'],capsule=context['ce_launch']['capsule'],
                     output_root=str(out/'predictions'),frozen_sources=rt['frozen_sources'])
            s = {k:v for k,v in p.items() if k not in ('source_output','frozen_sources')}
            s.update(prediction_output=p['output_root'],truth=context['ce_launch']['truth'],output_root=str(out/'scores'),
                     double_cross_views_root=h['tests']['cross_both']['views_root'],classes=h['classes'])
            pred_status = 'PREDICTIONS_COMPLETE'
        else:
            p = dict(run_id=RUN,row_id=row['row_id'],paper_row=row,protocol_id=row['protocol_id'],model_seed=row['model_seed'],
                     source_output=str(src),source_contract=h['source_contract'],classes=h['classes'],source_days=[1,2,3],
                     tests={t:dict(views_root=h['tests'][t]['views_root']) for t in TASKS},
                     restored_caches={k:v['path'] for k,v in context['caches'][row['protocol_id']].items()} if row['family']=='factorial' else {},
                     views=row['views'],readouts=row['readouts'],output_root=str(out/'predictions'),frozen_sources=rt['frozen_sources'])
            s = deepcopy(p)
            s.update(prediction_output=p['output_root'],output_root=str(out/'scores'),tests={t:h['tests'][t] for t in TASKS})
            for key in ('source_output','restored_caches','frozen_sources'):
                s.pop(key)
            pred_status = 'ALL_PREDICTIONS_COMPLETE'
        add('predict',row,p,out/'predictions/complete.json',pred_status,[str(src/'complete.json')])
        add('score',row,s,out/'scores/complete.json','SCORED_COMPLETE',[str(out/'predictions/complete.json')])
        sources[row['row_id']] = dict(row=row,source_output=str(src),predictions=p['output_root'],scores=s['output_root'],reused=False)
    if len(sources) != 226 or len(jobs) != 633 or Counter(j['kind'] for j in jobs) != dict(train=211,predict=211,score=211):
        raise ValueError('Registered 226-model matrix differs')
    write(root/'jobs.json',jobs)
    write(root/'sources.json',sources)
    write(root/'external_sources.json',external_sources(rt,context))
    return jobs
