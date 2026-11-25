"""Frozen cross-receiver seven-view backfill for the selected report models."""
import argparse
import csv
import statistics
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from experiments.cric_dura_comparison.io import read, write

RUN = '20261125-phase1-pamf-report-gap-sixscene-m18-r01'
MODULE = 'experiments.pamf_repl_source_adapt.report_gap_six'
SETTINGS = ('receiver_split_1', 'receiver_split_2', 'labeled_1pct', 'labeled_5pct', 'source_rx_6', 'ten_class')
PROTOCOLS = dict(receiver_split_1='F1_rx_repartition_r01', receiver_split_2='F2_rx_repartition_r01',
                 labeled_1pct='label01', labeled_5pct='label05', source_rx_6='m6-natural', ten_class='ManyRx')
SEEDS = (2026092701, 2026092702, 2026092703)
TASKS = ('cross_rx',)
ROUTES = ('raw', 'DURA-LU')


def matrix():
    return [dict(row_id=f'{setting}-tri-s{seed}', setting=setting, protocol_id=PROTOCOLS[setting],
                 model_seed=seed) for setting in SETTINGS for seed in SEEDS]


def source_for(row, project, profiles):
    seed, setting = row['model_seed'], row['setting']
    if setting == 'labeled_1pct':
        run = ('20261121-phase1-pamf-tri-label01-earlyanchor-reach-clean-m2-r01' if seed == SEEDS[0]
               else '20261121-phase1-pamf-tri-label01-earlyanchor-seedcheck-clean-m2-r01')
        key = f'labeled_1pct-tri-e20-repair60-s{seed}'
        return read(project/'runs'/run/'sources.json')[key], run, key
    key = f'{setting}-tri-profile-s{seed}'
    item = profiles[key]
    cfg = read(Path(item['source_output'])/'complete.json')['config']
    return item, cfg['run_id'], cfg['row_id']


def prepare(rt):
    from experiments.pamf_repl_source_adapt import full_suite, setting_profiles
    from experiments.pamf_repl_adaptive_kl import plan as adaptive
    from experiments.cric_dura_comparison.evaluation import VIEWS
    root, project = Path(rt['run_root']), Path(rt['project'])
    ctx = read(rt['context'])
    profiles = read(project/'runs/20261120-phase1-pamf-tri-setting-profiles-clean-m33-r01/sources.json')
    recipes = {(r['setting'], r['model_seed']): r for r in setting_profiles.matrix()}
    sources, jobs = {}, []
    (root/'configs').mkdir()
    for row in matrix():
        rid, protocol, seed = row['row_id'], row['protocol_id'], row['model_seed']
        item, source_run, source_row = source_for(row, project, profiles)
        cfg = full_suite.frozen_source(item, row, ctx)
        if (cfg['run_id'], cfg['row_id']) != (source_run, source_row):
            raise ValueError('Frozen model identity mismatch')
        pred_done = read(Path(item['predictions'])/'complete.json')
        if (pred_done['status'] != 'ALL_PREDICTIONS_COMPLETE' or pred_done['truth_read'] is not False
                or (pred_done['run_id'], pred_done['row_id']) != (source_run, source_row)):
            raise ValueError('Previous clean predictions belong to another model')
        if row['setting'] != 'labeled_1pct':
            recipe = recipes[(row['setting'], seed)]
            if row['setting'] == 'ten_class' and not recipe.get('reuse_run'):
                if cfg['tuning_recipe'] != adaptive.RECIPES[setting_profiles.TEN]:
                    raise ValueError('Ten-class recipe differs')
            else:
                setting_profiles.verify_recipe(recipe, cfg)
        h = ctx['handles'][protocol]
        cache = ctx['caches'][protocol]['LU']['path']
        cache_done = read(Path(cache)/'complete.json')
        if (cache_done['status'] != 'DURA_IQ_CACHE_COMPLETE' or cache_done['protocol_id'] != protocol
                or cache_done['truth_read'] or read(cache_done['G']['source_contract']) != read(h['source_contract'])):
            raise ValueError('DURA cache source contract mismatch')
        out = root/'readouts'/rid
        common = dict(run_id=RUN, protocol_id=protocol, row_id=rid, model_seed=seed,
                      classes=h['classes'], source_days=ctx['protocols'][protocol]['source_days'],
                      paper_row=dict(family='ssl', learning_regime='ssl', structure='full',
                                     signature_views=3, permuted=True, model_seed=seed),
                      views=list(VIEWS), readouts=list(ROUTES))
        pred = dict(common, source_contract=h['source_contract'], source_output=item['source_output'],
                    restored_caches={'LU':cache}, frozen_sources=rt['frozen_sources'],
                    tests={'cross_rx':{'views_root':h['tests']['cross_rx']['views_root']}},
                    output_root=str(out/'predictions'), reuse_predictions=item['predictions'])
        score = dict(common, tests={'cross_rx':h['tests']['cross_rx']},
                     prediction_output=str(out/'predictions'), output_root=str(out/'scores'))
        for kind, config, status, dependencies in (
                ('predict', pred, 'ALL_PREDICTIONS_COMPLETE', []),
                ('score', score, 'SCORED_COMPLETE', [str(out/'predictions/complete.json')])):
            path = root/'configs'/f'{kind}__{rid}.json'; write(path, config)
            jobs.append(dict(id=f'{kind}__{rid}', kind=kind, family='frozen', learning='ssl',
                             config=str(path), completion=str(out/('predictions' if kind=='predict' else 'scores')/'complete.json'),
                             successful_statuses=[status], dependencies=dependencies))
        sources[rid] = dict(row=row, source_output=item['source_output'], prior_predictions=item['predictions'],
                            scores=str(out/'scores'), predictions=str(out/'predictions'), dura_cache=cache)
    write(root/'sources.json', sources)
    write(root/'jobs.json', jobs)
    return jobs


@contextmanager
def cross_rx_scope():
    from experiments.pamf_single_view_paper import evaluation as native
    from experiments.pamf_paper_completion.model import load_reader
    saved = native.TASKS, native.READOUTS, native.load_reader
    native.TASKS, native.READOUTS, native.load_reader = TASKS, ROUTES, load_reader
    try:
        yield native
    finally:
        native.TASKS, native.READOUTS, native.load_reader = saved


def smoke(rt):
    import numpy as np
    import torch
    from experiments.cric_dura_comparison.io import tensor
    from experiments.pamf_paper_completion.model import load_reader
    root=Path(rt['run_root']); ctx=read(rt['context']); row=matrix()[0]
    entry=read(root/'sources.json')[row['row_id']]; h=ctx['handles'][row['protocol_id']]
    spec=dict(learning_regime='ssl', structure='full', signature_views=3, permuted=True, model_seed=row['model_seed'])
    torch.cuda.reset_peak_memory_stats()
    reader,_=load_reader(entry['source_output'],read(h['source_contract']),spec,'cuda:0')
    sample=np.load(Path(h['source_cache'])/'L_s_x.npy',mmap_mode='r',allow_pickle=False)[:4]
    with torch.inference_mode():
        logits=reader.classify_features(reader.features(tensor(np.asarray(sample,dtype=np.float32),'cuda:0')))
    if logits.shape!=(4,len(h['classes'])) or not bool(torch.isfinite(logits).all()):
        raise ValueError('No-query reader smoke failed')
    write(root/'smoke.json',dict(status='PASS', query_reads=0, real_checkpoint=True,
          checks=[dict(row=row['row_id'], peak_allocated_bytes=torch.cuda.max_memory_allocated())]))


def analyze(rt):
    root=Path(rt['run_root']); sources=read(root/'sources.json'); records=[]; grouped={}
    for rid,entry in sources.items():
        score=Path(entry['scores']); done=read(score/'complete.json'); data=read(score/'scores.json')
        if done['status']!='SCORED_COMPLETE' or not data['truth_last'] or not data['target_feedback_forbidden']:
            raise ValueError('Independent scores incomplete')
        row=entry['row']; results=[x for x in data['results'] if x['test']=='cross_rx' and x['dimension']=='overall']
        for route in ROUTES:
            selected=[x for x in results if x['readout']==route]
            if len(selected)!=7:raise ValueError('Missing seven-view results')
            for x in selected:
                records.append(dict(setting=row['setting'],seed=row['model_seed'],route=route,view=x['view'],
                                    count=x['count'],accuracy=100*x['accuracy'],macro_f1=100*x['macro_f1']))
            six=[100*x['accuracy'] for x in selected if x['view']!='clean']
            grouped.setdefault((row['setting'],route),[]).append((row['model_seed'],statistics.mean(six)))
    out=root/'analysis';out.mkdir(exist_ok=False)
    with (out/'overall.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summaries=[]
    for (setting,route), vals in grouped.items():
        if sorted(s for s,v in vals)!=list(SEEDS):raise ValueError('Missing seed')
        v=[v for _,v in vals]; summaries.append(dict(setting=setting,route=route,seed_n=3,
             mean_pct=statistics.mean(v),sd_pp=statistics.stdev(v),per_seed=dict(vals)))
    write(out/'summary.json',dict(status='ANALYZED', rows=summaries, truth_last=True, target_feedback_forbidden=True))
    write(out/'complete.json',dict(status='ANALYZED',rows=len(sources),seed_n=3))


def main():
    from experiments.pamf_repl_completion.runner import activate
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('dispatch','worker','publish'));parser.add_argument('--kind')
    for name in ('config','repo','output','context','ssh-config'):parser.add_argument('--'+name)
    a=parser.parse_args()
    if a.action=='publish':
        from .transfer_publish import publish
        return publish(a.repo,a.output,a.context,a.ssh_config,run_id=RUN,module=MODULE,decision_run=None,gpus=tuple(range(8)))
    c=read(a.config)
    if a.action=='dispatch':
        from experiments.pamf_branch_training import prepare as prep, runner
        from experiments.pamf_physics_route.runner import launch
        old=prep.prepare,runner.launch,runner.analyze,runner.admits
        base_admits=runner.admits
        prep.prepare,runner.launch,runner.analyze=prepare,partial(launch,module=MODULE),analyze
        runner.admits=lambda gpu,active,mib: (not any(x['gpu']==gpu for _,x in active.values()) and base_admits(gpu,active,max(12000,mib)))
        try:runner.dispatch(dict(c,prioritize_prediction=True))
        finally:prep.prepare,runner.launch,runner.analyze,runner.admits=old
    elif a.kind=='score':
        with cross_rx_scope() as native:native.score(c)
    else:
        activate(c)
        if a.kind=='smoke':smoke(c)
        elif a.kind=='predict':
            with cross_rx_scope() as native:native.predict(c)
        else:raise ValueError('Unknown worker kind')


if __name__=='__main__':main()
