"""Collect frozen PAMF score summaries; never load IQ, weights, or target truth.

The deliverable snapshot is sufficient for all subsequent plotting and reporting.
Already published studies are reused rather than rescored. New six-view summaries
use within-seed averages of frozen reported scores, never averages of scene SDs.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics as stats

SEEDS = (2026092701, 2026092702, 2026092703)
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
SIX = 'six_practical_mean'
PROTOCOLS = ('F0', 'F1', 'F2', 'ManyRx', 'chrono', 'label01', 'label05',
             'm1-natural', 'm2-natural', 'm3-natural', 'm4-natural', 'm6-natural',
             'm1-fixed', 'm2-fixed', 'm3-fixed', 'm4-fixed', 'm6-fixed')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False,
                                   indent=2) + '\n', encoding='utf-8')


def csv_rows(path, overall=True):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if overall:
        rows = [r for r in rows if r.get('dimension') == 'overall' and r.get('stratum') == 'ALL']
    for row in rows:
        for key in ('mean', 'sd', 'minimum', 'maximum', 'positive_seeds', 'accuracy_mean',
                    'accuracy_sd', 'macro_f1_mean', 'macro_f1_sd', 'mean_pp', 'sample_sd_pp'):
            if key in row and row[key] != '': row[key] = float(row[key])
    return rows


def save_csv(path, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def summarize(records, fields):
    grouped = defaultdict(list)
    for row in records:
        grouped[tuple(row[k] for k in fields)].append(row)
    out = []
    for key, rows in sorted(grouped.items()):
        if sorted(int(r['seed']) for r in rows) != list(SEEDS):
            raise ValueError('Each summary requires all three actual model seeds: ' + repr(key))
        vals = [r['value'] for r in rows]
        out.append(dict(zip(fields, key), mean=stats.mean(vals), sd=stats.stdev(vals),
            minimum=min(vals), maximum=max(vals), positive_seeds=sum(v > 0 for v in vals), n=3))
    return out


def selected_scores(score):
    if score['status'] != 'SCORED': raise ValueError('Unscored input')
    rows = [r for r in score['results'] if r['dimension'] == 'overall' and r['stratum'] == 'ALL']
    for row in rows:
        if row['view'] not in VIEWS or not 0 <= row['accuracy'] <= 1:
            raise ValueError('Unexpected score units or scene')
    return rows


def with_six(records, dimensions):
    out = list(records); grouped = defaultdict(list)
    for row in records:
        if row['view'] != 'clean': grouped[tuple(row[k] for k in dimensions)].append(row)
    for key, rows in grouped.items():
        if {r['view'] for r in rows} != set(VIEWS[1:]) or len(rows) != 6:
            raise ValueError('Incomplete six-view average: ' + repr(key))
        out.append(dict(zip(dimensions, key), view=SIX, value=stats.mean(r['value'] for r in rows)))
    return out


def collect(project, destination, resume=False):
    rep = project / 'automation_reports/CV-SincNet'
    final = project / 'local_artifacts/final_delivery_20261023/complete_inputs'
    manifest = read(final/'manifest.json')
    if (manifest['status'] != 'ALL_EFFECTIVE_PHASES_COMPLETE' or manifest['running'] != 0
            or manifest['queued'] != 0 or manifest['new_technical_failures'] != 0):
        raise ValueError('The final experiment matrix is not complete')
    if (destination/'snapshot.json').exists(): raise FileExistsError('Preserve the completed snapshot')
    destination.mkdir(parents=True, exist_ok=resume)
    sources = []
    def table(name, relative, overall=True):
        path = rep/relative
        rows = csv_rows(path, overall)
        save_csv(destination/(name+'.csv'), rows)
        sources.append(dict(id=name, file=path.relative_to(project).as_posix(), rows=len(rows),
                            grain='overall scored summaries' if overall else 'published source table'))
        return rows
    main_run = '20261021-phase1-T1-DURA-comparison-m153-r01'
    fine_run = '20261022-phase1-T1-DURA-F-ablation-binding-m63-r01'
    core_run = '20261023-phase1-PAMF-core-factorial-recovery-r01'
    g_run = '20261021-phase1-T1-DURA-G-ablation-m21-r01'
    l_run = '20261022-phase1-PAMF-supervised-Lonly-readouts-m6-r01'
    tables = dict(
        main=table('main',main_run+'/main_summary_overall.csv'),
        main_effects=table('main_effects',main_run+'/main_paired_overall.csv'),
        crossed=table('crossed',main_run+'/main_crossed_overall.csv'),
        core=table('core',core_run+'/core_overall_summary.csv'),
        core_effects=table('core_effects',core_run+'/core_overall_effects.csv'),
        core_geometry=table('core_geometry',core_run+'/core_source_geometry.csv',False),
        core_interventions=table('core_interventions',core_run+'/core_source_interventions.csv',False),
        fine=table('fine',fine_run+'/F_ablation_summary_overall.csv'),
        fine_effects=table('fine_effects',fine_run+'/F_ablation_paired_overall.csv'),
        cric_G=table('cric_G',fine_run+'/F_ablation_factorial_summary_overall.csv'),
        G_summary=table('G_summary',g_run+'/G_ablation_summary.csv',False),
        G_seed=table('G_seed',g_run+'/G_ablation_per_seed.csv',False),
        G_init_KD=table('G_init_KD',g_run+'/G_ablation_init_KD_contrasts.csv',False),
        Lonly=table('Lonly',l_run+'/Lonly_summary_overall.csv'),
        Lonly_effects=table('Lonly_effects',l_run+'/Lonly_paired_overall.csv'),
        unified_timing=table('unified_timing','20261021-phase1-T1-DURA-latency-r02/latency_summary.csv',False))
    # Previously verified studies are included verbatim at the published summary grain.
    for name in ('main','core','fine','Lonly'):
        for row in tables[name]:
            if not (0 <= row['mean'] <= 100 and row['sd'] >= 0):
                raise ValueError('Incorrect percentage units in '+name)
    protocols={r['protocol'] for r in tables['main']}
    if protocols != set(PROTOCOLS): raise ValueError('Incomplete main protocol matrix')
    for name, run in [('gfactor','20261023-phase1-PAMF-PL-Gfactor-analysis-m51-r01'),
                      ('external','20261023-phase1-external-DURA-analysis-m516-r01')]:
        result = final/'remote/runs'/run/'results'
        proof=read(result/'verification.json'); complete=read(result/'complete.json')
        expected_status = 'VERIFIED' if name == 'gfactor' else 'PASS'
        if proof['status'] != expected_status: raise ValueError(name+' result is unverified')
        for fn in (('gfactor_summary.csv','gfactor_paired_effects.csv') if name=='gfactor'
                   else ('external_DURA_summary.csv','external_DURA_vs_raw.csv')):
            key=name if 'summary' in fn else name+'_effects'
            tables[key]=csv_rows(result/fn)
            save_csv(destination/(key+'.csv'),tables[key])
            sources.append(dict(id=key,file=(result/fn).relative_to(project).as_posix(),
                                rows=len(tables[key]),grain='overall scored summaries'))
        write(destination/(name+'_verification.json'),dict(complete=complete,verification=proof))
    def resolve(path):
        if not path.startswith('PROJECT/'): raise ValueError('Expected portable frozen path')
        return final/'remote'/path.removeprefix('PROJECT/')
    # External six-view uncertainty is reconstructed from existing per-seed scores.
    # No confusion matrices, prediction arrays, or target truth are rescored here.
    ext_config=read(final/manifest['analysis_configs']['external'])
    if len(ext_config['rows'])!=516: raise ValueError('Expected all 516 frozen external F sources')
    ext_seed=[]
    for spec in ext_config['rows']:
        score=read(resolve(spec['scores']))
        if score['model_seed']!=spec['model_seed']: raise ValueError('External seed mismatch')
        if spec['variant']!='full': continue  # The user requested Full and Full+PL figures.
        for row in selected_scores(score):
            for metric in ('accuracy','macro_f1'):
                ext_seed.append(dict(protocol=spec['protocol_id'],family=spec['family'],
                    regime=spec['training_regime'],variant='full',seed=int(spec['model_seed']),
                    readout=row['readout'],test=row['test'],view=row['view'],metric=metric,
                    value=100*row[metric]))
    dims=('protocol','family','regime','variant','seed','readout','test','metric')
    ext_seed=with_six(ext_seed,dims)
    ext_summary=summarize(ext_seed,tuple(k for k in dims if k!='seed')+('view',))
    # Exact reconciliation to the existing 0..1 source summaries for every visible scene.
    ext_index={(r['protocol'],r['family'],r['regime'],r['variant'],r['readout'],r['task'],r['view']):r
               for r in tables['external']}
    for row in ext_summary:
        if row['view']==SIX: continue
        k=tuple(row[x] for x in ('protocol','family','regime','variant','readout','test','view'))
        old=ext_index[k]
        if any(abs(row[field]-100*float(old[row['metric']+'_'+field]))>1e-10 for field in ('mean','sd')):
            raise ValueError('External per-seed and published scene summaries disagree')
    tables['external_full_seed']=ext_seed;tables['external_full']=ext_summary
    save_csv(destination/'external_full_seed.csv',ext_seed)
    save_csv(destination/'external_full.csv',ext_summary)
    # Nine matched strict supervised structures, including the three reused full baselines.
    physics_seed=[];geometry=[];interventions=[];source_missing=[]
    for item in manifest['structure_rows']+manifest['structure_baselines']:
        rec=read(final/item['member']);score=rec['score'];arch=item['architecture'];seed=int(item['seed'])
        if score['model_seed']!=seed: raise ValueError('Structural seed mismatch')
        for row in selected_scores(score):
            if row['test']!='main' or row['count']!=168000: raise ValueError('Structural test scope changed')
            cm=row['confusion']
            if sum(map(sum,cm))!=168000 or abs(sum(cm[i][i] for i in range(6))/168000-row['accuracy'])>1e-12:
                raise ValueError('Structural overall score/count mismatch')
            for metric in ('accuracy','macro_f1'):
                physics_seed.append(dict(architecture=arch,seed=seed,readout=row['readout'],
                    test='main',view=row['view'],metric=metric,value=100*row[metric]))
        src=rec['source_physics']
        if src['source_U_reads']!=0 or src['target_reads']!=0 or src['fitted_probe_updates']!=0:
            raise ValueError('Structural source diagnosis is outside its declared permission scope')
        for metric,value in src['geometry'].items():
            if isinstance(value,(int,float)):
                geometry.append(dict(architecture=arch,seed=seed,metric=metric,value=float(value)))
        for row in src['interventions']:
            for metric in ('feature_cosine_distance','default_head_flip_fraction','reference_descriptor_relative_L2'):
                if metric not in row or row[metric] is None:
                    if metric != 'reference_descriptor_relative_L2':
                        raise ValueError('A required source sensitivity metric is missing')
                    source_missing.append(dict(architecture=arch,seed=seed,intervention=row['intervention'],
                        metric=metric,status='not_recorded_for_this_architecture'))
                    continue
                interventions.append(dict(architecture=arch,seed=seed,intervention=row['intervention'],
                    metric=metric,value=row[metric]))
    physics_seed=with_six(physics_seed,('architecture','seed','readout','test','metric'))
    tables['physics_seed']=physics_seed
    tables['physics']=summarize(physics_seed,('architecture','readout','test','metric','view'))
    tables['physics_geometry']=summarize(geometry,('architecture','metric'))
    tables['physics_interventions']=summarize(interventions,('architecture','intervention','metric'))
    tables['physics_source_missing']=source_missing
    if len({r['architecture'] for r in physics_seed})!=9: raise ValueError('Missing matched structure')
    for name in ('physics_seed','physics','physics_geometry','physics_interventions','physics_source_missing'):
        save_csv(destination/(name+'.csv'),tables[name])
    # Dedicated benchmarks remain separate cohorts. Pool calls, not round medians.
    timing_profiles={}
    for name in ('fine_timing','structure_timing'):
        config=read(final/manifest['analysis_configs'][name]);folder=resolve(config['output_root'])
        resolved=read(folder/'resolved.json');summary=read(folder/'summary.json')
        groups=defaultdict(list)
        with (folder/'samples.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                if not row['exclusive_GPU']: raise ValueError('Nonexclusive timing observation')
                groups[row['case'],int(row['batch'])].append(row)
        output=[]
        for (case,batch),rows in sorted(groups.items()):
            if sorted(r['round'] for r in rows)!=list(range(5)): raise ValueError('Incomplete timing rounds')
            vals=[v for r in rows for v in r['wall_ms']]
            if not vals or any(not math.isfinite(v) or v<=0 for v in vals): raise ValueError('Invalid timing values')
            v=sorted(vals);position=.95*(len(v)-1);lo=math.floor(position);alpha=position-lo
            p95=v[lo]*(1-alpha)+v[min(lo+1,len(v)-1)]*alpha
            median=stats.median(v)
            output.append(dict(cohort=name,case=case,batch=batch,rounds=5,samples=len(v),
                median_ms=median,P95_ms=p95,round_median_sd_ms=stats.stdev(stats.median(r['wall_ms']) for r in rows),
                throughput_IQ_s=1000*batch/median,
                H2D_D2H_median_ms=stats.median(v for r in rows for v in r['H2D_D2H_ms'])))
        if len(summary['results'])!=len(groups)*5: raise ValueError('Timing case/round coverage mismatch')
        tables[name]=output;save_csv(destination/(name+'.csv'),output)
        timing_profiles[name]=dict(F_parameters=resolved['F_parameters'],G_parameters=resolved['G_parameters'],
            mechanism_parameters=resolved['mechanism_parameters'],operator_profiles=read(folder/'operator_profiles.json'),
            cases=len(groups),case_rounds=len(summary['results']),exclusive_GPU=True,
            training_only_mechanisms_incremental_deployment_cost=summary.get('training_only_mechanisms_incremental_deployment_cost'))
    # Costs are recorded per study with explicit reuse; overlapping logical sources are never summed.
    cost_specs={
        'main_F_cost':main_run+'/main_F_cost.csv',
        'G_lineage_cost':'20261021-phase1-T1-DURA-G-lineage-m35-r01/G_lineage_cost_summary.csv',
        'fine_F_cost':fine_run+'/F_ablation_F_cost.csv',
        'core_F_cost':core_run+'/core_completed_F_cost.csv',
        'Lonly_F_cost':l_run+'/Lonly_F_cost.csv',
        'Lonly_G_cost':l_run+'/Lonly_G_cost.csv'}
    for name,path in cost_specs.items(): tables[name]=table(name,path,False)
    metadata=dict(schema='pamf_completed_results_v1',as_of_hkt='2026-10-23',
        scope='Current final T1-E50 (13L+37U) + DURA v1.0, completed Phase1 studies',
        frozen_completion=dict(effective_complete=manifest['effective_complete'],queued=0,running=0,
            excluded_historical_jobs=manifest['excluded_historical_jobs'],
            old_failed_replaced_by_r02=True),
        protocols=list(PROTOCOLS),F_model_seeds=list(SEEDS),
        final_F_parameters=232993,final_G_parameters=266951,final_total_parameters=499944,
        common_inference_precision='FP32; TF32 off',
        variance_boundary='Three F seeds conditional on fixed G, except explicitly paired F/G studies and m3 diagonal systems',
        no_new_training_inference_or_truth_access=True,
        external_six_mean='Within each model seed, average the six frozen scenario scores, then sample SD over three F seeds',
        timing_profiles=timing_profiles,sources=sources)
    write(destination/'snapshot.json',dict(metadata=metadata,tables=tables))
    print(json.dumps(dict(status='COLLECTED',destination=str(destination),
        tables={k:len(v) for k,v in tables.items()},timing_case_rounds={k:v['case_rounds'] for k,v in timing_profiles.items()})))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Repair this incomplete export, never a completed snapshot')
    a=p.parse_args();collect(a.project.resolve(),a.output.resolve(),a.resume)
