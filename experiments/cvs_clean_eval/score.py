"""Independent clean truth-last scoring. No model/training module imports."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from comparison_suite.score import metrics

SEEDS={2026092701,2026092702,2026092703,2026092704}
BASELINES=('native','cvcnn','real_cnn','resnet1d')
CANDIDATES={'orthogonal_pa','moment_pool','orthogonal_moment','shared_complex','residual_fusion','residual_fusion_moment','balanced_fusion','signed_balanced_fusion','tf_lowrank32','tf_bilinear','attentive_mean','attentive_moments','phase_delta','phase_dsq','coherence_phase','coherence_dsq','simplex_learned','simplex_fixed','rf_mp','rf_gmp','observable_phase','observable_affine','gauge_peak','gauge_coherent','reference_response','equivariant_memory','synchronized_equivariant','synchronized_gauge','coordinate_equivariant','coordinate_gauge','additive_equivariant','additive_gauge','fractional_half','fractional_learned','energy_equivariant','energy_half'}
REUSED_BASELINE_RUN='20261001-phase1-clean-baselines-manysig-m16-r01'
REUSED_RESIDUAL_RUN='20261001-phase1-cvs-selected-clean-manysig-m20-r01'
REUSED_ENERGY_RUN='20261002-phase1-cvs-energy-clean-manysig-m24-r01'
REUSED_COUPLED_RUN='20261002-phase1-cvs-coupled-clean-manysig-m28-r01'
CANDIDATES.update({'crossphase_raw','crossphase_half','coupled_lag1','coupled_lag4','volterra_lag1','volterra_lag4'})


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2)


def csvwrite(path,rows):
    fields=sorted({k for row in rows for k in row})
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def validate_matrix(spec):
    selection=read(spec['selection_file'])
    if selection.get('scope')=='baseline_only':
        if selection['status']!='FIXED_BASELINES_FROZEN' or selection.get('test_variants')!=list(BASELINES) or selection.get('model_seeds')!=sorted(SEEDS):
            raise ValueError('Fixed baseline matrix changed')
        variants=BASELINES
    else:
        if selection['status']!='SOURCE_SELECTION_FROZEN' or selection['selected_variant'] not in CANDIDATES:
            raise ValueError('No source-only frozen selection')
        variants=(*BASELINES,'residual_fusion',selection['selected_variant']) if selection.get('scope') in ('balanced_source','interaction_source','attentive_source','stability_source','coherence_source','simplex_source','rf_operator_source','observable_source','gauge_source','reference_source','equivariant_source','synchronized_source','coordinate_source','additive_source','fractional_source','energy_source') else (*BASELINES,selection['selected_variant'])
    if selection.get('scope')=='volterra_source':
        if selection['selected_variant'] not in {'volterra_lag1','volterra_lag4'} or selection.get('new_candidate_selected') is not True:
            raise ValueError('Only selected new Volterra candidate can be scored')
        variants=(*BASELINES,'residual_fusion','energy_equivariant','coupled_lag4',selection['selected_variant'])
    if selection.get('scope')=='coupled_source':
        if selection['selected_variant'] not in {'coupled_lag1','coupled_lag4'} or selection.get('new_candidate_selected') is not True:
            raise ValueError('Only selected new coupled candidate can be scored')
        variants=(*BASELINES,'residual_fusion','energy_equivariant',selection['selected_variant'])
    if selection.get('scope')=='crossphase_source':
        if selection['selected_variant'] not in {'crossphase_raw','crossphase_half'} or selection.get('new_candidate_selected') is not True:
            raise ValueError('Only selected new crossphase candidate can be scored')
        variants=(*BASELINES,'residual_fusion','energy_equivariant',selection['selected_variant'])
    if selection['target_access'] or selection['target_score_used']:raise ValueError('Target feedback forbidden')
    count=len(variants)*len(SEEDS)
    rows=spec['rows']
    if len(rows)!=count or {(r['variant'],r['model_seed']) for r in rows}!={(v,s) for v in variants for s in SEEDS}:
        raise ValueError('Incomplete registered clean matrix;truth remains closed')
    if len({r['row_id'] for r in rows})!=count or len({r['output_root'] for r in rows})!=count:
        raise ValueError('Duplicate clean rows/outputs')
    return selection


def preflight_predictions(spec):
    selection=validate_matrix(spec);predictions=[];reference=None
    for row in spec['rows']:
        cfg=read(row['config']);root=Path(row['output_root'])
        reused=bool(row.get('reuse_from_run'))
        if reused:
            original=read(cfg['selection_file']);old_marker=read(root.parents[1]/'scoring_clean_complete.json')
            residual=row['reuse_from_run']==REUSED_RESIDUAL_RUN and row['variant']=='residual_fusion'
            baseline=row['reuse_from_run']==REUSED_BASELINE_RUN and row['variant'] in BASELINES
            energy=row['reuse_from_run']==REUSED_ENERGY_RUN and row['variant']=='energy_equivariant'
            coupled=row['reuse_from_run']==REUSED_COUPLED_RUN and row['variant']=='coupled_lag4'
            old_run=REUSED_COUPLED_RUN if coupled else REUSED_ENERGY_RUN if energy else REUSED_RESIDUAL_RUN if residual else REUSED_BASELINE_RUN
            models=7 if coupled else 6 if energy else 5 if residual else 4
            if (not (baseline or residual or energy or coupled) or root.parts[-3:]!=(old_run,row['row_id'],'prediction') or
                original['target_access'] or original['target_score_used'] or
                old_marker['status']!='SCORED_COMPLETE' or old_marker['rows']!=models*4 or old_marker['models']!=models or old_marker['seeds']!=4):
                raise ValueError('Unauthorized/incomplete frozen reuse;truth remains closed')
            if baseline and (original.get('scope')!='baseline_only' or original['status']!='FIXED_BASELINES_FROZEN' or original['test_variants']!=list(BASELINES) or original['model_seeds']!=sorted(SEEDS)):
                raise ValueError('Original fixed baseline selection changed')
            if energy and (original.get('scope')!='energy_source' or original['status']!='SOURCE_SELECTION_FROZEN' or original['selected_variant']!='energy_equivariant'):
                raise ValueError('Original energy source selection changed')
            if coupled and (original.get('scope')!='coupled_source' or original['status']!='SOURCE_SELECTION_FROZEN' or original['selected_variant']!='coupled_lag4' or original.get('new_candidate_selected') is not True):
                raise ValueError('Original coupled source selection changed')
            if residual and (original.get('scope') in ('baseline_only','balanced_source','interaction_source','attentive_source','stability_source','coherence_source','simplex_source','rf_operator_source','observable_source','gauge_source','reference_source','equivariant_source','synchronized_source','coordinate_source','additive_source','fractional_source','energy_source') or original['status']!='SOURCE_SELECTION_FROZEN' or original['selected_variant']!='residual_fusion'):
                raise ValueError('Original residual source selection changed')
        flag=read(root/'clean_complete.json');resolved=read(root/'resolved_config.json');provenance=read(root/'provenance.json')
        if (flag['status']!='PREDICTIONS_COMPLETE' or flag['truth_read'] is not False or flag['query_fit'] is not False or flag['views']!=['clean'] or
            cfg['variant']!=row['variant'] or cfg['model_seed']!=row['model_seed'] or cfg['output_root']!=row['output_root'] or
            (not reused and cfg['selection_file']!=spec['selection_file']) or cfg['views']!=['clean'] or resolved['truth_read'] is not False or resolved['query_fit'] is not False or
            any(resolved.get(k)!=v for k,v in cfg.items()) or provenance['status']!='VERIFIED' or provenance['query_fit'] is not False):
            raise ValueError('Incomplete/contaminated clean predictions;truth remains closed')
        capsule=Path(cfg['p1_capsule']);manifest=read(capsule/'manifest.json')
        with np.load(capsule/'index.npz',allow_pickle=False) as index:ids=index['ids'].copy()
        with np.load(root/'clean_predictions.npz',allow_pickle=False) as values:
            if set(values.files)!={'ids','clean'} or not np.array_equal(ids,values['ids']):raise ValueError('Clean identity/view mismatch')
            pred=values['clean'].copy()
        if (not len(ids) or len(ids)!=len(set(ids.tolist())) or flag['count']!=len(ids) or pred.shape!=(len(ids),) or pred.dtype.kind not in 'iu' or
            pred.min()<0 or pred.max()>=len(manifest['classes']) or manifest['status']!='VALIDATED_ONCE'):
            raise ValueError('Invalid clean predictions')
        if reference is None:reference=(ids,manifest['classes'])
        elif not np.array_equal(ids,reference[0]) or manifest['classes']!=reference[1]:raise ValueError('Methods evaluated different physical queries/classes')
        predictions.append((row,ids,pred))
    return selection,reference,predictions


def score(spec):
    root=Path(spec['runtime_root'])
    if (root/'scoring_clean_complete.json').exists():return read(root/'scoring_clean_complete.json')
    selection,reference,predictions=preflight_predictions(spec)
    # First truth access in this process occurs only after every frozen file passes.
    truth=read(spec['p1_truth']);ids,classes=reference
    targets=[truth[s] for s in ids.tolist()]
    if any(type(t['label']) is not int or t['label']<0 or t['label']>=len(classes) for t in targets):raise ValueError('Truth label/class mismatch')
    y=np.asarray([t['label'] for t in targets],dtype=np.int64)
    receivers=np.asarray([str(t['receiver']) for t in targets])
    results=[]
    for row,row_ids,pred in predictions:
        for receiver in ['ALL',*sorted(set(receivers))]:
            mask=np.ones(len(ids),dtype=bool) if receiver=='ALL' else receivers==receiver
            results.append(dict(row_id=row['row_id'],method=row['variant'],model_seed=row['model_seed'],view='clean',receiver=receiver,
                **metrics(y[mask],pred[mask],len(classes))))
    groups=defaultdict(list)
    for row in results:groups[(row['method'],row['receiver'])].append(row)
    summary=[]
    for (method,receiver),rows in groups.items():
        entry=dict(method=method,receiver=receiver,view='clean',model_seeds=sorted(SEEDS),query_count_per_seed=rows[0]['query_count'])
        for metric in ('accuracy','macro_f1','macro_accuracy'):
            values=[r[metric] for r in rows]
            entry[metric+'_mean']=float(np.mean(values));entry[metric+'_seed_sd']=float(np.std(values,ddof=1))
        summary.append(entry)
    lookup={(r['method'],r['receiver'],r['model_seed']):r for r in results}
    paired=[]
    candidate='native' if selection.get('scope')=='baseline_only' else selection['selected_variant']
    if selection.get('scope')=='volterra_source':
        pair_baselines=(*BASELINES,'residual_fusion','energy_equivariant','coupled_lag4')
    elif selection.get('scope') in ('crossphase_source','coupled_source'):
        pair_baselines=(*BASELINES,'residual_fusion','energy_equivariant')
    else:
        pair_baselines=(*BASELINES,'residual_fusion') if selection.get('scope') in ('balanced_source','interaction_source','attentive_source','stability_source','coherence_source','simplex_source','rf_operator_source','observable_source','gauge_source','reference_source','equivariant_source','synchronized_source','coordinate_source','additive_source','fractional_source','energy_source') else BASELINES
    for receiver in ['ALL',*sorted(set(receivers))]:
        for baseline in pair_baselines:
            if baseline==candidate:continue
            deltas=[100*(lookup[(candidate,receiver,s)]['accuracy']-lookup[(baseline,receiver,s)]['accuracy']) for s in sorted(SEEDS)]
            paired.append(dict(candidate=candidate,baseline=baseline,receiver=receiver,view='clean',model_seeds=sorted(SEEDS),
                accuracy_delta_pp_by_seed=deltas,accuracy_delta_pp_mean=float(np.mean(deltas)),accuracy_delta_pp_seed_sd=float(np.std(deltas,ddof=1)),positive_seeds=sum(x>0 for x in deltas)))
    write(root/'clean_scored_results.json',dict(status='SCORED',results=results,target_feedback_forbidden=True))
    csvwrite(root/'clean_scored_results.csv',[{k:v for k,v in r.items() if k!='confusion'} for r in results])
    write(root/'clean_summary.json',dict(summary=summary,paired=paired,selection_ref=spec['selection_file'],target_feedback_forbidden=True))
    csvwrite(root/'clean_summary.csv',summary)
    csvwrite(root/'clean_paired.csv',[{k:v for k,v in r.items() if k not in {'model_seeds','accuracy_delta_pp_by_seed'}} for r in paired])
    marker=dict(status='SCORED_COMPLETE',models=len({r['variant'] for r in spec['rows']}),seeds=4,rows=len(spec['rows']),records=len(results),view='clean',query_count=len(ids),target_feedback_forbidden=True)
    write(root/'scoring_clean_complete.json',marker)
    return marker


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();print(json.dumps(score(read(a.spec))))
