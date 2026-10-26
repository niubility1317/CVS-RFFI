"""Recompute REPL contrasts from frozen public aggregate tables; no model access."""
import csv
import json
import statistics
from pathlib import Path


def analyze(root):
    root=Path(root)
    source=root/'exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data'
    def read(name):
        with (source/name).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
    paired=read('paired_effects.csv');summary=read('summary_overall.csv')
    selected=[r for r in paired if r['effect']=='REPL_minus_ordinary_PL' and r['test']=='cross_rx'
              and r['metric']=='accuracy' and r['readout'] in ('raw','DURA-LU')
              and r['view'] in ('clean','six_practical_mean')]
    assert len(selected)==44 and len({(r['protocol'],r['readout'],r['view']) for r in selected})==44
    main=[]
    for row in selected:
        values=[float(v) for v in row['seed_values'].split(';')]
        assert len(values)==3
        assert abs(statistics.mean(values)-float(row['mean']))<1e-10
        assert abs(statistics.stdev(values)-float(row['sd']))<1e-10
        assert sum(v>0 for v in values)==int(row['positive_seeds'])
        main.append(dict(protocol=row['protocol'],readout=row['readout'],view=row['view'],
                         difference_pp=float(row['mean']),paired_sd_pp=float(row['sd']),positive_seeds=int(row['positive_seeds']),
                         seed01_pp=values[0],seed02_pp=values[1],seed03_pp=values[2]))
    def endpoint(method,readout,view):
        rows=[r for r in summary if r['protocol']=='F0' and r['method']==method and r['test']=='cross_rx'
              and r['readout']==readout and r['view']==view and r['metric']=='accuracy'
              and r['dimension']=='overall' and r['stratum']=='ALL']
        assert len(rows)==1
        return float(rows[0]['mean'])
    ablations=[]
    for method in ('REPL','without_style','without_extrapolation','without_joint','without_guard'):
        for readout in ('raw','DURA-LU'):
            for view in ('clean','six_practical_mean'):
                value=endpoint(method,readout,view)
                ablations.append(dict(method=method,readout=readout,view=view,accuracy_percent=value,
                    delta_vs_REPL_pp=value-endpoint('REPL',readout,view),delta_vs_PL_pp=value-endpoint('ordinary_PL',readout,view)))
    geometry=[];interventions=[]
    original=read('source_geometry.csv');raw=read('source_interventions.csv')
    for method in ('ordinary_PL','REPL'):
        rows=[r for r in original if r['method']==method];assert len(rows)==3
        geometry.append(dict(method=method,**{k:statistics.mean(float(r[k]) for r in rows)
            for k in ('same_TX_cross_RX_distance','different_TX_same_RX_distance','separation_margin')}))
        for operation in sorted({r['intervention'] for r in raw}):
            rows=[r for r in raw if r['method']==method and r['intervention']==operation];assert len(rows)==3
            interventions.append(dict(method=method,intervention=operation,**{k:statistics.mean(float(r[k]) for r in rows)
                for k in ('feature_cosine_distance','default_head_flip_fraction')}))
    counts={readout+'_'+view:sum(r['difference_pp']>0 for r in main if r['readout']==readout and r['view']==view)
            for readout in ('raw','DURA-LU') for view in ('clean','six_practical_mean')}
    assert counts=={'raw_clean':7,'raw_six_practical_mean':0,'DURA-LU_clean':7,'DURA-LU_six_practical_mean':3}
    exposure=read('training_exposures.csv')
    one_rx=[r for r in exposure if r['protocol']=='m1-natural' and r['method']=='REPL'];assert len(one_rx)==3
    assert all(int(r['style_shallow_applied'])==int(r['style_middle_applied'])==0 for r in one_rx)
    result=dict(status='VERIFIED_FROZEN_AGGREGATES',source_root=str(source.relative_to(root)),
        endpoint='cross_rx only; each protocol retains own physical contract; no pooled grand accuracy',
        model_seeds=[2026092701,2026092702,2026092703],main=main,positive_settings=counts,ablations=ablations,
        source_geometry=geometry,source_interventions=interventions,one_source_RX_REPL_exposures=one_rx,
        raw_truth_reads=0,new_model_forwards=0,new_predictions=0,
        limitations=['Posthoc F0 clean method selection; not independent blind selection.',
                    'Three F seeds; fixed matched G within each contract. No G-seed inference.',
                    'Conditional ablations of full recipe do not identify every interaction.',
                    'Source geometry is diagnostic, not a causal decomposition of target errors.'])
    folder=root/'automation_reports/CV-SincNet/20261026-analysis-PAMF-SSL-retrospective-r01'
    for name,rows in [('repl_main_contrasts.csv',main),('repl_component_contrasts.csv',ablations),
                      ('repl_source_geometry.csv',geometry),('repl_source_interventions.csv',interventions)]:
        with (folder/name).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (folder/'repl_regression_evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='E:/type10-7');args=parser.parse_args()
    result=analyze(args.root)
    print(json.dumps(dict(status=result['status'],contrasts=len(result['main']),positive_settings=result['positive_settings'],
        one_RX_style_applied=0,one_RX_guard_conflicts=[int(r['gradient_guard_conflicts']) for r in result['one_source_RX_REPL_exposures']]),ensure_ascii=False))
