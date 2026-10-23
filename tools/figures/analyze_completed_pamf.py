"""Interpret the completed PAMF snapshot without new model or truth access."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as stats
from collect_completed_pamf import PROTOCOLS, SEEDS, SIX, VIEWS, save_csv, write


def rows(tables, name, **where):
    return [r for r in tables[name] if all(r.get(k)==v for k,v in where.items())]


def one(tables, name, **where):
    result=rows(tables,name,**where)
    if len(result)!=1: raise ValueError('Expected one reviewed row: '+repr((name,where,len(result))))
    return result[0]


def paired(records, dimensions, contrast_field, comparisons):
    index={tuple(r[k] for k in dimensions)+(r[contrast_field],int(r['seed'])):r['value'] for r in records}
    out=[]
    keys=sorted({tuple(r[k] for k in dimensions) for r in records})
    for key in keys:
        for effect,a,b in comparisons:
            if not all(key+(item,s) in index for item in (a,b) for s in SEEDS): continue
            values=[index[key+(a,s)]-index[key+(b,s)] for s in SEEDS]
            out.append(dict(zip(dimensions,key),effect=effect,mean=stats.mean(values),sd=stats.stdev(values),
                minimum=min(values),maximum=max(values),positive_seeds=sum(x>0 for x in values),n=3))
    return out


def analyze(source,output):
    data=json.loads(source.read_text(encoding='utf-8'));t=data['tables']
    common=dict(test='main',dimension='overall',stratum='ALL',metric='accuracy')
    def main(protocol='F0',regime='native_cric',readout='DURA-LU',view=SIX):
        return one(t,'main',protocol=protocol,regime=regime,readout=readout,view=view,**common)
    def effect(name,readout='DURA-LU',view=SIX):
        return [one(t,'main_effects',protocol=p,effect=name,readout=readout,view=view,**common) for p in PROTOCOLS]
    f0={view:main(view=view) for view in VIEWS+(SIX,)}
    raw={view:main(readout='raw',view=view) for view in VIEWS+(SIX,)}
    gain=effect('native_cric_G_minus_raw');clean=effect('native_cric_G_minus_raw',view='clean')
    comparisons={name:effect(name) for name in ('native_minus_supervised','native_minus_ordinaryPL')}
    facts=dict(final_F0={k:dict(mean=v['mean'],sd=v['sd']) for k,v in f0.items()},
        raw_F0={k:dict(mean=v['mean'],sd=v['sd']) for k,v in raw.items()},
        final_F0_gain=one(t,'main_effects',protocol='F0',effect='native_cric_G_minus_raw',
            readout='DURA-LU',view=SIX,**common),
        DURA_protocol_gain=dict(positive_mean=sum(r['mean']>0 for r in gain),total=len(gain),
            minimum=min(r['mean'] for r in gain),maximum=max(r['mean'] for r in gain)),
        DURA_clean_change=dict(nonpositive_mean=sum(r['mean']<=0 for r in clean),total=len(clean),
            worst=min(clean,key=lambda r:r['mean'])),
        native_comparisons={name:dict(nonpositive_mean=sum(r['mean']<=0 for r in rs),total=len(rs))
            for name,rs in comparisons.items()},
        low_urban_goal=dict(target_pct=70.0,actual_pct=f0['practical_low_urban']['mean'],
            shortfall_pp=70-f0['practical_low_urban']['mean']),
        five_other_channel_gaps=[dict(view=view,drop_from_raw_clean_pp=raw['clean']['mean']-f0[view]['mean'],
            within_five_pp=raw['clean']['mean']-f0[view]['mean']<=5) for view in VIEWS[1:-1]])
    physics_effects=paired(t['physics_seed'],('readout','test','view','metric'),'architecture',
        [(a,'X3_CE',a) for a in sorted({r['architecture'] for r in t['physics_seed']}) if a!='X3_CE'])
    external_effects=paired(t['external_full_seed'],('protocol','family','regime','variant','test','view','metric'),
        'readout',[('G-L_minus_raw','DURA-L','raw'),('G-LU_minus_raw','DURA-LU','raw'),
                   ('G-LU_minus_G-L','DURA-LU','DURA-L')])
    external_f0=rows(t,'external_full',protocol='F0',test='main',readout='DURA-LU',
        view=SIX,metric='accuracy')
    best=max(external_f0,key=lambda r:r['mean'])
    facts['external_F0_best_full']=best
    facts['native_vs_best_external_F0_GLU_pp']=f0[SIX]['mean']-best['mean']
    facts['external_Full_DURA']={}
    for effect_name in ('G-L_minus_raw','G-LU_minus_raw','G-LU_minus_G-L'):
        rs=[r for r in external_effects if r['effect']==effect_name and r['test']=='main'
            and r['view']==SIX and r['metric']=='accuracy']
        facts['external_Full_DURA'][effect_name]=dict(positive_mean=sum(r['mean']>0 for r in rs),
            total=len(rs),minimum=min(r['mean'] for r in rs),maximum=max(r['mean'] for r in rs))
    facts['fairness']={}
    for effect_name in sorted({r['effect'] for r in t['gfactor_effects']}):
        rs=rows(t,'gfactor_effects',effect=effect_name,test='main',view=SIX,metric='accuracy')
        facts['fairness'][effect_name]=dict(positive_mean=sum(r['mean']>0 for r in rs),total=len(rs),
            minimum=min(r['mean'] for r in rs),maximum=max(r['mean'] for r in rs),
            F0=one(t,'gfactor_effects',effect=effect_name,protocol_id='F0',test='main',view=SIX,metric='accuracy'))
    # Keep F0 architectures and recipe comparisons separate; they have different CE doses.
    facts['physics_effects']= [r for r in physics_effects if r['metric']=='accuracy'
        and r['view'] in ('clean',SIX)]
    facts['physics_recipe_scope']='All nine structures use matched L-only CE+sat.68/LS0, raw or fixed G-L; X3_CE is F15, not final T1.'
    facts['core']=rows(t,'core_effects',test='main',view=SIX,metric='accuracy')
    facts['core_clean']=rows(t,'core_effects',test='main',view='clean',metric='accuracy')
    facts['CRIC_G']=rows(t,'cric_G',test='main',view=SIX,metric='accuracy')
    facts['crossed']=rows(t,'crossed',test='main',view=SIX,metric='accuracy')
    facts['costs']={
        'main_new_F_shared_worker_hours':sum(float(r['shared_worker_seconds']) for r in t['main_F_cost'] if r['new_F']=='True')/3600,
        'main_logical_F_shared_worker_hours':sum(float(r['shared_worker_seconds']) for r in t['main_F_cost'])/3600,
        'G_35_lineage_shared_worker_hours':sum(float(r['total_seconds']) for r in t['G_lineage_cost'])/3600,
        'fine_63_new_F_shared_worker_hours':sum(float(r['shared_worker_seconds']) for r in t['fine_F_cost'] if r['newly_trained_for_fine_panel']=='True')/3600,
        'core_12_new_F_shared_worker_hours':sum(float(r['shared_worker_seconds']) for r in t['core_F_cost'] if r['new_F']=='True')/3600,
        'boundary':'Study scopes overlap through reuse; do not sum logical study totals. Elapsed is shared-worker wall time, not exclusive GPU time or added delivery wall time.'}
    facts['timing']={}
    for name,case in [('unified_timing','T4_graph_reuse_GF'),('fine_timing','T4_graph_reuse_GF'),
                       ('structure_timing','X3_CE_GF')]:
        rs=[r for r in t[name] if r['case']==case and int(r['batch'])==1]
        if len(rs)!=1:raise ValueError('Missing current timing control')
        facts['timing'][name]=rs[0]
    output.mkdir(exist_ok=False)
    save_csv(output/'physics_paired_effects.csv',physics_effects)
    save_csv(output/'external_full_paired_effects.csv',external_effects)
    write(output/'conclusions_data.json',facts)
    print(json.dumps({k:facts[k] for k in ('DURA_protocol_gain','DURA_clean_change','native_comparisons',
        'external_Full_DURA','native_vs_best_external_F0_GLU_pp','low_urban_goal','costs')},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();analyze(a.source,a.output)
